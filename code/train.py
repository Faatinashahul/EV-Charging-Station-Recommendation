"""
train.py
--------
Training loop, validation, loss computation, checkpointing,
and early stopping for the STGAT model.

The neighborhood consistency loss is computed and applied here —
this is the only file that knows about it. The model itself just
outputs sigmoid predictions; the constraint is a training-time signal.

Public API:
    train(model, train_loader, val_loader, bundle, cfg) → TrainResult
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import List
from tqdm import tqdm

from config import Config


# ---------------------------------------------------------------------------
# TrainResult — returned by train(), consumed by main.py and evaluate.py
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """
    Everything produced by the training run.

    train_losses : MSE loss per epoch on training set
    val_losses   : total loss (MSE + consistency) per epoch on validation set
    best_epoch   : epoch number where best val loss was achieved
    best_val_loss: the actual best val loss value
    duration_s   : total wall-clock training time in seconds
    stopped_early: True if early stopping triggered before n_epochs
    """
    train_losses:  List[float]
    val_losses:    List[float]
    best_epoch:    int
    best_val_loss: float
    duration_s:    float
    stopped_early: bool


# ---------------------------------------------------------------------------
# Consistency loss
# ---------------------------------------------------------------------------

def consistency_loss(
    predictions: torch.Tensor,   # (B, N)  sigmoid outputs
    adj_norm:    torch.Tensor,   # (N, N)  row-normalized adjacency
) -> torch.Tensor:
    """
    Neighborhood consistency loss.

    Penalizes each node for deviating too far from its neighbors'
    weighted average prediction. Encourages spatial coherence —
    adjacent zones should have reasonably similar occupancy forecasts
    since EV drivers redistribute demand across nearby stations.

    Formula:
        neighbor_avg[i] = sum_j( adj_norm[i,j] * pred[j] )
                        = (adj_norm @ pred)[i]

        L_consistency = mean( (pred - neighbor_avg)^2 )

    Properties:
        - Zero when every node's prediction exactly matches its
          neighbor-weighted average (perfect spatial smoothness).
        - Scale-free: predictions and neighbor_avg are both in (0,1)
          so the squared difference is also bounded in [0,1].
        - Does not force all nodes to the same value — only penalizes
          deviations from the local neighborhood average, not the global mean.
        - lambda_consistency=0.0 in TrainConfig disables this entirely.

    Args:
        predictions : (B, N) — sigmoid outputs from the model
        adj_norm    : (N, N) — row-normalized adjacency, from DataBundle.adj_norm
                               each row sums to 1 (or 0 for isolated nodes)

    Returns:
        scalar loss tensor
    """
    # neighbor_avg[b, i] = weighted average of neighbors of node i in batch b
    # adj_norm: (N, N), predictions: (B, N)
    # predictions.T: (N, B) → adj_norm @ predictions.T: (N, B) → .T: (B, N)
    neighbor_avg = torch.matmul(adj_norm, predictions.T).T   # (B, N)

    loss = torch.mean((predictions - neighbor_avg) ** 2)
    return loss

# Introduced to improve MAPE
def mape_aware_loss(
    predictions: torch.Tensor,   # (B, N)
    labels:      torch.Tensor,   # (B, N)
    eps:         float = 0.1,
    alpha:       float = 0.5,
) -> torch.Tensor:
    """
    Combines MSE with a relative error term that mimics MAPE.
    
    L = (1 - alpha) * MSE  +  alpha * mean(|pred - real| / (real + eps))

    The relative term directly penalizes percentage errors, training
    the model to be more accurate on low-occupancy zones.

    eps: smoothing to prevent division by zero on empty zones.
         0.1 means zones below 10% occupancy get a lighter penalty,
         preventing the loss from exploding on truly zero zones.
    alpha: blend weight. 0.5 = equal MSE and relative. 
           Increase toward 1.0 to prioritize MAPE reduction further.
    """
    mse_term      = torch.mean((predictions - labels) ** 2)
    relative_term = torch.mean(
        torch.abs(predictions - labels) / (labels + eps)
    )
    return (1 - alpha) * mse_term + alpha * relative_term


# ---------------------------------------------------------------------------
# Single epoch functions
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    mse_loss:     nn.Module,
    adj_norm:     torch.Tensor,
    lambda_c:     float,
    device:       torch.device,
    mape_eps:     float = 0.1,
    mape_alpha:   float = 0.5,
) -> float:
    """
    Runs one full pass over the training DataLoader.

    Loss:
        L_total = MSE(pred, label) + lambda_c * L_consistency

    The consistency loss is only computed when lambda_c > 0,
    avoiding unnecessary computation when it's disabled.

    Returns:
        mean total loss over all batches in this epoch
    """
    model.train()
    total_loss  = 0.0
    n_batches   = 0

    for occ, prc, label in loader:
        # Move batch to device
        occ   = occ.to(device)
        prc   = prc.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        predictions = model(occ, prc)                        # (B, N)

        # MSE reconstruction loss
        #loss = mse_loss(predictions, label)

        # NEW
        
        loss = mape_aware_loss(predictions, label, eps=mape_eps, alpha=mape_alpha)

        # Spatial consistency loss (training only)
        if lambda_c > 0.0:
            loss = loss + lambda_c * consistency_loss(predictions, adj_norm)

        loss.backward()

        # Gradient clipping — prevents occasional large gradient spikes
        # that can destabilize training with graph attention networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(
    model:    nn.Module,
    loader:   DataLoader,
    mse_loss: nn.Module,
    adj_norm: torch.Tensor,
    lambda_c: float,
    device:   torch.device,
) -> float:
    """
    Evaluates the model on the validation set.

    Uses the same total loss as training (MSE + consistency) so that
    the val curve is directly comparable to the train curve and
    checkpointing is consistent with what the optimizer minimizes.

    @torch.no_grad() decorator disables gradient tracking for the
    entire function — faster and uses less memory than wrapping
    individual calls.

    Returns:
        total validation loss (scalar)
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for occ, prc, label in loader:
        occ   = occ.to(device)
        prc   = prc.to(device)
        label = label.to(device)

        predictions = model(occ, prc)                        # (B, N)

        loss = mse_loss(predictions, label)
        if lambda_c > 0.0:
            loss = loss + lambda_c * consistency_loss(predictions, adj_norm)

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------

class EarlyStopper:
    """
    Tracks validation loss and signals when training should stop.

    Stops training if val loss hasn't improved by more than `min_delta`
    for `patience` consecutive epochs.

    min_delta: minimum improvement to count as a genuine improvement.
               Prevents stopping due to tiny numerical fluctuations.
               Default 0.0 means any improvement counts.
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0

    def should_stop(self, val_loss: float) -> bool:
        """
        Returns True if training should stop.
        Call once per epoch after computing val loss.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    adj_norm:     torch.Tensor,
    cfg:          Config,
) -> TrainResult:
    """
    Full training loop. Call this from main.py.

    Flow per epoch:
        1. Train one epoch → train loss
        2. Validate         → val loss
        3. Checkpoint if val loss improved
        4. Step LR scheduler
        5. Check early stopping

    Progress display:
        Outer tqdm bar: one row per epoch showing live train/val loss.
        No inner per-batch bar — keeps the terminal clean.
        At the end of training, a summary is printed.

    Args:
        model        : STGAT instance (already on device)
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        adj_norm     : (N, N) row-normalized adjacency from DataBundle
        cfg          : Config object

    Returns:
        TrainResult with loss curves, best epoch, and timing
    """
    device   = cfg.system.device
    tc       = cfg.train

    # Move adj_norm to device once — reused every batch
    adj_norm = adj_norm.to(device)

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = tc.lr,
        weight_decay = tc.weight_decay,
    )

    # ---- Loss function -----------------------------------------------------
    mse_loss = nn.MSELoss()

    # ---- LR Scheduler ------------------------------------------------------
    scheduler = None
    if tc.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode     = "min",
            factor   = tc.lr_scheduler_factor,
            patience = tc.lr_patience,
        )

    # ---- Early stopping ----------------------------------------------------
    stopper = EarlyStopper(patience=tc.patience) if tc.early_stopping else None

    # ---- State tracking ----------------------------------------------------
    train_losses:  List[float] = []
    val_losses:    List[float] = []
    best_val_loss: float       = float("inf")
    best_epoch:    int         = 0
    stopped_early: bool        = False
    start_time:    float       = time.time()

    checkpoint_path = cfg.paths.checkpoint_path("best")

    print(f"\n[train] Starting training for up to {tc.n_epochs} epochs")
    print(f"[train] Checkpointing to: {checkpoint_path}")
    if tc.early_stopping:
        print(f"[train] Early stopping enabled (patience={tc.patience})")
    if tc.use_lr_scheduler:
        print(f"[train] LR scheduler: ReduceLROnPlateau "
              f"(factor={tc.lr_scheduler_factor}, patience={tc.lr_patience})")
    print()

    # ---- Training loop -----------------------------------------------------
    pbar = tqdm(
        range(1, tc.n_epochs + 1),
        desc      = "Training",
        unit      = "epoch",
        dynamic_ncols = True,
    )

    for epoch in pbar:

        # Train
        t_loss = train_one_epoch(
        model, train_loader, optimizer,
        mse_loss, adj_norm, tc.lambda_consistency, device,
        mape_eps=tc.mape_loss_eps, mape_alpha=tc.mape_loss_alpha,
    )

        # Validate
        v_loss = validate(
            model, val_loader,
            mse_loss, adj_norm, tc.lambda_consistency, device
        )

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        # Update tqdm postfix with live loss values
        pbar.set_postfix({
            "train": f"{t_loss:.5f}",
            "val"  : f"{v_loss:.5f}",
            "best" : f"{best_val_loss:.5f}",
            "lr"   : f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        # Checkpoint if val loss improved
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_epoch    = epoch
            torch.save(model.state_dict(), checkpoint_path)

        # Optionally save last epoch checkpoint
        if tc.save_last:
            torch.save(
                model.state_dict(),
                cfg.paths.checkpoint_path("last")
            )

        # LR scheduler step
        if scheduler is not None:
            scheduler.step(v_loss)

        # Early stopping check
        if stopper is not None and stopper.should_stop(v_loss):
            stopped_early = True
            pbar.write(
                f"\n[train] Early stopping at epoch {epoch} "
                f"(no improvement for {tc.patience} epochs)"
            )
            break

    # ---- Summary -----------------------------------------------------------
    duration = time.time() - start_time

    print(f"\n[train] Finished.")
    print(f"  Best val loss : {best_val_loss:.6f}  (epoch {best_epoch})")
    print(f"  Total time    : {duration:.1f}s  "
          f"({duration/max(len(train_losses),1):.2f}s/epoch)")
    if stopped_early:
        print(f"  Stopped early : yes (after {len(train_losses)} epochs)")
    else:
        print(f"  Stopped early : no  (ran full {tc.n_epochs} epochs)")
    print(f"  Checkpoint    : {checkpoint_path}")

    return TrainResult(
        train_losses  = train_losses,
        val_losses    = val_losses,
        best_epoch    = best_epoch,
        best_val_loss = best_val_loss,
        duration_s    = duration,
        stopped_early = stopped_early,
    )


# ---------------------------------------------------------------------------
# Self-test — run directly: python train.py
# Uses random data and a tiny model — no dataset required.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from torch.utils.data import TensorDataset, DataLoader as DL
    from config import Config, TrainConfig, validate_config
    from model import build_model

    print("Running train.py self-test...\n")

    # ---- Minimal config override for fast self-test ------------------------
    cfg = Config(
        train=TrainConfig(
            n_epochs            = 5,
            batch_size          = 8,
            lr                  = 1e-3,
            lambda_consistency  = 0.1,
            use_lr_scheduler    = False,
            early_stopping      = False,
        )
    )
    validate_config(cfg)

    device = cfg.system.device
    N = cfg.model.n_nodes   # 247
    S = cfg.data.seq_len    # 12
    B = 8

    # ---- Random adjacency (sparse, ~2% density) ----------------------------
    torch.manual_seed(42)
    adj_dense  = (torch.rand(N, N) < 0.02).float()
    adj_dense  = ((adj_dense + adj_dense.T) > 0).float()
    adj_dense.fill_diagonal_(0)
    adj_sparse = adj_dense.to_sparse_coo()

    # Row-normalized adj (same as data.py _row_normalize_adj)
    degree   = adj_dense.sum(dim=1, keepdim=True).clamp(min=1e-8)
    adj_norm = adj_dense / degree

    # ---- Random dataset (4 batches of 8 samples) ---------------------------
    n_samples = 32
    occ_t   = torch.rand(n_samples, N, S)
    prc_t   = torch.rand(n_samples, N, S)
    label_t = torch.rand(n_samples, N)

    ds     = TensorDataset(occ_t, prc_t, label_t)
    loader = DL(ds, batch_size=B, shuffle=True, drop_last=True)

    # ---- Build model -------------------------------------------------------
    model = build_model(cfg, adj_sparse)

    # ---- Run training ------------------------------------------------------
    result = train(
        model        = model,
        train_loader = loader,
        val_loader   = loader,   # reuse train as val for self-test
        adj_norm     = adj_norm,
        cfg          = cfg,
    )

    # ---- Verify result object ----------------------------------------------
    assert len(result.train_losses) == 5,       "Wrong number of epochs recorded"
    assert len(result.val_losses)   == 5,       "Wrong number of val losses recorded"
    assert result.best_epoch >= 1,              "best_epoch not set"
    assert result.best_val_loss < float("inf"), "best_val_loss never updated"
    assert result.duration_s > 0,              "Duration not recorded"

    # Verify checkpoint was saved
    import os
    assert os.path.exists(cfg.paths.checkpoint_path("best")), \
        "Best checkpoint was not saved"

    # Verify loss decreased or stayed reasonable (not NaN/Inf)
    assert all(not torch.isnan(torch.tensor(l)) for l in result.train_losses), \
        "NaN in train losses"
    assert all(not torch.isinf(torch.tensor(l)) for l in result.train_losses), \
        "Inf in train losses"

    print("\n  TrainResult:")
    print(f"    train_losses  : {[round(l,5) for l in result.train_losses]}")
    print(f"    val_losses    : {[round(l,5) for l in result.val_losses]}")
    print(f"    best_epoch    : {result.best_epoch}")
    print(f"    best_val_loss : {result.best_val_loss:.6f}")
    print(f"    duration_s    : {result.duration_s:.2f}s")
    print(f"    stopped_early : {result.stopped_early}")
    print(f"\n  Checkpoint exists : "
          f"{os.path.exists(cfg.paths.checkpoint_path('best'))}")
    print("\ntrain.py self-test complete.")
