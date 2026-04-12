"""
evaluate.py
-----------
Inference, metric computation, results export, and optional
per-node error analysis for the trained STGAT model.

This file is strictly test-time — no gradients, no loss backward,
no consistency penalty. The model outputs raw sigmoid predictions
which are compared directly against ground truth labels.

Public API:
    evaluate(model, test_loader, cfg)          → EvalResult
    save_results(result, train_result, cfg)    → writes CSV to results/
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List
from tqdm import tqdm

from config import Config
from train import TrainResult


# ---------------------------------------------------------------------------
# EvalResult — returned by evaluate(), consumed by main.py
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    All outputs from a test-set evaluation run.

    predictions : (T_test, N)  model outputs, sigmoid-bounded in (0, 1)
    labels      : (T_test, N)  ground truth occupancy values
    metrics     : dict of scalar metric values (MSE, RMSE, MAE, MAPE, R2, RAE)
    node_mae    : (N,)  per-node MAE — identifies hardest zones to predict
    node_r2     : (N,)  per-node R²  — identifies zones where model fits well
    """
    predictions : np.ndarray
    labels      : np.ndarray
    metrics     : dict
    node_mae    : np.ndarray
    node_r2     : np.ndarray


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def _mse(pred: np.ndarray, real: np.ndarray) -> float:
    return float(np.mean((pred - real) ** 2))


def _rmse(pred: np.ndarray, real: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - real) ** 2)))


def _mae(pred: np.ndarray, real: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - real)))


def _mape(pred: np.ndarray, real: np.ndarray, eps: float = 0.01) -> float:
    """
    Mean Absolute Percentage Error.

    MAPE is undefined when real values are zero (division by zero).
    We add a small epsilon (0.01) to real values that are exactly zero —
    matching the original paper's approach. This affects a tiny fraction
    of samples (charging zones that are completely empty).

    eps=0.01 means a zero-occupancy zone is treated as 1% occupied
    for the purpose of percentage error calculation.
    """
    real_safe = real.copy()
    pred_safe = pred.copy()
    zero_mask = real_safe == 0
    real_safe[zero_mask] += eps
    pred_safe[zero_mask] += eps
    return float(np.mean(np.abs((pred_safe - real_safe) / real_safe)))


def _r2(pred: np.ndarray, real: np.ndarray) -> float:
    """
    R² (coefficient of determination).

    R² = 1 - SS_res / SS_tot
    Range: (-inf, 1]. Higher is better.
        1.0  = perfect prediction
        0.0  = model predicts the mean (no better than baseline)
        <0.0 = model is worse than predicting the mean

    SS_tot can be zero if all real values are identical (constant signal).
    We return 0.0 in that degenerate case.
    """
    ss_res = np.sum((real - pred) ** 2)
    ss_tot = np.sum((real - np.mean(real)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _rae(pred: np.ndarray, real: np.ndarray) -> float:
    """
    Relative Absolute Error.

    RAE = sum|pred - real| / sum|mean(real) - real|

    Measures error relative to a naive mean-predictor baseline.
    RAE < 1 means the model outperforms the mean baseline.
    RAE = 1 means the model is no better than predicting the mean.
    """
    numerator   = np.sum(np.abs(pred - real))
    denominator = np.sum(np.abs(np.mean(real) - real))
    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


def compute_metrics(
    predictions: np.ndarray,   # (T, N) or (T*N,) flattened
    labels:      np.ndarray,   # (T, N) or (T*N,) flattened
) -> dict:
    """
    Computes all metrics on flattened arrays.

    Flattening treats every (timestep, node) pair as one prediction,
    which gives a single scalar per metric representing overall model
    performance across all zones and all timesteps simultaneously.
    This matches the evaluation methodology of the original paper.

    Returns a dict with keys:
        MSE, RMSE, MAE, MAPE, R2, RAE
    All values are floats. MAPE is expressed as a fraction (not %).
    The summary() method of EvalResult prints them as percentages.
    """
    pred = predictions.flatten()
    real = labels.flatten()

    return {
        "MSE" : _mse(pred, real),
        "RMSE": _rmse(pred, real),
        "MAE" : _mae(pred, real),
        "MAPE": _mape(pred, real),
        "R2"  : _r2(pred, real),
        "RAE" : _rae(pred, real),
    }


def compute_node_metrics(
    predictions: np.ndarray,   # (T, N)
    labels:      np.ndarray,   # (T, N)
) -> tuple:
    """
    Computes MAE and R² independently for each of the N nodes.

    Returns:
        node_mae : (N,)  per-node MAE
        node_r2  : (N,)  per-node R²

    Use these to identify:
        - Hardest zones: highest node_mae
        - Best-fit zones: highest node_r2
        - Poorly-fit zones: lowest node_r2 or negative R²
    """
    N = predictions.shape[1]
    node_mae = np.array([_mae(predictions[:, i], labels[:, i]) for i in range(N)])
    node_r2  = np.array([_r2 (predictions[:, i], labels[:, i]) for i in range(N)])
    return node_mae, node_r2


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """
    Prints a formatted metric summary to stdout.
    Multiply-by-100 scaling matches the original paper's display convention.
    """
    scale = 100
    pad   = f"  {prefix}" if prefix else "  "
    print(f"{pad}MSE  : {metrics['MSE']  * scale:.4f} ×10⁻²")
    print(f"{pad}RMSE : {metrics['RMSE'] * scale:.4f} ×10⁻²")
    print(f"{pad}MAE  : {metrics['MAE']  * scale:.4f} ×10⁻²")
    print(f"{pad}MAPE : {metrics['MAPE'] * scale:.4f} %")
    print(f"{pad}R²   : {metrics['R2']   * scale:.4f} %")
    print(f"{pad}RAE  : {metrics['RAE']  * scale:.4f} %")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """
    Runs the model over the entire test DataLoader and collects
    predictions and labels as numpy arrays.

    Returns:
        predictions : (T_test, N)  float32 numpy array
        labels      : (T_test, N)  float32 numpy array

    The first dummy row used in the original code for concatenation
    is not needed here — we collect into lists and stack once,
    which is cleaner and avoids the off-by-one slice.
    """
    model.eval()
    pred_list  = []
    label_list = []

    for occ, prc, label in tqdm(loader, desc="Inference", leave=False):
        occ   = occ.to(device)
        prc   = prc.to(device)

        pred = model(occ, prc)                      # (B, N)

        pred_list.append(pred.cpu().numpy())
        label_list.append(label.numpy())            # label already on CPU

    predictions = np.concatenate(pred_list,  axis=0)   # (T_test, N)
    labels      = np.concatenate(label_list, axis=0)   # (T_test, N)

    return predictions, labels


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    model:       nn.Module,
    test_loader: torch.utils.data.DataLoader,
    cfg:         Config,
) -> EvalResult:
    """
    Runs full test-set evaluation and returns an EvalResult.

    Steps:
        1. Load best checkpoint (lowest val loss from training)
        2. Run inference over test set
        3. Compute global metrics (flattened over all nodes and timesteps)
        4. Compute per-node metrics (MAE and R² per zone)
        5. Print summary
        6. Return EvalResult

    The best checkpoint is always loaded here — even if called immediately
    after train(), the model weights in memory may be from the last epoch,
    not the best epoch. Loading the checkpoint guarantees we evaluate the
    correct weights.

    Args:
        model       : STGAT instance (architecture must match checkpoint)
        test_loader : test DataLoader from get_loaders()
        cfg         : Config object

    Returns:
        EvalResult with predictions, labels, and all metrics
    """
    device          = cfg.system.device
    checkpoint_path = cfg.paths.checkpoint_path("best")

    # ---- Load best checkpoint ----------------------------------------------
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at: {checkpoint_path}\n"
            f"Run training before evaluation."
        )

    print(f"\n[evaluate] Loading best checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("[evaluate] Checkpoint loaded.")

    # ---- Inference ---------------------------------------------------------
    print("[evaluate] Running inference on test set...")
    predictions, labels = run_inference(model, test_loader, device)
    print(f"[evaluate] Done. predictions: {predictions.shape}, "
          f"labels: {labels.shape}")

    # ---- Global metrics ----------------------------------------------------
    metrics  = compute_metrics(predictions, labels)

    # ---- Per-node metrics --------------------------------------------------
    node_mae, node_r2 = compute_node_metrics(predictions, labels)

    # ---- Print summary -----------------------------------------------------
    print("\n[evaluate] Test set metrics:")
    print_metrics(metrics)

    # Per-node summary
    top_k = 5
    worst_nodes = np.argsort(node_mae)[-top_k:][::-1]
    best_nodes  = np.argsort(node_r2) [-top_k:][::-1]

    print(f"\n[evaluate] Per-node analysis ({predictions.shape[1]} nodes total):")
    print(f"  Mean node MAE : {node_mae.mean():.6f}")
    print(f"  Std  node MAE : {node_mae.std():.6f}")
    print(f"  Mean node R²  : {node_r2.mean():.6f}")
    print(f"  Nodes with R² < 0 (worse than mean predictor): "
          f"{(node_r2 < 0).sum()}")
    print(f"\n  Hardest zones to predict (highest MAE):")
    for rank, node_idx in enumerate(worst_nodes, 1):
        print(f"    {rank}. Node {node_idx:3d} — MAE={node_mae[node_idx]:.6f}, "
              f"R²={node_r2[node_idx]:.6f}")
    print(f"\n  Best-fit zones (highest R²):")
    for rank, node_idx in enumerate(best_nodes, 1):
        print(f"    {rank}. Node {node_idx:3d} — MAE={node_mae[node_idx]:.6f}, "
              f"R²={node_r2[node_idx]:.6f}")


    # ✅ ADD HERE
    #latest_pred = predictions[-1]
    latest_pred = predictions.mean(axis=0)
    
    df = pd.DataFrame({
        "station_id": list(range(len(latest_pred))),
        "predicted_demand": latest_pred
    })
    
    df.to_csv("results/predictions.csv", index=False)
    
    print("[evaluate] Saved predictions for recommendation → results/predictions.csv")



    return EvalResult(
        predictions = predictions,
        labels      = labels,
        metrics     = metrics,
        node_mae    = node_mae,
        node_r2     = node_r2,
    )


# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------

def save_results(
    result:       EvalResult,
    cfg:          Config,
    train_result: Optional[TrainResult] = None,
) -> None:
    """
    Saves evaluation metrics (and optionally training info) to a CSV.

    Output columns:
        MSE, RMSE, MAE, MAPE, R2, RAE
        + if train_result provided:
          best_epoch, best_val_loss, n_epochs_run, duration_s, stopped_early

    File is saved to cfg.paths.results_path("test").

    Args:
        result       : EvalResult from evaluate()
        cfg          : Config object
        train_result : optional TrainResult from train() — adds training info
    """
    row = {k: v for k, v in result.metrics.items()}

    if train_result is not None:
        row["best_epoch"]    = train_result.best_epoch
        row["best_val_loss"] = train_result.best_val_loss
        row["n_epochs_run"]  = len(train_result.train_losses)
        row["duration_s"]    = round(train_result.duration_s, 2)
        row["stopped_early"] = train_result.stopped_early

    # Also save key config values for experiment reproducibility
    row["seq_len"]           = cfg.data.seq_len
    row["pred_len"]          = cfg.data.pred_len
    row["lambda_consistency"]= cfg.train.lambda_consistency
    row["lr"]                = cfg.train.lr
    row["n_nodes"]           = cfg.model.n_nodes

    df   = pd.DataFrame([row])
    path = cfg.paths.results_path("test")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"\n[evaluate] Results saved to: {path}")


# ---------------------------------------------------------------------------
# Self-test — run directly: python evaluate.py
# Requires train.py self-test to have run first (needs the checkpoint).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader as DL
    from config import Config, TrainConfig, validate_config
    from model import build_model
    from train import train

    print("Running evaluate.py self-test...\n")

    # ---- Config: fast run --------------------------------------------------
    cfg = Config(
        train=TrainConfig(
            n_epochs           = 5,
            batch_size         = 8,
            lr                 = 1e-3,
            lambda_consistency = 0.1,
            use_lr_scheduler   = False,
            early_stopping     = False,
        )
    )
    validate_config(cfg)

    device = cfg.system.device
    N = cfg.model.n_nodes
    S = cfg.data.seq_len

    # ---- Random graph ------------------------------------------------------
    torch.manual_seed(42)
    adj_dense  = (torch.rand(N, N) < 0.02).float()
    adj_dense  = ((adj_dense + adj_dense.T) > 0).float()
    adj_dense.fill_diagonal_(0)
    adj_sparse = adj_dense.to_sparse_coo()
    degree     = adj_dense.sum(dim=1, keepdim=True).clamp(min=1e-8)
    adj_norm   = adj_dense / degree

    # ---- Random data -------------------------------------------------------
    n_samples = 32
    occ_t   = torch.rand(n_samples, N, S)
    prc_t   = torch.rand(n_samples, N, S)
    label_t = torch.rand(n_samples, N)
    ds      = TensorDataset(occ_t, prc_t, label_t)
    loader  = DL(ds, batch_size=8, shuffle=False, drop_last=False)

    # ---- Train (to produce checkpoint) ------------------------------------
    model        = build_model(cfg, adj_sparse)
    train_result = train(model, loader, loader, adj_norm, cfg)

    # ---- Evaluate ----------------------------------------------------------
    result = evaluate(model, loader, cfg)

    # ---- Save results ------------------------------------------------------
    save_results(result, cfg, train_result)

    # ---- Assertions --------------------------------------------------------
    assert result.predictions.shape == (n_samples, N), \
        f"Wrong predictions shape: {result.predictions.shape}"
    assert result.labels.shape == (n_samples, N), \
        f"Wrong labels shape: {result.labels.shape}"
    assert set(result.metrics.keys()) == {"MSE","RMSE","MAE","MAPE","R2","RAE"}, \
        f"Missing metrics: {result.metrics.keys()}"
    assert result.node_mae.shape == (N,), \
        f"Wrong node_mae shape: {result.node_mae.shape}"
    assert result.node_r2.shape == (N,), \
        f"Wrong node_r2 shape: {result.node_r2.shape}"
    assert os.path.exists(cfg.paths.results_path("test")), \
        "Results CSV not saved"
    assert np.all(result.predictions >= 0) and np.all(result.predictions <= 1), \
        "Predictions outside [0, 1]"

    print("\n  All assertions passed.")
    print(f"  Results CSV exists : {os.path.exists(cfg.paths.results_path('test'))}")
    print("\nevaluate.py self-test complete.")
