"""
data.py
-------
Everything related to data — loading raw CSVs, preprocessing,
graph construction, sliding window, train/val/test splits,
PyTorch Dataset, and DataLoaders.

Nothing in this file knows about the model or training loop.
The only external dependency is config.py.

Public API (what other files import from here):
    load_data(cfg)      → returns a DataBundle (all tensors + graph + metadata)
    get_loaders(bundle, cfg) → returns (train_loader, val_loader, test_loader)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Tuple

from config import Config, validate_config


# ---------------------------------------------------------------------------
# DataBundle — the single object passed around after loading
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """
    Holds every preprocessed artifact produced by load_data().
    Passed to get_loaders() and also available in main.py for inspection.

    Shapes (T = total timesteps, N = nodes, S = seq_len):
        occupancy   : (T, N)   float32, normalized to [0, 1]
        price       : (T, N)   float32
        adj_dense   : (N, N)   float32, binary adjacency
        adj_sparse  : sparse COO tensor (N, N)
        adj_norm    : (N, N)   float32, row-normalized adjacency
                               used by the consistency loss in train.py
        capacity    : (N,)     float32, pile count per zone (raw, not normalized)
                               kept for reference / result interpretation
        n_nodes     : int      number of graph nodes (should match cfg.model.n_nodes)
        n_timesteps : int      total number of timesteps in the dataset
    """
    occupancy:   np.ndarray         # (T, N)
    price:       np.ndarray         # (T, N)
    adj_dense:   torch.Tensor       # (N, N)
    adj_sparse:  torch.Tensor       # (N, N) sparse COO
    adj_norm:    torch.Tensor       # (N, N) row-normalized, for consistency loss
    capacity:    np.ndarray         # (N,)
    n_nodes:     int
    n_timesteps: int


# ---------------------------------------------------------------------------
# Step 1 — Load and normalize raw CSVs
# ---------------------------------------------------------------------------

def load_raw(cfg: Config) -> DataBundle:
    """
    Reads all dataset CSVs and returns a DataBundle with preprocessed arrays.

    Occupancy normalization:
        Raw occupancy counts busy piles per zone.
        Divided by capacity (pile count) → values in [0, 1].
        This matches how the original dataset was used in the paper.

    Price:
        Used as-is (no normalization).
        The model receives price as a second input feature alongside occupancy.
        If price scales vary wildly across zones, consider z-score normalization
        here in future — add a flag to DataConfig for that.

    Adjacency:
        adj.csv contains a binary (0/1) matrix: 1 = zones are neighbors.
        We build three versions:
          - adj_dense  : the raw binary tensor, used for graph propagation
          - adj_sparse : sparse COO version, used in GATLayer for efficiency
          - adj_norm   : row-normalized (divide each row by its degree sum),
                         used in the neighborhood consistency loss in train.py
    """
    paths = cfg.paths

    # ---- Load CSVs ---------------------------------------------------------
    _check_files_exist(paths)

    occ_df = pd.read_csv(paths.occupancy_csv,   index_col=0, header=0)
    inf_df = pd.read_csv(paths.information_csv, index_col=None, header=0)
    prc_df = pd.read_csv(paths.price_csv,       index_col=0, header=0)
    adj_df = pd.read_csv(paths.adj_csv,         index_col=0, header=0)

    # ---- Arrays ------------------------------------------------------------
    capacity = np.array(inf_df['count'], dtype=np.float32).reshape(1, -1)  # (1, N)
    occupancy = np.array(occ_df, dtype=np.float32) / capacity               # (T, N)
    price     = np.array(prc_df, dtype=np.float32)                          # (T, N)
    # Z-score normalize price across the time axis per node
    # This equalizes the scale of price and occupancy (both ~mean 0, std 1 range)
    # before they enter the Conv2d fusion layer.
    # Added for performance improvement 🙏
    price_mean = price.mean(axis=0, keepdims=True)   # (1, N)
    price_std  = price.std(axis=0, keepdims=True)    # (1, N)
    price_std  = np.where(price_std < 1e-8, 1.0, price_std)  # avoid div/0
    price      = (price - price_mean) / price_std
    adj       = np.array(adj_df, dtype=np.float32)                          # (N, N)

    n_timesteps, n_nodes = occupancy.shape

    # ---- Sanity checks -----------------------------------------------------
    assert occupancy.shape == price.shape, (
        f"Occupancy shape {occupancy.shape} doesn't match price shape {price.shape}. "
        f"Check your CSVs."
    )
    assert adj.shape == (n_nodes, n_nodes), (
        f"Adjacency matrix shape {adj.shape} doesn't match node count {n_nodes}."
    )
    assert n_nodes == cfg.model.n_nodes, (
        f"Loaded {n_nodes} nodes from data but cfg.model.n_nodes={cfg.model.n_nodes}. "
        f"Update n_nodes in ModelConfig."
    )
    assert np.all(occupancy >= 0), "Occupancy contains negative values after normalization."
    #assert np.all(price >= 0),     "Price contains negative values — check price.csv."
    # Price range is validated informally via the print below — no hard assertion
    # since z-score normalized or negative prices are both valid inputs.

    # Warn (don't crash) if occupancy exceeds 1 — can happen with data errors
    over_cap = np.sum(occupancy > 1.0)
    if over_cap > 0:
        print(f"  [data] Warning: {over_cap} occupancy values exceed 1.0 "
              f"({100*over_cap/occupancy.size:.2f}% of data). "
              f"Clipping to [0, 1].")
        occupancy = np.clip(occupancy, 0.0, 1.0)

    # ---- Build graph tensors -----------------------------------------------
    adj_dense  = torch.tensor(adj, dtype=torch.float32)
    adj_sparse = adj_dense.to_sparse_coo()
    adj_norm   = _row_normalize_adj(adj_dense)

    print(f"  [data] Loaded: {n_timesteps} timesteps × {n_nodes} nodes")
    print(f"  [data] Occupancy range: [{occupancy.min():.4f}, {occupancy.max():.4f}]")
    print(f"  [data] Price range:     [{price.min():.4f}, {price.max():.4f}]")
    print(f"  [data] Adjacency edges: {int(adj.sum())} (density: "
          f"{adj.sum() / (n_nodes * n_nodes):.4f})")

    return DataBundle(
        occupancy   = occupancy,
        price       = price,
        adj_dense   = adj_dense,
        adj_sparse  = adj_sparse,
        adj_norm    = adj_norm,
        capacity    = capacity.flatten(),
        n_nodes     = n_nodes,
        n_timesteps = n_timesteps,
    )


def _check_files_exist(paths) -> None:
    """Raises a clear error if any required CSV is missing."""
    required = [
        paths.occupancy_csv,
        paths.price_csv,
        paths.adj_csv,
        paths.information_csv,
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files:\n" +
            "\n".join(f"  {p}" for p in missing) +
            f"\nExpected in: {paths.data_dir}"
        )


def _row_normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    Row-normalizes the adjacency matrix: each row divided by its degree sum.
    Result[i, j] = adj[i, j] / degree[i]

    Used in the neighborhood consistency loss:
        neighbors_avg = adj_norm @ predictions
    This gives each node a weighted average of its neighbors' predictions,
    which the loss then compares against that node's own prediction.

    Nodes with degree 0 (isolated nodes) would cause division by zero —
    we set those rows to 0 to handle gracefully.
    """
    degree = adj.sum(dim=1, keepdim=True)          # (N, 1)
    degree = torch.clamp(degree, min=1e-8)          # avoid div by zero
    adj_norm = adj / degree                         # (N, N)
    return adj_norm


# ---------------------------------------------------------------------------
# Step 2 — Train / Val / Test split
# ---------------------------------------------------------------------------

def split_data(
    data: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits a (T, N) array into train / val / test along the time axis.
    Maintains temporal order — no shuffling.

    Split is index-based (not ratio-based rounding on both ends) to ensure
    no timestep is lost or double-counted:
        train : [0,         train_end)
        val   : [train_end, val_end)
        test  : [val_end,   T)

    Args:
        data        : array of shape (T, N)
        train_ratio : fraction of T for training
        val_ratio   : fraction of T for validation
                      test gets the remainder (1 - train - val)

    Returns:
        (train, val, test) as numpy arrays
    """
    T = len(data)
    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))

    train = data[:train_end]
    val   = data[train_end:val_end]
    test  = data[val_end:]

    return train, val, test


# ---------------------------------------------------------------------------
# Step 3 — Sliding window
# ---------------------------------------------------------------------------

def sliding_window(
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a (T, N) time series into (X, y) pairs via a sliding window.

    For each valid position i:
        X[i] = data[i : i + seq_len]              shape: (seq_len, N)
        y[i] = data[i + seq_len + pred_len - 1]   shape: (N,)

    The label y is a single future snapshot, pred_len steps beyond the
    end of the input window. With seq_len=12 and pred_len=6:
        - X covers the past 60 minutes (12 × 5min steps)
        - y is the occupancy 30 minutes into the future

    Total samples = T - seq_len - pred_len + 1

    Args:
        data     : (T, N) array
        seq_len  : input window length
        pred_len : forecast horizon

    Returns:
        X : (samples, seq_len, N)
        y : (samples, N)
    """
    X, y = [], []
    T = len(data)
    for i in range(T - seq_len - pred_len + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len + pred_len - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 4 — PyTorch Dataset
# ---------------------------------------------------------------------------

class EVDataset(Dataset):
    """
    PyTorch Dataset for EV charging demand prediction.

    Each sample contains:
        occ   : occupancy sequence,  shape (N, seq_len)  — node-first for GAT
        prc   : price sequence,      shape (N, seq_len)
        label : target occupancy,    shape (N,)

    The transpose from (seq_len, N) → (N, seq_len) happens here once,
    so the model always receives node-first tensors without any runtime
    transposition overhead in the training loop.

    Tensors are kept on CPU here and moved to the target device inside
    the training loop (via .to(device)). This is the standard pattern —
    moving data to GPU inside the Dataset causes issues with DataLoader
    workers and wastes GPU memory on the full dataset.
    """

    def __init__(
        self,
        occupancy: np.ndarray,   # (T_split, N)
        price:     np.ndarray,   # (T_split, N)
        seq_len:   int,
        pred_len:  int,
    ):
        occ_X, occ_y = sliding_window(occupancy, seq_len, pred_len)
        prc_X, _     = sliding_window(price,     seq_len, pred_len)

        # Convert to tensors: (samples, seq_len, N) → store as-is,
        # transpose to (N, seq_len) per sample in __getitem__
        self.occ   = torch.from_numpy(occ_X)   # (samples, seq_len, N)
        self.prc   = torch.from_numpy(prc_X)   # (samples, seq_len, N)
        self.label = torch.from_numpy(occ_y)   # (samples, N)

    def __len__(self) -> int:
        return len(self.occ)

    def __getitem__(self, idx: int):
        # Transpose: (seq_len, N) → (N, seq_len)
        # Model expects [batch, node, seq] for both occ and prc
        occ   = self.occ[idx].T      # (N, seq_len)
        prc   = self.prc[idx].T      # (N, seq_len)
        label = self.label[idx]      # (N,)
        return occ, prc, label


# ---------------------------------------------------------------------------
# Step 5 — Public API: load_data and get_loaders
# ---------------------------------------------------------------------------

def load_data(cfg: Config) -> DataBundle:
    """
    Full data loading pipeline. Call this once in main.py.

    Steps:
        1. Load and normalize raw CSVs
        2. Validate against config
        3. Return DataBundle

    The split and Dataset creation happens inside get_loaders() so that
    DataBundle stays lightweight and reusable for analysis outside training.
    """
    print("\n[data] Loading dataset...")
    bundle = load_raw(cfg)
    print("[data] Done.\n")
    return bundle


def get_loaders(
    bundle: DataBundle,
    cfg:    Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits data, creates EVDatasets, wraps in DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)

    DataLoader details:
        train : shuffled, drop_last=True  (keeps batch sizes uniform)
        val   : not shuffled, full dataset in one batch for fast val loss
        test  : not shuffled, full dataset in one batch for evaluation

    Val and test use batch_size = full split length.
    This avoids partial-batch edge cases during metric computation.
    You can always iterate them with a standard for loop — there will
    just be one batch per epoch.
    """

    # ---- Split raw arrays --------------------------------------------------
    train_occ, val_occ, test_occ = split_data(
        bundle.occupancy, cfg.data.train_ratio, cfg.data.val_ratio
    )
    train_prc, val_prc, test_prc = split_data(
        bundle.price, cfg.data.train_ratio, cfg.data.val_ratio
    )

    # ---- Build Datasets ----------------------------------------------------
    train_ds = EVDataset(train_occ, train_prc, cfg.data.seq_len, cfg.data.pred_len)
    val_ds   = EVDataset(val_occ,   val_prc,   cfg.data.seq_len, cfg.data.pred_len)
    test_ds  = EVDataset(test_occ,  test_prc,  cfg.data.seq_len, cfg.data.pred_len)

    print(f"  [data] Samples — train: {len(train_ds)}, "
          f"val: {len(val_ds)}, test: {len(test_ds)}")

    # ---- Wrap in DataLoaders -----------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.train.batch_size,
        shuffle     = True,
        drop_last   = True,
        num_workers = cfg.system.num_workers,
        pin_memory  = cfg.system.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = len(val_ds),
        shuffle     = False,
        drop_last   = False,
        num_workers = cfg.system.num_workers,
        pin_memory  = cfg.system.pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = len(test_ds),
        shuffle     = False,
        drop_last   = False,
        num_workers = cfg.system.num_workers,
        pin_memory  = cfg.system.pin_memory,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Self-test — run directly: python data.py
# Requires datasets/ folder to be populated with the CSVs.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = Config()
    validate_config(cfg)

    print(cfg.summary())

    # Load
    bundle = load_data(cfg)

    # Inspect bundle
    print("DataBundle contents:")
    print(f"  occupancy   : {bundle.occupancy.shape}  dtype={bundle.occupancy.dtype}")
    print(f"  price       : {bundle.price.shape}  dtype={bundle.price.dtype}")
    print(f"  adj_dense   : {tuple(bundle.adj_dense.shape)}  dtype={bundle.adj_dense.dtype}")
    print(f"  adj_norm    : {tuple(bundle.adj_norm.shape)}  dtype={bundle.adj_norm.dtype}")
    print(f"  adj_sparse  : {bundle.adj_sparse.shape}  nnz={bundle.adj_sparse._nnz()}")
    print(f"  capacity    : {bundle.capacity.shape}  "
          f"range=[{bundle.capacity.min():.0f}, {bundle.capacity.max():.0f}]")
    print(f"  n_nodes     : {bundle.n_nodes}")
    print(f"  n_timesteps : {bundle.n_timesteps}")

    # Loaders
    print("\nBuilding DataLoaders...")
    train_loader, val_loader, test_loader = get_loaders(bundle, cfg)

    print(f"\n  train batches : {len(train_loader)}")
    print(f"  val batches   : {len(val_loader)}")
    print(f"  test batches  : {len(test_loader)}")

    # Inspect one batch
    occ, prc, label = next(iter(train_loader))
    print(f"\nSample train batch:")
    print(f"  occ   : {tuple(occ.shape)}   (batch, node, seq)")
    print(f"  prc   : {tuple(prc.shape)}   (batch, node, seq)")
    print(f"  label : {tuple(label.shape)}  (batch, node)")
    print(f"  occ range  : [{occ.min():.4f}, {occ.max():.4f}]")
    print(f"  label range: [{label.min():.4f}, {label.max():.4f}]")

    # Verify adj_norm rows sum to 1 (or 0 for isolated nodes)
    row_sums = bundle.adj_norm.sum(dim=1)
    assert torch.allclose(
        row_sums[row_sums > 0],
        torch.ones_like(row_sums[row_sums > 0]),
        atol=1e-5
    ), "adj_norm rows do not sum to 1 — normalization error."
    print("\n  adj_norm row-sum check : PASSED")

    print("\ndata.py loaded and validated successfully.")