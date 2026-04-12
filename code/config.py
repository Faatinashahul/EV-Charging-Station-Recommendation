"""
config.py
---------
Single source of truth for the entire STGAT project.
Every hyperparameter, path, flag, and constant lives here.
Nothing in any other file should contain a magic number or hardcoded path.

To run a different experiment, only this file should need to change.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    """
    All filesystem paths used across the project.
    ROOT is the project root — adjust this to wherever you clone the repo.
    Everything else is derived from ROOT so the project is fully portable.
    """

    # Project root — change this one line if you move the project
    root: str = os.path.dirname(os.path.abspath(__file__))

    # Input data directory — expects the 6 CSVs described in the README
    data_dir: str = field(init=False)

    # Where trained model checkpoints (.pt files) are saved
    checkpoint_dir: str = field(init=False)

    # Where evaluation results (metric CSVs) are saved
    results_dir: str = field(init=False)

    def __post_init__(self):
        self.data_dir        = os.path.join(self.root, "datasets")
        self.checkpoint_dir  = os.path.join(self.root, "checkpoints")
        self.results_dir     = os.path.join(self.root, "results")

        # Create directories if they don't exist yet — safe to call repeatedly
        for d in [self.data_dir, self.checkpoint_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)

    # ---- Individual file paths (derived) -----------------------------------

    @property
    def occupancy_csv(self) -> str:
        return os.path.join(self.data_dir, "occupancy.csv")

    @property
    def price_csv(self) -> str:
        return os.path.join(self.data_dir, "price.csv")

    @property
    def adj_csv(self) -> str:
        return os.path.join(self.data_dir, "adj.csv")

    @property
    def distance_csv(self) -> str:
        return os.path.join(self.data_dir, "distance.csv")

    @property
    def information_csv(self) -> str:
        return os.path.join(self.data_dir, "information.csv")

    @property
    def time_csv(self) -> str:
        return os.path.join(self.data_dir, "time.csv")

    def checkpoint_path(self, tag: str = "best") -> str:
        """
        Returns the full path for a checkpoint file.
        tag: a short label to distinguish checkpoints, e.g. 'best', 'last'.
        Example: checkpoint_path('best') -> '.../checkpoints/stgat_best.pt'
        """
        return os.path.join(self.checkpoint_dir, f"stgat_{tag}.pt")

    def results_path(self, tag: str = "test") -> str:
        """
        Returns the full path for a results CSV.
        Example: results_path('test') -> '.../results/stgat_test_metrics.csv'
        """
        return os.path.join(self.results_dir, f"stgat_{tag}_metrics.csv")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """
    Controls how raw data is loaded, windowed, and split.

    Sequence terminology used throughout the project:
      seq_len   — number of past timesteps fed as input (the lookback window)
      pred_len  — number of steps ahead being predicted (the forecast horizon)

    With 5-minute data intervals:
      seq_len=12  →  1 hour of lookback
      pred_len=1  →  predicting 5 minutes ahead
      pred_len=6  →  predicting 30 minutes ahead
      pred_len=12 →  predicting 1 hour ahead

    Split ratios must sum to 1.0.
    The original paper uses 60/20/20.
    """

    # Sliding window parameters
    seq_len:  int = 12    # input sequence length (timesteps)
    pred_len: int = 6     # forecast horizon (timesteps)

    # Train / validation / test proportions
    train_ratio: float = 0.6
    val_ratio:   float = 0.2
    test_ratio:  float = 0.2

    # Number of input features per node per timestep
    # Currently: occupancy + price = 2
    # If you add weather, POI density, etc. in the future, increment this
    n_features: int = 2

    def __post_init__(self):
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        assert total == 1.0, (
            f"Split ratios must sum to 1.0, got {total}. "
            f"Check train_ratio={self.train_ratio}, "
            f"val_ratio={self.val_ratio}, test_ratio={self.test_ratio}."
        )
        assert self.seq_len  > 0, "seq_len must be a positive integer."
        assert self.pred_len > 0, "pred_len must be a positive integer."
        assert self.pred_len < self.seq_len, (
            f"pred_len ({self.pred_len}) should be less than seq_len ({self.seq_len}) "
            f"for meaningful temporal context."
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Defines the STGAT architecture.

    Architecture overview:
      1. Conv2d          — fuses occupancy + price into a single feature stream
                           per node. Reduces the 2-feature input to 1 channel,
                           and slightly compresses the sequence dimension.
                           kernel: (conv_kernel, n_features) → output seq = seq_len - conv_kernel + 1

      2. GATLayer × 2   — learns spatially-aware node embeddings by attending
                           over graph neighbors. Two layers = 2-hop receptive field.
                           Residual connections (alpha) prevent over-smoothing.

      3. LSTM            — models temporal dynamics across the compressed sequence.
                           Input features per timestep = 2 (outputs from GAT layer 1 + layer 2).
                           lstm_hidden controls the hidden state dimensionality.

      4. TPA             — Temporal Pattern Attention. Attends over all LSTM
                           hidden states to surface recurring temporal patterns
                           (e.g. morning/evening peaks) rather than relying only
                           on the final hidden state.

      5. Linear → Sigmoid — projects to a scalar per node, sigmoid clamps to (0, 1).

    Node count (n_nodes) is set at runtime from the adjacency matrix shape.
    Set it explicitly here only if you want to override / sanity-check.
    """

    # Graph attention parameters
    gat_heads:   int   = 4      # Number of attention heads per GAT layer
    gat_dropout: float = 0.0    # Dropout on GAT attention coefficients
                                 # 0.0 = disabled; increase if overfitting spatially
    gat_alpha:   float = 0.2    # LeakyReLU negative slope in attention scoring

    # Residual connection weight between GAT layers
    # output = (1 - alpha) * aggregated + alpha * input
    # Higher alpha = stronger skip, less aggregation
    residual_alpha: float = 0.5

    # Conv2d parameters (feature fusion before GAT)
    conv_kernel: int = 2        # Temporal kernel size. After conv: seq = seq_len - conv_kernel + 1
                                 # With seq_len=12, conv_kernel=2 → internal seq = 11

    # LSTM parameters
    lstm_layers:  int = 2       # Number of stacked LSTM layers
    lstm_hidden:  int = 16       # Hidden state size per timestep
                                 # Kept at 2 (matching the 2 GAT layer outputs stacked as features)
                                 # Increase to 8/16/32 if you want a wider temporal model

    # TPA parameters
    tpa_k: int = 10              # Projection dimension inside TPA attention scoring
                                 # Must satisfy: tpa_k ≤ (internal_seq - 1)
                                 # With conv_kernel=2 and seq_len=12: internal_seq=11, so tpa_k ≤ 10

    # Regularization
    dropout: float = 0.5        # Dropout rate applied after each GAT layer

    # Node count — populated at runtime from adj.csv shape
    # Shenzhen ST-EVCDP dataset: 247 nodes
    # Override here to sanity-check against loaded data
    n_nodes: int = 247

    def __post_init__(self):
        assert 0.0 <= self.gat_dropout  <= 1.0, "gat_dropout must be in [0, 1]."
        assert 0.0 <= self.dropout      <= 1.0, "dropout must be in [0, 1]."
        assert 0.0 <= self.residual_alpha <= 1.0, "residual_alpha must be in [0, 1]."
        assert self.gat_heads  >= 1, "gat_heads must be at least 1."
        assert self.lstm_layers >= 1, "lstm_layers must be at least 1."
        assert self.lstm_hidden >= 1, "lstm_hidden must be at least 1."
        assert self.conv_kernel >= 1, "conv_kernel must be at least 1."
        assert self.tpa_k >= 1, "tpa_k must be at least 1."


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Controls the training loop, optimizer, loss, and checkpointing.

    Loss function:
      L_total = MSE(pred, label) + lambda_consistency * L_consistency

      L_consistency enforces that each node's prediction doesn't deviate
      wildly from its graph neighbors' weighted average prediction.
      It is computed as:
        L_consistency = mean( (pred - A_norm @ pred)^2 )
      where A_norm is the row-normalized adjacency matrix.

      lambda_consistency = 0.0 disables the constraint entirely.
      Recommended starting range: 0.05 – 0.2.

    Early stopping:
      Controlled by early_stopping flag.
      If enabled, training halts when val loss hasn't improved for
      `patience` consecutive epochs.
      The best checkpoint (lowest val loss) is always saved regardless.
    """

    # Core training
    n_epochs:   int   = 200
    batch_size: int   = 128 
    lr:         float = 1e-3      # Adam learning rate
    weight_decay: float = 1e-5    # Adam L2 regularization

    # Loss
    lambda_consistency: float = 0.1   # Weight of neighborhood consistency loss
                                       # 0.0 = pure MSE (disables spatial constraint)
                                       # 0.1 = gentle spatial regularization (recommended)
                                       # 0.5+ = strong spatial smoothing (may hurt accuracy)

    # Learning rate schedule
    # ReduceLROnPlateau: halves LR when val loss plateaus for `lr_patience` epochs
    use_lr_scheduler:  bool  = True
    lr_scheduler_factor: float = 0.5   # Factor to reduce LR by
    lr_patience:       int   = 10      # Epochs to wait before reducing LR

    # Early stopping
    early_stopping: bool = False       # Toggle via config — False = run full n_epochs
    patience:       int  = 20          # Epochs without val improvement before stopping
                                        # Only used when early_stopping=True
    
    # MAPE-aware loss parameters
    mape_loss_eps:   float = 0.1   # smoothing for near-zero zones
    mape_loss_alpha: float = 0.5   # blend: 0=pure MSE, 1=pure relative

    # Checkpointing
    # Best model (lowest val loss) is always saved.
    # save_last also saves the final epoch checkpoint separately.
    save_last: bool = False

    def __post_init__(self):
        assert self.n_epochs    > 0,   "n_epochs must be positive."
        assert self.batch_size  > 0,   "batch_size must be positive."
        assert self.lr          > 0.0, "Learning rate must be positive."
        assert self.weight_decay >= 0.0, "weight_decay must be non-negative."
        assert 0.0 <= self.lambda_consistency, "lambda_consistency must be non-negative."
        assert self.patience    > 0,   "patience must be positive."
        assert 0.0 < self.lr_scheduler_factor < 1.0, (
            "lr_scheduler_factor must be between 0 and 1 (exclusive)."
        )


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    """
    Hardware and reproducibility settings.

    Device selection:
      'auto'  — best available: CUDA → MPS → CPU (recommended for all platforms)
      'cuda'  — forces NVIDIA GPU (errors if unavailable)
      'mps'   — forces Apple Silicon GPU (errors if unavailable)
                 Use this on M1/M2/M3 Macs for meaningful speedup over CPU.
                 PyTorch MPS backend is mature as of torch 2.x.
      'cpu'   — forces CPU (useful for debugging or if MPS causes issues)

    MPS notes (Apple Silicon):
      - MPS is Metal Performance Shaders — Apple's GPU compute framework.
      - On an M2 Pro, MPS gives roughly 3-6x speedup over CPU for this model size.
      - pin_memory should stay False with MPS (it's a CUDA-specific optimization).
      - If you hit any MPS-specific op errors, fall back to 'cpu' temporarily.

    Seeding:
      Setting a seed makes runs deterministic and reproducible.
      seed=None disables seeding (non-deterministic).

    DataLoader workers:
      num_workers=0 means data loading happens in the main process.
      On macOS with MPS, keep num_workers=0 to avoid multiprocessing conflicts.
    """

    device_preference: str  = "auto"   # 'auto' | 'cuda' | 'mps' | 'cpu'
    seed:              int  = 2023     # Random seed for reproducibility. None = no seeding.
    num_workers:       int  = 0        # Keep at 0 on macOS/MPS
    pin_memory:        bool = False    # CUDA only — leave False for MPS/CPU

    def __post_init__(self):
        assert self.device_preference in ("auto", "cuda", "mps", "cpu"), (
            f"device_preference must be 'auto', 'cuda', 'mps', or 'cpu'. "
            f"Got '{self.device_preference}'."
        )
        assert self.num_workers >= 0, "num_workers must be non-negative."

    @property
    def device(self) -> torch.device:
        """
        Resolves the actual torch.device based on device_preference and availability.
        Priority for 'auto': CUDA > MPS > CPU.

        Kept as a @property (not stored in __init__) so the config object
        stays picklable — important for DataLoader workers.
        """
        if self.device_preference == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")

        elif self.device_preference == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device_preference='cuda' but CUDA is not available. "
                    "Switch to 'auto' or 'cpu'."
                )
            return torch.device("cuda")

        elif self.device_preference == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "device_preference='mps' but MPS is not available. "
                    "Requires macOS 12.3+ with Apple Silicon and torch 1.12+. "
                    "Switch to 'auto' or 'cpu'."
                )
            return torch.device("mps")

        else:
            return torch.device("cpu")


# ---------------------------------------------------------------------------
# Master config — the one object you import everywhere
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Master configuration object.
    Import and instantiate this in every other file:

        from config import Config
        cfg = Config()

    Then access sub-configs as attributes:
        cfg.paths.data_dir
        cfg.data.seq_len
        cfg.model.gat_heads
        cfg.train.lr
        cfg.system.device

    To override defaults for an experiment, pass keyword args:
        cfg = Config(train=TrainConfig(lr=5e-4, lambda_consistency=0.2))

    Or mutate after construction:
        cfg = Config()
        cfg.train.early_stopping = True
        cfg.train.patience = 15
    """

    paths:  PathConfig  = field(default_factory=PathConfig)
    data:   DataConfig  = field(default_factory=DataConfig)
    model:  ModelConfig = field(default_factory=ModelConfig)
    train:  TrainConfig = field(default_factory=TrainConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def summary(self) -> str:
        """
        Returns a human-readable summary of the full configuration.
        Useful to print at the start of a training run for experiment logging.

        Usage:
            cfg = Config()
            print(cfg.summary())
        """
        device_str = str(self.system.device)
        sep = "-" * 52

        lines = [
            sep,
            "  STGAT — Configuration Summary",
            sep,
            "",
            "  [Paths]",
            f"    data_dir        : {self.paths.data_dir}",
            f"    checkpoint_dir  : {self.paths.checkpoint_dir}",
            f"    results_dir     : {self.paths.results_dir}",
            "",
            "  [Data]",
            f"    seq_len         : {self.data.seq_len} steps "
              f"({self.data.seq_len * 5} min lookback)",
            f"    pred_len        : {self.data.pred_len} steps "
              f"({self.data.pred_len * 5} min forecast)",
            f"    n_features      : {self.data.n_features}  (occ + price)",
            f"    split           : {self.data.train_ratio:.0%} train / "
              f"{self.data.val_ratio:.0%} val / "
              f"{self.data.test_ratio:.0%} test",
            "",
            "  [Model]",
            f"    n_nodes         : {self.model.n_nodes}",
            f"    gat_heads       : {self.model.gat_heads}",
            f"    residual_alpha  : {self.model.residual_alpha}",
            f"    conv_kernel     : {self.model.conv_kernel}  "
              f"(internal seq = {self.data.seq_len - self.model.conv_kernel + 1})",
            f"    lstm_layers     : {self.model.lstm_layers}",
            f"    lstm_hidden     : {self.model.lstm_hidden}",
            f"    tpa_k           : {self.model.tpa_k}",
            f"    dropout         : {self.model.dropout}",
            "",
            "  [Training]",
            f"    n_epochs        : {self.train.n_epochs}",
            f"    batch_size      : {self.train.batch_size}",
            f"    lr              : {self.train.lr}",
            f"    weight_decay    : {self.train.weight_decay}",
            f"    lambda_consist  : {self.train.lambda_consistency}"
              + (" (disabled)" if self.train.lambda_consistency == 0.0 else ""),
            f"    lr_scheduler    : {'on' if self.train.use_lr_scheduler else 'off'}"
              + (f"  (factor={self.train.lr_scheduler_factor}, "
                 f"patience={self.train.lr_patience})"
                 if self.train.use_lr_scheduler else ""),
            f"    early_stopping  : {'on' if self.train.early_stopping else 'off'}"
              + (f"  (patience={self.train.patience})"
                 if self.train.early_stopping else ""),
            f"    save_last       : {self.train.save_last}",
            "",
            "  [System]",
            f"    device          : {device_str}"
              + (" \u2190 Apple Silicon GPU" if device_str == "mps"
                 else " \u2190 NVIDIA GPU"   if device_str.startswith("cuda")
                 else " \u2190 CPU only"),
            f"    seed            : {self.system.seed}",
            f"    num_workers     : {self.system.num_workers}",
            "",
            sep,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: validate internal consistency across sub-configs
# ---------------------------------------------------------------------------

def validate_config(cfg: Config) -> None:
    """
    Cross-checks constraints that span multiple sub-configs.
    Call this once at the start of main.py before doing anything else.

    Raises AssertionError with a clear message if anything is inconsistent.
    """

    # Derived sequence length after Conv2d
    internal_seq = cfg.data.seq_len - cfg.model.conv_kernel + 1
    assert internal_seq > 1, (
        f"conv_kernel={cfg.model.conv_kernel} is too large for seq_len={cfg.data.seq_len}. "
        f"internal_seq would be {internal_seq}, must be > 1."
    )

    # TPA needs at least tpa_k past hidden states to attend over
    # hw covers h_1 ... h_(internal_seq - 1), so length = internal_seq - 1
    assert cfg.model.tpa_k <= internal_seq - 1, (
        f"tpa_k={cfg.model.tpa_k} exceeds available hidden states for attention "
        f"(internal_seq - 1 = {internal_seq - 1}). "
        f"Reduce tpa_k or increase seq_len / reduce conv_kernel."
    )

    # pred_len must not overshoot seq_len
    assert cfg.data.pred_len < cfg.data.seq_len, (
        f"pred_len={cfg.data.pred_len} must be less than seq_len={cfg.data.seq_len}."
    )

    # Patience for early stopping should be meaningfully less than n_epochs
    if cfg.train.early_stopping:
        assert cfg.train.patience < cfg.train.n_epochs, (
            f"Early stopping patience ({cfg.train.patience}) is >= n_epochs "
            f"({cfg.train.n_epochs}), making early stopping unreachable. "
            f"Reduce patience or disable early_stopping."
        )


# ---------------------------------------------------------------------------
# Quick self-test — run this file directly to verify config instantiates
# cleanly and prints correctly: `python config.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = Config()
    validate_config(cfg)
    print(cfg.summary())
    print("  config.py loaded and validated successfully.\n")