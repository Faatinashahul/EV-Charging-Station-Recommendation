# EV Charging Station Recommendation System

A Spatio-Temporal Graph Attention Network (STGAT) that predicts EV charging station occupancy across Shenzhen and recommends nearby low-demand stations to users in real time.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Saving and Loading the Model (.pkl)](#saving-and-loading-the-model-pkl)
- [Running the Recommender](#running-the-recommender)
- [Configuration](#configuration)
- [Training Details](#training-details)
- [Evaluation Results](#evaluation-results)
- [Web App](#web-app)

---

## Overview

This system forecasts occupancy at 247 EV charging stations in Shenzhen, China, using a graph-based deep learning model. Given a user's location (latitude/longitude or Chinese postcode), the recommender finds nearby stations with the lowest predicted demand — minimising waiting time.

**Key features:**
- Spatial modelling with Graph Attention Networks (GAT)
- Temporal modelling with LSTM + Temporal Pattern Attention (TPA)
- Neighbourhood consistency loss for spatially coherent predictions
- MAPE-aware loss for accurate forecasts at low-occupancy zones
- Streamlit web app for interactive recommendations

---

## Architecture

The model is called **STGAT** (Spatio-Temporal Graph Attention Network) and consists of four stacked components:

```
occupancy (B, N, S)  ──┐
                        ├──► FeatureFusion (Conv2d) ──► (B, N, S')
price (B, N, S)      ──┘
                                    │
                                    ▼
                        SpatialEncoder (GATLayer × 2 + residuals)
                        → layer1 (B, N, S'),  layer2 (B, N, S')
                                    │
                                    ▼
                        TPADecoder (LSTM + Temporal Pattern Attention)
                        → logits (B, N)
                                    │
                                    ▼
                              Sigmoid → predictions (B, N) ∈ (0, 1)
```

| Component | Role |
|---|---|
| **FeatureFusion** | `Conv2d` fuses occupancy + price into a single feature stream per node. Learns joint local patterns (e.g. high price + rising occupancy). |
| **GATLayer** | Sparse multi-head graph attention. Each node aggregates weighted signals from its graph neighbours. Parameters stored in `nn.ParameterDict` so they are correctly trained. |
| **SpatialEncoder** | Two stacked `GATLayer`s with residual connections (weight `residual_alpha`). Gives every node a 2-hop receptive field while preventing over-smoothing. |
| **TPADecoder** | 2-layer LSTM over the spatial features, then Temporal Pattern Attention weights all past hidden states against the final one. Better than using only the last hidden state for periodic demand patterns. |

---

## Project Structure

```
EV-charging-station-recommendation-system/
├── code/
│   ├── config.py          # All hyperparameters and paths — edit here only
│   ├── data.py            # Data loading, normalisation, windowing, DataLoader
│   ├── model.py           # STGAT architecture (FeatureFusion, GAT, TPA)
│   ├── train.py           # Training loop, validation, early stopping
│   ├── evaluate.py        # Test-set evaluation and metrics
│   ├── recommend.py       # Station recommendation logic
│   ├── app.py             # Streamlit web application
│   ├── main.py            # Entry point: config → data → train → evaluate
│   ├── datasets/          # Raw CSV data files (see Datasets section)
│   ├── results/           # Saved predictions, plots, metric CSVs
│   └── checkpoints/       # Saved model checkpoints (.pt files)
├── save_model.py          # Train and/or export model to .pkl
├── stgat_model.pkl        # Serialised model (created by save_model.py)
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Datasets

All files live in `code/datasets/`. The dataset covers **247 EV charging stations in Shenzhen** sampled at **5-minute intervals** from 19 June 2022 to 18 July 2022.

| File | Description |
|---|---|
| `occupancy.csv` | Timestep × station occupancy ratios (primary target) |
| `price.csv` | Timestep × station charging prices |
| `adj.csv` | 247×247 binary adjacency matrix (spatial graph edges) |
| `distance.csv` | Pairwise station distances (km) |
| `information.csv` | Station metadata (ID, location, pile count) |
| `time.csv` | Timestamps for each row in occupancy/price |
| `stations.csv` | Station coordinates + capacity (used by recommender) |
| `duration.csv` | Charging session durations |
| `volume.csv` | Charging volume per session |
| `SZweather*.xls` | Shenzhen weather data for the same period |
| `SZ_districts/` | Shenzhen district shapefiles (GIS visualisation) |

---

## Installation

**Requirements:** Python 3.9+, pip

```bash
# 1. Clone or unzip the project
cd EV-charging-station-recommendation-system

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Key dependencies: `torch`, `pandas`, `numpy`, `scikit-learn`, `streamlit`, `geopy`, `tqdm`, `matplotlib`, `seaborn`.

---

## Quick Start

### Train + evaluate end-to-end

```bash
cd code
python main.py
```

This runs the full pipeline: loads data → builds model → trains for up to 200 epochs → evaluates on the test set → saves metrics to `results/`.

### Train only

```bash
cd code
python train.py   # self-test with random data (no dataset needed)
```

### Evaluate only (requires a saved checkpoint)

```bash
cd code
python evaluate.py
```

---

## Saving and Loading the Model (.pkl)

The `save_model.py` script trains the model (or reloads an existing checkpoint) and serialises everything needed for inference into a single `.pkl` file.

### Create the .pkl

```bash
# Train from scratch, then save
python save_model.py

# Skip training — load the best existing checkpoint and save as pkl
python save_model.py --from-checkpoint

# Custom output path
python save_model.py --output /path/to/my_model.pkl
```

The default output path is `stgat_model.pkl` in the project root.

### What is stored inside the .pkl

The file contains a Python dict with these keys:

| Key | Type | Description |
|---|---|---|
| `model_state_dict` | `OrderedDict` | Trained model weights |
| `model_class` | `str` | `"STGAT"` (documentation tag) |
| `cfg` | `Config` | Full configuration used for training |
| `adj_sparse` | `torch.Tensor` | Sparse adjacency matrix (CPU) |
| `metadata` | `dict` | Training summary (best epoch, val loss, etc.) |

### Reload the model for inference

```python
import pickle
import torch
import sys
sys.path.insert(0, "code")   # so Python can find model.py and config.py

from model import build_model

with open("stgat_model.pkl", "rb") as f:
    payload = pickle.load(f)

cfg        = payload["cfg"]
adj_sparse = payload["adj_sparse"]
state_dict = payload["model_state_dict"]

# Reconstruct architecture and load weights
model = build_model(cfg, adj_sparse)
model.load_state_dict(state_dict)
model.eval()

# Run inference
occ = torch.rand(1, cfg.model.n_nodes, cfg.data.seq_len)  # (B, N, S)
prc = torch.rand(1, cfg.model.n_nodes, cfg.data.seq_len)  # (B, N, S)

with torch.no_grad():
    predictions = model(occ, prc)   # (B, N) — occupancy in (0, 1)

print(predictions.shape)  # torch.Size([1, 247])
```

Or use the helper function from `save_model.py`:

```python
import sys
sys.path.insert(0, ".")   # project root where save_model.py lives

from save_model import load_from_pkl

model = load_from_pkl("stgat_model.pkl")
# model is ready for inference immediately
```

### Why pickle and not just torch.save?

`torch.save` saves only the weights (`state_dict`) or the whole model object — it does not bundle the `Config` and `adj_sparse` matrix needed to reconstruct the model on a different machine. The `.pkl` approach stores everything in one portable file so the model can be reloaded with a single `pickle.load()` call, without needing to re-specify any hyperparameters.

---

## Running the Recommender

### Command-line

```bash
cd code
python recommend.py
# Enter your Chinese postcode when prompted (e.g. 518000)
```

The script geocodes the postcode, finds the 10 nearest stations, scores them by predicted demand, and returns the top 3 with distances and availability estimates.

### Programmatic use

```python
import sys
sys.path.insert(0, "code")

from recommend import recommend

# User coordinates (latitude, longitude)
results = recommend(user_lat=22.5431, user_lon=114.0579, top_k=3)
print(results)
```

### Streamlit web app

```bash
cd code
streamlit run app.py
```

Opens a browser-based map interface where users can enter their location and see recommended stations overlaid on a Shenzhen map.

---

## Configuration

All hyperparameters live in `code/config.py`. The most commonly changed values:

```python
# code/config.py

@dataclass
class DataConfig:
    seq_len:  int   = 12    # lookback window (12 × 5 min = 1 hour)
    pred_len: int   = 6     # forecast horizon (6 × 5 min = 30 min)
    train_ratio: float = 0.6
    val_ratio:   float = 0.2
    test_ratio:  float = 0.2

@dataclass
class ModelConfig:
    gat_heads:      int   = 4      # attention heads per GAT layer
    lstm_hidden:    int   = 16     # LSTM hidden state size
    lstm_layers:    int   = 2      # stacked LSTM layers
    tpa_k:          int   = 10     # TPA projection dimension
    conv_kernel:    int   = 2      # temporal fusion kernel size
    residual_alpha: float = 0.5    # skip-connection weight in SpatialEncoder
    dropout:        float = 0.5    # dropout after each GAT layer

@dataclass
class TrainConfig:
    n_epochs:           int   = 200
    batch_size:         int   = 128
    lr:                 float = 1e-3
    lambda_consistency: float = 0.1   # spatial consistency loss weight
    early_stopping:     bool  = False
    patience:           int   = 20
```

To run an experiment with different settings, change values in `config.py` — nothing else needs to be modified.

---

## Training Details

### Loss function

```
L_total = MAPE-aware loss + λ · L_consistency
```

**MAPE-aware loss** combines MSE with a relative error term:

```
L_mape = (1 - α) · MSE(pred, label) + α · mean(|pred - label| / (label + ε))
```

This prevents the model from ignoring low-occupancy stations (where absolute errors are small but percentage errors can be large). Default: `α = 0.5`, `ε = 0.1`.

**Neighbourhood consistency loss** encourages spatially coherent predictions:

```
L_consistency = mean((pred - A_norm @ pred)²)
```

Where `A_norm` is the row-normalised adjacency matrix. A node's prediction should not deviate wildly from its graph neighbours' weighted average.

### Optimiser

Adam with `lr=1e-3`, `weight_decay=1e-5`, gradient clipping at `max_norm=1.0`.

LR scheduler: `ReduceLROnPlateau` — halves LR when validation loss plateaus for 10 epochs.

### Checkpointing

Best model (lowest validation loss) is saved to `code/checkpoints/stgat_best.pt` automatically during training.

---

## Evaluation Results

Results are saved to `code/results/` after running `main.py` or `evaluate.py`.

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| R² | Coefficient of determination |

Sample output files:
- `stgat_test_metrics.csv` — per-node and aggregate metrics
- `prediction_vs_actual.png` — time-series comparison plot
- `scatter_pred_vs_actual.png` — scatter plot of predictions vs ground truth
- `error_distribution.png` — histogram of prediction errors
- `spatial_performance.png` — per-district accuracy on Shenzhen map
- `demand_heatmap.png` — predicted demand heatmap

---

## Web App

The Streamlit app (`code/app.py`) provides an interactive interface:

- Enter latitude/longitude or a Chinese postcode
- View recommended stations on an interactive map
- See predicted occupancy, capacity, and estimated waiting time
- Filter by distance radius or minimum availability

```bash
cd code
streamlit run app.py
```
