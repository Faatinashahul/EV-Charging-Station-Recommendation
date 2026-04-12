"""
main.py
-------
Entry point for the STGAT project.
Wires config → data → model → train → evaluate into one clean run.

Usage:
    python main.py

To change any hyperparameter, edit config.py — do not modify this file.
This file contains zero logic, only orchestration.
"""

import torch
from config import Config, validate_config
from data import load_data, get_loaders
from model import build_model
from train import train
from evaluate import evaluate, save_results


def main():

    # ------------------------------------------------------------------ #
    # 1. Configuration                                                     #
    # ------------------------------------------------------------------ #
    cfg = Config()
    validate_config(cfg)

    print(cfg.summary())

    # Reproducibility — seed everything before any tensor is created
    if cfg.system.seed is not None:
        torch.manual_seed(cfg.system.seed)
        if cfg.system.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.system.seed)
        # MPS does not have a dedicated seed call — torch.manual_seed covers it

    # ------------------------------------------------------------------ #
    # 2. Data                                                              #
    # ------------------------------------------------------------------ #
    bundle = load_data(cfg)

    train_loader, val_loader, test_loader = get_loaders(bundle, cfg)

    # ------------------------------------------------------------------ #
    # 3. Model                                                             #
    # ------------------------------------------------------------------ #
    model = build_model(cfg, bundle.adj_sparse)

    # ------------------------------------------------------------------ #
    # 4. Training                                                          #
    # ------------------------------------------------------------------ #
    train_result = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        adj_norm     = bundle.adj_norm,
        cfg          = cfg,
    )

    # ------------------------------------------------------------------ #
    # 5. Evaluation                                                        #
    # ------------------------------------------------------------------ #
    eval_result = evaluate(
        model       = model,
        test_loader = test_loader,
        cfg         = cfg,
    )

    save_results(eval_result, cfg, train_result)


if __name__ == "__main__":
    main()