import os
import copy
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_prognosis import (
    Config,
    GATSurv,
    read_prognosis_csv,
    zscore_fit_transform,
    zscore_transform,
    minmax_fit_transform,
    minmax_transform,
    load_adjacency_from_csv,
    prepare_adjacency,
    build_node_features_for_model,
    SurvivalDataset,
    evaluate,
    c_index,
    set_seed,
)


# =============================
# Configuration
# =============================

# Evaluation dataset path (replace before running)
DATA_CSV_PATH: str = "REPLACE_WITH_EVAL_CSV_PATH"

# Pretrained checkpoint path (replace before running)
CKPT_PATH: str = "REPLACE_WITH_MODEL_CKPT_PATH"

def read_generic_dataset_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read evaluation data.
    Expected column layout (1-based):
      1: sample_id
      3-82: lncRNA (80)
      83-322: RNA (240)
      323: event
      324: time
    """
    df = pd.read_csv(path)
    sample_ids = df.iloc[:, 0].astype(str).values
    X_lnc = df.iloc[:, 1:81].astype(float).values
    X_rna = df.iloc[:, 81:321].astype(float).values
    event = df.iloc[:, 321].astype(float).values
    time = df.iloc[:, 322].astype(float).values
    return sample_ids, X_lnc, X_rna, event, time


def restore_config_from_ckpt(base_cfg: Config, ckpt: dict) -> Config:
    """Restore training configuration from a checkpoint."""
    cfg = copy.deepcopy(base_cfg)
    cfg_dict = ckpt.get("config", None)
    if isinstance(cfg_dict, dict):
        for k, v in cfg_dict.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def main():
    # Load checkpoint and restore configuration
    base_cfg = Config()

    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"CKPT_PATH '{CKPT_PATH}' not found")
    print(f"Using checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    cfg = restore_config_from_ckpt(base_cfg, ckpt)
    set_seed(cfg.seed)
    device = torch.device("cuda" if (cfg.use_cuda_if_available and torch.cuda.is_available()) else "cpu")
    print(f"Restored config: use_lnc={cfg.use_lnc}, adj_threshold={cfg.adj_threshold}, seed={cfg.seed}")

    # Load training and evaluation datasets
    _, Xlnc_tr, Xrna_tr, _, _ = read_prognosis_csv(cfg.train_csv)
    data_ids, Xlnc_data, Xrna_data, event_data, time_data = read_generic_dataset_csv(DATA_CSV_PATH)

    # Normalize using training-set statistics
    if getattr(cfg, "minmax_norm", False):
        Xrna_tr, rna_min, rna_max = minmax_fit_transform(Xrna_tr)
        Xrna_data = minmax_transform(Xrna_data, rna_min, rna_max)
        if cfg.use_lnc:
            Xlnc_tr, lnc_min, lnc_max = minmax_fit_transform(Xlnc_tr)
            Xlnc_data = minmax_transform(Xlnc_data, lnc_min, lnc_max)

    if getattr(cfg, "zscore_norm", False):
        Xrna_tr, rna_mean, rna_std = zscore_fit_transform(Xrna_tr)
        Xrna_data = zscore_transform(Xrna_data, rna_mean, rna_std)
        if cfg.use_lnc:
            Xlnc_tr, lnc_mean, lnc_std = zscore_fit_transform(Xlnc_tr)
            Xlnc_data = zscore_transform(Xlnc_data, lnc_mean, lnc_std)

    # Load and process adjacency matrix
    adj_all = load_adjacency_from_csv(cfg.adj_csv_path)
    adj_use = prepare_adjacency(adj_all, cfg).to(device)

    # Build evaluation inputs
    Xnodes = build_node_features_for_model(Xrna_data, Xlnc_data if cfg.use_lnc else None, cfg)
    event_t = torch.tensor(event_data, dtype=torch.float32)
    time_t = torch.tensor(time_data, dtype=torch.float32)

    dataset = SurvivalDataset(Xnodes, event_t, time_t)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Build model and load weights
    model = GATSurv(cfg).to(device)
    state_dict = ckpt.get("state_dict", None)
    if state_dict is None:
        raise RuntimeError(f"Checkpoint '{CKPT_PATH}' has no 'state_dict'")
    model.load_state_dict(state_dict)

    # Inference and metric computation
    _, haz_pred, evt_np, time_np = evaluate(model, loader, adj_use, device)
    eval_cindex = c_index(haz_pred, evt_np, time_np)
    print(f"Evaluation C-index on dataset '{DATA_CSV_PATH}': {eval_cindex:.4f}")

    # Save predictions
    out_df = pd.DataFrame({
        "sample_id": data_ids,
        "hazard_pred": haz_pred,
        "event": evt_np,
        "time": time_np,
    })
    base_name = os.path.splitext(os.path.basename(DATA_CSV_PATH))[0]
    out_csv = os.path.join(cfg.results_dir, f"{base_name}_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Predictions saved to: {out_csv}")


if __name__ == "__main__":
    main()


