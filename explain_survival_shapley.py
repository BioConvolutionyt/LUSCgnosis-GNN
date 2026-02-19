import os
import copy
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch

from train_prognosis import (
    Config,
    GATSurv,
    read_prognosis_csv,
    minmax_fit_transform,
    minmax_transform,
    zscore_fit_transform,
    zscore_transform,
    load_adjacency_from_csv,
    prepare_adjacency,
    build_node_features_for_model,
    set_seed,
)

"""
Node-level Shapley explanation using Monte Carlo permutation sampling.

Goal: explain each gene node's marginal contribution (lnc + mRNA)
to patient-level hazard predictions from GATSurv.
"""

# Pretrained checkpoint path (replace before running)
CKPT_PATH: str = "REPLACE_WITH_MODEL_CKPT_PATH"

# Input dataset for explanation (replace before running)
CSV_PATH: str = "REPLACE_WITH_SHAP_INPUT_CSV_PATH"


########################################
# Checkpoint and configuration
########################################

def find_best_checkpoint() -> Tuple[str, float, dict]:
    """Load the checkpoint specified by CKPT_PATH."""
    if not CKPT_PATH:
        raise ValueError("CKPT_PATH is empty. Please set CKPT_PATH to a .pt checkpoint file.")
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"CKPT_PATH '{CKPT_PATH}' not found")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cidx = ckpt.get("cindex_test", float("nan"))
    print(f"[Checkpoint] Using specified model: {CKPT_PATH} (cindex_test={cidx})")
    return CKPT_PATH, float(cidx) if cidx == cidx else float("nan"), ckpt


def restore_config_from_ckpt(base_cfg: Config, ckpt: dict) -> Config:
    """Restore training configuration from a checkpoint."""
    cfg = copy.deepcopy(base_cfg)
    cfg_dict = ckpt.get("config", None)
    if isinstance(cfg_dict, dict):
        for k, v in cfg_dict.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


########################################
# Data loading
########################################

def read_analysis_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read data used for explanation.
    Expected column layout (1-based):
      1: sample_id
      2-81: lncRNA (80)
      82-321: RNA (240)
      second-to-last column: OS
      last column: OS.time
    """
    df = pd.read_csv(path)
    sample_ids = df.iloc[:, 0].astype(str).values
    X_lnc = df.iloc[:, 1:81].astype(float).values
    X_rna = df.iloc[:, 81:321].astype(float).values
    event = df.iloc[:, -2].astype(float).values
    time = df.iloc[:, -1].astype(float).values
    return sample_ids, X_lnc, X_rna, event, time


########################################
# Hazard evaluation under node masking
########################################

def compute_hazard_for_mask(
    model: GATSurv,
    x_real: torch.Tensor,        # [V, 1]
    baseline: torch.Tensor,      # [V, 1]
    adj_bin: torch.Tensor,       # [V, V]
    mask: torch.Tensor,          # [V], 1=keep node, 0=mask node
    device: torch.device,
) -> float:
    """
    Compute hazard under a given node mask.
    Masked nodes use baseline expression and their incident edges are removed.
    """
    model.eval()
    x_real = x_real.to(device)
    baseline = baseline.to(device)
    adj_bin = adj_bin.to(device)
    m = mask.to(device=device, dtype=torch.float32).view(-1, 1)

    # Feature masking
    x_eff = baseline * (1.0 - m) + x_real * m
    x_eff = x_eff.unsqueeze(0)

    # Structural masking
    mask_mat = m @ m.T
    adj_eff = adj_bin * mask_mat

    with torch.no_grad():
        _, _, hazard = model(x_eff, adj_eff)
        return float(hazard.item())


########################################
# Single-patient Monte Carlo Shapley
########################################

def explain_one_patient_shapley_mc(
    cfg: Config,
    model: GATSurv,
    x_real: torch.Tensor,        # [V,1]
    baseline: torch.Tensor,      # [V,1]
    adj_bin: torch.Tensor,       # [V,V]
    num_permutations: int = 64,
    random_seed: int = 0,
    device: torch.device = torch.device("cuda"),
) -> Tuple[np.ndarray, float, float]:
    """
    Approximate Shapley values by permutation sampling:
      φ_i ≈ E_π[ f(S_π(i) ∪ {i}) - f(S_π(i)) ].
    Complexity is approximately O(num_permutations * V) forward passes.
    Returns (phi, f_full, f_empty).
    """
    model = model.to(device)
    x_real = x_real.to(device)
    baseline = baseline.to(device)
    adj_bin = adj_bin.to(device)

    V = x_real.shape[0]
    rng = np.random.RandomState(random_seed)

    # Empty graph
    mask_empty = torch.zeros(V, dtype=torch.float32, device=device)
    f_empty = compute_hazard_for_mask(model, x_real, baseline, adj_bin, mask_empty, device)

    # Full graph
    mask_full = torch.ones(V, dtype=torch.float32, device=device)
    f_full = compute_hazard_for_mask(model, x_real, baseline, adj_bin, mask_full, device)

    print(f"    [Debug] f_empty={f_empty:.6f}, f_full={f_full:.6f}, delta={f_full - f_empty:.6f}")

    phi = np.zeros(V, dtype=np.float64)

    for t in range(num_permutations):
        perm = rng.permutation(V)
        mask = torch.zeros(V, dtype=torch.float32, device=device)
        f_prev = f_empty

        for idx in perm:
            mask[idx] = 1.0
            f_curr = compute_hazard_for_mask(model, x_real, baseline, adj_bin, mask, device)
            contrib = f_curr - f_prev
            phi[idx] += contrib
            f_prev = f_curr

    phi /= float(num_permutations)

    # Sanity check: sum(phi) ≈ f_full - f_empty
    print(f"    [Debug] sum(phi)={phi.sum():.6f}, target={f_full - f_empty:.6f}")

    return phi, f_full, f_empty


########################################
# Main workflow
########################################

def main():
    # Device and configuration
    base_cfg = Config()
    set_seed(base_cfg.seed)

    device = torch.device("cuda" if (base_cfg.use_cuda_if_available and torch.cuda.is_available()) else "cpu")

    # Load checkpoint and model
    best_path, _, best_ckpt = find_best_checkpoint()
    cfg = restore_config_from_ckpt(base_cfg, best_ckpt)
    set_seed(cfg.seed)

    print(f"[Info] Restored cfg.use_lnc={cfg.use_lnc}, adj_threshold={cfg.adj_threshold}, seed={cfg.seed}")

    model = GATSurv(cfg).to(device)
    state_dict = best_ckpt.get("state_dict", None)
    if state_dict is None:
        raise RuntimeError(f"Checkpoint '{best_path}' has no 'state_dict'")
    model.load_state_dict(state_dict)
    model.eval()

    # Load training data and fit normalization statistics
    train_ids, Xlnc_tr, Xrna_tr, event_tr, time_tr = read_prognosis_csv(cfg.train_csv)

    # Min-max normalization (optional)
    if getattr(cfg, "minmax_norm", False):
        Xrna_tr, rna_min, rna_max = minmax_fit_transform(Xrna_tr)
        if cfg.use_lnc:
            Xlnc_tr, lnc_min, lnc_max = minmax_fit_transform(Xlnc_tr)
        else:
            Xlnc_tr = None
            lnc_min = lnc_max = None
    else:
        rna_min = rna_max = None
        Xlnc_tr = Xlnc_tr if cfg.use_lnc else None
        lnc_min = lnc_max = None

    # Z-score normalization (optional)
    if getattr(cfg, "zscore_norm", False):
        Xrna_tr, rna_mean, rna_std = zscore_fit_transform(Xrna_tr)
        if cfg.use_lnc:
            Xlnc_tr, lnc_mean, lnc_std = zscore_fit_transform(Xlnc_tr)
        else:
            Xlnc_tr = None
            lnc_mean = lnc_std = None
    else:
        rna_mean = rna_std = None
        Xlnc_tr = Xlnc_tr if cfg.use_lnc else None
        lnc_mean = lnc_std = None

    Xnodes_tr = build_node_features_for_model(Xrna_tr, Xlnc_tr if cfg.use_lnc else None, cfg)
    baseline_nodes = Xnodes_tr.mean(dim=0).squeeze(-1)

    # Load explanation data and apply the same normalization
    test_ids, Xlnc_te, Xrna_te, event_te, time_te = read_analysis_csv(CSV_PATH)

    if getattr(cfg, "minmax_norm", False) and rna_min is not None:
        Xrna_te = minmax_transform(Xrna_te, rna_min, rna_max)
        if cfg.use_lnc and lnc_min is not None:
            Xlnc_te = minmax_transform(Xlnc_te, lnc_min, lnc_max)
        else:
            Xlnc_te = Xlnc_te if cfg.use_lnc else None
    else:
        Xlnc_te = Xlnc_te if cfg.use_lnc else None

    if getattr(cfg, "zscore_norm", False) and rna_mean is not None:
        Xrna_te = zscore_transform(Xrna_te, rna_mean, rna_std)
        if cfg.use_lnc and lnc_mean is not None:
            Xlnc_te = zscore_transform(Xlnc_te, lnc_mean, lnc_std)
        else:
            Xlnc_te = Xlnc_te if cfg.use_lnc else None
    else:
        Xlnc_te = Xlnc_te if cfg.use_lnc else None

    Xnodes_te = build_node_features_for_model(Xrna_te, Xlnc_te if cfg.use_lnc else None, cfg)

    # Adjacency matrix
    adj_all = load_adjacency_from_csv(cfg.adj_csv_path)
    adj_bin = prepare_adjacency(adj_all, cfg)

    # Samples to explain
    patient_indices = list(range(Xnodes_te.shape[0]))

    # Output directory keyed by dataset name
    dataset_tag = os.path.splitext(os.path.basename(CSV_PATH))[0]
    subdir = f"surv_shapley_mc_{dataset_tag}"
    out_dir = os.path.join(cfg.results_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    for idx in patient_indices:
        x_nodes_patient = Xnodes_te[idx].squeeze(-1)
        x_nodes_patient = x_nodes_patient.unsqueeze(-1)

        print(f"\n[Explain] patient index={idx}, sample_id={test_ids[idx]}")
        phi, f_full, f_empty = explain_one_patient_shapley_mc(
            cfg=cfg,
            model=model,
            x_real=x_nodes_patient,
            baseline=baseline_nodes.unsqueeze(-1),
            adj_bin=adj_bin,
            num_permutations=64,
            random_seed=cfg.seed + idx,
            device=device,
        )

        # L1-normalized contribution
        abs_phi = np.abs(phi)
        phi_rel = phi / (abs_phi.sum() + 1e-12)

        # Save results
        out_path = os.path.join(out_dir, f"patient_{idx}_node_shapley.csv")
        df = pd.DataFrame({
            "node_index": np.arange(len(phi)),
            "phi": phi,
            "phi_rel": phi_rel,
        })
        df.to_csv(out_path, index=False)
        print(f"  -> Saved node shapley to {out_path}")

    print("\nDone Monte Carlo survival Shapley explanations.")


if __name__ == "__main__":
    main()