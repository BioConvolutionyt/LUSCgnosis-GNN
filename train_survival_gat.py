import os
import math
import copy
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================
# Configuration
# =============================
@dataclass
class Config:
    # Input data paths (replace before running)
    train_csv: str = "REPLACE_WITH_TRAIN_CSV_PATH"
    val_csv: str = "REPLACE_WITH_VAL_CSV_PATH"

    # Runtime
    seed: int = 24
    use_cuda_if_available: bool = True

    # Input feature layout
    label_dim: int = 1
    input_node_feature_dim: int = 1
    use_lnc: bool = True

    # Graph
    num_rna_nodes: int = 240
    num_lnc_nodes: int = 80
    adj_threshold: float = 0.09

    # Model architecture
    nhids: Tuple[int, int] = (18, 46)
    nheads: Tuple[int, int] = (8, 4)
    fc_dims: Tuple[int, int, int] = (104, 45, 84)
    gat_dropout: float = 0.22
    fc_dropout: float = 0.3
    leakyrelu_alpha: float = 0.2

    # Optimization
    batch_size: int = 18
    num_epochs: int = 150
    lr: float = 0.0003
    weight_decay: float = 0.00001
    lambda_cox: float = 1.0
    lambda_reg_l1: float = 0.00955

    # Normalization
    # If both are True, min-max is applied before z-score.
    zscore_norm: bool = True
    minmax_norm: bool = False

    # Output paths (replace before running)
    model_path: str = "REPLACE_WITH_MODEL_OUTPUT_PATH"
    results_dir: str = "REPLACE_WITH_RESULTS_DIR"
    # Precomputed adjacency CSV (replace before running)
    adj_csv_path: str = "REPLACE_WITH_ADJACENCY_CSV_PATH"


# =============================
# Utils
# =============================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_prognosis_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read survival modeling data.
    Expected column layout (1-based):
      1: sample_id
      2-81: lncRNA (80)
      82-321: RNA (240)
      322: OS
      323: OS.time
    """
    df = pd.read_csv(path)

    sample_ids = df.iloc[:, 0].astype(str).values
    X_lnc = df.iloc[:, 1:81].astype(float).values
    X_rna = df.iloc[:, 81:321].astype(float).values
    event = df.iloc[:, 321].astype(float).values
    time = df.iloc[:, 322].astype(float).values

    return sample_ids, X_lnc, X_rna, event, time



def load_adjacency_from_csv(path: str) -> torch.Tensor:
    """Load a precomputed adjacency matrix."""
    mat = pd.read_csv(path).values.astype(np.float32)
    return torch.tensor(mat)


def prepare_adjacency(adj: torch.Tensor, cfg: Config) -> torch.Tensor:
    """
    Select adjacency shape based on feature setup and binarize it.
    - use_lnc=True: accepts 320x320 adjacency (lncRNA + RNA graph)
    - use_lnc=False: accepts 240x240 adjacency, or crops RNA subgraph from 320x320
    """
    if cfg.use_lnc:
        if adj.ndim != 2 or adj.size(0) != adj.size(1) or adj.size(0) not in (320,):
            raise ValueError(f"Adjacency expected 320x320 when use_lnc=True, got {tuple(adj.size())}")
        adj_use = adj
    else:
        if adj.ndim != 2 or adj.size(0) != adj.size(1):
            raise ValueError(f"Adjacency must be square, got {tuple(adj.size())}")
        if adj.size(0) == 320:
            adj_use = adj[cfg.num_lnc_nodes:, cfg.num_lnc_nodes:]
        if adj.size(0) == 240:
            adj_use = adj
        if adj.size(0) not in (240, 320):
            raise ValueError(f"Adjacency expected 320x320 or 240x240 when use_lnc=False, got {tuple(adj.size())}")

    # Binary adjacency mask
    adj_bin = (adj_use >= cfg.adj_threshold).to(dtype=torch.float32)
    return adj_bin


def zscore_fit_transform(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mean) / std, mean, std


def zscore_transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / (std + 1e-8)


def minmax_fit_transform(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply column-wise min-max normalization and return training statistics."""
    vmin = train.min(axis=0, keepdims=True)
    vmax = train.max(axis=0, keepdims=True)
    denom = (vmax - vmin) + 1e-8
    return (train - vmin) / denom, vmin, vmax


def minmax_transform(x: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    denom = (vmax - vmin) + 1e-8
    return (x - vmin) / denom


class SurvivalDataset(Dataset):
    def __init__(self, Xnodes: torch.Tensor, event: torch.Tensor, time: torch.Tensor):
        self.Xnodes = Xnodes
        self.event = event.float()
        self.time = time.float()

    def __len__(self):
        return self.Xnodes.size(0)

    def __getitem__(self, idx):
        return self.Xnodes[idx], self.event[idx], self.time[idx]


# =============================
# Model
# =============================
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float, alpha: float, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [batch, nodes, features], adj: [nodes, nodes]
        h = torch.matmul(x, self.W)  # [bs, N, F']
        bs, N, Fp = h.size()

        a_input = torch.cat([
            h.repeat(1, 1, N).view(bs, N * N, Fp),
            h.repeat(1, N, 1)
        ], dim=-1).view(bs, N, N, 2 * Fp)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [bs, N, N]
        batch_adj = adj.unsqueeze(0).repeat(bs, 1, 1)
        zero_vec = (-9e15) * torch.ones_like(e)
        attention = torch.where(batch_adj > 0, e, zero_vec)
        attention = self.dropout_layer(F.softmax(attention, dim=-1))  # [bs, N, N]
        h_prime = torch.bmm(attention, h)  # [bs, N, F']

        if self.concat:
            return F.elu(h_prime)
        return h_prime


class GATSurv(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.dropout_layer = nn.Dropout(p=cfg.gat_dropout)

        nhids = cfg.nhids
        nheads = cfg.nheads
        fc_dims = cfg.fc_dims

        self.attentions1 = nn.ModuleList([
            GraphAttentionLayer(cfg.input_node_feature_dim, nhids[0], dropout=cfg.gat_dropout, alpha=cfg.leakyrelu_alpha, concat=True)
            for _ in range(nheads[0])
        ])
        self.attentions2 = nn.ModuleList([
            GraphAttentionLayer(nhids[0] * nheads[0], nhids[1], dropout=cfg.gat_dropout, alpha=cfg.leakyrelu_alpha, concat=True)
            for _ in range(nheads[1])
        ])

        # Node-level pooling after each GAT block
        self.pool1 = nn.Linear(nhids[0] * nheads[0], 1)
        self.pool2 = nn.Linear(nhids[1] * nheads[1], 1)

        # Multi-scale concatenation: [x0, x1, x2]
        num_nodes = (cfg.num_lnc_nodes + cfg.num_rna_nodes) if cfg.use_lnc else cfg.num_rna_nodes
        lin_input_dim = 3 * num_nodes

        self.encoder = nn.Sequential(
            nn.Linear(lin_input_dim, fc_dims[0]),
            nn.ELU(),
            nn.AlphaDropout(p=cfg.fc_dropout, inplace=False),
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.ELU(),
            nn.AlphaDropout(p=cfg.fc_dropout, inplace=False),
            nn.Linear(fc_dims[1], fc_dims[2]),
            nn.ELU(),
            nn.AlphaDropout(p=cfg.fc_dropout, inplace=False),
        )

        self.classifier = nn.Linear(fc_dims[2], 1)

    def forward(self, xnodes: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # xnodes: [batch, nodes, 1], adj: [nodes, nodes]
        x0 = torch.mean(xnodes, dim=-1)  # [bs, nodes]
        x = self.dropout_layer(xnodes)
        x = torch.cat([att(x, adj) for att in self.attentions1], dim=-1)  # [bs, nodes, nhid1*heads1]
        x1 = self.pool1(x).squeeze(-1)  # [bs, nodes]

        x = self.dropout_layer(x)
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=-1)  # [bs, nodes, nhid2*heads2]
        x2 = self.pool2(x).squeeze(-1)  # [bs, nodes]

        x_concat = torch.cat([x0, x1, x2], dim=1)  # [bs, 3*nodes]
        features = self.encoder(x_concat)
        hazard = self.classifier(features).reshape(-1)  # [bs]
        return x_concat, features, hazard


# =============================
# Loss & Metrics
# =============================
def cox_ph_loss(surv_time: torch.Tensor, event: torch.Tensor, hazard: torch.Tensor) -> torch.Tensor:
    """Mini-batch negative partial log-likelihood for Cox PH."""
    device = hazard.device
    with torch.no_grad():
        risk_mat = (surv_time.view(1, -1) >= surv_time.view(-1, 1)).float().to(device)
    theta = hazard.view(-1)
    exp_theta = torch.exp(theta)
    loss = -torch.mean((theta - torch.log(torch.sum(exp_theta * risk_mat, dim=1))) * event)
    return loss


def c_index(hazard: np.ndarray, event: np.ndarray, time: np.ndarray) -> float:
    """Harrell's C-index. Higher hazard indicates higher risk."""
    N = len(hazard)
    concord, total = 0.0, 0.0
    for i in range(N):
        if event[i] != 1:
            continue
        for j in range(N):
            if time[j] <= time[i]:
                continue
            total += 1.0
            if hazard[j] < hazard[i]:
                concord += 1.0
            elif math.isclose(hazard[j], hazard[i], rel_tol=1e-9, abs_tol=1e-9):
                concord += 0.5
    return float(concord / total) if total > 0 else float("nan")


# =============================
# Train / Evaluate
# =============================
def train_one_epoch(model: GATSurv, loader: DataLoader, adj240: torch.Tensor, optimizer: torch.optim.Optimizer, cfg: Config, device: torch.device) -> float:
    model.train()
    epoch_loss = 0.0
    total = 0
    for X320, event, time in loader:
        X320 = X320.to(device)
        event = event.to(device)
        time = time.to(device)

        optimizer.zero_grad()
        _, _, hazard = model(X320, adj240)
        loss = cfg.lambda_cox * cox_ph_loss(time, event, hazard)

        # L1 regularization
        l1 = 0.0
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
        loss = loss + cfg.lambda_reg_l1 * l1

        loss.backward()
        optimizer.step()

        bs = X320.size(0)
        epoch_loss += loss.item() * bs
        total += bs

    return epoch_loss / max(total, 1)


@torch.no_grad()
def evaluate(model: GATSurv, loader: DataLoader, adj240: torch.Tensor, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_hazard, all_event, all_time = [], [], []
    loss_sum, total = 0.0, 0

    for X320, event, time in loader:
        X320 = X320.to(device)
        event = event.to(device)
        time = time.to(device)
        _, _, hazard = model(X320, adj240)
        loss = cox_ph_loss(time, event, hazard)

        bs = X320.size(0)
        loss_sum += loss.item() * bs
        total += bs

        all_hazard.append(hazard.detach().cpu().numpy())
        all_event.append(event.detach().cpu().numpy())
        all_time.append(time.detach().cpu().numpy())

    hazard_np = np.concatenate(all_hazard, axis=0)
    event_np = np.concatenate(all_event, axis=0)
    time_np = np.concatenate(all_time, axis=0)
    avg_loss = loss_sum / max(total, 1)
    return avg_loss, hazard_np, event_np, time_np


def build_node_features_for_model(X_rna: np.ndarray, X_lnc: Optional[np.ndarray], cfg: Config) -> torch.Tensor:
    """
    Build node feature tensors:
    - use_lnc=True: [N, 320, 1] (lnc80 + RNA240)
    - use_lnc=False: [N, 240, 1] (RNA only)
    """
    N = X_rna.shape[0]
    if cfg.use_lnc:
        assert X_lnc is not None and X_lnc.shape[1] == cfg.num_lnc_nodes, "X_lnc must be (N, 80) when use_lnc=True"
        Xnodes = np.zeros((N, cfg.num_lnc_nodes + cfg.num_rna_nodes), dtype=np.float32)
        Xnodes[:, :cfg.num_lnc_nodes] = X_lnc.astype(np.float32)
        Xnodes[:, cfg.num_lnc_nodes:] = X_rna.astype(np.float32)
    else:
        Xnodes = X_rna.astype(np.float32)
    Xnodes = torch.tensor(Xnodes, dtype=torch.float32).unsqueeze(-1)
    return Xnodes


def main():
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if (cfg.use_cuda_if_available and torch.cuda.is_available()) else "cpu")

    # 1) Load train/validation data
    train_ids, Xlnc_tr, Xrna_tr, event_tr, time_tr = read_prognosis_csv(cfg.train_csv)
    _, Xlnc_val, Xrna_val, event_val, time_val = read_prognosis_csv(cfg.val_csv)

    # 2) Normalize features using training statistics
    if cfg.minmax_norm:
        Xrna_tr, rna_min, rna_max = minmax_fit_transform(Xrna_tr)
        Xrna_val = minmax_transform(Xrna_val, rna_min, rna_max)
        if cfg.use_lnc:
            Xlnc_tr, lnc_min, lnc_max = minmax_fit_transform(Xlnc_tr)
            Xlnc_val = minmax_transform(Xlnc_val, lnc_min, lnc_max)

    if cfg.zscore_norm:
        Xrna_tr, rna_mean, rna_std = zscore_fit_transform(Xrna_tr)
        Xrna_val = zscore_transform(Xrna_val, rna_mean, rna_std)
        if cfg.use_lnc:
            Xlnc_tr, lnc_mean, lnc_std = zscore_fit_transform(Xlnc_tr)
            Xlnc_val = zscore_transform(Xlnc_val, lnc_mean, lnc_std)

    # 3) Load adjacency matrix
    adj_all = load_adjacency_from_csv(cfg.adj_csv_path)
    adj_use = prepare_adjacency(adj_all, cfg).to(device)

    # 4) Build node-feature tensors and labels
    Xtr_nodes = build_node_features_for_model(Xrna_tr, Xlnc_tr if cfg.use_lnc else None, cfg)
    Xval_nodes = build_node_features_for_model(Xrna_val, Xlnc_val if cfg.use_lnc else None, cfg)
    event_tr_t = torch.tensor(event_tr, dtype=torch.float32)
    time_tr_t = torch.tensor(time_tr, dtype=torch.float32)
    event_val_t = torch.tensor(event_val, dtype=torch.float32)
    time_val_t = torch.tensor(time_val, dtype=torch.float32)

    # 5) Create data loaders
    train_ds = SurvivalDataset(Xtr_nodes, event_tr_t, time_tr_t)
    val_ds = SurvivalDataset(Xval_nodes, event_val_t, time_val_t)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # 6) Initialize model and optimizer
    model = GATSurv(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_cindex = -np.inf
    best_state = None

    # 7) Train with validation monitoring
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, adj_use, optimizer, cfg, device)
        val_loss, haz_val, evt_val, tim_val = evaluate(model, val_loader, adj_use, device)
        cind_val = c_index(haz_val, evt_val, tim_val)

        print(f"Epoch {epoch:03d}/{cfg.num_epochs} | TrainLoss {train_loss:.6f} | ValLoss {val_loss:.6f} | Val C-index {cind_val:.4f}")

        if np.isfinite(cind_val) and cind_val > best_val_cindex:
            best_val_cindex = cind_val
            best_state = copy.deepcopy(model.state_dict())

    # 8) Restore best model and save artifacts
    if best_state is not None:
        model.load_state_dict(best_state)
    cindex_to_save = float(best_val_cindex) if np.isfinite(best_val_cindex) else float("nan")
    print(f"Best Val C-index during training: {cindex_to_save:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "cindex_val": cindex_to_save
    }, cfg.model_path)
    print(f"Model saved to: {cfg.model_path}")

    # Save adjacency used in training
    n_nodes = (cfg.num_lnc_nodes + cfg.num_rna_nodes) if cfg.use_lnc else cfg.num_rna_nodes
    adj_path = os.path.join(cfg.results_dir, f"adjacency_{n_nodes}x{n_nodes}.npy")
    np.save(adj_path, adj_use.detach().cpu().numpy())
    print(f"Adjacency saved to: {adj_path}")


if __name__ == "__main__":
    main()


