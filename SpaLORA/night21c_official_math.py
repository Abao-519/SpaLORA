"""spaMGCN official-math compatibility utilities for Night-21C.

The neural modules are imported from an immutable snapshot of upstream commit
77dfe67d4fd80c124722e68a0f71af36d10fa5fa.  This file only adapts sanitized
numeric inputs, label-free fixed-epoch training, checkpoint replay, and audit
metadata.  The dense loss and propagation formulas follow upstream train3.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
import random
import sys
from typing import Mapping, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


UPSTREAM_COMMIT = "77dfe67d4fd80c124722e68a0f71af36d10fa5fa"


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def robust_standardize(value: np.ndarray) -> np.ndarray:
    """Night-21B-compatible input scaling for already reduced feature views."""
    value = np.asarray(value, dtype=np.float64)
    median = np.median(value, axis=0, keepdims=True)
    mad = np.median(np.abs(value - median), axis=0, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-6)
    result = np.clip((value - median) / scale, -8.0, 8.0)
    if not np.isfinite(result).all():
        raise ValueError("non-finite standardized input")
    return result.astype(np.float32)


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def official_preprocess_graph(graph: sp.spmatrix) -> sp.csr_matrix:
    """Upstream preprocess_graph on a registered graph support.

    Upstream constructs an unweighted coordinate kNN graph, symmetrizes it,
    clips duplicate edges to one, adds identity, then applies D^-1/2 A D^-1/2.
    Carriers may store row-normalized weights, so only their registered support
    is used before applying the upstream normalization.
    """
    graph = sp.csr_matrix(graph, dtype=np.float64)
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be square")
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.nnz == 0:
        raise ValueError("registered graph is empty")
    graph.data[:] = 1.0
    adjacent = graph + sp.eye(graph.shape[0], format="csr", dtype=np.float64)
    degree = np.asarray(adjacent.sum(axis=1)).ravel()
    if np.any(degree <= 0) or not np.isfinite(degree).all():
        raise ValueError("invalid graph degree")
    inv = 1.0 / np.sqrt(degree)
    normalized = (sp.diags(inv) @ adjacent @ sp.diags(inv)).tocsr()
    if not np.isfinite(normalized.data).all():
        raise ValueError("non-finite normalized adjacency")
    return normalized.astype(np.float32)


def scipy_to_torch_sparse(graph: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = graph.tocoo()
    indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()


@dataclass(frozen=True)
class OfficialMathConfig:
    profile_id: str
    epochs: int
    learning_rate: float
    sigma: float
    loss_n: float
    loss_w: float = 0.1
    loss_s: float = 0.1
    loss_a: float = 0.1
    n_z: int = 20
    ae_hidden_1: int = 128
    ae_hidden_2: int = 256
    graph_order: int = 4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def import_official_class(snapshot_root: str):
    mgcn_root = os.path.join(os.path.abspath(snapshot_root), "MGCN-main")
    required = [
        os.path.join(mgcn_root, "model", "AE.py"),
        os.path.join(mgcn_root, "model", "IGAE.py"),
        os.path.join(mgcn_root, "model", "spaMGCN.py"),
    ]
    if not all(os.path.isfile(path) for path in required):
        raise FileNotFoundError("incomplete fixed upstream model snapshot")
    if mgcn_root not in sys.path:
        sys.path.insert(0, mgcn_root)
    module = importlib.import_module("model.spaMGCN")
    return module.spaMGCN


def build_official_model(
    snapshot_root: str,
    view1_dim: int,
    view2_dim: int,
    k: int,
    config: OfficialMathConfig,
    device: torch.device,
):
    cls = import_official_class(snapshot_root)
    model = cls(
        ae_n_enc_1=config.ae_hidden_1,
        ae_n_enc_2=config.ae_hidden_2,
        ae_n_enc_3=config.n_z,
        ae_n_dec_1=config.n_z,
        ae_n_dec_2=config.ae_hidden_2,
        ae_n_dec_3=config.ae_hidden_1,
        gae_n_enc_1=config.ae_hidden_1,
        gae_n_enc_2=config.ae_hidden_2,
        gae_n_enc_3=config.n_z,
        gae_n_dec_1=config.n_z,
        gae_n_dec_2=config.ae_hidden_2,
        gae_n_dec_3=config.ae_hidden_1,
        n_input=int(view1_dim),
        n_input1=int(view2_dim),
        n_z=config.n_z,
        n_clusters=int(k),
        sigma=float(config.sigma),
        v=1.0,
    )
    return model.to(device)


def official_fused_embedding(model, x1: torch.Tensor, x2: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """Exact encoder/fusion subpath from upstream spaMGCN.forward without unused dense outputs."""
    sigma = model.sigma
    za11, za12, za13 = model.ae.encoder(x1)
    zg11 = model.gae.encoder.gnn_1(x1, adjacency)
    zg12 = model.gae.encoder.gnn_2((1 - sigma) * za11 + sigma * zg11, adjacency)
    zg13 = model.gae.encoder.gnn_3((1 - sigma) * za12 + sigma * zg12, adjacency, active=False)
    z1 = (1 - sigma) * za13 + sigma * zg13

    za21, za22, za23 = model.ae1.encoder(x2)
    zg21 = model.gae1.encoder.gnn_1(x2, adjacency)
    zg22 = model.gae1.encoder.gnn_2((1 - sigma) * za21 + sigma * zg21, adjacency)
    zg23 = model.gae1.encoder.gnn_3((1 - sigma) * za22 + sigma * zg22, adjacency, active=False)
    z2 = (1 - sigma) * za23 + sigma * zg23
    return torch.cat([z1, z2], dim=1)


def upstream_cosine_similarity(embedding: torch.Tensor) -> torch.Tensor:
    matrix = torch.matmul(embedding, embedding.T)
    length = torch.norm(embedding, p=2, dim=1)
    denom = torch.matmul(length.reshape((-1, 1)), length.reshape((-1, 1)).T) - 5e-12
    result = torch.div(matrix, denom)
    if torch.any(torch.isnan(result)):
        result = torch.where(torch.isnan(result), torch.full_like(result, 0.4868), result)
    return result


def upstream_noise_cross_entropy(embedding: torch.Tensor, dense_adjacency: torch.Tensor) -> torch.Tensor:
    similarity = upstream_cosine_similarity(embedding)
    exponential = torch.exp(similarity)
    negative = torch.mul(exponential, 1 - dense_adjacency).sum(axis=1)
    positive = torch.mul(exponential, dense_adjacency).sum(axis=1)
    return -torch.log(torch.div(positive, negative)).mean()


def official_dense_loss(
    model,
    x1: torch.Tensor,
    x2: torch.Tensor,
    adjacency: torch.Tensor,
    dense_adjacency: torch.Tensor,
    config: OfficialMathConfig,
    epoch: int,
) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor], torch.Tensor]:
    outputs = model(x1, x2, adjacency)
    x1_hat, z1_hat, adj1_hat, z1_ae, z1_graph = outputs[:5]
    x2_hat, z2_hat, adj2_hat, z2_ae, z2_graph, fused = outputs[5:]
    fused_logits = torch.mm(fused, fused.T)
    adjacency_bce = F.binary_cross_entropy_with_logits(fused_logits, dense_adjacency)
    noise_cross_entropy = upstream_noise_cross_entropy(fused, dense_adjacency)
    reconstruction = F.mse_loss(x1_hat, x1) + F.mse_loss(x2_hat, x2)
    graph_feature = (
        F.mse_loss(z1_hat, torch.spmm(adjacency, x1))
        + F.mse_loss(z2_hat, torch.spmm(adjacency, x2))
    )
    consistency = F.mse_loss(z1_graph, z1_ae) + F.mse_loss(z2_graph, z2_ae)
    dense_adj_hat_unused = F.mse_loss(adj1_hat, dense_adjacency) + F.mse_loss(adj2_hat, dense_adjacency)
    total = reconstruction + config.loss_w * graph_feature + config.loss_a * adjacency_bce + config.loss_s * consistency
    nce_active = bool(epoch > config.epochs * 0.1)
    if nce_active:
        total = total + config.loss_n * noise_cross_entropy
    pieces = {
        "total": total,
        "reconstruction": reconstruction,
        "graph_feature": graph_feature,
        "adjacency_bce": adjacency_bce,
        "consistency": consistency,
        "noise_cross_entropy": noise_cross_entropy,
        "dense_adj_hat_unused": dense_adj_hat_unused,
        "nce_active": torch.as_tensor(float(nce_active), device=total.device),
    }
    return total, pieces, fused


def _state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def train_official_math(
    view1: np.ndarray,
    view2: np.ndarray,
    graph: sp.csr_matrix,
    k: int,
    config: OfficialMathConfig,
    snapshot_root: str,
    seed: int,
    device: str = "cuda",
):
    set_determinism(seed)
    dev = torch.device(device)
    x1_np, x2_np = robust_standardize(view1), robust_standardize(view2)
    graph_np = official_preprocess_graph(graph)
    x1 = torch.as_tensor(x1_np, device=dev)
    x2 = torch.as_tensor(x2_np, device=dev)
    adjacency = scipy_to_torch_sparse(graph_np, dev)
    dense_adjacency = adjacency.to_dense()
    model = build_official_model(snapshot_root, x1_np.shape[1], x2_np.shape[1], k, config, dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    before = _state_sha(model)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    checkpoints = {0, max(0, int(math.floor(config.epochs * 0.1))), config.epochs // 2, config.epochs - 1}
    snapshots: list[dict[str, float]] = []
    loss_history: list[float] = []
    max_grad = 0.0
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, pieces, _ = official_dense_loss(model, x1, x2, adjacency, dense_adjacency, config, epoch)
        if not torch.isfinite(total):
            raise RuntimeError(f"non-finite official loss at epoch {epoch}")
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        if not torch.isfinite(grad):
            raise RuntimeError(f"non-finite official gradient at epoch {epoch}")
        max_grad = max(max_grad, float(grad.detach().cpu()))
        optimizer.step()
        loss_history.append(float(total.detach().cpu()))
        if epoch in checkpoints:
            snapshots.append({"epoch": epoch + 1, **{name: float(value.detach().cpu()) for name, value in pieces.items()}})
        del total, pieces
    model.eval()
    with torch.no_grad():
        representation = official_fused_embedding(model, x1, x2, adjacency).detach().cpu().numpy().astype(np.float32)
    after = _state_sha(model)
    if before == after or max_grad <= 0:
        raise RuntimeError("official optimizer did not change parameters")
    if not np.isfinite(representation).all():
        raise RuntimeError("non-finite official representation")
    window = min(50, max(1, len(loss_history) // 2))
    previous = float(np.mean(loss_history[-2 * window:-window]))
    recent = float(np.mean(loss_history[-window:]))
    diagnostics = {
        "optimizer_steps": config.epochs,
        "parameter_changed": True,
        "parameter_state_sha_before": before,
        "parameter_state_sha_after": after,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "max_gradient_norm": max_grad,
        "loss_snapshots": snapshots,
        "loss_relative_change_last_windows": (recent - previous) / max(abs(previous), 1e-12),
        "representation_sha256": sha256_array(representation),
        "input_shapes": {"view1": list(view1.shape), "view2": list(view2.shape)},
        "graph_shape": list(graph.shape),
        "graph_input_nnz": int(graph.nnz),
        "graph_official_nnz": int(graph_np.nnz),
        "dense_nxn_shape": [int(graph.shape[0]), int(graph.shape[0])],
        "dense_nxn_allocated": True,
        "dense_target_bytes": int(graph.shape[0] * graph.shape[0] * 4),
        "peak_gpu_allocated_mb": float(torch.cuda.max_memory_allocated(dev) / (1024 ** 2)) if dev.type == "cuda" else 0.0,
    }
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    return representation, state, diagnostics


def reload_official_embedding(
    view1: np.ndarray,
    view2: np.ndarray,
    graph: sp.csr_matrix,
    k: int,
    config: OfficialMathConfig,
    snapshot_root: str,
    seed: int,
    state: Mapping[str, torch.Tensor],
    device: str = "cpu",
) -> np.ndarray:
    set_determinism(seed)
    dev = torch.device(device)
    x1_np, x2_np = robust_standardize(view1), robust_standardize(view2)
    graph_np = official_preprocess_graph(graph)
    x1 = torch.as_tensor(x1_np, device=dev)
    x2 = torch.as_tensor(x2_np, device=dev)
    adjacency = scipy_to_torch_sparse(graph_np, dev)
    model = build_official_model(snapshot_root, x1_np.shape[1], x2_np.shape[1], k, config, dev)
    incompatible = model.load_state_dict(dict(state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint reload failed")
    model.eval()
    with torch.no_grad():
        result = official_fused_embedding(model, x1, x2, adjacency)
    return result.detach().cpu().numpy().astype(np.float32)


def common_kmeans(representation: np.ndarray, k: int, seed: int = 0, n_init: int = 20) -> np.ndarray:
    partition = KMeans(n_clusters=int(k), n_init=int(n_init), random_state=int(seed), algorithm="lloyd").fit_predict(
        np.asarray(representation, dtype=np.float32)
    )
    if np.unique(partition).size != int(k):
        raise RuntimeError("KMeans failed exact K")
    return partition.astype(np.int32)
