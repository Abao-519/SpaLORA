"""Dataset-name-blind unified residual fusion primitives for Night-13B.

The scientific core receives only a fused observation embedding and two sparse
spatial operators.  Assay-specific preprocessing and label-based evaluation live
outside this module.  No dataset, tissue, family, file path, label or metric is
accepted by the model API.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


FORBIDDEN_CONFIG_KEYS = {
    "dataset", "dataset_name", "family", "tissue", "path", "label", "labels",
    "ground_truth", "ari", "nmi", "ami", "fmi", "q",
}


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def ordered_id_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest()


def canonical_config_sha256(value: Mapping[str, object]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def canonical_csr(matrix: sp.spmatrix) -> sp.csr_matrix:
    value = matrix.tocsr().astype(np.float64)
    value.sum_duplicates()
    value.eliminate_zeros()
    value.sort_indices()
    return value


def row_stochastic(matrix: sp.spmatrix) -> sp.csr_matrix:
    value = canonical_csr(matrix)
    degree = np.asarray(value.sum(axis=1)).ravel()
    if np.any(~np.isfinite(degree)) or np.any(degree <= 0):
        raise ValueError("sparse operator has a zero or non-finite degree")
    return canonical_csr(sp.diags(1.0 / degree) @ value)


def scipy_to_torch(matrix: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = canonical_csr(matrix).tocoo()
    indices = torch.as_tensor(np.vstack((coo.row, coo.col)),
                              dtype=torch.long, device=device)
    values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape,
                                   device=device).coalesce()


class ContentAdaptiveGraphResidual(nn.Module):
    """One global continuous rule for a sparse multi-scale graph residual.

    The global discrepancy is measured on a row-normalized copy of the input,
    while the residual is applied to the original embedding.  This preserves the
    common endpoint's Euclidean geometry when the gate is effectively zero.
    """

    def __init__(self, config: Mapping[str, object]):
        super().__init__()
        if FORBIDDEN_CONFIG_KEYS & set(config):
            raise ValueError("model config contains an identity, label or metric key")
        required = {"threshold", "slope", "max_residual", "fine_k", "broad_k"}
        if set(config) != required:
            raise ValueError("unified residual config schema mismatch")
        if int(config["fine_k"]) >= int(config["broad_k"]):
            raise ValueError("fine_k must be smaller than broad_k")
        self.config = dict(config)
        # Kept in the checkpoint and used by the gradient smoke. Formal runs use
        # the frozen zero value; no labels or within-run selection touch it.
        self.threshold_offset = nn.Parameter(torch.zeros(()))

    def forward(self, embedding: torch.Tensor, fine: torch.Tensor,
                broad: torch.Tensor) -> Dict[str, torch.Tensor]:
        if embedding.ndim != 2 or embedding.shape[0] != fine.shape[0]:
            raise ValueError("embedding/operator shape contract failed")
        normalized = F.normalize(embedding, p=2, dim=1, eps=1e-12)
        fine_normalized = F.normalize(torch.sparse.mm(fine, normalized),
                                      p=2, dim=1, eps=1e-12)
        discrepancy = (1.0 - (normalized * fine_normalized).sum(dim=1)).mean()
        threshold = float(self.config["threshold"]) + self.threshold_offset
        gate = torch.sigmoid(float(self.config["slope"]) *
                             (discrepancy - threshold))
        beta = float(self.config["max_residual"]) * gate
        fine_embedding = torch.sparse.mm(fine, embedding)
        broad_embedding = torch.sparse.mm(broad, embedding)
        multiscale = (1.0 - gate) * fine_embedding + gate * broad_embedding
        fused = (1.0 - beta) * embedding + beta * multiscale
        return {
            "fused": fused,
            "fine_embedding": fine_embedding,
            "broad_embedding": broad_embedding,
            "discrepancy": discrepancy,
            "gate": gate,
            "beta": beta,
        }


def gradient_probe_loss(output: Mapping[str, torch.Tensor],
                        embedding: torch.Tensor) -> torch.Tensor:
    """Unsupervised-only probe; used to verify a finite nonzero gradient."""
    spatial = F.mse_loss(output["fused"], output["fine_embedding"])
    preservation = F.mse_loss(output["fused"], embedding)
    return spatial + 0.25 * preservation


def model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def mean_binary_geary(labels: Sequence[int], graph: sp.spmatrix) -> float:
    y = np.asarray(labels)
    w = canonical_csr(graph)
    total = float(w.sum())
    if w.shape != (len(y), len(y)) or total <= 0:
        raise ValueError("Geary graph contract failed")
    rows, cols = w.nonzero()
    values = []
    for group in np.unique(y):
        x = (y == group).astype(np.float64)
        denominator = float(np.sum((x - x.mean()) ** 2))
        if denominator <= 0:
            continue
        numerator = float(np.sum(w.data * (x[rows] - x[cols]) ** 2))
        values.append((len(x) - 1.0) * numerator / (2.0 * total * denominator))
    return float(np.mean(values)) if values else 0.0


__all__ = [
    "ContentAdaptiveGraphResidual", "array_sha256", "canonical_config_sha256",
    "gradient_probe_loss", "mean_binary_geary", "model_state_sha256",
    "ordered_id_sha256", "row_stochastic", "scipy_to_torch",
]
