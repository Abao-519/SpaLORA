"""Label-free MF-RACF primitives for the locked Night-9B registry.

The module receives only opaque arrays, observation ids, coordinates and a
registered candidate record.  It deliberately has no dataset loader, label
path, evaluator, identity router, or metric-selection code.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F


VIEW_KEYS = ("emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused",
             "alpha_omics1", "alpha_omics2", "alpha_cross")
FORBIDDEN_KEYS = {"label", "labels", "ground_truth", "ari", "nmi", "q",
                  "dataset", "dataset_name", "tissue", "platform", "file_name"}


def canonical_json_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def array_sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode())
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def canonical_csr(value: sp.spmatrix) -> sp.csr_matrix:
    result = value.tocsr().astype(np.float64)
    result.sum_duplicates(); result.eliminate_zeros(); result.sort_indices()
    return result


def sparse_sha(value: sp.spmatrix) -> str:
    matrix = canonical_csr(value)
    h = hashlib.sha256()
    for array in (matrix.indptr.astype(np.int64), matrix.indices.astype(np.int64),
                  matrix.data.astype(np.float64)):
        h.update(np.ascontiguousarray(array).tobytes())
    h.update(json.dumps(list(matrix.shape), separators=(",", ":")).encode())
    return h.hexdigest()


def row_l2_np(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def row_l2(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value, p=2, dim=1, eps=1e-12)


def _directed_knn(values: np.ndarray, ids: Sequence[str], k: int,
                  metric: str) -> sp.csr_matrix:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n <= int(k) or len(ids) != n:
        raise ValueError("kNN shape or k contract failed")
    model = NearestNeighbors(n_neighbors=int(k) + 1, metric=metric, n_jobs=1)
    model.fit(values)
    distances, indices = model.kneighbors(values, return_distance=True)
    names = np.asarray(ids, dtype=str)
    rows, cols = [], []
    for i in range(n):
        choices = [(float(d), str(names[j]), int(j))
                   for d, j in zip(distances[i], indices[i]) if int(j) != i]
        for _, _, j in sorted(choices)[:int(k)]:
            rows.append(i); cols.append(j)
    return canonical_csr(sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                       shape=(n, n)))


def symmetric_knn(values: np.ndarray, ids: Sequence[str], k: int,
                  metric: str) -> sp.csr_matrix:
    directed = _directed_knn(values, ids, k, metric)
    return canonical_csr(directed.maximum(directed.T))


def row_stochastic_with_loops(value: sp.spmatrix) -> sp.csr_matrix:
    matrix = canonical_csr(value + sp.eye(value.shape[0], format="csr"))
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(~np.isfinite(degree)) or np.any(degree <= 0):
        raise RuntimeError("common graph contains a zero or non-finite degree")
    return canonical_csr(sp.diags(1.0 / degree) @ matrix)


def rna_common_graph(rna: np.ndarray, coordinates: np.ndarray,
                     ids: Sequence[str], k: int) -> tuple[sp.csr_matrix, dict]:
    """Cosine RNA-feature kNN intersect Euclidean spatial kNN, then symmetrize."""
    feature = symmetric_knn(row_l2_np(rna), ids, int(k), "cosine")
    spatial = symmetric_knn(np.asarray(coordinates), ids, int(k), "euclidean")
    intersection = canonical_csr(feature.multiply(spatial))
    intersection = canonical_csr(intersection.maximum(intersection.T))
    result = row_stochastic_with_loops(intersection)
    support = canonical_csr((result != 0).astype(np.float64))
    feature_support = canonical_csr((feature != 0).astype(np.float64))
    spatial_support = canonical_csr((spatial != 0).astype(np.float64))
    outside = canonical_csr(support - support.multiply(feature_support.maximum(sp.eye(len(rna)))))
    outside2 = canonical_csr(support - support.multiply(spatial_support.maximum(sp.eye(len(rna)))))
    if outside.nnz or outside2.nnz:
        raise RuntimeError("common graph is not the registered support intersection")
    return result, {
        "k": int(k), "combine": "edge_intersection_then_symmetrize",
        "feature_metric": "cosine", "spatial_metric": "euclidean",
        "self_loops": True, "union_fallback": False,
        "feature_support_sha256": sparse_sha(feature),
        "spatial_support_sha256": sparse_sha(spatial),
        "common_graph_sha256": sparse_sha(result),
        "nnz": int(result.nnz), "zero_degree_count": int(np.sum(np.asarray(result.sum(1)).ravel() <= 0)),
    }


def spatial_operator(coordinates: np.ndarray, ids: Sequence[str], k: int) -> sp.csr_matrix:
    return row_stochastic_with_loops(symmetric_knn(coordinates, ids, int(k), "euclidean"))


def torch_sparse(value: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = canonical_csr(value).tocoo()
    index = torch.as_tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)
    data = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(index, data, coo.shape, device=device).coalesce()


def fixed_permutation(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(910000 + int(seed)).permutation(int(n)).astype(np.int64)


def robust_reliability(residual_rna: torch.Tensor, residual_aux: torch.Tensor,
                       temperature: float = 1.0, lower: float = .1,
                       upper: float = .9) -> torch.Tensor:
    residual = torch.column_stack((residual_rna, residual_aux))
    # CUDA median-with-indices is not deterministic in torch 2.0.  The
    # registered statistics are stop-gradient, so compute their exact
    # deterministic NumPy medians on CPU and return them to the same device.
    detached_np = residual.detach().cpu().numpy().astype(np.float64, copy=False)
    median_np = np.median(detached_np, axis=0)
    mad_np = np.maximum(np.median(np.abs(detached_np - median_np), axis=0), 1e-12)
    median = torch.as_tensor(median_np, dtype=residual.dtype, device=residual.device)
    mad = torch.as_tensor(mad_np, dtype=residual.dtype, device=residual.device)
    standardized = (residual.detach() - median) / mad
    weights = torch.softmax(-standardized / float(temperature), dim=1)
    weights = weights.clamp(float(lower), float(upper))
    weights = weights / weights.sum(dim=1, keepdim=True)
    return weights


class RACFModel(nn.Module):
    """Small cooperative adapter over immutable family-reference views."""

    def __init__(self, input_dim: int, candidate: Mapping[str, object],
                 latent_dim: int | None = None):
        super().__init__()
        if FORBIDDEN_KEYS & set(candidate):
            raise ValueError("candidate contains identity or label fields")
        self.candidate = dict(candidate)
        latent_dim = int(input_dim if latent_dim is None else latent_dim)
        self.input_dim = int(input_dim); self.latent_dim = latent_dim
        self.rna_feature = nn.Linear(input_dim, latent_dim, bias=False)
        self.rna_spatial = nn.Linear(input_dim, latent_dim, bias=False)
        self.aux_common = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder_rna = nn.Linear(latent_dim, input_dim)
        self.decoder_aux = nn.Linear(latent_dim, input_dim)
        self.stage1_logits = nn.Parameter(torch.zeros(2))
        self.stage2_logits = nn.Parameter(torch.zeros(2))
        self.dgi_discriminator = nn.Linear(latent_dim, 1, bias=False)
        # Identity initialization preserves the inherited family representation
        # at the first fixed endpoint while still allowing every registered
        # cooperative branch to learn.  It is common across all families.
        for layer in (self.rna_feature, self.rna_spatial, self.aux_common,
                      self.decoder_rna, self.decoder_aux):
            nn.init.zeros_(layer.weight)
            with torch.no_grad():
                diagonal = min(layer.weight.shape)
                layer.weight[:diagonal, :diagonal].copy_(torch.eye(diagonal))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x_rna: torch.Tensor, x_aux: torch.Tensor,
                spatial: torch.Tensor, common: torch.Tensor | None,
                reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        feature = row_l2(self.rna_feature(x_rna))
        spatial_branch = row_l2(self.rna_spatial(torch.sparse.mm(spatial, x_rna)))
        propagated_aux = x_aux if common is None else torch.sparse.mm(common, x_aux)
        auxiliary = row_l2(self.aux_common(propagated_aux))
        if bool(self.candidate["hierarchical_fusion"]):
            w1 = torch.softmax(self.stage1_logits, dim=0)
            w2 = torch.softmax(self.stage2_logits, dim=0)
        else:
            w1 = feature.new_tensor([.5, .5])
            w2 = feature.new_tensor([.5, .5])
        rna = row_l2(w1[0] * feature + w1[1] * spatial_branch)
        recon_rna = self.decoder_rna(rna)
        recon_aux = self.decoder_aux(auxiliary)
        residual_rna = (recon_rna - x_rna).square().mean(dim=1)
        residual_aux = (recon_aux - x_aux).square().mean(dim=1)
        if bool(self.candidate["reliability_gate"]):
            reliability = robust_reliability(residual_rna, residual_aux)
            # Reliability is a per-spot modulation of the registered second
            # hierarchical fusion, not a replacement for that learned stage.
            reliability = reliability * w2[None, :]
            reliability = reliability / reliability.sum(dim=1, keepdim=True)
            reliability = reliability.clamp(.1, .9)
            reliability = reliability / reliability.sum(dim=1, keepdim=True)
            fused = row_l2(reliability[:, :1] * rna + reliability[:, 1:] * auxiliary)
        else:
            reliability = torch.column_stack((torch.ones_like(residual_rna) * w2[0],
                                                torch.ones_like(residual_aux) * w2[1]))
            fused = row_l2(w2[0] * rna + w2[1] * auxiliary)
        # The immutable reference regularizes the fixed endpoint without routing.
        fused = row_l2(.75 * reference + .25 * fused)
        return {
            "rna_feature": feature, "rna_spatial": spatial_branch,
            "rna": rna, "auxiliary": auxiliary, "fused": fused,
            "recon_rna": recon_rna, "recon_aux": recon_aux,
            "residual_rna": residual_rna, "residual_aux": residual_aux,
            "stage1": w1, "stage2": w2, "reliability": reliability,
        }


def racf_loss(model: RACFModel, output: Mapping[str, torch.Tensor],
              x_rna: torch.Tensor, x_aux: torch.Tensor,
              reference: torch.Tensor, permutation: torch.Tensor) -> tuple[torch.Tensor, dict]:
    reconstruction = .5 * (F.mse_loss(output["recon_rna"], x_rna) +
                             F.mse_loss(output["recon_aux"], x_aux))
    cooperation = F.mse_loss(output["rna"], output["auxiliary"])
    reference_loss = F.mse_loss(output["fused"], reference)
    total = reconstruction + .10 * cooperation + .10 * reference_loss
    dgi = output["fused"].new_zeros(())
    if bool(model.candidate["dgi"]):
        positive = model.dgi_discriminator(output["fused"]).squeeze(1)
        negative = model.dgi_discriminator(output["fused"][permutation]).squeeze(1)
        dgi = .5 * (F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive)) +
                    F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative)))
        total = total + .05 * dgi
    return total, {
        "reconstruction": float(reconstruction.detach().cpu()),
        "cooperation": float(cooperation.detach().cpu()),
        "reference": float(reference_loss.detach().cpu()),
        "dgi": float(dgi.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def output_views(output: Mapping[str, torch.Tensor]) -> Mapping[str, np.ndarray]:
    n = len(output["fused"])
    stage1 = output["stage1"].detach().cpu().numpy()[None, :].repeat(n, axis=0)
    stage2 = output["stage2"].detach().cpu().numpy()[None, :].repeat(n, axis=0)
    return {
        "emb_latent_omics1": output["rna"].detach().cpu().numpy().astype(np.float32),
        "emb_latent_omics2": output["auxiliary"].detach().cpu().numpy().astype(np.float32),
        "SpaLORA_fused": output["fused"].detach().cpu().numpy().astype(np.float32),
        "alpha_omics1": stage1.astype(np.float32),
        "alpha_omics2": output["reliability"].detach().cpu().numpy().astype(np.float32),
        "alpha_cross": stage2.astype(np.float32),
    }


def tensor_state_sha(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(tensor.dtype).encode())
        h.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()
