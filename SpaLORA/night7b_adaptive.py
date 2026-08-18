"""Night-7B label-free adaptive relational fusion primitives.

This module intentionally contains no label paths, metric code, dataset names,
or evaluator imports.  Every public training/transform function operates only
on opaque unit inputs, a fixed K, and the SHA-locked Night-7B registry.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
import torch
from torch import nn
import torch.nn.functional as F

from .night6c_pipeline import (
    NumericalHeadFailure,
    _neighbors,
    array_sha,
    mclust,
    self_tuning_affinity,
    sparse_sha,
    spectral,
)
from .night7a_consensus import canonical_csr, canonical_partition


G00 = "G00_SP18_F20_CORR_UNION"
G04 = "G04_SP10_F10_EUC_UNION"
VIEWS = ("emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused")
HEAD_ORDER = tuple("H%02d" % i for i in range(18))
RECIPE_ORDER = tuple("R%02d" % i for i in range(10))
CROSS_VIEW_PAIRS = tuple((i, i + 3) for i in range(3))


def canonical_json_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def row_l2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def row_sparse_strict(matrix: sp.spmatrix) -> sp.csr_matrix:
    value = canonical_csr(matrix)
    degree = np.asarray(value.sum(axis=1)).ravel()
    if np.any(~np.isfinite(degree)) or np.any(degree <= 0):
        raise NumericalHeadFailure("zero or non-finite sparse row")
    return canonical_csr(sp.diags(1.0 / degree) @ value)


def sym_zero(matrix: sp.spmatrix) -> sp.csr_matrix:
    value = canonical_csr((matrix + matrix.T) * 0.5)
    value.setdiag(0.0)
    value.eliminate_zeros()
    value.sort_indices()
    return value


def graph_summary(views: Mapping[str, np.ndarray]) -> np.ndarray:
    return row_l2(sum((row_l2(views[key]) for key in VIEWS)) / 3.0)


def _top_weight_neighbors(matrix: sp.spmatrix, k: int,
                          ids: Sequence[str]) -> np.ndarray:
    value = canonical_csr(matrix)
    names = np.asarray(ids, dtype=str)
    out = np.empty((value.shape[0], k), dtype=np.int64)
    for row in range(value.shape[0]):
        start, stop = value.indptr[row], value.indptr[row + 1]
        cols = value.indices[start:stop]
        vals = value.data[start:stop]
        mask = cols != row
        cols, vals = cols[mask], vals[mask]
        if len(cols) < k:
            raise NumericalHeadFailure("affinity row has fewer than k neighbors")
        order = np.lexsort((names[cols], -vals))[:k]
        out[row] = cols[order]
    return out


def reliability_inputs(s00: sp.spmatrix, s04: sp.spmatrix,
                       z00: np.ndarray, z04: np.ndarray,
                       ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Return local/global graph weights and the four fixed MoE scalars.

    The four scalars are, in registry order, own and cross prediction error for
    G00 followed by own and cross prediction error for G04.
    """
    n00 = _top_weight_neighbors(s00, 10, ids)
    n04 = _top_weight_neighbors(s04, 10, ids)
    own00 = np.linalg.norm(z00 - z00[n00].mean(axis=1), axis=1)
    cross00 = np.linalg.norm(z00 - z00[n04].mean(axis=1), axis=1)
    own04 = np.linalg.norm(z04 - z04[n04].mean(axis=1), axis=1)
    cross04 = np.linalg.norm(z04 - z04[n00].mean(axis=1), axis=1)
    raw = np.column_stack((cross00 - own00, cross04 - own04))
    scales = np.maximum(np.median(np.abs(raw), axis=0), 1e-12)
    score = np.clip(raw / scales[None, :], -10.0, 10.0)
    shifted = score - score.max(axis=1, keepdims=True)
    local = np.exp(shifted); local /= local.sum(axis=1, keepdims=True)
    gscore = score.mean(axis=0)
    gscore -= gscore.max()
    global_weight = np.exp(gscore); global_weight /= global_weight.sum()
    scalars = np.column_stack((own00, cross00, own04, cross04))
    audit = {
        "raw_scale": scales.tolist(),
        "local_weight_quantiles": np.quantile(local, [0, .01, .05, .25, .5, .75, .95, .99, 1], axis=0).tolist(),
        "global_weight": global_weight.tolist(),
        "scalar_sha256": array_sha(scalars),
    }
    return local, global_weight, {"scalars": scalars, "audit": audit}


def build_head_affinity(head_id: str, s00: sp.spmatrix, s04: sp.spmatrix,
                        views00: Mapping[str, np.ndarray],
                        views04: Mapping[str, np.ndarray],
                        ids: Sequence[str]) -> Tuple[sp.csr_matrix, dict]:
    if head_id not in HEAD_ORDER:
        raise KeyError(head_id)
    r00, r04 = row_sparse_strict(s00), row_sparse_strict(s04)
    z00, z04 = graph_summary(views00), graph_summary(views04)
    local, global_weight, extra = reliability_inputs(s00, s04, z00, z04, ids)
    diag = {}
    if head_id in {"H00", "H08", "H12", "H14", "H16"}:
        result = canonical_csr(s04)
    elif head_id in {"H01", "H09", "H13", "H15", "H17"}:
        result = sym_zero((r00 + r04) * 0.5)
    elif head_id == "H02":
        result = sym_zero(r00 * .40 + r04 * .60)
    elif head_id in {"H03", "H10"}:
        result = sym_zero(r00 * .30 + r04 * .70)
    elif head_id == "H04":
        result = sym_zero(r00 * .20 + r04 * .80)
    elif head_id == "H05":
        result = sym_zero(r00 * .10 + r04 * .90)
    elif head_id in {"H06", "H11"}:
        directed = sp.diags(local[:, 0]) @ r00 + sp.diags(local[:, 1]) @ r04
        result = sym_zero(directed)
        diag = extra["audit"]
    elif head_id == "H07":
        directed = r00 * float(global_weight[0]) + r04 * float(global_weight[1])
        result = sym_zero(directed)
        diag = extra["audit"]
    else:  # pragma: no cover - exhaustive guard
        raise KeyError(head_id)
    result.setdiag(0.0); result.eliminate_zeros(); result.sort_indices()
    return canonical_csr(result), diag


def affinity_audit(matrix: sp.spmatrix) -> dict:
    value = canonical_csr(matrix)
    diff = canonical_csr(value - value.T)
    degree = np.asarray(value.sum(axis=1)).ravel()
    components, labels = connected_components(value, directed=False)
    sizes = np.bincount(labels, minlength=components)
    return {
        "shape": list(value.shape), "nnz": int(value.nnz),
        "finite": bool(np.isfinite(value.data).all()),
        "symmetry_max_error": float(np.max(np.abs(diff.data))) if diff.nnz else 0.0,
        "diagonal_max_abs": float(np.max(np.abs(value.diagonal()))) if value.shape[0] else 0.0,
        "zero_degree_count": int(np.sum(degree == 0)),
        "connected_component_count": int(components),
        "connected_component_sizes": sorted(map(int, sizes), reverse=True),
        "canonical_sparse_sha256": sparse_sha(value),
    }


def _eigen_embedding(matrix: sp.spmatrix, k: int) -> np.ndarray:
    value = canonical_csr((matrix + matrix.T) * .5)
    degree = np.asarray(value.sum(axis=1)).ravel()
    if np.any(degree <= 0):
        raise NumericalHeadFailure("zero degree before normalized Laplacian")
    lap = laplacian(value, normed=True).astype(np.float64)
    v0 = np.linspace(1.0, 2.0, value.shape[0], dtype=np.float64)
    _, vectors = eigsh(lap, k=int(k), which="SM", v0=v0, tol=1e-10,
                       maxiter=max(10000, value.shape[0] * 20))
    for col in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, col])))
        if vectors[pivot, col] < 0:
            vectors[:, col] *= -1.0
    return row_l2(vectors)


def _leiden_exact_k(matrix: sp.spmatrix, k: int,
                    resolutions: Sequence[float]) -> Tuple[np.ndarray, dict]:
    import igraph as ig
    import leidenalg
    upper = sp.triu(canonical_csr(matrix), k=1).tocoo()
    graph = ig.Graph(n=matrix.shape[0],
                     edges=list(zip(upper.row.tolist(), upper.col.tolist())),
                     directed=False)
    graph.es["weight"] = upper.data.astype(float).tolist()
    tried = []
    for resolution in resolutions:
        part = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=float(resolution),
            n_iterations=-1, seed=2020,
        )
        count = len(part)
        tried.append([float(resolution), int(count)])
        if count == int(k):
            return np.asarray(part.membership, dtype=np.int64) + 1, {
                "resolution": float(resolution), "resolution_trace": tried,
            }
    raise NumericalHeadFailure("no registered Leiden resolution produced exact K")


def run_partition(head_id: str, affinity: sp.spmatrix, k: int,
                  resolutions: Sequence[float]) -> Tuple[np.ndarray, dict]:
    if head_id in {"H00", "H01", "H02", "H03", "H04", "H05", "H06", "H07"}:
        labels = spectral(affinity, k); aux = {"partition_head": "SPECTRAL_DISCRETIZE"}
    elif head_id in {"H08", "H09", "H10", "H11", "H12", "H13"}:
        embedding = _eigen_embedding(affinity, k)
        model = "EEE" if head_id in {"H08", "H09", "H10", "H11"} else "VVV"
        out = mclust(embedding, k, model, 2020)
        labels = out["labels"]
        aux = {"partition_head": "EIGEN_MCLUST_" + model,
               "selected_model": out["selected_model"],
               "eigen_embedding_sha256": array_sha(embedding)}
    elif head_id in {"H14", "H15"}:
        from sklearn.cluster import KMeans
        embedding = _eigen_embedding(affinity, k)
        labels = KMeans(n_clusters=k, n_init=100, random_state=2020,
                        algorithm="lloyd").fit_predict(embedding) + 1
        aux = {"partition_head": "EIGEN_KMEANS100",
               "eigen_embedding_sha256": array_sha(embedding)}
    elif head_id in {"H16", "H17"}:
        labels, aux = _leiden_exact_k(affinity, k, resolutions)
        aux["partition_head"] = "LEIDEN_EXACT_K"
    else:
        raise KeyError(head_id)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != affinity.shape[0] or len(np.unique(labels)) != int(k):
        raise NumericalHeadFailure("partition did not produce fixed K")
    canonical = canonical_partition(labels)
    aux["canonical_partition_sha256"] = array_sha(canonical)
    return labels, aux


def _torch_row_l2(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value, p=2, dim=1, eps=1e-12)


class AdaptiveFusion(nn.Module):
    def __init__(self, input_dims: Sequence[int], k: int, moe: bool,
                 semantic: bool):
        super().__init__()
        self.input_dims = tuple(map(int, input_dims))
        self.moe_enabled = bool(moe)
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 64, bias=False), nn.LayerNorm(64))
            for dim in self.input_dims
        ])
        self.adapter = nn.Sequential(nn.Linear(64, 128), nn.GELU(),
                                     nn.Dropout(.1), nn.Linear(128, 64))
        self.decoders = nn.ModuleList([nn.Linear(64, dim) for dim in self.input_dims])
        self.gate = (nn.Sequential(nn.Linear(196, 128), nn.GELU(),
                                   nn.Linear(128, 2)) if moe else None)
        self.semantic = nn.Linear(64, int(k)) if semantic else None

    def forward(self, inputs: Sequence[torch.Tensor],
                reliability_scalars: torch.Tensor,
                masks: Optional[Sequence[torch.Tensor]] = None) -> dict:
        projected = []
        for i, (module, value) in enumerate(zip(self.projectors, inputs)):
            work = value
            if masks is not None:
                if masks[i].dtype != torch.bool or masks[i].ndim != 1 or masks[i].numel() != value.shape[0]:
                    raise RuntimeError("mask forward contract requires one boolean per spot")
                work = torch.where(masks[i][:, None], torch.zeros_like(value), value)
            projected.append(_torch_row_l2(module(work)))
        u00 = _torch_row_l2(torch.stack(projected[:3], 0).mean(0))
        u04 = _torch_row_l2(torch.stack(projected[3:], 0).mean(0))
        if self.gate is None:
            weights = torch.full((u00.shape[0], 2), .5, device=u00.device,
                                 dtype=u00.dtype)
        else:
            features = torch.cat((u00, u04, torch.abs(u00-u04),
                                  reliability_scalars), dim=1)
            weights = torch.softmax(self.gate(features), dim=1)
        fused0 = weights[:, :1] * u00 + weights[:, 1:] * u04
        z = _torch_row_l2(fused0 + self.adapter(fused0))
        decoded = [module(z) for module in self.decoders]
        return {"projected": projected, "u00": u00, "u04": u04,
                "gate_weights": weights, "z": z, "decoded": decoded,
                "semantic_logits": self.semantic(z) if self.semantic is not None else None}


def deterministic_masks(n: int, recipe_id: str, seed: int) -> Sequence[np.ndarray]:
    result = []
    count = int(math.floor(.15 * n))
    for view in range(6):
        token = ("%s|%d|%d" % (recipe_id, int(seed), view)).encode("utf-8")
        state = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
        rng = np.random.default_rng(state)
        result.append(np.sort(rng.choice(n, size=count, replace=False)).astype(np.int64))
    return result


def fixed_mnn_triplets(z00: np.ndarray, z04: np.ndarray,
                       ids: Sequence[str], recipe_id: str,
                       seed: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    z00, z04 = row_l2(z00), row_l2(z04)
    a = _neighbors(z04, 10, "euclidean", ids)
    b = _neighbors(z00, 10, "euclidean", ids)
    # Cross-view nearest lists: explicit deterministic O(N^2) row blocks, no
    # resident dense N-by-N matrix.
    n = len(z00); cross04 = np.empty((n, 10), dtype=np.int64)
    cross00 = np.empty((n, 10), dtype=np.int64)
    names = np.asarray(ids, dtype=str)
    for start in range(0, n, 256):
        stop = min(n, start + 256)
        d = 2.0 - 2.0 * (z00[start:stop] @ z04.T)
        for off, row in enumerate(range(start, stop)):
            cross04[row] = np.lexsort((names, d[off]))[:10]
        d2 = d.T
        # reverse rows cannot be completed blockwise here; compute below.
    for start in range(0, n, 256):
        stop = min(n, start + 256)
        d = 2.0 - 2.0 * (z04[start:stop] @ z00.T)
        for off, row in enumerate(range(start, stop)):
            cross00[row] = np.lexsort((names, d[off]))[:10]
    positives = np.empty(n, dtype=np.int64)
    negatives = np.empty(n, dtype=np.int64)
    for i in range(n):
        mutual = [int(j) for j in cross04[i] if i in set(cross00[int(j)])]
        positives[i] = min(mutual, key=lambda j: names[j]) if mutual else int(cross04[i, 0])
        # Deterministic negative from the farthest half of the opposite view.
        sim = z04 @ z00[i]
        order = np.lexsort((names, sim))
        far = order[:max(1, n // 2)]
        token = ("%s|%d|%d|negative" % (recipe_id, int(seed), i)).encode("utf-8")
        pick = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % len(far)
        negatives[i] = int(far[pick])
    return positives, negatives, {
        "positive_sha256": array_sha(positives),
        "negative_sha256": array_sha(negatives),
        "fallback_nearest_nonmutual_count": int(sum(i not in set(cross00[int(positives[i])]) for i in range(n))),
    }


def sparse_relation_edges(inputs: Sequence[np.ndarray], ids: Sequence[str],
                          k: int = 20) -> Tuple[np.ndarray, np.ndarray, Sequence[np.ndarray]]:
    n = len(inputs[0]); per = []
    unions = [set() for _ in range(n)]
    for values in inputs:
        idx = _neighbors(row_l2(values), k, "euclidean", ids)
        per.append(idx)
        for i in range(n): unions[i].update(map(int, idx[i]))
    rows, cols = [], []
    for i, values in enumerate(unions):
        for j in sorted(values, key=lambda x: str(ids[x])):
            rows.append(i); cols.append(j)
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64), per


def grouped_softmax(logits: torch.Tensor, rows: torch.Tensor, n: int) -> torch.Tensor:
    if rows.numel() and not bool(torch.all(rows[1:] >= rows[:-1])):
        raise RuntimeError("relation edges must be row-sorted")
    unique_rows, counts = torch.unique_consecutive(rows, return_counts=True)
    expected = torch.arange(n, device=rows.device, dtype=rows.dtype)
    if not torch.equal(unique_rows, expected):
        raise RuntimeError("relation edge union has an empty row")
    maximum = torch.segment_reduce(logits, reduce="max", lengths=counts)
    exp = torch.exp(logits - maximum[rows])
    denom = torch.segment_reduce(exp, reduce="sum", lengths=counts)
    return exp / torch.clamp(denom[rows], min=1e-12)


def relation_kl(z: torch.Tensor, inputs: Sequence[torch.Tensor],
                rows: torch.Tensor, cols: torch.Tensor, tau: float = .1) -> torch.Tensor:
    p = grouped_softmax((z[rows] * z[cols]).sum(1) / tau, rows, z.shape[0])
    terms = []
    for value in inputs:
        target = value.detach()
        q = grouped_softmax((target[rows] * target[cols]).sum(1) / tau,
                            rows, z.shape[0])
        terms.append(torch.sum(q * (torch.log(q + 1e-12) - torch.log(p + 1e-12))) / z.shape[0])
    return torch.stack(terms).mean()


def symmetric_clip(a: torch.Tensor, b: torch.Tensor, tau: float = .1,
                   batch: int = 256) -> torch.Tensor:
    n = a.shape[0]; target = torch.arange(n, device=a.device); total = a.new_tensor(0.0)
    for start in range(0, n, batch):
        stop = min(n, start + batch)
        total = total + F.cross_entropy(a[start:stop] @ b.T / tau,
                                        target[start:stop], reduction="sum")
        total = total + F.cross_entropy(b[start:stop] @ a.T / tau,
                                        target[start:stop], reduction="sum")
    return total / (2.0 * n)


def dcca_loss(a: torch.Tensor, b: torch.Tensor, ridge: float = 1e-3,
              top: int = 32) -> torch.Tensor:
    a = a - a.mean(0, keepdim=True); b = b - b.mean(0, keepdim=True)
    denom = max(1, a.shape[0] - 1)
    eye = torch.eye(a.shape[1], device=a.device, dtype=a.dtype)
    ca = a.T @ a / denom + ridge * eye
    cb = b.T @ b / denom + ridge * eye
    cab = a.T @ b / denom
    ea, va = torch.linalg.eigh(ca); eb, vb = torch.linalg.eigh(cb)
    wa = va @ torch.diag(torch.rsqrt(torch.clamp(ea, min=1e-12))) @ va.T
    wb = vb @ torch.diag(torch.rsqrt(torch.clamp(eb, min=1e-12))) @ vb.T
    corr = torch.linalg.svdvals(wa @ cab @ wb)
    return -corr[:min(int(top), len(corr))].sum()


def moe_balance(weights: torch.Tensor) -> torch.Tensor:
    importance = weights.mean(0)
    hard = F.one_hot(weights.argmax(1), num_classes=2).float().mean(0)
    def cv2(x):
        return x.var(unbiased=False) / torch.clamp(x.mean().square(), min=1e-12)
    return cv2(importance) + cv2(hard)


def loss_components(output: Mapping[str, object], targets: Sequence[torch.Tensor],
                    recipe_losses: Sequence[str], relation_rows: torch.Tensor,
                    relation_cols: torch.Tensor, positives: torch.Tensor,
                    negatives: torch.Tensor, mask_indices: Sequence[torch.Tensor],
                    semantic_target: Optional[torch.Tensor],
                    semantic_keep: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
    z = output["z"]; projected = output["projected"]
    result = {"RECON": torch.stack([F.mse_loss(x, y) for x, y in zip(output["decoded"], targets)]).mean()}
    if "RELKL" in recipe_losses:
        result["RELKL"] = relation_kl(z, targets, relation_rows, relation_cols)
    if "CLIP" in recipe_losses:
        result["CLIP"] = torch.stack([symmetric_clip(projected[i], projected[j]) for i, j in CROSS_VIEW_PAIRS]).mean()
    if "MNN" in recipe_losses:
        result["MNN"] = F.triplet_margin_loss(z, projected[3][positives],
                                               projected[3][negatives], margin=.5)
    if "MASK" in recipe_losses:
        pieces = [F.mse_loss(output["decoded"][i][mask_indices[i]], targets[i][mask_indices[i]])
                  for i in range(6)]
        result["MASK"] = torch.stack(pieces).mean()
    if "SEMANTIC" in recipe_losses:
        if semantic_target is None or semantic_keep is None or output["semantic_logits"] is None:
            raise RuntimeError("semantic target contract missing")
        result["SEMANTIC"] = F.cross_entropy(output["semantic_logits"][semantic_keep] / .5,
                                              semantic_target[semantic_keep])
    if "MOE_BALANCE" in recipe_losses:
        result["MOE_BALANCE"] = moe_balance(output["gate_weights"])
    if "DCCA" in recipe_losses:
        result["DCCA"] = dcca_loss(output["u00"], output["u04"])
    return result


LOSS_WEIGHTS = {"RECON": 1.0, "RELKL": 1.0, "CLIP": .2, "MNN": .2,
                "MASK": .5, "SEMANTIC": .2, "MOE_BALANCE": .01,
                "DCCA": .05}


def total_loss(components: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return sum(components[key] * LOSS_WEIGHTS[key] for key in components)


def pseudo_confidence(affinity: sp.spmatrix, labels: np.ndarray) -> np.ndarray:
    value = canonical_csr(affinity); degree = np.asarray(value.sum(axis=1)).ravel()
    same = np.zeros(value.shape[0], dtype=np.float64)
    for i in range(value.shape[0]):
        start, stop = value.indptr[i], value.indptr[i + 1]
        cols, weights = value.indices[start:stop], value.data[start:stop]
        same[i] = weights[labels[cols] == labels[i]].sum()
    return same / np.maximum(degree, 1e-12)


def pseudo_keep(labels: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        count = max(1, int(math.ceil(.60 * len(idx))))
        order = np.lexsort((idx, -confidence[idx]))[:count]
        keep[idx[order]] = True
    return keep
