"""Night-5A clean-room, label-free method candidates and training utilities.

This module deliberately has no semantic-label reader.  C00/C01/C02 delegate
to the locked Night-3B implementations.  Every other candidate is defined by
the authoritative Night-5A registry and consumes frozen preprocessing
artifacts that are shared by every model seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from .model_corrected import EncoderOverallCorrected, GraphLinear
from .night3a_ige import (
    LOSS_KEYS, _clone_state, calibrated_total, global_gradient_norm,
    model_state_sha256, raw_losses, required_record_steps, run_initial_probe,
    state_dict_sha256,
)
from .night3b_ablation import Night3BTrainer, active_ige_coefficients
from .preprocess import fix_seed


LEGACY_DELEGATES = {
    "C00_FULL_IGE": "FULL_IGE",
    "C01_DROP_CORR2": "DROP_CORR2",
    "C02_UNIFORM_ALL": "UNIFORM_ALL",
}
REGISTERED_PREFIXES = {"C%02d" % value for value in range(17)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_registry(path: Path) -> dict:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = registry.get("candidates", [])
    ids = [row.get("id") for row in candidates]
    expected = ["C%02d" % value for value in range(17)]
    if len(ids) != 17 or [value.split("_", 1)[0] for value in ids] != expected or len(set(ids)) != 17:
        raise RuntimeError("Registry must contain exactly C00-C16 once and in order")
    if registry.get("parent_commit") != "4a22cfb4afe331e1ca2edcc0b86a01fa3892a452":
        raise RuntimeError("Registry parent commit drift")
    return registry


def registry_contracts(registry: Mapping[str, object]) -> Dict[str, dict]:
    result = {}
    for row in registry["candidates"]:
        contract = dict(row)
        contract["config_sha256"] = canonical_sha256(row)
        result[row["id"]] = contract
    if len({row["config_sha256"] for row in result.values()}) != 17:
        raise RuntimeError("Candidate configuration SHAs are not unique")
    return result


def _pca_representation(values: np.ndarray, components: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("PCA input must be a two-dimensional observation matrix")
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, ddof=0, keepdims=True)
    centered = centered / np.maximum(scale, 1e-12)
    n_components = min(int(components), centered.shape[0] - 1, centered.shape[1])
    if n_components < 1 or float(np.max(np.abs(centered))) <= 1e-12:
        return np.zeros((len(centered), 1), dtype=np.float64)
    transformed = PCA(n_components=n_components, svd_solver="full").fit_transform(centered)
    transformed -= transformed.mean(axis=0, keepdims=True)
    transformed /= np.maximum(transformed.std(axis=0, ddof=0, keepdims=True), 1e-12)
    return transformed.astype(np.float64, copy=False)


def local_reliability_weights(first: np.ndarray, second: np.ndarray, k: int = 20,
                              epsilon: float = 1e-12,
                              clip: Tuple[float, float] = (0.001, 0.999)) -> np.ndarray:
    """Frozen local-predictability weights; swapping modalities swaps columns."""
    first = _pca_representation(first)
    second = _pca_representation(second)
    if first.shape[0] != second.shape[0]:
        raise ValueError("Modalities must have the same observation count")
    n = len(first)
    k = min(max(1, int(k)), n - 1)

    def neighbors(values):
        return NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(values).kneighbors(
            return_distance=False
        )[:, 1:]

    first_neighbors = neighbors(first)
    second_neighbors = neighbors(second)

    def score(target, own_neighbors, other_neighbors):
        own_pred = target[own_neighbors].mean(axis=1)
        other_pred = target[other_neighbors].mean(axis=1)
        own_error = np.mean((target - own_pred) ** 2, axis=1)
        other_error = np.mean((target - other_pred) ** 2, axis=1)
        denom = own_error + other_error
        result = np.divide(other_error, denom + epsilon)
        result[denom <= epsilon] = 0.5
        return np.clip(result, float(clip[0]), float(clip[1]))

    scores = np.column_stack((
        score(first, first_neighbors, second_neighbors),
        score(second, second_neighbors, first_neighbors),
    ))
    total = scores.sum(axis=1, keepdims=True)
    weights = scores / np.maximum(total, epsilon)
    weights[~np.isfinite(weights).all(axis=1)] = 0.5
    if not np.isfinite(weights).all() or np.max(np.abs(weights.sum(axis=1) - 1.0)) > 1e-12:
        raise AssertionError("Reliability weights are invalid")
    return weights.astype(np.float32)


def _support(graph: torch.Tensor, remove_diagonal: bool = True) -> sp.csr_matrix:
    graph = graph.coalesce().cpu()
    rows, cols = graph.indices().numpy()
    values = np.ones(len(rows), dtype=np.float64)
    matrix = sp.coo_matrix((values, (rows, cols)), shape=tuple(graph.shape)).tocsr()
    matrix = matrix.maximum(matrix.T)
    if remove_diagonal:
        matrix.setdiag(0)
        matrix.eliminate_zeros()
    matrix.data[:] = 1.0
    return matrix


def anchor_graph(spatial: torch.Tensor, rna_feature: torch.Tensor, eta: float) -> Tuple[torch.Tensor, dict]:
    spatial_support = _support(spatial)
    feature_support = _support(rna_feature)
    common = spatial_support.multiply(feature_support)
    raw = spatial_support + float(eta) * common + sp.eye(spatial_support.shape[0], format="csr")
    degree = np.asarray(raw.sum(axis=1)).reshape(-1)
    inv = np.power(np.maximum(degree, 1e-12), -0.5)
    normalized = sp.diags(inv) @ raw @ sp.diags(inv)
    normalized = normalized.maximum(normalized.T).tocoo()
    indices = torch.as_tensor(np.vstack((normalized.row, normalized.col)), dtype=torch.long)
    values = torch.as_tensor(normalized.data, dtype=torch.float32)
    result = torch.sparse_coo_tensor(indices, values, size=normalized.shape).coalesce()
    n_components = int(connected_components(spatial_support, directed=False, return_labels=False))
    common_edges = int(common.nnz // 2)
    spatial_edges = int(spatial_support.nnz // 2)
    support_rows, support_cols = spatial_support.nonzero()
    preserved = bool(np.all(np.asarray(raw[support_rows, support_cols]).reshape(-1) >= 1.0))
    stats = {
        "eta": float(eta),
        "n_observations": int(spatial_support.shape[0]),
        "spatial_undirected_edges": spatial_edges,
        "common_undirected_edges": common_edges,
        "common_edge_fraction": float(common_edges / max(spatial_edges, 1)),
        "spatial_connected_components": n_components,
        "minimum_raw_degree_with_self_loop": float(degree.min()),
        "maximum_raw_degree_with_self_loop": float(degree.max()),
        "original_spatial_support_preserved": preserved,
        "symmetric_normalized": bool((normalized - normalized.T).nnz == 0),
    }
    if not stats["original_spatial_support_preserved"] or degree.min() <= 0:
        raise AssertionError("Anchor graph deleted a spatial edge or created an isolate")
    return result, stats


def _mutual_pairs(representation: np.ndarray, k: int) -> np.ndarray:
    k = min(max(1, int(k)), len(representation) - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(representation).kneighbors(
        return_distance=False
    )[:, 1:]
    sets = [set(map(int, row)) for row in neighbors]
    pairs = [(i, int(j)) for i, row in enumerate(neighbors) for j in row if i < j and i in sets[int(j)]]
    return np.asarray(sorted(set(pairs)), dtype=np.int64).reshape(-1, 2)


def frozen_triplets(first: np.ndarray, second: np.ndarray, seed: int, k: int = 3,
                    farthest_fraction: float = 0.4) -> Tuple[np.ndarray, dict]:
    rng = np.random.default_rng(int(seed))
    representations = (_pca_representation(first), _pca_representation(second))
    rows = []
    source_counts = {}
    for source, representation in enumerate(representations):
        pairs = _mutual_pairs(representation, k)
        distances = pairwise_distances(representation, metric="euclidean")
        farthest_count = max(1, int(math.ceil(float(farthest_fraction) * len(representation))))
        for anchor, positive in pairs:
            candidates = np.argpartition(distances[anchor], -farthest_count)[-farthest_count:]
            candidates = candidates[(candidates != anchor) & (candidates != positive)]
            negative = int(candidates[int(rng.integers(0, len(candidates)))])
            rows.append((int(anchor), int(positive), negative, int(source)))
        source_counts[str(source)] = int(len(pairs))
    result = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    coverage = len(set(result[:, :3].reshape(-1).tolist())) / float(len(first)) if len(result) else 0.0
    stats = {
        "triplet_count": int(len(result)), "source_counts": source_counts,
        "spot_coverage": float(coverage),
        "duplicate_fraction": float(1.0 - len(set(map(tuple, result.tolist()))) / max(len(result), 1)),
        "mnn_k": int(k), "negative_farthest_fraction": float(farthest_fraction),
        "preprocessing_seed": int(seed),
    }
    return result, stats


def frozen_contrast_pairs(spatial: torch.Tensor, seed: int) -> Tuple[np.ndarray, dict]:
    graph = _support(spatial)
    rows, cols = graph.nonzero()
    positives = np.asarray([(int(i), int(j)) for i, j in zip(rows, cols) if i < j], dtype=np.int64)
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    quantiles = np.unique(np.quantile(degree, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if len(quantiles) < 2:
        bins = np.zeros(len(degree), dtype=int)
    else:
        bins = np.digitize(degree, quantiles[1:-1], right=True)
    by_bin = {value: np.flatnonzero(bins == value) for value in np.unique(bins)}
    neighbor_sets = [set(graph.indices[graph.indptr[i]:graph.indptr[i + 1]].tolist()) for i in range(len(degree))]
    rng = np.random.default_rng(int(seed))
    negatives = []
    for anchor, positive in positives:
        pool = by_bin[int(bins[positive])]
        valid = pool[(pool != anchor)]
        valid = np.asarray([value for value in valid if int(value) not in neighbor_sets[int(anchor)]], dtype=np.int64)
        if not len(valid):
            valid = np.asarray([value for value in range(len(degree))
                                if value != anchor and value not in neighbor_sets[int(anchor)]], dtype=np.int64)
        negatives.append((int(anchor), int(valid[int(rng.integers(0, len(valid)))])))
    negative_array = np.asarray(negatives, dtype=np.int64).reshape(-1, 2)
    packed = np.column_stack((positives, negative_array[:, 1])) if len(positives) else np.empty((0, 3), dtype=np.int64)
    stats = {
        "positive_edge_count": int(len(positives)), "negative_edge_count": int(len(negative_array)),
        "degree_stratified": True, "preprocessing_seed": int(seed),
        "contains_positive_as_negative": bool(any(int(n) in neighbor_sets[int(a)] for a, _, n in packed)),
    }
    if stats["contains_positive_as_negative"]:
        raise AssertionError("Contrastive negative overlaps a spatial edge")
    return packed, stats


def save_sparse_graph(path: Path, graph: torch.Tensor) -> None:
    graph = graph.coalesce().cpu()
    np.savez_compressed(path, indices=graph.indices().numpy(), values=graph.values().numpy(),
                        shape=np.asarray(graph.shape, dtype=np.int64))


def load_sparse_graph(path: Path) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as archive:
        return torch.sparse_coo_tensor(torch.as_tensor(archive["indices"], dtype=torch.long),
                                       torch.as_tensor(archive["values"], dtype=torch.float32),
                                       size=tuple(map(int, archive["shape"]))).coalesce()


def build_label_free_artifacts(data: Mapping[str, object], directory: Path,
                               preprocessing_seed: int) -> dict:
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("Refusing to overwrite label-free artifacts: %s" % directory)
    directory.mkdir(parents=True, exist_ok=True)
    first = np.asarray(data["rna_pca_scores"], dtype=np.float64)
    second = np.asarray(data["features_omics2"], dtype=np.float64)
    reliability = local_reliability_weights(first, second, k=20)
    np.save(directory / "reliability_weights.npy", reliability, allow_pickle=False)
    triplets, triplet_stats = frozen_triplets(first, second, preprocessing_seed, k=3,
                                              farthest_fraction=0.4)
    np.save(directory / "mnn_triplets.npy", triplets, allow_pickle=False)
    contrast, contrast_stats = frozen_contrast_pairs(data["adj_spatial_omics1"], preprocessing_seed)
    np.save(directory / "contrast_pairs.npy", contrast, allow_pickle=False)
    permutation = np.random.default_rng(int(preprocessing_seed)).permutation(len(first)).astype(np.int64)
    np.save(directory / "dgi_permutation.npy", permutation, allow_pickle=False)
    anchor_stats = {}
    for suffix, eta in (("05", 0.5), ("10", 1.0)):
        graph, stats = anchor_graph(data["adj_spatial_omics1"], data["adj_feature_omics1"], eta)
        save_sparse_graph(directory / ("rna_anchor_%s.npz" % suffix), graph)
        anchor_stats[suffix] = stats
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
    manifest = {
        "schema_version": 1, "semantic_label_access": False,
        "preprocessing_seed": int(preprocessing_seed),
        "reliability": {
            "shape": list(reliability.shape), "mean": reliability.mean(axis=0).tolist(),
            "minimum": reliability.min(axis=0).tolist(), "maximum": reliability.max(axis=0).tolist(),
            "pca_components": 20, "pca_solver": "full", "k": 20,
            "distance": "euclidean", "epsilon": 1e-12, "clip": [0.001, 0.999],
        },
        "triplets": triplet_stats, "contrast": contrast_stats,
        "dgi": {"permutation_is_bijection": bool(np.array_equal(np.sort(permutation), np.arange(len(first)))),
                "preprocessing_seed": int(preprocessing_seed)},
        "anchor_graphs": anchor_stats, "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_label_free_artifacts(directory: Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, row in manifest["files"].items():
        if sha256_file(directory / name) != row["sha256"]:
            raise RuntimeError("Frozen artifact SHA mismatch: %s" % name)
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(directory / "manifest.json"),
        "reliability": np.load(directory / "reliability_weights.npy", allow_pickle=False),
        "triplets": np.load(directory / "mnn_triplets.npy", allow_pickle=False),
        "contrast": np.load(directory / "contrast_pairs.npy", allow_pickle=False),
        "dgi_permutation": np.load(directory / "dgi_permutation.npy", allow_pickle=False),
        "anchor05": load_sparse_graph(directory / "rna_anchor_05.npz"),
        "anchor10": load_sparse_graph(directory / "rna_anchor_10.npz"),
    }


class ResidualSparseEncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, second_scale: float = 0.5):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(in_features, out_features))
        self.weight2 = nn.Parameter(torch.empty(out_features, out_features))
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.second_scale = float(second_scale)
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h1 = self.norm1(F.gelu(torch.sparse.mm(adjacency, torch.mm(features, self.weight1))))
        update = F.gelu(torch.sparse.mm(adjacency, torch.mm(h1, self.weight2)))
        return self.norm2(h1 + self.second_scale * update)


class Night5AModel(EncoderOverallCorrected):
    def __init__(self, in1: int, out1: int, in2: int, out2: int,
                 attention_policy: str, learned_fraction: Optional[float] = None,
                 reliability_weights: Optional[np.ndarray] = None,
                 residual_encoder: bool = False, dgi_head: bool = False):
        super().__init__(in1, out1, in2, out2)
        self.attention_policy = str(attention_policy)
        self.learned_fraction = None if learned_fraction is None else float(learned_fraction)
        if reliability_weights is None:
            self.register_buffer("reliability_weights", torch.empty(0, 2), persistent=True)
        else:
            self.register_buffer("reliability_weights", torch.as_tensor(reliability_weights, dtype=torch.float32),
                                 persistent=True)
        if residual_encoder:
            self.encoder1 = ResidualSparseEncoder(in1, out1, 0.5)
            self.encoder2 = ResidualSparseEncoder(in2, out2, 0.5)
        if dgi_head:
            self.dgi_bilinear = nn.Parameter(torch.empty(out1, out1))
            nn.init.xavier_uniform_(self.dgi_bilinear)
        else:
            self.register_parameter("dgi_bilinear", None)

    @staticmethod
    def _uniform(first: torch.Tensor) -> torch.Tensor:
        return torch.full((len(first), 2), 0.5, dtype=first.dtype, device=first.device)

    def _combine(self, first: torch.Tensor, second: torch.Tensor, layer, location: str):
        stacked = torch.stack((first, second), dim=1)
        policy = self.attention_policy
        if policy == "uniform_all" or (policy == "frozen_local_reliability" and location == "within"):
            alpha = self._uniform(first)
        elif policy == "frozen_local_reliability" and location == "cross":
            alpha = self.reliability_weights.to(dtype=first.dtype, device=first.device)
        elif policy == "shrink_to_uniform":
            _, learned = layer(first, second, uniform=False)
            alpha = (1.0 - float(self.learned_fraction)) * self._uniform(first) + float(self.learned_fraction) * learned
        elif policy == "learned":
            _, alpha = layer(first, second, uniform=False)
        else:
            raise ValueError("Unknown Night-5A attention policy: %s" % policy)
        return torch.sum(stacked * alpha.unsqueeze(-1), dim=1), alpha

    def forward(self, features1, features2, spatial1, feature1, spatial2, feature2):
        latent_spatial1 = self.encoder1(features1, spatial1)
        latent_spatial2 = self.encoder2(features2, spatial2)
        latent_feature1 = self.encoder1(features1, feature1)
        latent_feature2 = self.encoder2(features2, feature2)
        latent1, alpha1 = self._combine(latent_spatial1, latent_feature1, self.attention1, "within")
        latent2, alpha2 = self._combine(latent_spatial2, latent_feature2, self.attention2, "within")
        combined, alpha = self._combine(latent1, latent2, self.cross_attention, "cross")
        recon1 = self.decoder1(combined, spatial1)
        recon2 = self.decoder2(combined, spatial2)
        latent1_across = self.encoder2(self.decoder2(latent1, spatial2), spatial2)
        latent2_across = self.encoder1(self.decoder1(latent2, spatial1), spatial1)
        return {
            "emb_latent_omics1": latent1, "emb_latent_omics2": latent2,
            "emb_latent_combined": combined, "emb_recon_omics1": recon1,
            "emb_recon_omics2": recon2,
            "emb_latent_omics1_across_recon": latent1_across,
            "emb_latent_omics2_across_recon": latent2_across,
            "alpha_omics1": alpha1, "alpha_omics2": alpha2, "alpha": alpha,
        }


class _Forward:
    def __init__(self, features1, features2, adjacencies):
        self.features1 = features1
        self.features2 = features2
        self.adjacencies = tuple(adjacencies)

    def __call__(self, model):
        return model(self.features1, self.features2, *self.adjacencies)


def active_mask_corr2_off() -> Dict[str, bool]:
    return {name: name != "L_corr2_raw" for name in LOSS_KEYS}


def hybrid_coefficients(gradients: Mapping[str, float], legacy_factors: Sequence[float],
                        beta: float, eps: float) -> Dict[str, float]:
    mask = active_mask_corr2_off()
    active = [name for name in LOSS_KEYS if mask[name]]
    legacy = {name: float(value) for name, value in zip(LOSS_KEYS, legacy_factors)}
    raw = np.asarray([
        legacy[name] ** (1.0 - float(beta)) * (float(gradients[name]) + float(eps)) ** (-float(beta))
        for name in active
    ], dtype=np.float64)
    normalized = 4.0 * raw / raw.sum()
    result = {name: 0.0 for name in LOSS_KEYS}
    for name, value in zip(active, normalized):
        result[name] = float(value)
    if abs(sum(result.values()) - 4.0) > 1e-6 or result["L_corr2_raw"] != 0.0:
        raise AssertionError("Hybrid coefficient contract failed")
    return result


def candidate_model_policy(candidate: Mapping[str, object]) -> dict:
    candidate_id = candidate["id"]
    if candidate_id in LEGACY_DELEGATES:
        return {"delegate": LEGACY_DELEGATES[candidate_id]}
    attention = candidate.get("attention", "uniform_all")
    return {
        "attention_policy": attention,
        "learned_fraction": candidate.get("learned_fraction_rho"),
        "residual_encoder": candidate.get("encoder") == "two_layer_sparse_residual_gelu_layernorm",
        "dgi_head": "dgi_weight" in candidate,
    }


@dataclass
class Night5ATrainingResult:
    output: Dict[str, np.ndarray]
    logs: Sequence[dict]
    model: nn.Module
    probe: Optional[dict]
    coefficients: Dict[str, float]
    initial_losses: Dict[str, float]
    initial_state_sha256: str
    final_state_sha256: str
    auxiliary: Dict[str, object]


class Night5ATrainer:
    def __init__(self, data: Mapping[str, object], cfg: Mapping[str, object],
                 candidate: Mapping[str, object], seed: int, device: torch.device,
                 artifacts: Optional[Mapping[str, object]] = None, eps: float = 1e-12):
        self.data = data
        self.cfg = dict(cfg)
        self.candidate = dict(candidate)
        self.candidate_id = str(candidate["id"])
        if self.candidate_id.split("_", 1)[0] not in REGISTERED_PREFIXES:
            raise ValueError("Unregistered Night-5A candidate: %s" % self.candidate_id)
        self.seed = int(seed)
        self.device = device
        self.eps = float(eps)
        self.artifacts = dict(artifacts or {})
        self.features1 = torch.as_tensor(data["features_omics1"], dtype=torch.float32, device=device)
        self.features2 = torch.as_tensor(data["features_omics2"], dtype=torch.float32, device=device)
        adjacencies = [data[name].to(device) for name in (
            "adj_spatial_omics1", "adj_feature_omics1", "adj_spatial_omics2", "adj_feature_omics2"
        )]
        if self.candidate_id in ("C08_RNA_ANCHOR05", "C09_RNA_ANCHOR10"):
            key = "anchor05" if self.candidate_id.endswith("05") else "anchor10"
            shared = self.artifacts[key].to(device)
            adjacencies[0] = shared
            adjacencies[2] = shared
        if any(not value.is_sparse for value in adjacencies):
            raise AssertionError("Night-5A adjacency must remain sparse")
        self.adjacencies = tuple(adjacencies)
        self.forward = _Forward(self.features1, self.features2, self.adjacencies)

    def new_model(self) -> nn.Module:
        fix_seed(self.seed)
        policy = candidate_model_policy(self.candidate)
        reliability = None
        if policy.get("attention_policy") == "frozen_local_reliability":
            rho = float(self.candidate["reliability_fraction_rho"])
            raw = np.asarray(self.artifacts["reliability"], dtype=np.float32)
            reliability = (1.0 - rho) * 0.5 + rho * raw
        return Night5AModel(
            self.features1.shape[1], int(self.cfg["embedding_dim"]),
            self.features2.shape[1], int(self.cfg["embedding_dim"]),
            attention_policy=policy["attention_policy"],
            learned_fraction=policy.get("learned_fraction"),
            reliability_weights=reliability,
            residual_encoder=bool(policy.get("residual_encoder")),
            dgi_head=bool(policy.get("dgi_head")),
        ).to(self.device)

    def _auxiliary_loss(self, model: nn.Module, result: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        zero = result["emb_latent_combined"].sum() * 0.0
        z = result["emb_latent_combined"]
        if self.candidate_id in ("C10_MNN_TRIPLET01", "C11_MNN_TRIPLET03"):
            rows = torch.as_tensor(self.artifacts["triplets"], dtype=torch.long, device=self.device)
            raw = F.triplet_margin_loss(z[rows[:, 0]], z[rows[:, 1]], z[rows[:, 2]],
                                        margin=float(self.candidate["triplet_margin"]), reduction="mean")
            weight = float(self.candidate["triplet_weight"])
            return raw * weight, {"name": "mnn_triplet", "raw": float(raw.detach().cpu()), "weight": weight}
        if self.candidate_id == "C12_NEIGHBOR_CONTRAST":
            rows = torch.as_tensor(self.artifacts["contrast"], dtype=torch.long, device=self.device)
            normalized = F.normalize(z, p=2, dim=1, eps=1e-12)
            positive = torch.sum(normalized[rows[:, 0]] * normalized[rows[:, 1]], dim=1)
            negative = torch.sum(normalized[rows[:, 0]] * normalized[rows[:, 2]], dim=1)
            logits = torch.stack((positive, negative), dim=1) / float(self.candidate["temperature"])
            raw = F.cross_entropy(logits, torch.zeros(len(rows), dtype=torch.long, device=self.device))
            weight = float(self.candidate["contrastive_weight"])
            return raw * weight, {"name": "neighbor_contrast", "raw": float(raw.detach().cpu()), "weight": weight}
        if self.candidate_id == "C16_DGI01":
            permutation = torch.as_tensor(self.artifacts["dgi_permutation"], dtype=torch.long, device=self.device)
            corrupted = model(self.features1[permutation], self.features2[permutation], *self.adjacencies)
            summary = torch.sigmoid(z.mean(dim=0))
            positive = torch.sum(torch.mm(z, model.dgi_bilinear) * summary, dim=1)
            negative = torch.sum(torch.mm(corrupted["emb_latent_combined"], model.dgi_bilinear) * summary, dim=1)
            raw = 0.5 * (F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive)) +
                         F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative)))
            weight = float(self.candidate["dgi_weight"])
            return raw * weight, {
                "name": "dgi", "raw": float(raw.detach().cpu()), "weight": weight,
                "positive_logit_mean": float(positive.detach().mean().cpu()),
                "negative_logit_mean": float(negative.detach().mean().cpu()),
            }
        return zero, {"name": "none", "raw": 0.0, "weight": 0.0}

    def _coefficients(self, probe: Mapping[str, object]) -> Dict[str, float]:
        calibration = self.candidate.get("loss_calibration")
        if calibration == "legacy_gradient_geometric_mix":
            return hybrid_coefficients(probe["gradients"], self.cfg["loss_factors"],
                                       float(self.candidate["gradient_beta"]), self.eps)
        return active_ige_coefficients(probe["gradients"], active_mask_corr2_off(), self.eps)

    def _record(self, model, step, coefficients, gradient_norm, started):
        model.eval()
        with torch.no_grad():
            result = self.forward(model)
            losses = raw_losses(result, self.features1, self.features2)
            auxiliary, aux = self._auxiliary_loss(model, result)
            contributions = {name: float(losses[name].cpu()) * float(coefficients[name]) for name in LOSS_KEYS}
            total = float(sum(contributions.values()) + auxiliary.detach().cpu())
            row = {
                "step": int(step), "fraction_of_training": float(step / int(self.cfg["epochs"])),
                "wall_seconds": float(time.perf_counter() - started), "gradient_norm": float(gradient_norm),
                "total_loss": total, "embedding_finite": bool(torch.isfinite(result["emb_latent_combined"]).all().cpu()),
                "auxiliary_name": aux["name"], "auxiliary_raw": aux["raw"],
                "auxiliary_weight": aux["weight"], "auxiliary_contribution": float(auxiliary.detach().cpu()),
                "attention_cross_entropy": float((-torch.clamp(result["alpha"], 1e-12, 1.0) *
                                                  torch.log(torch.clamp(result["alpha"], 1e-12, 1.0))).sum(1).mean().cpu()),
                "checkpoint_state_sha256": model_state_sha256(model),
            }
            for name in LOSS_KEYS:
                short = name.replace("L_", "").replace("_raw", "")
                row[name] = float(losses[name].cpu())
                row[short + "_coefficient"] = float(coefficients[name])
                row[short + "_contribution"] = contributions[name]
            for key, alpha in (("cross", result["alpha"]), ("rna_within", result["alpha_omics1"]),
                               ("mod2_within", result["alpha_omics2"])):
                row[key + "_weight0_mean"] = float(alpha[:, 0].mean().cpu())
                row[key + "_weight1_mean"] = float(alpha[:, 1].mean().cpu())
        model.train()
        return row

    def _embedding_geometry(self, embedding: torch.Tensor) -> dict:
        value = F.normalize(embedding.detach(), p=2, dim=1, eps=1e-12)
        variance = float(torch.var(value, dim=0, unbiased=False).mean().cpu())
        graph = self.adjacencies[0].coalesce()
        rows, cols = graph.indices()
        keep = rows != cols
        rows, cols = rows[keep], cols[keep]
        if len(rows):
            neighbor = float(torch.sum(value[rows] * value[cols], dim=1).mean().cpu())
        else:
            neighbor = float("nan")
        generator = torch.Generator(device="cpu"); generator.manual_seed(20260813)
        sample_count = max(1, min(int(len(rows)), 10000))
        left = torch.randint(0, len(value), (sample_count,), generator=generator, device="cpu").to(value.device)
        right = torch.randint(0, len(value), (sample_count,), generator=generator, device="cpu").to(value.device)
        nonneighbor = float(torch.sum(value[left] * value[right], dim=1).mean().cpu())
        return {"pairwise_variance": variance, "neighbor_cosine": neighbor,
                "sampled_non_neighbor_cosine": nonneighbor,
                "neighbor_non_neighbor_cosine_gap": neighbor - nonneighbor}

    def train(self):
        if self.candidate_id in LEGACY_DELEGATES:
            return Night3BTrainer(self.data, self.cfg, LEGACY_DELEGATES[self.candidate_id],
                                  self.seed, self.device, self.eps).train()
        model = self.new_model()
        initial_state = _clone_state(model)
        initial_state_hash = state_dict_sha256(initial_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
        initial_result = self.forward(model)
        initial_geometry = self._embedding_geometry(initial_result["emb_latent_combined"])
        initial_raw = raw_losses(initial_result, self.features1, self.features2)
        initial_losses = {name: float(value.detach().cpu()) for name, value in initial_raw.items()}
        probe = run_initial_probe(model, self.forward, self.eps)
        model.load_state_dict(initial_state)
        if model_state_sha256(model) != initial_state_hash:
            raise AssertionError("Initial probe did not restore exact state")
        coefficients = self._coefficients(probe)
        started = time.perf_counter()
        record_steps = set(required_record_steps(int(self.cfg["epochs"])))
        logs = []
        initial_result = self.forward(model)
        initial_base = calibrated_total(raw_losses(initial_result, self.features1, self.features2), coefficients)
        initial_aux, _ = self._auxiliary_loss(model, initial_result)
        gradients = torch.autograd.grad(initial_base + initial_aux,
                                        [p for p in model.parameters() if p.requires_grad], allow_unused=True)
        grad_norm = math.sqrt(sum(float(torch.sum(g.detach().double() ** 2).cpu())
                                  for g in gradients if g is not None))
        logs.append(self._record(model, 0, coefficients, grad_norm, started))
        for step in range(1, int(self.cfg["epochs"]) + 1):
            result = self.forward(model)
            base = calibrated_total(raw_losses(result, self.features1, self.features2), coefficients)
            auxiliary, _ = self._auxiliary_loss(model, result)
            total = base + auxiliary
            optimizer.zero_grad(); total.backward()
            grad_norm = global_gradient_norm(model)
            optimizer.step()
            if step in record_steps:
                logs.append(self._record(model, step, coefficients, grad_norm, started))
        if [row["step"] for row in logs] != sorted(record_steps):
            raise AssertionError("Night-5A checkpoints incomplete")
        if not all(row["embedding_finite"] and np.isfinite(row["total_loss"]) for row in logs):
            raise FloatingPointError("Night-5A training produced non-finite values")
        model.eval()
        with torch.no_grad():
            result = self.forward(model)
        output = {
            "emb_latent_omics1": F.normalize(result["emb_latent_omics1"], p=2, eps=1e-12, dim=1).cpu().numpy(),
            "emb_latent_omics2": F.normalize(result["emb_latent_omics2"], p=2, eps=1e-12, dim=1).cpu().numpy(),
            "SpaLORA": F.normalize(result["emb_latent_combined"], p=2, eps=1e-12, dim=1).cpu().numpy(),
            "alpha_omics1": result["alpha_omics1"].cpu().numpy(),
            "alpha_omics2": result["alpha_omics2"].cpu().numpy(),
            "alpha": result["alpha"].cpu().numpy(),
        }
        active = [name for name in LOSS_KEYS if coefficients[name] != 0.0]
        auxiliary = {
            "active_loss_names": active, "active_loss_coefficient_sum": float(sum(coefficients[name] for name in active)),
            "corr2_objective_contribution_exact_zero": coefficients["L_corr2_raw"] == 0.0,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "initial_embedding_geometry": initial_geometry,
            "final_embedding_geometry": self._embedding_geometry(result["emb_latent_combined"]),
        }
        return Night5ATrainingResult(
            output=output, logs=logs, model=model, probe=probe, coefficients=coefficients,
            initial_losses=initial_losses, initial_state_sha256=initial_state_hash,
            final_state_sha256=model_state_sha256(model), auxiliary=auxiliary,
        )


def run_identity(dataset: str, candidate_id: str, seed: int, config_sha256: str,
                 artifact_manifest_sha256: str) -> dict:
    payload = {
        "dataset": str(dataset), "candidate_id": str(candidate_id), "seed": int(seed),
        "config_sha256": str(config_sha256),
        "label_free_artifact_manifest_sha256": str(artifact_manifest_sha256),
    }
    payload["run_identity_sha256"] = canonical_sha256(payload)
    return payload


__all__ = [
    "LEGACY_DELEGATES", "Night5AModel", "Night5ATrainer", "ResidualSparseEncoder",
    "active_mask_corr2_off", "anchor_graph", "build_label_free_artifacts",
    "canonical_sha256", "candidate_model_policy", "frozen_contrast_pairs",
    "frozen_triplets", "hybrid_coefficients", "load_label_free_artifacts",
    "load_registry", "local_reliability_weights", "registry_contracts", "run_identity",
    "sha256_file",
]
