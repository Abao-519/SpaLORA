"""Sparse, study-normalized edge features and shared edge models for Night-23A."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import torch
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigsh


def file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def stable_seed(text: str, base: int = 2301) -> int:
    return (int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) + base) % (2**31 - 1)


def source_lanes(all_lanes: Iterable[str], heldout: str) -> list[str]:
    lanes = list(all_lanes)
    if heldout not in lanes or len(set(lanes)) != len(lanes):
        raise RuntimeError("invalid held-out/source lane contract")
    output = [lane for lane in lanes if lane != heldout]
    if heldout in output or len(output) != len(lanes) - 1:
        raise RuntimeError("held-out leakage in source lanes")
    return output


def load_csr(archive, prefix: str) -> sp.csr_matrix:
    graph = sp.csr_matrix(
        (archive[prefix + "__data"], archive[prefix + "__indices"], archive[prefix + "__indptr"]),
        shape=tuple(int(x) for x in archive[prefix + "__shape"]),
        dtype=np.float64,
    )
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def robust_columns(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    median = np.median(value, axis=0)
    mad = np.median(np.abs(value - median), axis=0) * 1.4826
    std = np.std(value, axis=0)
    scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
    return (value - median) / scale


def row_unit(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norm, 1e-12)


def percentile(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if not len(value):
        return value.copy()
    return (rankdata(value, method="average") - 0.5) / len(value)


def node_percentile(value: np.ndarray) -> np.ndarray:
    return percentile(np.asarray(value, dtype=np.float64))


def edge_keys(rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    lo, hi = np.minimum(rows, cols), np.maximum(rows, cols)
    if np.any(lo == hi) or np.any(lo < 0) or np.any(hi >= n):
        raise RuntimeError("invalid undirected edge")
    return lo * np.int64(n) + hi


def knn_state(value: np.ndarray, k: int) -> dict:
    value = robust_columns(value)
    n = len(value)
    if n <= k:
        raise RuntimeError("kNN requires n > k")
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1)
    model.fit(value)
    distances, indices = model.kneighbors(value)
    indices, distances = indices[:, 1:], distances[:, 1:]
    directed_rank = {}
    neighbor_sets = []
    for i in range(n):
        current = set()
        for rank, j in enumerate(indices[i], start=1):
            j = int(j)
            directed_rank[(i, j)] = rank
            current.add(j)
        neighbor_sets.append(current)
    mutual = []
    rr = {}
    for (i, j), rank_ij in directed_rank.items():
        rank_ji = directed_rank.get((j, i))
        if rank_ji is None or i >= j:
            continue
        mutual.append((i, j))
        rr[(i, j)] = 0.5 * ((k + 1 - rank_ij) / k + (k + 1 - rank_ji) / k)
    kth = distances[:, -1]
    return {
        "standardized": value,
        "unit": row_unit(value),
        "indices": indices.astype(np.int32),
        "distances": distances.astype(np.float64),
        "directed_rank": directed_rank,
        "neighbor_sets": neighbor_sets,
        "mutual": np.asarray(mutual, dtype=np.int64).reshape(-1, 2),
        "rr": rr,
        "degree": None,
        "local_scale_rank": node_percentile(kth),
    }


FEATURE_NAMES = [
    "retained_similarity_rank",
    "view1_similarity_rank",
    "view2_similarity_rank",
    "view_similarity_min",
    "view_similarity_mean",
    "view_similarity_conflict",
    "retained_mutual",
    "view1_mutual",
    "view2_mutual",
    "retained_reciprocal_rank",
    "view1_reciprocal_rank",
    "view2_reciprocal_rank",
    "registered_spatial_edge",
    "registered_spatial_weight_rank",
    "spatial_degree_mean_rank",
    "spatial_degree_absdiff_rank",
    "retained_degree_mean_rank",
    "retained_degree_absdiff_rank",
    "view1_degree_mean_rank",
    "view1_degree_absdiff_rank",
    "view2_degree_mean_rank",
    "view2_degree_absdiff_rank",
    "retained_local_scale_mean_rank",
    "retained_local_scale_absdiff_rank",
    "view1_local_scale_mean_rank",
    "view1_local_scale_absdiff_rank",
    "view2_local_scale_mean_rank",
    "view2_local_scale_absdiff_rank",
    "cross_modal_neighbor_overlap_mean",
    "cross_modal_neighbor_overlap_absdiff",
]


def _edge_lookup(keys: np.ndarray, values: np.ndarray) -> dict[int, float]:
    return {int(k): float(v) for k, v in zip(keys, values)}


def build_union_edge_features(
    view1: np.ndarray,
    view2: np.ndarray,
    retained: np.ndarray,
    spatial: sp.csr_matrix,
    k: int = 12,
) -> dict:
    n = len(view1)
    if len(view2) != n or len(retained) != n or spatial.shape != (n, n):
        raise RuntimeError("carrier shape mismatch")
    states = {
        "retained": knn_state(retained, k),
        "view1": knn_state(view1, k),
        "view2": knn_state(view2, k),
    }
    spatial_upper = sp.triu(spatial, k=1).tocoo()
    spatial_keys = edge_keys(spatial_upper.row, spatial_upper.col, n)
    all_keys = set(int(x) for x in spatial_keys.tolist())
    for state in states.values():
        if len(state["mutual"]):
            all_keys.update(int(x) for x in edge_keys(state["mutual"][:, 0], state["mutual"][:, 1], n))
    keys = np.asarray(sorted(all_keys), dtype=np.int64)
    rows, cols = (keys // n).astype(np.int32), (keys % n).astype(np.int32)
    if len(keys) != len(np.unique(keys)):
        raise RuntimeError("duplicate union edges")

    spatial_weight_rank = percentile(spatial_upper.data)
    spatial_lookup = _edge_lookup(spatial_keys, spatial_weight_rank)
    spatial_indicator = np.asarray([int(key) in spatial_lookup for key in keys], dtype=np.float64)
    spatial_rank = np.asarray([spatial_lookup.get(int(key), 0.0) for key in keys], dtype=np.float64)

    similarity_rank = {}
    mutual_indicator, reciprocal_rank = {}, {}
    node_degree_rank = {}
    for name, state in states.items():
        similarity = np.sum(state["unit"][rows] * state["unit"][cols], axis=1)
        similarity_rank[name] = percentile(similarity)
        mutual_keys = (
            edge_keys(state["mutual"][:, 0], state["mutual"][:, 1], n)
            if len(state["mutual"])
            else np.empty(0, dtype=np.int64)
        )
        mutual_set = set(int(x) for x in mutual_keys.tolist())
        rr_lookup = {
            int(i * n + j): float(value)
            for (i, j), value in state["rr"].items()
        }
        mutual_indicator[name] = np.asarray([int(key) in mutual_set for key in keys], dtype=np.float64)
        reciprocal_rank[name] = np.asarray([rr_lookup.get(int(key), 0.0) for key in keys], dtype=np.float64)
        degree = np.zeros(n, dtype=np.float64)
        if len(state["mutual"]):
            np.add.at(degree, state["mutual"][:, 0], 1)
            np.add.at(degree, state["mutual"][:, 1], 1)
        node_degree_rank[name] = node_percentile(degree)

    spatial_degree = np.asarray(spatial.getnnz(axis=1), dtype=np.float64)
    spatial_degree_rank = node_percentile(spatial_degree)
    overlap = np.empty(n, dtype=np.float64)
    for i, (left, right) in enumerate(zip(states["view1"]["neighbor_sets"], states["view2"]["neighbor_sets"])):
        union = left | right
        overlap[i] = len(left & right) / max(len(union), 1)
    overlap = node_percentile(overlap)

    def endpoint_pair(node_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return 0.5 * (node_value[rows] + node_value[cols]), np.abs(node_value[rows] - node_value[cols])

    columns = [
        similarity_rank["retained"],
        similarity_rank["view1"],
        similarity_rank["view2"],
        np.minimum(similarity_rank["view1"], similarity_rank["view2"]),
        0.5 * (similarity_rank["view1"] + similarity_rank["view2"]),
        np.abs(similarity_rank["view1"] - similarity_rank["view2"]),
        mutual_indicator["retained"],
        mutual_indicator["view1"],
        mutual_indicator["view2"],
        reciprocal_rank["retained"],
        reciprocal_rank["view1"],
        reciprocal_rank["view2"],
        spatial_indicator,
        spatial_rank,
    ]
    columns.extend(endpoint_pair(spatial_degree_rank))
    for name in ("retained", "view1", "view2"):
        columns.extend(endpoint_pair(node_degree_rank[name]))
    for name in ("retained", "view1", "view2"):
        columns.extend(endpoint_pair(states[name]["local_scale_rank"]))
    columns.extend(endpoint_pair(overlap))
    features = np.column_stack(columns).astype(np.float32)
    if features.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(features)):
        raise RuntimeError("feature schema/finite failure")
    return {
        "rows": rows,
        "cols": cols,
        "features": features,
        "feature_names": np.asarray(FEATURE_NAMES, dtype="U"),
        "spatial_score": spatial_rank.astype(np.float32),
        "intersection_score": (spatial_indicator * mutual_indicator["retained"]).astype(np.float32),
        "edge_count": len(rows),
        "component_edge_counts": {
            "registered_spatial": int(spatial_indicator.sum()),
            "retained_mutual": int(mutual_indicator["retained"].sum()),
            "view1_mutual": int(mutual_indicator["view1"].sum()),
            "view2_mutual": int(mutual_indicator["view2"].sum()),
        },
    }


def teacher_relations(partition: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    partition = np.asarray(partition)
    if partition.ndim != 1 or np.any(rows >= len(partition)) or np.any(cols >= len(partition)):
        raise RuntimeError("teacher relation shape failure")
    return (partition[rows] == partition[cols]).astype(np.uint8)


def validate_teacher(partition: np.ndarray, k: int, spatial: sp.csr_matrix) -> dict:
    partition = np.asarray(partition)
    labels, counts = np.unique(partition, return_counts=True)
    upper = sp.triu(spatial, k=1).tocoo()
    internal = [int(np.sum((partition[upper.row] == label) & (partition[upper.col] == label))) for label in labels]
    return {
        "observed_k": int(len(labels)),
        "exact_k": int(len(labels)) == int(k),
        "cluster_sizes": counts.astype(int).tolist(),
        "min_cluster_size": int(counts.min()),
        "no_singleton": bool(counts.min() > 1),
        "internal_spatial_edges_per_cluster": internal,
        "all_clusters_have_internal_spatial_edge": bool(all(value > 0 for value in internal)),
    }


def deterministic_balanced_indices(y: np.ndarray, limit_per_class: int, seed_text: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.uint8)
    rng = np.random.default_rng(stable_seed(seed_text))
    pools = [np.flatnonzero(y == label) for label in (0, 1)]
    if any(not len(index) for index in pools):
        raise RuntimeError("teacher relation lacks a class")
    take = min(int(limit_per_class), *(len(index) for index in pools))
    chosen = []
    for index in pools:
        chosen.append(np.sort(rng.choice(index, size=take, replace=False)))
    output = np.concatenate(chosen)
    return output[rng.permutation(len(output))]


@dataclass(frozen=True)
class EdgeModelConfig:
    sample_per_class_per_study: int = 20000
    logistic_c: float = 1.0
    mlp_hidden: int = 32
    mlp_steps: int = 240
    mlp_batch_per_study: int = 2048
    mlp_learning_rate: float = 0.003
    weight_decay: float = 0.0001


class EdgeMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def fit_logistic(source: list[dict], config: EdgeModelConfig, shuffled: bool = False) -> dict:
    xs, ys = [], []
    for item in source:
        y = np.asarray(item["target"], dtype=np.uint8)
        if shuffled:
            rng = np.random.default_rng(stable_seed(item["lane"] + "__shuffled"))
            y = y[rng.permutation(len(y))]
        index = deterministic_balanced_indices(y, config.sample_per_class_per_study, item["lane"] + str(shuffled))
        xs.append(np.asarray(item["features"], dtype=np.float64)[index])
        ys.append(y[index])
    x = np.vstack(xs)
    y = np.concatenate(ys)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    x = (x - mean) / scale
    model = LogisticRegression(C=config.logistic_c, solver="lbfgs", max_iter=500, random_state=23)
    model.fit(x, y)
    return {
        "mean": mean,
        "scale": scale,
        "coef": model.coef_.reshape(-1),
        "intercept": float(model.intercept_[0]),
        "iterations": int(model.n_iter_[0]),
        "sample_count": int(len(y)),
    }


def predict_logistic(state: dict, features: np.ndarray) -> np.ndarray:
    x = (np.asarray(features, dtype=np.float64) - state["mean"]) / state["scale"]
    return expit(x @ state["coef"] + state["intercept"]).astype(np.float64)


def fit_mlp(source: list[dict], config: EdgeModelConfig, device: str = "cpu") -> dict:
    torch.manual_seed(2301)
    np.random.seed(2301)
    sampled = []
    for item in source:
        y = np.asarray(item["target"], dtype=np.uint8)
        index = deterministic_balanced_indices(y, config.sample_per_class_per_study, item["lane"] + "__mlp")
        sampled.append((np.asarray(item["features"], dtype=np.float32)[index], y[index].astype(np.float32)))
    all_x = np.vstack([item[0] for item in sampled]).astype(np.float64)
    mean = all_x.mean(axis=0)
    scale = all_x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    normalized = [(((x - mean) / scale).astype(np.float32), y) for x, y in sampled]
    model = EdgeMLP(len(mean), config.mlp_hidden).to(device)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config.mlp_learning_rate, weight_decay=config.weight_decay)
    generators = [np.random.default_rng(stable_seed(item["lane"] + "__batch")) for item in source]
    ledger = []
    model.train()
    for step in range(config.mlp_steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for (x, y), rng in zip(normalized, generators):
            size = min(config.mlp_batch_per_study, len(y))
            index = rng.choice(len(y), size=size, replace=False)
            tx = torch.as_tensor(x[index], device=device)
            ty = torch.as_tensor(y[index], device=device)
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(model(tx), ty))
        loss = torch.stack(losses).mean()
        loss.backward()
        grad_norm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)).cpu())
        optimizer.step()
        if step in {0, 1, 5, 20, 60, 120, config.mlp_steps - 1}:
            ledger.append({"step": step, "loss": float(loss.detach().cpu()), "grad_norm": grad_norm})
    change = float(sum((model.state_dict()[name].cpu() - value).abs().sum() for name, value in initial.items()))
    if not np.isfinite(change) or change <= 0:
        raise RuntimeError("MLP parameter update failed")
    return {
        "mean": mean,
        "scale": scale,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "config": asdict(config),
        "input_dim": len(mean),
        "parameter_l1_change": change,
        "loss_ledger": ledger,
    }


def predict_mlp(state: dict, features: np.ndarray, device: str = "cpu") -> np.ndarray:
    model = EdgeMLP(int(state["input_dim"]), int(state["config"]["mlp_hidden"]))
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device).eval()
    x = ((np.asarray(features, dtype=np.float64) - state["mean"]) / state["scale"]).astype(np.float32)
    output = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            logits = model(torch.as_tensor(x[start : start + 65536], device=device))
            output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def model_state_manifest(config: EdgeModelConfig) -> dict:
    return {"schema": "night23a-edge-model-config-v1", **asdict(config)}


STAGE_B_ARMS = (
    "RAW_EQUAL_UNION",
    "ARISE_LIKE_INTERSECTION",
    "SPATIAL_ONLY",
    "RETAINED_ONLY",
    "VIEW1_ONLY",
    "VIEW2_ONLY",
    "LOGISTIC_DIRECT",
    "MLP_NO_CONFIDENCE_ABSTENTION",
    "SHUFFLED_TEACHER_FULL_TRANSFORM",
    "FULL_XBED",
)


def confidence_abstention_weights(probability: np.ndarray) -> np.ndarray:
    """Conservative edge capacity: uncertain predictions return to raw-union weight one."""
    probability = np.asarray(probability, dtype=np.float64)
    if not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise RuntimeError("invalid edge probability")
    confidence = 2.0 * np.abs(probability - 0.5)
    output = (1.0 - confidence) + confidence * probability
    if not np.all(np.isfinite(output)) or np.any(output < 0):
        raise RuntimeError("invalid confidence-abstention capacity")
    return output


def stage_b_arm_weights(features: np.ndarray, prediction: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build all pre-registered Stage-B nonnegative capacities on one locked edge union."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("Stage-B feature schema mismatch")
    required = {
        "logistic_probability",
        "mlp_probability",
        "shuffled_teacher_probability",
        "arise_like_intersection_score",
    }
    if not required.issubset(prediction):
        raise RuntimeError("Stage-B prediction schema mismatch")
    m = len(features)
    for key in required:
        value = np.asarray(prediction[key])
        if value.shape != (m,):
            raise RuntimeError(f"Stage-B prediction shape mismatch: {key}")
    weights = {
        "RAW_EQUAL_UNION": np.ones(m, dtype=np.float64),
        "ARISE_LIKE_INTERSECTION": np.asarray(prediction["arise_like_intersection_score"], dtype=np.float64),
        "SPATIAL_ONLY": features[:, FEATURE_NAMES.index("registered_spatial_edge")],
        "RETAINED_ONLY": features[:, FEATURE_NAMES.index("retained_mutual")],
        "VIEW1_ONLY": features[:, FEATURE_NAMES.index("view1_mutual")],
        "VIEW2_ONLY": features[:, FEATURE_NAMES.index("view2_mutual")],
        "LOGISTIC_DIRECT": np.asarray(prediction["logistic_probability"], dtype=np.float64),
        "MLP_NO_CONFIDENCE_ABSTENTION": np.asarray(prediction["mlp_probability"], dtype=np.float64),
        "SHUFFLED_TEACHER_FULL_TRANSFORM": confidence_abstention_weights(
            prediction["shuffled_teacher_probability"]
        ),
        "FULL_XBED": confidence_abstention_weights(prediction["logistic_probability"]),
    }
    if tuple(weights) != STAGE_B_ARMS:
        raise RuntimeError("Stage-B arm ordering mismatch")
    for name, value in weights.items():
        if value.shape != (m,) or not np.all(np.isfinite(value)) or np.any(value < 0):
            raise RuntimeError(f"invalid Stage-B arm capacity: {name}")
        if float(value.sum()) <= 0:
            raise RuntimeError(f"zero Stage-B arm capacity: {name}")
    if np.array_equal(weights["FULL_XBED"], weights["RAW_EQUAL_UNION"]):
        raise RuntimeError("FULL_XBED is a no-op on the locked prediction")
    return weights


def spectral_exact_k_partition(
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    k: int,
    seed: int = 23,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (len(rows) == len(cols) == len(weights)) or np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise RuntimeError("invalid sparse partition weights")
    graph = sp.coo_matrix((weights, (rows, cols)), shape=(n, n))
    graph = (graph + graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    if np.any(degree <= 0):
        # Fixed, label-free epsilon self-degree repair without adding dense edges.
        graph = graph + sp.eye(n, format="csr") * 1e-8
        degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    inv = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized = sp.diags(inv) @ graph @ sp.diags(inv)
    v0 = np.linspace(1.0, 2.0, n, dtype=np.float64)
    v0 /= np.linalg.norm(v0)
    _, vectors = eigsh(normalized, k=k, which="LA", tol=1e-5, maxiter=max(5000, n * 2), v0=v0)
    vectors = row_unit(vectors)
    partition = KMeans(n_clusters=k, n_init=20, random_state=seed, algorithm="lloyd").fit_predict(vectors)
    if len(np.unique(partition)) != k:
        raise RuntimeError("exact-K partition failure")
    return partition.astype(np.int32)
