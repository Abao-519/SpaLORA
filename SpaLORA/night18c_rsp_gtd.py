"""Night-18C robust shared/private graph-trend decomposition.

The module is intentionally label-free and dataset-agnostic.  It decomposes
two co-registered molecular views into one boundary-preserving shared graph
trend and two spot-wise group-sparse private residuals.  Stage A is a direct,
deterministic convex-objective probe; it is not a learned encoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def robust_standardize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("view must be a finite matrix")
    center = np.median(value, axis=0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(value - center), axis=0, keepdims=True)
    std = np.std(value, axis=0, keepdims=True)
    scale = np.where(mad > 1e-7, mad, np.where(std > 1e-7, std, 1.0))
    return np.clip((value - center) / scale, -12.0, 12.0)


def row_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-10)


def orthogonal_procrustes_align(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align target score coordinates to reference score coordinates."""

    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if reference.shape != target.shape or reference.ndim != 2:
        raise ValueError("Procrustes score matrices must have the same two-dimensional shape")
    left, singular, right_t = np.linalg.svd(target.T @ reference, full_matrices=False)
    rotation = left @ right_t
    return row_normalize(target @ rotation), rotation, singular


def shared_view_basis(
    view1: np.ndarray, view2: np.ndarray, dimension: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Map views to a coordinated score basis without annotation.

    Independent PCA axes are not semantically aligned even when their shapes
    match.  We therefore solve the same-spot orthogonal Procrustes problem
    ``min_Q ||score2 Q - score1||_F`` and freeze the resulting rotation.
    """

    outputs = []
    for view in (view1, view2):
        standardized = robust_standardize(view)
        target = min(int(dimension), standardized.shape[1], standardized.shape[0] - 1)
        if target < 2:
            raise ValueError("insufficient rank for shared graph-trend basis")
        reduced = PCA(n_components=target, svd_solver="full", whiten=False).fit_transform(standardized)
        outputs.append(row_normalize(robust_standardize(reduced)))
    if outputs[0].shape[1] != outputs[1].shape[1]:
        target = min(outputs[0].shape[1], outputs[1].shape[1])
        outputs = [value[:, :target] for value in outputs]
    aligned, rotation, singular = orthogonal_procrustes_align(outputs[0], outputs[1])
    diagnostics = {
        "alignment": "SAME_SPOT_ORTHOGONAL_PROCRUSTES",
        "rotation_sha256": sha256_array(rotation.astype(np.float64)),
        "rotation_orthogonality_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(rotation.shape[0]))),
        "cross_singular_values": [float(x) for x in singular],
        "pre_alignment_frobenius": float(np.linalg.norm(outputs[1] - outputs[0])),
        "post_alignment_frobenius": float(np.linalg.norm(aligned - outputs[0])),
    }
    if diagnostics["rotation_orthogonality_error"] > 1e-8:
        raise RuntimeError("view-coordinate Procrustes rotation is not orthogonal")
    return outputs[0], aligned, diagnostics


def undirected_upper_graph(graph: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    upper = sp.triu(graph, k=1, format="coo")
    if upper.nnz == 0 or not np.isfinite(upper.data).all() or np.any(upper.data <= 0):
        raise ValueError("registered graph must contain finite positive undirected edges")
    return upper.row.astype(np.int64), upper.col.astype(np.int64), upper.data.astype(np.float64), graph.shape[0]


@dataclass(frozen=True)
class TrendConfig:
    config_id: str
    lambda_tv: float
    gamma_private: float
    huber_delta: float = 0.35
    step_z: float = 0.08
    step_r: float = 0.20
    iterations: int = 40
    trend_weight: float = 0.35
    trend_dimension: int = 24
    tv_epsilon: float = 1e-4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def stage_a_configs() -> tuple[TrendConfig, ...]:
    """Small pre-evaluation family grid; values are not lane-specific."""

    return (
        TrendConfig("T01_BALANCED", lambda_tv=0.06, gamma_private=0.06),
        TrendConfig("T02_MORE_TV", lambda_tv=0.12, gamma_private=0.06),
        TrendConfig("T03_MORE_PRIVATE", lambda_tv=0.06, gamma_private=0.12),
    )


def _huber_value_and_gradient(residual: np.ndarray, delta: float) -> tuple[float, np.ndarray]:
    norm = np.linalg.norm(residual, axis=1)
    small = norm <= delta
    value = np.empty_like(norm)
    value[small] = 0.5 * norm[small] ** 2 / delta
    value[~small] = norm[~small] - 0.5 * delta
    gradient = residual / np.maximum(norm[:, None], delta)
    return float(np.sum(value)), gradient


def decomposition_objective(
    h1: np.ndarray,
    h2: np.ndarray,
    z: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    weight: np.ndarray,
    config: TrendConfig,
    lambda_override: float | None = None,
) -> dict[str, float]:
    data1, _ = _huber_value_and_gradient(z + r1 - h1, config.huber_delta)
    data2, _ = _huber_value_and_gradient(z + r2 - h2, config.huber_delta)
    edge_norm = np.sqrt(np.sum((z[rows] - z[cols]) ** 2, axis=1) + config.tv_epsilon ** 2)
    private = float(np.sum(np.linalg.norm(r1, axis=1)) + np.sum(np.linalg.norm(r2, axis=1)))
    lam = config.lambda_tv if lambda_override is None else float(lambda_override)
    data = 0.5 * (data1 + data2)
    tv = lam * float(np.sum(weight * edge_norm))
    sparse = 0.5 * config.gamma_private * private
    return {"objective": data + tv + sparse, "data": data, "graph_tv": tv, "private": sparse}


def _tv_gradient(z: np.ndarray, rows: np.ndarray, cols: np.ndarray, weight: np.ndarray, epsilon: float) -> np.ndarray:
    difference = z[rows] - z[cols]
    scaled = weight[:, None] * difference / np.sqrt(np.sum(difference ** 2, axis=1, keepdims=True) + epsilon ** 2)
    gradient = np.zeros_like(z)
    np.add.at(gradient, rows, scaled)
    np.add.at(gradient, cols, -scaled)
    return gradient


def _group_shrink(value: np.ndarray, threshold: float) -> np.ndarray:
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    factor = np.maximum(1.0 - threshold / np.maximum(norm, 1e-12), 0.0)
    return value * factor


def _z_step(
    h1: np.ndarray,
    h2: np.ndarray,
    z: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    weight: np.ndarray,
    config: TrendConfig,
    lambda_value: float,
) -> tuple[np.ndarray, float]:
    _, grad1 = _huber_value_and_gradient(z + r1 - h1, config.huber_delta)
    _, grad2 = _huber_value_and_gradient(z + r2 - h2, config.huber_delta)
    gradient = 0.5 * (grad1 + grad2)
    if lambda_value > 0:
        gradient += lambda_value * _tv_gradient(z, rows, cols, weight, config.tv_epsilon)
    before = decomposition_objective(h1, h2, z, r1, r2, rows, cols, weight, config, lambda_value)["objective"]
    step = config.step_z
    for _ in range(18):
        candidate = z - step * gradient
        after = decomposition_objective(h1, h2, candidate, r1, r2, rows, cols, weight, config, lambda_value)["objective"]
        if after <= before + 1e-10:
            return candidate, step
        step *= 0.5
    return z.copy(), 0.0


def _residual_step(
    h1: np.ndarray,
    h2: np.ndarray,
    z: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    weight: np.ndarray,
    config: TrendConfig,
    lambda_value: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    before = decomposition_objective(h1, h2, z, r1, r2, rows, cols, weight, config, lambda_value)["objective"]
    _, grad1 = _huber_value_and_gradient(z + r1 - h1, config.huber_delta)
    _, grad2 = _huber_value_and_gradient(z + r2 - h2, config.huber_delta)
    step = config.step_r
    for _ in range(18):
        candidate1 = _group_shrink(r1 - 0.5 * step * grad1, 0.5 * step * config.gamma_private)
        candidate2 = _group_shrink(r2 - 0.5 * step * grad2, 0.5 * step * config.gamma_private)
        after = decomposition_objective(h1, h2, z, candidate1, candidate2, rows, cols, weight, config, lambda_value)["objective"]
        if after <= before + 1e-10:
            return candidate1, candidate2, step
        step *= 0.5
    return r1.copy(), r2.copy(), 0.0


def solve_shared_private(
    h1: np.ndarray,
    h2: np.ndarray,
    graph: sp.csr_matrix,
    config: TrendConfig,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Solve a registered Stage-A arm with monotone block updates."""

    if h1.shape != h2.shape:
        raise ValueError("modalities must share shape after fixed adapters")
    rows, cols, weight, n = undirected_upper_graph(graph)
    if n != len(h1):
        raise ValueError("graph/view observation mismatch")
    z = 0.5 * (h1 + h2)
    r1 = np.zeros_like(z)
    r2 = np.zeros_like(z)
    lambda_value = 0.0 if mode == "PRIVATE_ONLY" else config.lambda_tv
    update_private = mode in {"FULL", "PRIVATE_ONLY"}
    if mode not in {"FULL", "GRAPH_TV_ONLY", "PRIVATE_ONLY"}:
        raise ValueError(mode)
    trace: list[dict[str, float]] = []
    initial = decomposition_objective(h1, h2, z, r1, r2, rows, cols, weight, config, lambda_value)
    trace.append({"iteration": 0, **initial, "z_step": 0.0, "r_step": 0.0})
    previous_objective = initial["objective"]
    for iteration in range(1, config.iterations + 1):
        z, z_step = _z_step(h1, h2, z, r1, r2, rows, cols, weight, config, lambda_value)
        r_step = 0.0
        if update_private:
            r1, r2, r_step = _residual_step(h1, h2, z, r1, r2, rows, cols, weight, config, lambda_value)
        current = decomposition_objective(h1, h2, z, r1, r2, rows, cols, weight, config, lambda_value)
        if current["objective"] > previous_objective + 1e-8:
            raise RuntimeError("registered objective increased")
        previous_objective = current["objective"]
        if iteration in {1, config.iterations // 2, config.iterations}:
            trace.append({"iteration": iteration, **current, "z_step": z_step, "r_step": r_step})
    diagnostics = {
        "mode": mode,
        "trace": trace,
        "objective_monotone": True,
        "private1_nonzero_fraction": float(np.mean(np.linalg.norm(r1, axis=1) > 1e-8)),
        "private2_nonzero_fraction": float(np.mean(np.linalg.norm(r2, axis=1) > 1e-8)),
        "private1_norm": float(np.linalg.norm(r1)),
        "private2_norm": float(np.linalg.norm(r2)),
        "changed_shared_frobenius": float(np.linalg.norm(z - 0.5 * (h1 + h2))),
    }
    return z, r1, r2, diagnostics


def deterministic_permutation(ids: Iterable[str], salt: str) -> np.ndarray:
    keys = []
    for index, identifier in enumerate(ids):
        digest = hashlib.sha256((salt + "\0" + str(identifier)).encode("utf-8")).digest()
        keys.append((digest, index))
    order = np.asarray([index for _, index in sorted(keys)], dtype=np.int64)
    if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    return order


def solve_permuted_private(
    h1: np.ndarray,
    h2: np.ndarray,
    graph: sp.csr_matrix,
    config: TrendConfig,
    ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    full_z, r1, r2, full_diagnostics = solve_shared_private(h1, h2, graph, config, "FULL")
    order1 = deterministic_permutation(ids, "night18c-private-view1")
    order2 = deterministic_permutation(ids, "night18c-private-view2")
    permuted1, permuted2 = r1[order1], r2[order2]
    rows, cols, weight, _ = undirected_upper_graph(graph)
    z = 0.5 * (h1 + h2)
    trace = []
    previous = decomposition_objective(h1, h2, z, permuted1, permuted2, rows, cols, weight, config)["objective"]
    for iteration in range(1, config.iterations + 1):
        z, step = _z_step(h1, h2, z, permuted1, permuted2, rows, cols, weight, config, config.lambda_tv)
        current = decomposition_objective(h1, h2, z, permuted1, permuted2, rows, cols, weight, config)["objective"]
        if current > previous + 1e-8:
            raise RuntimeError("permuted-private objective increased")
        previous = current
        if iteration in {1, config.iterations // 2, config.iterations}:
            trace.append({"iteration": iteration, "objective": current, "z_step": step})
    return z, {
        "mode": "PERMUTED_PRIVATE",
        "full_private1_norm": float(np.linalg.norm(r1)),
        "full_private2_norm": float(np.linalg.norm(r2)),
        "permutation1_sha256": sha256_array(order1),
        "permutation2_sha256": sha256_array(order2),
        "permutation_nonidentity": bool(not np.array_equal(order1, np.arange(len(order1))) and not np.array_equal(order2, np.arange(len(order2)))),
        "trace": trace,
        "objective_monotone": True,
        "source_full_diagnostics": full_diagnostics,
        "changed_shared_frobenius": float(np.linalg.norm(z - full_z)),
    }


def l2_lowpass(trend: np.ndarray, graph: sp.csr_matrix, strength: float, steps: int = 8) -> np.ndarray:
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph, dtype=np.float64).T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(1)).ravel()
    inv = np.zeros_like(degree)
    inv[degree > 0] = 1.0 / degree[degree > 0]
    transition = sp.diags(inv) @ graph
    result = np.asarray(trend, dtype=np.float64).copy()
    for _ in range(int(steps)):
        result = (trend + strength * (transition @ result)) / (1.0 + strength)
    return result


def compose_representation(retained: np.ndarray, trend: np.ndarray | None, trend_weight: float) -> np.ndarray:
    anchor = row_normalize(robust_standardize(retained))
    if trend is None or trend_weight == 0:
        block = np.zeros((len(anchor), 24), dtype=np.float64)
    else:
        block = row_normalize(robust_standardize(trend))
    representation = np.concatenate([anchor, float(trend_weight) * block], axis=1)
    return row_normalize(representation).astype(np.float32)


def common_kmeans_endpoint(representation: np.ndarray, k: int) -> np.ndarray:
    partition = KMeans(n_clusters=int(k), n_init=20, random_state=0).fit_predict(representation)
    _, encoded = np.unique(partition, return_inverse=True)
    encoded = encoded.astype(np.int32)
    if len(np.unique(encoded)) != int(k):
        raise RuntimeError("endpoint violated exact K")
    return encoded
