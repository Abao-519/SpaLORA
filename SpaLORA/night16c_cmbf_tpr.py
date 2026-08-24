"""Night-16C cross-modal boundary field and trusted prototype refinement.

The producer accepts only numeric views, a sparse spatial graph, a start bank,
an initial exact-K partition and numeric configuration.  It has no dataset or
reference-label argument.  Public annotations are consumed by a separate
evaluator after candidate partitions have been materialized and hashed.

The cross-modal boundary field (CMBF) represents every sparse edge by three
continuous states: within-domain support, consensus boundary, and modality
conflict.  Trusted prototype refinement (TPR) anchors stable/high-margin cores
and only permits low-trust boundary observations to move under a common unary
+ anisotropic Potts + rejected-mass self-return energy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


@dataclass(frozen=True)
class CMBFTPRConfig:
    directional_mix: float = 0.0
    conflict_pass: float = 0.0
    pairwise_strength: float = 0.10
    self_return_strength: float = 0.10
    trust_threshold: float = 0.55
    anchor_strength: float = 0.10
    unary_view1_weight: float = 0.50
    move_boundary_floor: float = 0.10
    sweeps: int = 1
    min_cluster_fraction_of_equal: float = 0.01
    boundary_enabled: bool = True
    conflict_enabled: bool = True
    trust_gate_enabled: bool = True
    directional_enabled: bool = True

    def validate(self) -> None:
        bounded = {
            "directional_mix": self.directional_mix,
            "conflict_pass": self.conflict_pass,
            "trust_threshold": self.trust_threshold,
            "unary_view1_weight": self.unary_view1_weight,
            "move_boundary_floor": self.move_boundary_floor,
        }
        for name, value in bounded.items():
            if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")
        nonnegative = {
            "pairwise_strength": self.pairwise_strength,
            "self_return_strength": self.self_return_strength,
            "anchor_strength": self.anchor_strength,
            "min_cluster_fraction_of_equal": self.min_cluster_fraction_of_equal,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if int(self.sweeps) < 0:
            raise ValueError("sweeps must be nonnegative")


@dataclass(frozen=True)
class BoundaryEvidence:
    graph: sp.csr_matrix
    support: np.ndarray
    boundary: np.ndarray
    conflict: np.ndarray
    conductance: np.ndarray
    rejected_mass: np.ndarray
    node_support: np.ndarray
    node_boundary: np.ndarray
    node_conflict: np.ndarray
    start_stability: np.ndarray
    prototype_confidence: np.ndarray
    node_trust: np.ndarray
    view1: np.ndarray
    view2: np.ndarray


def encode_partition(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 1:
        raise ValueError("partition must be one-dimensional")
    return np.unique(value, return_inverse=True)[1].astype(np.int32)


def partition_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value, dtype=np.int32))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def canonical_graph(graph: sp.spmatrix, n: int | None = None) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    if graph.shape[0] != graph.shape[1] or (n is not None and graph.shape != (n, n)):
        raise ValueError("graph must be square and match observations")
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sum_duplicates()
    graph.sort_indices()
    if graph.nnz == 0 or not np.isfinite(graph.data).all() or np.any(graph.data < 0):
        raise ValueError("graph must have finite nonnegative sparse edges")
    return graph


def robust_standardize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("views must be finite two-dimensional matrices")
    median = np.median(value, axis=0, keepdims=True)
    mad = np.median(np.abs(value - median), axis=0, keepdims=True)
    fallback = np.std(value, axis=0, keepdims=True)
    scale = np.where(mad > 1e-8, 1.4826 * mad, np.maximum(fallback, 1e-8))
    standardized = (value - median) / scale
    standardized = np.clip(standardized, -12.0, 12.0)
    return standardized.astype(np.float32)


def _row_normalized(graph: sp.csr_matrix) -> sp.csr_matrix:
    row_sum = np.asarray(graph.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum, dtype=np.float64)
    valid = row_sum > 0
    inverse[valid] = 1.0 / row_sum[valid]
    return sp.diags(inverse).dot(graph).tocsr()


def _edge_change(value: np.ndarray, graph: sp.csr_matrix, directional_mix: float) -> np.ndarray:
    row = np.repeat(np.arange(graph.shape[0], dtype=np.int64), np.diff(graph.indptr))
    col = graph.indices.astype(np.int64, copy=False)
    direct = np.sqrt(np.mean((value[row] - value[col]) ** 2, axis=1))
    if directional_mix <= 0:
        return direct
    local_mean = _row_normalized(graph).dot(value)
    residual = value - local_mean
    directional = np.sqrt(np.mean((residual[row] - residual[col]) ** 2, axis=1))
    return (1.0 - directional_mix) * direct + directional_mix * directional


def _empirical_rank(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all() or len(value) == 0:
        raise ValueError("edge changes must be finite and nonempty")
    return (rankdata(value, method="average") - 0.5) / float(len(value))


def align_partition(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference = encode_partition(reference)
    candidate = encode_partition(candidate)
    nr = int(reference.max()) + 1
    nc = int(candidate.max()) + 1
    table = np.zeros((nr, nc), dtype=np.int64)
    np.add.at(table, (reference, candidate), 1)
    row, col = linear_sum_assignment(-table)
    mapping = {int(c): int(r) for r, c in zip(row, col)}
    for group in range(nc):
        if group not in mapping:
            mapping[group] = int(np.argmax(table[:, group]))
    return np.asarray([mapping[int(x)] for x in candidate], dtype=np.int32)


def start_stability(reference: np.ndarray, start_bank: np.ndarray | None) -> np.ndarray:
    reference = encode_partition(reference)
    if start_bank is None:
        return np.ones(len(reference), dtype=np.float32)
    bank = np.asarray(start_bank)
    if bank.ndim != 2 or bank.shape[1] != len(reference):
        raise ValueError("start bank must have shape [starts, observations]")
    aligned = [align_partition(reference, row) for row in bank]
    aligned.append(reference)
    return np.mean(np.stack(aligned, axis=0) == reference[None, :], axis=0).astype(np.float32)


def _robust_centers(value: np.ndarray, partition: np.ndarray, k: int, core: np.ndarray | None = None) -> np.ndarray:
    centers = []
    for group in range(int(k)):
        mask = partition == group
        if core is not None and int(np.sum(mask & core)) >= 3:
            mask &= core
        if not np.any(mask):
            raise ValueError("empty cluster in prototype construction")
        centers.append(np.median(value[mask], axis=0))
    return np.stack(centers).astype(np.float32)


def _prototype_distances(value: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.mean((value[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    scale = float(np.median(distances[np.isfinite(distances)]))
    return (distances / max(scale, 1e-8)).astype(np.float32)


def _prototype_confidence(dist1: np.ndarray, dist2: np.ndarray, partition: np.ndarray) -> np.ndarray:
    n = len(partition)
    rows = np.arange(n)
    current = 0.5 * (dist1[rows, partition] + dist2[rows, partition])
    merged = 0.5 * (dist1 + dist2)
    alternatives = merged.copy()
    alternatives[rows, partition] = np.inf
    alternative = np.min(alternatives, axis=1)
    margin = alternative - current
    scale = float(np.median(np.abs(margin - np.median(margin)))) * 1.4826
    z = np.clip(margin / max(scale, 1e-8), -12.0, 12.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _node_mean(graph: sp.csr_matrix, edge_value: np.ndarray) -> np.ndarray:
    weighted = graph.copy()
    weighted.data = np.asarray(edge_value, dtype=np.float64)
    count = np.diff(graph.indptr).astype(np.float64)
    total = np.asarray(weighted.sum(axis=1)).reshape(-1)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0).astype(np.float32)


def prepare_boundary_evidence(
    view1: np.ndarray,
    view2: np.ndarray,
    graph: sp.spmatrix,
    initial: np.ndarray,
    start_bank: np.ndarray | None,
    config: CMBFTPRConfig,
) -> BoundaryEvidence:
    config.validate()
    initial = encode_partition(initial)
    k = int(initial.max()) + 1
    view1 = robust_standardize(view1)
    view2 = robust_standardize(view2)
    if len(view1) != len(view2) or len(view1) != len(initial):
        raise ValueError("view/partition observation mismatch")
    graph = canonical_graph(graph, len(initial))
    directional_mix = float(config.directional_mix) if config.directional_enabled else 0.0
    rank1 = _empirical_rank(_edge_change(view1, graph, directional_mix))
    rank2 = _empirical_rank(_edge_change(view2, graph, directional_mix))

    support = (1.0 - rank1) * (1.0 - rank2)
    boundary = rank1 * rank2
    conflict = rank1 * (1.0 - rank2) + rank2 * (1.0 - rank1)
    # The three probabilities are an exact soft partition of edge state.
    total = support + boundary + conflict
    support = support / total
    boundary = boundary / total
    conflict = conflict / total

    if not config.boundary_enabled:
        conductance = support + boundary + float(config.conflict_pass) * conflict
    elif not config.conflict_enabled:
        conductance = support + conflict
    else:
        conductance = support + float(config.conflict_pass) * conflict
    conductance = np.clip(conductance, 0.0, 1.0).astype(np.float32)

    base_mass = np.asarray(graph.sum(axis=1)).reshape(-1)
    accepted = graph.copy()
    accepted.data = accepted.data * conductance
    accepted_mass = np.asarray(accepted.sum(axis=1)).reshape(-1)
    rejected_mass = np.divide(
        base_mass - accepted_mass,
        base_mass,
        out=np.zeros_like(base_mass),
        where=base_mass > 0,
    ).astype(np.float32)

    centers1 = _robust_centers(view1, initial, k)
    centers2 = _robust_centers(view2, initial, k)
    dist1 = _prototype_distances(view1, centers1)
    dist2 = _prototype_distances(view2, centers2)
    stability = start_stability(initial, start_bank)
    prototype = _prototype_confidence(dist1, dist2, initial)
    node_support = _node_mean(graph, support)
    node_boundary = _node_mean(graph, boundary)
    node_conflict = _node_mean(graph, conflict)
    # A geometric conjunction prevents a single strong cue from masking an
    # unstable prototype or a boundary-heavy neighbourhood.  The actual move
    # gate below is stricter still: stability and prototype confidence must
    # both be below threshold, while boundary evidence must exceed its floor.
    node_trust = np.clip(
        np.power(
            np.maximum(stability, 1e-6)
            * np.maximum(prototype, 1e-6)
            * np.maximum(1.0 - node_boundary, 1e-6)
            * np.maximum(1.0 - node_conflict, 1e-6),
            0.25,
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    return BoundaryEvidence(
        graph=graph,
        support=support.astype(np.float32),
        boundary=boundary.astype(np.float32),
        conflict=conflict.astype(np.float32),
        conductance=conductance,
        rejected_mass=rejected_mass,
        node_support=node_support,
        node_boundary=node_boundary,
        node_conflict=node_conflict,
        start_stability=stability,
        prototype_confidence=prototype,
        node_trust=node_trust,
        view1=view1,
        view2=view2,
    )


def _accepted_graph(evidence: BoundaryEvidence) -> sp.csr_matrix:
    accepted = evidence.graph.copy().astype(np.float64)
    accepted.data = accepted.data * evidence.conductance
    row_sum = np.asarray(accepted.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum)
    valid = row_sum > 0
    inverse[valid] = 1.0 / row_sum[valid]
    return sp.diags(inverse).dot(accepted).tocsr()


def trusted_prototype_refinement(
    initial: np.ndarray,
    evidence: BoundaryEvidence,
    config: CMBFTPRConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    config.validate()
    initial = encode_partition(initial)
    partition = initial.copy()
    k = int(partition.max()) + 1
    n = len(partition)
    threshold = max(5, int(np.ceil(float(config.min_cluster_fraction_of_equal) * n / k)))
    accepted = _accepted_graph(evidence)
    core = (
        (evidence.start_stability >= float(config.trust_threshold))
        | (evidence.prototype_confidence >= float(config.trust_threshold))
        | (evidence.node_boundary < float(config.move_boundary_floor))
    )
    total_moves = 0
    sweep_rows: list[dict[str, int]] = []

    for sweep in range(int(config.sweeps)):
        centers1 = _robust_centers(evidence.view1, partition, k, core)
        centers2 = _robust_centers(evidence.view2, partition, k, core)
        dist1 = _prototype_distances(evidence.view1, centers1)
        dist2 = _prototype_distances(evidence.view2, centers2)
        unary = float(config.unary_view1_weight) * dist1 + (1.0 - float(config.unary_view1_weight)) * dist2
        counts = np.bincount(partition, minlength=k).astype(np.int64)
        moves = 0
        start_of_sweep = partition.copy()
        for node in range(n):
            if config.trust_gate_enabled:
                jointly_low_trust = (
                    evidence.start_stability[node] < float(config.trust_threshold)
                    and evidence.prototype_confidence[node] < float(config.trust_threshold)
                    and evidence.node_boundary[node] >= float(config.move_boundary_floor)
                )
                if not jointly_low_trust:
                    continue
            begin, end = evidence.graph.indptr[node : node + 2]
            neighbours = evidence.graph.indices[begin:end]
            if len(neighbours) == 0:
                continue
            neighbour_labels = partition[neighbours]
            if evidence.node_boundary[node] < float(config.move_boundary_floor):
                continue
            candidates = np.unique(np.concatenate(([partition[node]], neighbour_labels)))
            abegin, aend = accepted.indptr[node : node + 2]
            accepted_neighbours = accepted.indices[abegin:aend]
            accepted_weights = accepted.data[abegin:aend]
            costs = []
            for group in candidates:
                pairwise = float(np.sum(accepted_weights * (partition[accepted_neighbours] != group)))
                move = int(group != start_of_sweep[node])
                anchor = int(group != initial[node])
                cost = float(unary[node, group])
                cost += float(config.pairwise_strength) * pairwise
                cost += float(config.self_return_strength) * float(evidence.rejected_mass[node]) * move
                cost += float(config.anchor_strength) * float(evidence.node_trust[node]) * anchor
                costs.append(cost)
            costs = np.asarray(costs, dtype=np.float64)
            old = int(partition[node])
            new = int(candidates[int(np.argmin(costs))])
            if new == old or counts[old] <= threshold:
                continue
            old_cost = float(costs[np.flatnonzero(candidates == old)[0]])
            new_cost = float(np.min(costs))
            if new_cost < old_cost - 1e-10:
                partition[node] = new
                counts[old] -= 1
                counts[new] += 1
                moves += 1
        total_moves += moves
        sweep_rows.append({"sweep": int(sweep), "moves": int(moves)})
        if moves == 0:
            break

    partition = encode_partition(partition)
    sizes = np.bincount(partition, minlength=k)
    if len(np.unique(partition)) != k or int(sizes.min()) < threshold:
        raise RuntimeError("TPR violated exact-K/minimum-cluster contract")
    diagnostics: dict[str, object] = {
        "config": asdict(config),
        "initial_partition_sha256": partition_sha256(initial),
        "partition_sha256": partition_sha256(partition),
        "changed_observations": int(np.sum(partition != initial)),
        "total_moves": int(total_moves),
        "sweeps": sweep_rows,
        "min_cluster_threshold": int(threshold),
        "min_cluster_size": int(sizes.min()),
        "cluster_sizes": [int(x) for x in sizes],
        "support_mean": float(np.mean(evidence.support)),
        "boundary_mean": float(np.mean(evidence.boundary)),
        "conflict_mean": float(np.mean(evidence.conflict)),
        "conductance_mean": float(np.mean(evidence.conductance)),
        "rejected_mass_mean": float(np.mean(evidence.rejected_mass)),
        "trust_mean": float(np.mean(evidence.node_trust)),
        "anchored_observations": int(np.sum(core)),
        "dense_n_by_n_count": 0,
        "label_reads": 0,
    }
    return partition, diagnostics


def cmbf_tpr(
    initial: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    graph: sp.spmatrix,
    start_bank: np.ndarray | None,
    config: CMBFTPRConfig,
) -> tuple[np.ndarray, BoundaryEvidence, dict[str, object]]:
    evidence = prepare_boundary_evidence(view1, view2, graph, initial, start_bank, config)
    partition, diagnostics = trusted_prototype_refinement(initial, evidence, config)
    return partition, evidence, diagnostics


def sparse_graph_from_csr_arrays(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: Iterable[int],
) -> sp.csr_matrix:
    shape_tuple = tuple(int(x) for x in shape)
    return canonical_graph(sp.csr_matrix((data, indices, indptr), shape=shape_tuple))
