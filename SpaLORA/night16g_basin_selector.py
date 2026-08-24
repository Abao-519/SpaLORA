"""Label-free evidence for selecting from locked multimodal partitions.

The module consumes only numeric views, sparse spatial graphs, a locked bank of
candidate partitions, and candidate provenance.  It never accepts annotations
or dataset identifiers.  Dense work is restricted to the small candidate by
candidate matrix; no observation by observation dense matrix is constructed.
Persistence is an optional evidence axis and may be disabled by a fitted
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True)
class BasinSelectorConfig:
    basin_ari_threshold: float = 0.82
    persistent_ari_threshold: float = 0.90
    molecular_weight: float = 1.0
    topology_weight: float = 1.0
    persistence_weight: float = 1.0
    risk_weight: float = 1.0
    representative_evidence_weight: float = 0.35
    ordered_path_weight: float = 0.0

    def validate(self) -> None:
        if not 0.0 < self.basin_ari_threshold <= self.persistent_ari_threshold <= 1.0:
            raise ValueError("invalid basin/persistence thresholds")
        for name in (
            "molecular_weight",
            "topology_weight",
            "persistence_weight",
            "risk_weight",
            "representative_evidence_weight",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 <= float(self.ordered_path_weight) <= 1.0:
            raise ValueError("ordered_path_weight must be between zero and one")


def partition_sha256(partition: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(partition, dtype=np.int32))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _encode(partition: np.ndarray) -> np.ndarray:
    _, encoded = np.unique(np.asarray(partition), return_inverse=True)
    return encoded.astype(np.int32)


def robust_standardize(view: np.ndarray) -> np.ndarray:
    view = np.asarray(view, dtype=np.float64)
    if view.ndim != 2 or not np.isfinite(view).all():
        raise ValueError("view must be a finite matrix")
    center = np.median(view, axis=0)
    scale = np.median(np.abs(view - center), axis=0) * 1.4826
    fallback = np.std(view, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    result = (view - center) / scale
    return np.clip(result, -12.0, 12.0)


def molecular_separation(view: np.ndarray, partition: np.ndarray) -> dict[str, float]:
    view = np.asarray(view, dtype=np.float64)
    partition = _encode(partition)
    n, _ = view.shape
    k = int(partition.max()) + 1
    global_center = np.mean(view, axis=0)
    total = float(np.sum((view - global_center) ** 2))
    within = 0.0
    centroids: list[np.ndarray] = []
    for group in range(k):
        block = view[partition == group]
        if len(block) == 0:
            raise ValueError("empty cluster")
        centroid = np.mean(block, axis=0)
        centroids.append(centroid)
        within += float(np.sum((block - centroid) ** 2))
    between = max(total - within, 0.0)
    explained = between / max(total, 1e-12)
    ch = (between / max(k - 1, 1)) / max(within / max(n - k, 1), 1e-12)
    centroid_matrix = np.stack(centroids)
    distances = np.sqrt(
        np.maximum(
            np.sum(centroid_matrix**2, axis=1)[:, None]
            + np.sum(centroid_matrix**2, axis=1)[None, :]
            - 2.0 * centroid_matrix @ centroid_matrix.T,
            0.0,
        )
    )
    distances[np.eye(k, dtype=bool)] = np.inf
    min_centroid_distance = float(np.min(distances)) if k > 1 else 0.0
    return {
        "explained": float(explained),
        "ch": float(ch),
        "within_per_observation": float(within / n),
        "min_centroid_distance": min_centroid_distance,
    }


def spatial_evidence(graph: sp.csr_matrix, partition: np.ndarray) -> dict[str, float]:
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph).T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    partition = _encode(partition)
    rows = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    total = float(graph.data.sum())
    agreement = float(
        np.sum(graph.data * (partition[rows] == partition[graph.indices])) / max(total, 1e-12)
    )
    probabilities = np.bincount(partition).astype(np.float64) / len(partition)
    chance = float(np.sum(probabilities**2))
    excess = (agreement - chance) / max(1.0 - chance, 1e-12)
    return {"agreement": agreement, "excess": float(excess)}


def candidate_similarity(partitions: np.ndarray) -> np.ndarray:
    partitions = np.asarray(partitions)
    m = len(partitions)
    similarity = np.eye(m, dtype=np.float64)
    for left in range(m):
        for right in range(left):
            value = adjusted_rand_score(partitions[left], partitions[right])
            similarity[left, right] = similarity[right, left] = float(value)
    return similarity


def plain_medoid_index(
    records: Sequence[dict[str, object]], similarity: np.ndarray
) -> int:
    """Return a deterministic partition medoid with candidate-ID tie break."""

    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.shape != (len(records), len(records)) or len(records) == 0:
        raise ValueError("record/similarity mismatch")
    centrality = (similarity.sum(axis=1) - 1.0) / max(len(records) - 1, 1)
    return min(
        range(len(records)),
        key=lambda index: (-float(centrality[index]), str(records[index]["candidate_id"])),
    )


def _percentile(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("ranking axis must be finite")
    ranks = rankdata(values if higher_is_better else -values, method="average")
    if len(values) == 1:
        return np.ones(1, dtype=np.float64)
    return (ranks - 1.0) / (len(values) - 1.0)


def _connected_components(similarity: np.ndarray, threshold: float) -> np.ndarray:
    n = len(similarity)
    parent = np.arange(n)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for left in range(n):
        for right in range(left):
            if similarity[left, right] >= threshold:
                a, b = find(left), find(right)
                if a != b:
                    parent[b] = a
    roots = [find(i) for i in range(n)]
    mapping = {value: index for index, value in enumerate(sorted(set(roots)))}
    return np.asarray([mapping[value] for value in roots], dtype=np.int32)


def select_basin(
    records: Sequence[dict[str, object]],
    partitions: np.ndarray,
    config: BasinSelectorConfig,
    similarity: np.ndarray | None = None,
) -> tuple[int, list[dict[str, object]], dict[str, object]]:
    """Select a basin, then its evidence-aware medoid representative."""

    config.validate()
    partitions = np.asarray(partitions, dtype=np.int32)
    if len(records) != len(partitions) or len(records) == 0:
        raise ValueError("candidate record/partition mismatch")
    if similarity is None:
        similarity = candidate_similarity(partitions)
    else:
        similarity = np.asarray(similarity, dtype=np.float64)
        if similarity.shape != (len(partitions), len(partitions)):
            raise ValueError("candidate similarity shape mismatch")
    basin_ids = _connected_components(similarity, config.basin_ari_threshold)
    m = len(records)
    molecular_raw = np.asarray([float(row["molecular_joint"]) for row in records])
    topology_raw = np.asarray([float(row["topology_joint"]) for row in records])
    risk_raw = np.asarray([float(row["microcluster_score"]) for row in records])
    molecular = _percentile(molecular_raw)
    topology = _percentile(topology_raw)
    risk = _percentile(risk_raw)
    persistence = np.zeros(m, dtype=np.float64)
    basin_size = np.zeros(m, dtype=np.float64)
    for index in range(m):
        peers = np.flatnonzero(basin_ids == basin_ids[index])
        basin_size[index] = len(peers) / m
        other = peers[peers != index]
        local = float(np.mean(similarity[index, other])) if len(other) else 0.0
        persistent = float(np.mean(similarity[index] >= config.persistent_ari_threshold))
        persistence[index] = 0.55 * local + 0.30 * persistent + 0.15 * basin_size[index]
    persistence_rank = _percentile(persistence)
    evidence = (
        config.molecular_weight * molecular
        + config.topology_weight * topology
        + config.persistence_weight * persistence_rank
        + config.risk_weight * risk
    ) / max(
        config.molecular_weight
        + config.topology_weight
        + config.persistence_weight
        + config.risk_weight,
        1e-12,
    )

    basin_rows: list[dict[str, object]] = []
    for basin in sorted(set(int(x) for x in basin_ids)):
        members = np.flatnonzero(basin_ids == basin)
        cross_start = len({str(records[i].get("start_id", "")) for i in members})
        cross_arm = len({str(records[i].get("arm", "")) for i in members})
        score = float(
            0.45 * np.median(evidence[members])
            + 0.25 * np.max(evidence[members])
            + 0.15 * min(cross_start / 3.0, 1.0)
            + 0.10 * min(cross_arm / 3.0, 1.0)
            + 0.05 * min(len(members) / 5.0, 1.0)
        )
        basin_rows.append(
            {
                "basin_id": basin,
                "member_count": int(len(members)),
                "cross_start_count": int(cross_start),
                "cross_arm_count": int(cross_arm),
                "score": score,
                "members": [int(x) for x in members],
            }
        )
    winning_basin = max(
        basin_rows,
        key=lambda row: (float(row["score"]), int(row["member_count"]), -int(row["basin_id"])),
    )
    members = np.asarray(winning_basin["members"], dtype=np.int32)
    representative_scores = []
    for index in members:
        peers = members[members != index]
        medoid = float(np.mean(similarity[index, peers])) if len(peers) else 0.0
        representative_scores.append(
            medoid + config.representative_evidence_weight * float(evidence[index])
        )
    selected = int(members[int(np.argmax(representative_scores))])
    diagnostics = {
        "selected_index": selected,
        "selected_partition_sha256": partition_sha256(partitions[selected]),
        "winning_basin_id": int(winning_basin["basin_id"]),
        "candidate_count": m,
        "basin_count": len(basin_rows),
        "dense_observation_by_observation_count": 0,
        "candidate_similarity_shape": list(similarity.shape),
        "producer_label_reads": 0,
    }
    enriched: list[dict[str, object]] = []
    for index, row in enumerate(records):
        enriched.append(
            {
                **row,
                "candidate_index": index,
                "basin_id": int(basin_ids[index]),
                "molecular_rank": float(molecular[index]),
                "topology_rank": float(topology[index]),
                "persistence": float(persistence[index]),
                "persistence_rank": float(persistence_rank[index]),
                "risk_rank": float(risk[index]),
                "evidence_score": float(evidence[index]),
                "selected": index == selected,
            }
        )
    return selected, enriched, diagnostics


def select_evidence_rank(
    records: Sequence[dict[str, object]],
    partitions: np.ndarray,
    config: BasinSelectorConfig,
    similarity: np.ndarray | None = None,
) -> tuple[int, list[dict[str, object]], dict[str, object]]:
    """Select a basin representative by multimodal/topology rank evidence.

    Unlike a plain partition medoid, this ranks candidates with molecular,
    sparse-topology, cluster-risk, and optionally persistence evidence.  A
    fitted configuration may set any axis weight to zero; callers and reports
    must describe only the axes that remain active after fitting.
    """

    config.validate()
    partitions = np.asarray(partitions, dtype=np.int32)
    if len(records) != len(partitions) or len(records) == 0:
        raise ValueError("candidate record/partition mismatch")
    if similarity is None:
        similarity = candidate_similarity(partitions)
    else:
        similarity = np.asarray(similarity, dtype=np.float64)
        if similarity.shape != (len(partitions), len(partitions)):
            raise ValueError("candidate similarity shape mismatch")
    basin_ids = _connected_components(similarity, config.basin_ari_threshold)
    molecular = _percentile(
        np.asarray([float(row["molecular_joint"]) for row in records])
    )
    topology = _percentile(
        np.asarray([float(row["topology_joint"]) for row in records])
    )
    risk = _percentile(
        np.asarray([float(row["microcluster_score"]) for row in records])
    )
    persistence = np.zeros(len(records), dtype=np.float64)
    basin_persistence = np.zeros(len(records), dtype=np.float64)
    ordered_path_persistence = np.zeros(len(records), dtype=np.float64)
    ordered_path_span = np.zeros(len(records), dtype=np.float64)
    local_medoid = np.zeros(len(records), dtype=np.float64)
    for index in range(len(records)):
        peers = np.flatnonzero(basin_ids == basin_ids[index])
        other = peers[peers != index]
        local_medoid[index] = (
            float(np.mean(similarity[index, other])) if len(other) else 0.0
        )
        start_ids = {str(records[i].get("start_id", "")) for i in peers}
        arms = {str(records[i].get("arm", "")) for i in peers}
        path_support = 0.5 * min(len(start_ids) / 3.0, 1.0) + 0.5 * min(
            len(arms) / 3.0, 1.0
        )
        persistent_fraction = float(
            np.mean(similarity[index] >= config.persistent_ari_threshold)
        )
        basin_persistence[index] = (
            0.50 * local_medoid[index]
            + 0.30 * persistent_fraction
            + 0.20 * path_support
        )
    path_groups: dict[str, list[tuple[int, int]]] = {}
    for index, row in enumerate(records):
        path_id = str(row.get("path_id", "")).strip()
        path_index_text = str(row.get("path_index", "")).strip()
        if path_id and path_index_text:
            path_groups.setdefault(path_id, []).append((int(float(path_index_text)), index))
    for members in path_groups.values():
        ordered = [index for _, index in sorted(members)]
        for position, index in enumerate(ordered):
            adjacent = []
            if position > 0:
                adjacent.append(similarity[index, ordered[position - 1]])
            if position + 1 < len(ordered):
                adjacent.append(similarity[index, ordered[position + 1]])
            adjacent_mean = float(np.mean(adjacent)) if adjacent else 0.0
            left = position
            while left > 0 and similarity[ordered[left], ordered[left - 1]] >= config.persistent_ari_threshold:
                left -= 1
            right = position
            while right + 1 < len(ordered) and similarity[ordered[right], ordered[right + 1]] >= config.persistent_ari_threshold:
                right += 1
            span = float((right - left + 1) / len(ordered))
            ordered_path_span[index] = span
            ordered_path_persistence[index] = 0.70 * adjacent_mean + 0.30 * span
    persistence[:] = basin_persistence
    path_mask = ordered_path_persistence > 0
    persistence[path_mask] = (
        (1.0 - float(config.ordered_path_weight)) * basin_persistence[path_mask]
        + float(config.ordered_path_weight) * ordered_path_persistence[path_mask]
    )
    persistence_rank = _percentile(persistence)
    denominator = max(
        config.molecular_weight
        + config.topology_weight
        + config.persistence_weight
        + config.risk_weight,
        1e-12,
    )
    score = (
        config.molecular_weight * molecular
        + config.topology_weight * topology
        + config.persistence_weight * persistence_rank
        + config.risk_weight * risk
    ) / denominator
    order = sorted(
        range(len(records)),
        key=lambda index: (-float(score[index]), str(records[index]["candidate_id"])),
    )
    selected = int(order[0])
    enriched = []
    for index, row in enumerate(records):
        enriched.append(
            {
                **row,
                "candidate_index": index,
                "basin_id": int(basin_ids[index]),
                "molecular_rank": float(molecular[index]),
                "topology_rank": float(topology[index]),
                "persistence": float(persistence[index]),
                "basin_persistence": float(basin_persistence[index]),
                "ordered_path_persistence": float(ordered_path_persistence[index]),
                "ordered_path_span": float(ordered_path_span[index]),
                "persistence_rank": float(persistence_rank[index]),
                "risk_rank": float(risk[index]),
                "local_basin_medoid": float(local_medoid[index]),
                "evidence_score": float(score[index]),
                "selected": index == selected,
            }
        )
    diagnostics = {
        "selected_index": selected,
        "selected_partition_sha256": partition_sha256(partitions[selected]),
        "selected_basin_id": int(basin_ids[selected]),
        "selected_evidence_score": float(score[selected]),
        "candidate_count": len(records),
        "basin_count": len(set(int(value) for value in basin_ids)),
        "ordered_path_candidate_count": int(np.sum(path_mask)),
        "candidate_similarity_shape": list(similarity.shape),
        "dense_observation_by_observation_count": 0,
        "producer_label_reads": 0,
    }
    return selected, enriched, diagnostics
