"""Night-18D external transfer helpers around the frozen Night-15F energy."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import ExpansionEnergyConfig, PreparedExpansionEvidence, continuous_multiscale_expansion, continuous_multiscale_single_site


PRIMARY_ARMS = (
    "NO_OP_START", "L2_LOWPASS_MATCHED", "FULL_FROZEN_ENERGY", "REGISTERED_SCALE_ONLY",
    "NO_SELF_RETURN_STAY", "PAIRWISE_ZERO_KEEP_STAY", "PURE_DYNAMIC_UNARY", "SINGLE_SITE_SAME_ENERGY",
)


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value); digest = hashlib.sha256(); digest.update(value.dtype.str.encode()); digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes()); return digest.hexdigest()


def config_from_dict(raw: dict) -> ExpansionEnergyConfig:
    return ExpansionEnergyConfig(**{**raw, "local": ContinuousEnergyConfig(**raw["local"])})


def transition(graph: sp.csr_matrix) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph, dtype=np.float64).T).tocsr(); graph.setdiag(0); graph.eliminate_zeros()
    degree = np.asarray(graph.sum(1)).ravel(); inverse = np.zeros_like(degree); inverse[degree > 0] = 1.0 / degree[degree > 0]
    return (sp.diags(inverse) @ graph).tocsr()


def l2_lowpass_partition(retained: np.ndarray, graph: sp.csr_matrix, initial: np.ndarray, k: int, cycles: int) -> np.ndarray:
    value = np.asarray(retained, dtype=np.float64).copy(); operator = transition(graph)
    for _ in range(max(1, int(cycles))): value = 0.65 * value + 0.35 * np.asarray(operator @ value)
    prototypes = np.stack([value[np.asarray(initial) == group].mean(0) for group in range(k)])
    partition = KMeans(k, init=prototypes, n_init=1, random_state=0, max_iter=200).fit_predict(value)
    _, partition = np.unique(partition, return_inverse=True); partition = partition.astype(np.int32)
    if len(np.unique(partition)) != k: raise RuntimeError("L2 endpoint violated exact K")
    return partition


def arm_config(config: ExpansionEnergyConfig, arm: str) -> ExpansionEnergyConfig:
    if arm in {"NO_OP_START", "L2_LOWPASS_MATCHED", "FULL_FROZEN_ENERGY", "SINGLE_SITE_SAME_ENERGY"}: return config
    if arm == "REGISTERED_SCALE_ONLY": return replace(config, scale_fine=0.0, scale_registered=1.0, scale_broad=0.0)
    if arm == "NO_SELF_RETURN_STAY": return replace(config, self_return_strength=0.0)
    if arm == "PAIRWISE_ZERO_KEEP_STAY": return replace(config, pairwise_beta=0.0)
    if arm == "PURE_DYNAMIC_UNARY": return replace(config, pairwise_beta=0.0, self_return_strength=0.0)
    raise ValueError(arm)


def run_arm(initial: np.ndarray, k: int, evidence: PreparedExpansionEvidence, retained: np.ndarray, registered_graph: sp.csr_matrix,
            config: ExpansionEnergyConfig, arm: str) -> tuple[np.ndarray, dict[str, object]]:
    initial = np.asarray(initial, dtype=np.int32)
    if arm == "NO_OP_START": return initial.copy(), {"solver": "NO_OP", "changed_observations": 0.0}
    if arm == "L2_LOWPASS_MATCHED":
        partition = l2_lowpass_partition(retained, registered_graph, initial, k, config.expansion_cycles)
        return partition, {"solver": "L2_LOWPASS_PROTOTYPE_KMEANS", "changed_observations": float(np.sum(partition != initial))}
    selected = arm_config(config, arm)
    if arm == "SINGLE_SITE_SAME_ENERGY": return continuous_multiscale_single_site(initial, k, evidence, selected)
    return continuous_multiscale_expansion(initial, k, evidence, selected)


def structure_feasibility(partition: np.ndarray, fine_graph: sp.csr_matrix, k: int) -> dict[str, object]:
    partition = np.asarray(partition, dtype=np.int32); sizes = np.bincount(partition, minlength=k)
    if len(np.unique(partition)) != k or len(sizes) != k or np.any(sizes == 0): return {"feasible": False, "exact_k": False, "min_cluster_size": 0, "min_internal_edges": 0}
    upper = sp.triu(sp.csr_matrix(fine_graph).maximum(sp.csr_matrix(fine_graph).T), k=1, format="coo")
    same = partition[upper.row] == partition[upper.col]; internal = np.bincount(partition[upper.row[same]], minlength=k)
    feasible = bool(sizes.min() >= 2 and internal.min() >= 1)
    return {"feasible": feasible, "exact_k": True, "min_cluster_size": int(sizes.min()), "cluster_sizes": [int(x) for x in sizes], "min_internal_edges": int(internal.min())}


def medoid_index(partitions: np.ndarray, eligible: Sequence[int] | None = None) -> int:
    indices = list(range(len(partitions))) if eligible is None else list(eligible)
    if not indices: raise RuntimeError("no eligible partition for medoid")
    score = np.zeros(len(indices), dtype=np.float64)
    for left in range(len(indices)):
        for right in range(left):
            value = adjusted_rand_score(partitions[indices[left]], partitions[indices[right]])
            score[left] += value; score[right] += value
    best = max(range(len(indices)), key=lambda offset: (score[offset], -indices[offset]))
    return int(indices[best])
