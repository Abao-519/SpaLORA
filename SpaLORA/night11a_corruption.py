"""Frozen sparse-patch corruptions for Night-11A."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from typing import Sequence, Tuple

import numpy as np
from scipy import sparse


def _hash(parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def sparse_patch(graph: sparse.spmatrix, observation_ids: Sequence[str],
                 unit_id: str, direction: str, condition: str,
                 replicate_seed: int, fraction: float = 0.20) -> np.ndarray:
    graph = graph.tocsr()
    n = graph.shape[0]
    if graph.shape != (n, n) or len(observation_ids) != n:
        raise ValueError("sparse observation graph contract mismatch")
    anchor = _hash((unit_id, direction, condition, replicate_seed, "anchor")) % n
    target = int(math.ceil(fraction * n))
    seen = np.zeros(n, dtype=bool); seen[anchor] = True
    queue = deque([anchor]); selected = []
    ids = np.asarray(observation_ids, dtype=str)
    while queue and len(selected) < target:
        i = queue.popleft(); selected.append(i)
        nb = graph.indices[graph.indptr[i]:graph.indptr[i + 1]]
        for j in nb[np.argsort(ids[nb], kind="mergesort")]:
            if not seen[j]: seen[j] = True; queue.append(int(j))
    if len(selected) < target:
        remaining = np.flatnonzero(~seen)
        for i in remaining[np.argsort(ids[remaining], kind="mergesort")]:
            if len(selected) == target: break
            seen[i] = True; queue.append(int(i))
            while queue and len(selected) < target:
                j = queue.popleft(); selected.append(j)
                nb = graph.indices[graph.indptr[j]:graph.indptr[j + 1]]
                for k in nb[np.argsort(ids[nb], kind="mergesort")]:
                    if not seen[k]: seen[k] = True; queue.append(int(k))
    return np.asarray(selected, dtype=np.int64)


def conflict_permutation(graph: sparse.spmatrix, observation_ids: Sequence[str],
                         patch: np.ndarray, unit_id: str, direction: str,
                         condition: str, replicate_seed: int) -> np.ndarray:
    graph = graph.tocsr(); patch = np.asarray(patch, dtype=np.int64)
    ids = np.asarray(observation_ids, dtype=str)
    keyed = sorted(((_hash((unit_id, direction, condition, replicate_seed,
                            ids[int(i)], "permutation-order")), ids[int(i)], int(i))
                    for i in patch))
    order = np.asarray([i for _, _, i in keyed], dtype=np.int64)
    m = len(order)
    if m < 3: raise ValueError("patch too small")
    start = 1 + _hash((unit_id, direction, condition, replicate_seed, "shift")) % (m - 1)
    def invalid(a, b):
        return a == b or graph[a, b] != 0
    for delta in range(m - 1):
        shift = 1 + (start - 1 + delta) % (m - 1)
        mapped = np.roll(order, shift).copy()
        # A raw cyclic shift preserves marginals and is almost a derangement,
        # but a spatial patch can contain many direct-neighbour pairs. Repair
        # those pairs by deterministic destination swaps while preserving the
        # same one-to-one permutation.
        for i in range(m):
            if not invalid(int(order[i]), int(mapped[i])):
                continue
            repaired = False
            for offset in range(1, m):
                j = (i + offset) % m
                if (not invalid(int(order[i]), int(mapped[j])) and
                        not invalid(int(order[j]), int(mapped[i]))):
                    mapped[i], mapped[j] = mapped[j], mapped[i]
                    repaired = True
                    break
            if not repaired:
                break
        if all(not invalid(int(a), int(b)) for a, b in zip(order, mapped)):
            lookup = {int(a): int(b) for a, b in zip(order, mapped)}
            return np.asarray([lookup[int(a)] for a in patch], dtype=np.int64)
    raise RuntimeError("no registered fixed-point-free non-neighbor permutation")


def patch_and_permutation(graph, observation_ids, unit_id, direction,
                          condition, replicate_seed) -> Tuple[np.ndarray, np.ndarray]:
    patch = sparse_patch(graph, observation_ids, unit_id, direction,
                         condition, replicate_seed)
    perm = conflict_permutation(graph, observation_ids, patch, unit_id,
                                direction, condition, replicate_seed)
    return patch, perm
