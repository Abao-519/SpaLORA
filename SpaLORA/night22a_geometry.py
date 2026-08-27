"""Sparse geometry endpoints for the Night-22A label-closed ceiling study."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def standardize(value: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(value, dtype=np.float64))


def undirected_no_diag(graph: sp.spmatrix) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float64)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.nnz == 0 or not np.isfinite(graph.data).all() or np.any(graph.data < 0):
        raise ValueError("invalid sparse affinity")
    return graph


def mass_normalize(graph: sp.spmatrix) -> sp.csr_matrix:
    graph = undirected_no_diag(graph)
    mass = float(graph.sum())
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("non-positive graph mass")
    return (graph / mass).tocsr()


def self_tuning_knn(value: np.ndarray, neighbors: int) -> sp.csr_matrix:
    x = standardize(value)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    count = min(int(neighbors) + 1, len(x))
    distances, indices = NearestNeighbors(
        n_neighbors=count, metric="cosine", algorithm="brute", n_jobs=1
    ).fit(x).kneighbors(x)
    local = distances[:, -1].astype(np.float64)
    positive = local[local > 1e-12]
    fallback = float(np.median(positive)) if positive.size else 1.0
    local[local <= 1e-12] = max(fallback, 1e-6)
    rows = np.repeat(np.arange(len(x)), count)
    cols = indices.ravel()
    dist = distances.ravel()
    keep = rows != cols
    rows, cols, dist = rows[keep], cols[keep], dist[keep]
    weights = np.exp(-np.square(dist) / np.maximum(local[rows] * local[cols], 1e-12))
    directed = sp.csr_matrix((weights, (rows, cols)), shape=(len(x), len(x)))
    # A union graph is used so a locally asymmetric neighbour choice does not
    # create an artificial isolated node. The local scales remain node-wise.
    return undirected_no_diag(directed.maximum(directed.T))


def combine_graphs(feature: sp.spmatrix, spatial: sp.spmatrix, spatial_ratio: float) -> sp.csr_matrix:
    if spatial_ratio < 0:
        raise ValueError("spatial_ratio must be non-negative")
    feature = mass_normalize(feature)
    spatial = mass_normalize(spatial)
    return undirected_no_diag(feature + float(spatial_ratio) * spatial)


def density_correct(graph: sp.spmatrix) -> sp.csr_matrix:
    graph = undirected_no_diag(graph)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inv = sp.diags(1.0 / np.maximum(degree, 1e-12))
    return undirected_no_diag(inv @ graph @ inv)


def normalized_eigensystem(graph: sp.spmatrix, dimensions: int) -> Tuple[np.ndarray, np.ndarray]:
    graph = undirected_no_diag(graph)
    graph = graph + 0.25 * sp.eye(graph.shape[0], format="csr")
    degree = np.asarray(graph.sum(axis=1)).ravel()
    if np.any(degree <= 0):
        raise ValueError("zero degree in spectral operator")
    inv_sqrt = sp.diags(1.0 / np.sqrt(degree))
    operator = (inv_sqrt @ graph @ inv_sqrt).tocsr()
    dimensions = min(int(dimensions), operator.shape[0] - 2)
    values, vectors = eigsh(
        operator,
        k=dimensions,
        which="LA",
        v0=np.linspace(1.0, 2.0, operator.shape[0]),
        tol=1e-7,
        maxiter=20000,
    )
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    return values, vectors


def spectral_partition(graph: sp.spmatrix, k: int, extra: int, seed: int = 0) -> np.ndarray:
    _, vectors = normalized_eigensystem(graph, k + extra)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return KMeans(k, n_init=20, random_state=seed, algorithm="lloyd").fit_predict(vectors).astype(np.int32)


def diffusion_partition(graph: sp.spmatrix, k: int, time_power: int, seed: int = 0) -> np.ndarray:
    values, vectors = normalized_eigensystem(graph, k + 7)
    # Drop the Perron vector. Eigenvalue powers expose whether a stable
    # diffusion plateau, rather than a raw spectral cut, explains recovery.
    coordinates = vectors[:, 1:] * np.power(np.clip(values[1:], 0.0, 1.0), int(time_power))[None, :]
    coordinates = StandardScaler().fit_transform(coordinates)
    return KMeans(k, n_init=20, random_state=seed, algorithm="lloyd").fit_predict(coordinates).astype(np.int32)


def leiden_exact_k(graph: sp.spmatrix, k: int, seed: int = 0) -> Tuple[np.ndarray | None, float | None]:
    import igraph as ig
    import leidenalg

    upper = sp.triu(undirected_no_diag(graph), k=1).tocoo()
    edges = list(zip(upper.row.tolist(), upper.col.tolist()))
    base = ig.Graph(n=graph.shape[0], edges=edges, directed=False)
    base.es["weight"] = upper.data.astype(float).tolist()
    resolutions = np.geomspace(0.03, 8.0, 32)
    matches: List[Tuple[float, np.ndarray, float]] = []
    for resolution in resolutions:
        partition = leidenalg.find_partition(
            base,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(resolution),
            seed=int(seed),
            n_iterations=-1,
        )
        membership = np.asarray(partition.membership, dtype=np.int32)
        if np.unique(membership).size == int(k):
            matches.append((float(partition.quality()), membership, float(resolution)))
    if not matches:
        return None, None
    _, membership, resolution = max(matches, key=lambda item: (item[0], -item[2]))
    return membership, resolution


def exact_k(partition: np.ndarray, k: int) -> np.ndarray:
    partition = np.asarray(partition, dtype=np.int32)
    if partition.ndim != 1 or np.unique(partition).size != int(k):
        raise ValueError("partition does not have exact K")
    return partition


@dataclass
class GeometryBank:
    candidate_ids: np.ndarray
    partitions: np.ndarray
    diagnostics: Dict[str, object]


def generate_geometry_bank(representation: np.ndarray, spatial_graph: sp.spmatrix, k: int) -> GeometryBank:
    x = standardize(representation)
    spatial = undirected_no_diag(spatial_graph)
    feature12 = self_tuning_knn(x, 12)
    feature24 = self_tuning_knn(x, 24)
    joint025 = combine_graphs(feature12, spatial, 0.25)
    joint050 = combine_graphs(feature12, spatial, 0.50)
    joint100 = combine_graphs(feature12, spatial, 1.00)
    corrected = density_correct(feature12)

    rows: List[Tuple[str, np.ndarray]] = []
    rows.append(("GEOM_KMEANS_RAW", KMeans(k, n_init=20, random_state=0, algorithm="lloyd").fit_predict(representation)))
    rows.append(("GEOM_GMM_FULL", GaussianMixture(k, covariance_type="full", n_init=3, random_state=0, reg_covar=1e-6).fit_predict(x)))
    rows.append(("GEOM_FEATURE_NCUT_K12", spectral_partition(feature12, k, 3)))
    rows.append(("GEOM_FEATURE_NCUT_K24", spectral_partition(feature24, k, 3)))
    rows.append(("GEOM_JOINT_NCUT_S025", spectral_partition(joint025, k, 3)))
    rows.append(("GEOM_JOINT_NCUT_S100", spectral_partition(joint100, k, 3)))
    rows.append(("GEOM_DENSITY_CORRECTED_NCUT", spectral_partition(corrected, k, 3)))
    rows.append(("GEOM_DIFFUSION_T2", diffusion_partition(joint050, k, 2)))
    rows.append(("GEOM_DIFFUSION_T4", diffusion_partition(joint050, k, 4)))
    rows.append(("GEOM_SPATIAL_NCUT", spectral_partition(spatial, k, 3)))

    leiden_meta: Dict[str, object] = {}
    for name, graph in (("GEOM_LEIDEN_FEATURE", feature12), ("GEOM_LEIDEN_JOINT", joint050)):
        partition, resolution = leiden_exact_k(graph, k)
        if partition is not None:
            rows.append((name, partition))
            leiden_meta[name] = {"exact_k_found": True, "resolution": resolution}
        else:
            leiden_meta[name] = {"exact_k_found": False, "resolution": None}

    candidate_ids: List[str] = []
    partitions: List[np.ndarray] = []
    seen: Dict[str, str] = {}
    aliases: Dict[str, str] = {}
    for name, partition in rows:
        partition = exact_k(partition, k)
        digest = array_sha(partition)
        if digest in seen:
            aliases[name] = seen[digest]
            continue
        seen[digest] = name
        candidate_ids.append(name)
        partitions.append(partition)

    diagnostics = {
        "feature12_edges": int(sp.triu(feature12, k=1).nnz),
        "feature24_edges": int(sp.triu(feature24, k=1).nnz),
        "spatial_edges": int(sp.triu(spatial, k=1).nnz),
        "candidate_aliases": aliases,
        "leiden": leiden_meta,
        "candidate_count_before_deduplication": len(rows),
        "candidate_count_after_deduplication": len(candidate_ids),
    }
    return GeometryBank(
        candidate_ids=np.asarray(candidate_ids, dtype="U64"),
        partitions=np.stack(partitions).astype(np.int32),
        diagnostics=diagnostics,
    )
