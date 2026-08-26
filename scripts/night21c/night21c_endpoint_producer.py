"""Label-closed endpoint bank for one locked Night-21C representation."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_graph(archive) -> sp.csr_matrix:
    return sp.csr_matrix(
        (archive["graph0__data"], archive["graph0__indices"], archive["graph0__indptr"]),
        shape=tuple(int(x) for x in archive["graph0__shape"]),
    )


def standardize(value: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(value, dtype=np.float64))


def mutual_knn(value: np.ndarray, neighbors: int) -> sp.csr_matrix:
    x = standardize(value)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    count = min(int(neighbors) + 1, len(x))
    dist, ind = NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute", n_jobs=1).fit(x).kneighbors(x)
    rows = np.repeat(np.arange(len(x)), count)
    cols, distances = ind.ravel(), dist.ravel()
    keep = rows != cols
    rows, cols, distances = rows[keep], cols[keep], distances[keep]
    positive = distances[distances > 1e-10]
    fallback = float(np.median(positive)) if positive.size else 1.0
    local = dist[:, -1].astype(np.float64)
    local[local <= 1e-10] = max(fallback, 1e-6)
    weights = np.exp(-np.square(distances) / np.maximum(local[rows] * local[cols], 1e-12))
    directed = sp.csr_matrix((weights, (rows, cols)), shape=(len(x), len(x)))
    return directed.minimum(directed.T).tocsr()


def support_graph(graph: sp.spmatrix) -> sp.csr_matrix:
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph, dtype=np.float64).T)
    graph.setdiag(0); graph.eliminate_zeros()
    graph.data[:] = 1.0
    return graph


def spectral_partition(operator: sp.csr_matrix, k: int, extra: int, seed: int) -> np.ndarray:
    operator = operator.maximum(operator.T).tocsr() + 0.25 * sp.eye(operator.shape[0], format="csr")
    degree = np.asarray(operator.sum(axis=1)).ravel()
    if np.any(degree <= 0):
        raise RuntimeError("zero degree in sparse spectral head")
    inv = sp.diags(1.0 / np.sqrt(degree))
    normalized = (inv @ operator @ inv).tocsr()
    dim = min(int(k + extra), operator.shape[0] - 2)
    values, vectors = eigsh(normalized, k=dim, which="LA", v0=np.linspace(1.0, 2.0, operator.shape[0]), tol=1e-7, maxiter=10000)
    vectors = vectors[:, np.argsort(values)[::-1]]
    for col in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, col])))
        if vectors[pivot, col] < 0: vectors[:, col] *= -1
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return KMeans(n_clusters=k, n_init=20, random_state=seed, algorithm="lloyd").fit_predict(vectors).astype(np.int32)


def exact_k(value: np.ndarray, k: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.int32)
    if value.shape != (len(value),) or np.unique(value).size != k:
        raise RuntimeError("endpoint violated exact K")
    return value


def neighbor_sets(value: np.ndarray, k: int = 10) -> np.ndarray:
    count = min(k + 1, len(value))
    ind = NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute", n_jobs=1).fit(standardize(value)).kneighbors(return_distance=False)
    return np.asarray([row[row != i][:k] for i, row in enumerate(ind)], dtype=np.int32)


def overlap(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean([len(set(a.tolist()) & set(b.tolist())) / max(len(a), 1) for a, b in zip(left, right)]))


def representation_diagnostics(rep, retained, view1, view2, graph):
    x = standardize(rep)
    singular = np.linalg.svd(x, compute_uv=False)
    prob = np.square(singular) / max(float(np.sum(np.square(singular))), 1e-12)
    effective_rank = float(np.exp(-np.sum(prob[prob > 0] * np.log(prob[prob > 0]))))
    rows, cols = sp.triu(support_graph(graph), k=1).nonzero()
    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    edge_cosine = float(np.mean(np.sum(normalized[rows] * normalized[cols], axis=1))) if len(rows) else float("nan")
    nr, nt, n1, n2 = neighbor_sets(rep), neighbor_sets(retained), neighbor_sets(view1), neighbor_sets(view2)
    return {
        "effective_rank": effective_rank,
        "spatial_edge_cosine_mean": edge_cosine,
        "knn_overlap_retained": overlap(nr, nt),
        "knn_overlap_view1": overlap(nr, n1),
        "knn_overlap_view2": overlap(nr, n2),
        "input_modality_knn_overlap": overlap(n1, n2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--embedding-key", default="representation")
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--representation-source", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    start = time.time(); embedding_path = Path(args.embedding); carrier_path = Path(args.carrier); output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(embedding_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"]); representation = np.asarray(archive[args.embedding_key], dtype=np.float32)
        input_partition = np.asarray(archive["partition"], dtype=np.int32) if "partition" in archive.files else None
    with np.load(carrier_path, allow_pickle=False) as archive:
        carrier_ids = np.asarray(archive["ids"]); retained = np.asarray(archive["retained"], dtype=np.float32)
        view1 = np.asarray(archive["view1"], dtype=np.float32); view2 = np.asarray(archive["view2"], dtype=np.float32); graph = load_graph(archive)
    if not np.array_equal(ids, carrier_ids): raise RuntimeError("ordered ID mismatch")
    if len(representation) != len(ids) or not np.isfinite(representation).all(): raise RuntimeError("invalid representation")
    raw = np.asarray(representation, dtype=np.float32); x = standardize(representation); k = int(args.k); candidates = []
    def add(name, partition): candidates.append((name, exact_k(partition, k)))
    # Project common endpoint and upstream notebook both consume the locked
    # embedding directly. Standardization is reserved for GMM/Ward controls.
    add("COMMON_KMEANS_N20_S0", KMeans(k, n_init=20, random_state=0, algorithm="lloyd").fit_predict(raw))
    add("OFFICIAL_KMEANS_N10_S0", KMeans(k, n_init=10, random_state=0, algorithm="lloyd").fit_predict(raw))
    add("GMM_DIAG_N3_S0", GaussianMixture(k, covariance_type="diag", n_init=3, random_state=0, reg_covar=1e-6).fit_predict(x))
    add("GMM_FULL_N3_S0", GaussianMixture(k, covariance_type="full", n_init=3, random_state=0, reg_covar=1e-6).fit_predict(x))
    spatial = support_graph(graph); feature12 = mutual_knn(x, 12); feature20 = mutual_knn(x, 20)
    try:
        add("SPARSE_WARD_REGISTERED", AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=spatial).fit_predict(x))
    except TypeError:
        add("SPARSE_WARD_REGISTERED", AgglomerativeClustering(n_clusters=k, linkage="ward", connectivity=spatial, affinity="euclidean").fit_predict(x))
    add("SPARSE_GRAPH_FEATURE12_SPATIAL50", spectral_partition(feature12 + spatial, k, 2, 0))
    add("HISTORICAL_CONCAT_KNN_S01", spectral_partition(feature12, k, 2, 0))
    add("HISTORICAL_CONCAT_KNN_S02", spectral_partition(feature20, k, 3, 0))
    if input_partition is not None and not np.array_equal(candidates[0][1], input_partition):
        raise RuntimeError("project common endpoint failed to reproduce the input artifact partition")
    candidate_ids = np.asarray([name for name, _ in candidates], dtype="U64")
    partitions = np.stack([part for _, part in candidates])
    np.savez_compressed(output, ids=ids, candidate_ids=candidate_ids, partitions=partitions)
    with np.load(output, allow_pickle=False) as saved:
        if not np.array_equal(saved["partitions"], partitions) or not np.array_equal(saved["candidate_ids"], candidate_ids): raise RuntimeError("candidate reload mismatch")
    manifest = {
        "schema": "night21c-endpoint-bank-v1", "lane": args.lane, "k": k,
        "representation_source": args.representation_source, "embedding_path": str(embedding_path),
        "embedding_file_sha256": file_sha(embedding_path), "embedding_array_sha256": array_sha(representation),
        "carrier_sha256": file_sha(carrier_path), "ordered_ids_sha256": array_sha(ids),
        "candidate_ids": candidate_ids.tolist(), "candidate_partition_sha256": {name: array_sha(part) for name, part in candidates},
        "input_partition_present": input_partition is not None,
        "input_partition_sha256": array_sha(input_partition) if input_partition is not None else None,
        "common_endpoint_matches_input_partition": bool(input_partition is not None and np.array_equal(candidates[0][1], input_partition)),
        "candidate_bank_sha256": file_sha(output), "labels_read": 0,
        "representation_diagnostics": representation_diagnostics(representation, retained, view1, view2, graph),
        "wall_seconds": time.time() - start, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__": main()
