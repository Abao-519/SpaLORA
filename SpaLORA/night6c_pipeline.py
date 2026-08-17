"""Night-6C clean graph and clustering rescue pipeline.

The module is deliberately label-free.  Evaluation lives in a separate script.
All graph and head parameters are resolved from the SHA-locked registry.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from .night3a_ige import model_state_sha256
from .night3af_cache import load_cache, sha256_file
from .night5a_rnd import Night5ATrainer, _Forward


BASE_C04 = {
    "id": "C04_SHRINK25",
    "family": "fusion",
    "description": "Learned fusion shrunk strongly toward equal weights",
    "attention": "shrink_to_uniform",
    "learned_fraction_rho": 0.25,
    "corr2": False,
    "loss_calibration": "active_set_IGE",
}

DATASET_CFG = {
    "a1": {
        "n_clusters": 10, "embedding_dim": 64, "epochs": 200,
        "loss_factors": [1.9, 2.5, 1.5, 10.0],
        "locked_m_bad_expected": 2.289938091,
    },
    "tonsil": {
        "n_clusters": 4, "embedding_dim": 64, "epochs": 200,
        "loss_factors": [1.9, 2.5, 1.5, 10.0],
        "locked_m_bad_expected": 2.289938091,
    },
}

VIEW_KEYS = (
    "emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused",
    "alpha_omics1", "alpha_omics2", "alpha_cross",
)


def canonical_json_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        fh.write("\n"); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    with tmp.open("rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def observation_sha(ids: Sequence[str]) -> str:
    return sha_bytes("\n".join(map(str, ids)).encode("utf-8"))


def sparse_sha(matrix: sp.spmatrix) -> str:
    value = matrix.tocsr().astype(np.float64)
    value.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.indptr.astype(np.int64).tobytes())
    digest.update(value.indices.astype(np.int64).tobytes())
    digest.update(value.data.tobytes())
    return digest.hexdigest()


def parse_registry(registry: Mapping[str, object]) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    graphs = {row["id"]: dict(row) for row in registry["graph_candidates"]}
    heads = {row["id"]: dict(row) for row in registry["head_candidates"]}
    expected_g = [f"G{i:02d}_" for i in range(9)]
    expected_h = [f"H{i:02d}_" for i in range(12)]
    if len(graphs) != 9 or not all(any(k.startswith(p) for k in graphs) for p in expected_g):
        raise RuntimeError("registry graph contract is not 9/9")
    if len(heads) != 12 or not all(any(k.startswith(p) for k in heads) for p in expected_h):
        raise RuntimeError("registry head contract is not 12/12")
    hashes = [canonical_json_sha(x) for x in graphs.values()]
    if len(set(hashes)) != 9:
        raise RuntimeError("graph config SHA collision")
    hashes = [canonical_json_sha(x) for x in heads.values()]
    if len(set(hashes)) != 12:
        raise RuntimeError("head config SHA collision")
    return graphs, heads


def _neighbors(values: np.ndarray, k: int, metric: str, ids: Sequence[str]) -> np.ndarray:
    """Deterministic non-self kNN, with lexical observation-id tie breaking."""
    x = np.asarray(values, dtype=np.float64)
    n = len(x)
    if n <= k:
        raise ValueError("k must be smaller than observation count")
    if metric not in {"euclidean", "correlation"}:
        raise ValueError(f"unsupported metric: {metric}")
    probe = min(n, max(k + 1, k + 65))
    nn = NearestNeighbors(n_neighbors=probe, metric=metric, algorithm="brute").fit(x)
    distances, indices = nn.kneighbors(x, return_distance=True)
    names = np.asarray(ids, dtype=str)
    out = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        pairs = [(float(d), str(names[j]), int(j)) for d, j in zip(distances[i], indices[i]) if int(j) != i]
        pairs.sort(key=lambda z: (z[0], z[1]))
        if len(pairs) < k:
            raise RuntimeError("insufficient non-self neighbors")
        # If the queried boundary is tied with the last returned point, fail closed
        # rather than silently using backend ordering.
        if probe < n and len(pairs) > k and math.isclose(pairs[k-1][0], pairs[-1][0], rel_tol=0.0, abs_tol=1e-14):
            all_dist = pairwise_distances(x[i:i+1], x, metric=metric).ravel()
            pairs = [(float(all_dist[j]), str(names[j]), int(j)) for j in range(n) if j != i]
            pairs.sort(key=lambda z: (z[0], z[1]))
        out[i] = [z[2] for z in pairs[:k]]
    return out


def binary_knn(values: np.ndarray, k: int, metric: str, ids: Sequence[str], sym: str) -> sp.csr_matrix:
    idx = _neighbors(values, int(k), metric, ids)
    n = len(idx)
    directed = sp.coo_matrix((np.ones(n * int(k), dtype=np.float64),
                              (np.repeat(np.arange(n), int(k)), idx.reshape(-1))),
                             shape=(n, n)).tocsr()
    if sym == "union":
        graph = directed.maximum(directed.T)
    elif sym == "mutual":
        graph = directed.multiply(directed.T)
    else:
        raise ValueError(f"unsupported symmetrization: {sym}")
    graph.setdiag(0); graph.eliminate_zeros(); graph.data[:] = 1.0
    return graph


def normalize_support(graph: sp.spmatrix) -> sp.csr_matrix:
    graph = graph.tocsr().astype(np.float64)
    graph.setdiag(0); graph.eliminate_zeros(); graph.data[:] = 1.0
    raw = graph + sp.eye(graph.shape[0], format="csr")
    degree = np.asarray(raw.sum(axis=1)).ravel()
    inv = np.power(np.maximum(degree, 1e-12), -0.5)
    return (sp.diags(inv) @ raw @ sp.diags(inv)).tocsr()


def moran_scores(values: np.ndarray, graph: sp.spmatrix) -> np.ndarray:
    """Per-feature Moran I using the exact candidate RNA spatial support."""
    x = np.asarray(values, dtype=np.float64)
    w = graph.tocsr().astype(np.float64)
    w.setdiag(0); w.eliminate_zeros()
    s0 = float(w.sum())
    if s0 <= 0:
        return np.full(x.shape[1], np.nan, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    denominator = np.sum(centered * centered, axis=0)
    numerator = np.sum(centered * w.dot(centered), axis=0)
    return np.divide(len(x) * numerator, s0 * denominator,
                     out=np.zeros_like(numerator), where=denominator > 1e-12)


def scipy_to_torch(matrix: sp.spmatrix) -> torch.Tensor:
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(
        torch.as_tensor(np.vstack((coo.row, coo.col)), dtype=torch.long),
        torch.as_tensor(coo.data, dtype=torch.float32), size=coo.shape,
    ).coalesce()


def torch_support(value: torch.Tensor) -> sp.csr_matrix:
    value = value.coalesce().cpu()
    rows, cols = value.indices().numpy()
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=tuple(value.shape)).tocsr()
    graph.setdiag(0); graph.eliminate_zeros(); graph.data[:] = 1.0
    return graph.maximum(graph.T).tocsr()


def build_graph_data(prepared, candidate: Mapping[str, object], cache_dir: Path,
                     base_manifest_sha: str) -> Tuple[dict, dict]:
    ids = prepared.obs_names.astype(str).to_numpy()
    coords = np.asarray(prepared.coordinates, dtype=np.float64)
    f1 = np.asarray(prepared.data["rna_pca_scores"], dtype=np.float64)
    f2 = np.asarray(prepared.data["features_omics2"], dtype=np.float64)
    spatial = binary_knn(coords, int(candidate["spatial_k"]), "euclidean", ids,
                         str(candidate["spatial_symmetrization"]))
    feature1 = binary_knn(f1, int(candidate["feature_k"]), str(candidate["feature_metric"]), ids, "union")
    feature2 = binary_knn(f2, int(candidate["feature_k"]), str(candidate["feature_metric"]), ids, "union")
    refinement = str(candidate["spatial_refinement"])
    if refinement == "per_modality_support_intersection_with_its_feature_graph":
        spatial1 = spatial.multiply(feature1).tocsr()
        spatial2 = spatial.multiply(feature2).tocsr()
        spatial1.eliminate_zeros(); spatial2.eliminate_zeros()
    elif refinement == "none":
        spatial1 = spatial.copy(); spatial2 = spatial.copy()
    else:
        raise ValueError(f"unsupported spatial refinement: {refinement}")
    graphs = {
        "adj_spatial_omics1": spatial1, "adj_spatial_omics2": spatial2,
        "adj_feature_omics1": feature1, "adj_feature_omics2": feature2,
    }
    data = dict(prepared.data)
    for key, graph in graphs.items():
        data[key] = scipy_to_torch(normalize_support(graph))
    cache_dir.mkdir(parents=True, exist_ok=False)
    graph_rows = {}
    for key, graph in graphs.items():
        target = cache_dir / f"{key}_support.npz"
        sp.save_npz(target, graph.astype(np.uint8), compressed=True)
        isolated = int(np.sum(np.asarray(graph.sum(axis=1)).ravel() == 0))
        components = int(connected_components(graph, directed=False, return_labels=False))
        graph_rows[key] = {
            "support_sha256": sparse_sha(graph), "file": target.name,
            "file_sha256": sha256_file(target), "size_bytes": target.stat().st_size,
            "nnz": int(graph.nnz), "undirected_edges": int(graph.nnz // 2),
            "isolated_nodes_before_normalization": isolated,
            "connected_components_before_normalization": components,
            "normalization_self_loop_only_for_isolates": True,
        }
    asr_scores = moran_scores(np.asarray(prepared.data["features_omics1"]), spatial1)
    asr_path = cache_dir / "asr_moran_scores_selected_genes.npy"
    np.save(asr_path, asr_scores, allow_pickle=False)
    array_rows = {}
    for key in ("features_omics1", "features_omics2", "weight_vector_omics1",
                "rna_pca_scores", "coordinates"):
        value = prepared.coordinates if key == "coordinates" else prepared.data[key]
        value = np.asarray(value)
        array_rows[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                           "canonical_array_sha256": array_sha(value)}
    manifest = {
        "schema_version": 1, "candidate": dict(candidate),
        "candidate_config_sha256": canonical_json_sha(candidate),
        "base_cache_manifest_sha256": base_manifest_sha,
        "canonical_observation_sha256": observation_sha(ids),
        "arrays": array_rows, "graphs": graph_rows,
        "asr_moran_spatial_rule": candidate["id"],
        "asr_moran_scoring_executed": True,
        "asr_moran_spatial_support_sha256": graph_rows["adj_spatial_omics1"]["support_sha256"],
        "asr_moran_scores": {"file": asr_path.name, "file_sha256": sha256_file(asr_path),
                             "size_bytes": asr_path.stat().st_size,
                             "shape": list(asr_scores.shape), "dtype": str(asr_scores.dtype),
                             "canonical_array_sha256": array_sha(asr_scores)},
        "asr_training_effect": "diagnostic_only_under_corrected_unweighted_HVG_only_C04; selection_and_weights_remain_fixed",
        "silent_fallback": False,
    }
    manifest["canonical_graph_cache_sha256"] = canonical_json_sha(manifest)
    atomic_json(cache_dir / "manifest.json", manifest)
    return data, manifest


def load_graph_data(prepared, cache_dir: Path) -> Tuple[dict, dict]:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed = dict(manifest); expected = observed.pop("canonical_graph_cache_sha256")
    if canonical_json_sha(observed) != expected:
        raise RuntimeError("graph-cache canonical hash mismatch")
    data = dict(prepared.data)
    for key, row in manifest["graphs"].items():
        path = cache_dir / row["file"]
        if sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"graph-cache file mismatch: {path}")
        graph = sp.load_npz(path).tocsr()
        if sparse_sha(graph) != row["support_sha256"]:
            raise RuntimeError(f"graph support mismatch: {key}")
        data[key] = scipy_to_torch(normalize_support(graph))
    return data, manifest


def normalized_views(result: Mapping[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    norm = torch.nn.functional.normalize
    return {
        "emb_latent_omics1": norm(result["emb_latent_omics1"], p=2, dim=1, eps=1e-12).detach().cpu().numpy(),
        "emb_latent_omics2": norm(result["emb_latent_omics2"], p=2, dim=1, eps=1e-12).detach().cpu().numpy(),
        "SpaLORA_fused": norm(result["emb_latent_combined"], p=2, dim=1, eps=1e-12).detach().cpu().numpy(),
        "alpha_omics1": result["alpha_omics1"].detach().cpu().numpy(),
        "alpha_omics2": result["alpha_omics2"].detach().cpu().numpy(),
        "alpha_cross": result["alpha"].detach().cpu().numpy(),
    }


def deterministic_pca(values: np.ndarray, n_components: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = min(int(n_components), values.shape[0] - 1, values.shape[1])
    return PCA(n_components=n, svd_solver="full").fit_transform(values)


def mclust(values: np.ndarray, k: int, model_names="EEE", seed: int = 2020) -> dict:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    ro.r.library("mclust")
    ro.r["set.seed"](int(seed))
    converted = numpy2ri.py2rpy(np.asarray(values, dtype=np.float64))
    result = ro.r["Mclust"](converted, int(k), modelNames=model_names)
    labels = np.asarray(result.rx2("classification"), dtype=np.int64)
    posterior = np.asarray(result.rx2("z"), dtype=np.float64)
    model = str(result.rx2("modelName")[0])
    bic = np.asarray(result.rx2("bic"), dtype=np.float64)
    return {"labels": labels, "posterior": posterior, "selected_model": model,
            "bic_max": float(np.nanmax(bic))}


def h00(values: np.ndarray, k: int) -> dict:
    x = row_normalize(values)
    return mclust(deterministic_pca(x, 20), k, "EEE", 2020)


def row_normalize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _neighbor_sets(values: np.ndarray, k: int = 10, ids: Sequence[str] | None = None) -> np.ndarray:
    if ids is None:
        ids = np.asarray([f"{i:012d}" for i in range(len(values))])
    return _neighbors(row_normalize(values), k, "euclidean", ids)


def reliability_weights(views: Sequence[np.ndarray], k: int = 10,
                        ids: Sequence[str] | None = None) -> np.ndarray:
    neighbors = [_neighbor_sets(v, k, ids) for v in views]
    scores = []
    for i in range(len(views)):
        pair = []
        for j in range(len(views)):
            if i == j: continue
            vals = []
            for a, b in zip(neighbors[i], neighbors[j]):
                aa, bb = set(map(int, a)), set(map(int, b))
                vals.append(len(aa & bb) / max(1, len(aa | bb)))
            pair.append(float(np.mean(vals)))
        scores.append(float(np.mean(pair)))
    scores = np.asarray(scores, dtype=np.float64)
    return scores / max(float(scores.sum()), 1e-12)


def self_tuning_affinity(values: np.ndarray, k: int = 10,
                         ids: Sequence[str] | None = None) -> sp.csr_matrix:
    x = row_normalize(values)
    n = len(x)
    if ids is None:
        ids = np.asarray([f"{i:012d}" for i in range(n)])
    idx = _neighbors(x, k, "euclidean", ids)
    rows = np.repeat(np.arange(n), k); cols = idx.reshape(-1)
    distances = np.linalg.norm(x[rows] - x[cols], axis=1).reshape(n, k)
    sigma = np.maximum(distances[:, -1], 1e-12)
    weights = np.exp(-(distances.reshape(-1) ** 2) /
                     np.maximum(sigma[rows] * sigma[cols], 1e-12))
    graph = sp.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    graph = graph.maximum(graph.T); graph.setdiag(0); graph.eliminate_zeros()
    return graph


def spectral(affinity: sp.spmatrix, k: int) -> np.ndarray:
    affinity = affinity.tocsr().astype(np.float64)
    affinity = ((affinity + affinity.T) * 0.5).tocsr()
    return SpectralClustering(n_clusters=int(k), affinity="precomputed",
                              assign_labels="discretize", n_init=20,
                              random_state=2020).fit_predict(affinity).astype(np.int64) + 1


def spatial_k6(coords: np.ndarray, ids: Sequence[str]) -> sp.csr_matrix:
    return binary_knn(np.asarray(coords), 6, "euclidean", ids, "union")


def row_stochastic(graph: sp.spmatrix) -> sp.csr_matrix:
    graph = graph.tocsr().astype(np.float64)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(degree, 1.0)) @ graph


def _agreement_affinity(affinities: Sequence[sp.csr_matrix], mean: sp.csr_matrix,
                        ids: Sequence[str]) -> sp.csr_matrix:
    supports = [(x.copy().astype(bool)).astype(np.int8) for x in affinities]
    count = supports[0] + supports[1] + supports[2]
    result = mean.multiply(count >= 2).tocsr(); result.eliminate_zeros()
    degree = np.asarray(result.sum(axis=1)).ravel()
    names = np.asarray(ids, dtype=str)
    mean = mean.tocsr()
    for i in np.flatnonzero(degree == 0):
        row = mean.getrow(i)
        if row.nnz == 0: continue
        maximum = float(row.data.max())
        choices = row.indices[np.isclose(row.data, maximum, atol=1e-15, rtol=0)]
        j = int(sorted(choices, key=lambda z: names[z])[0])
        result[i, j] = maximum; result[j, i] = maximum
    result.eliminate_zeros()
    return result


def _icm(posterior: np.ndarray, initial: np.ndarray, spatial: sp.csr_matrix,
         ids: Sequence[str], beta: float) -> np.ndarray:
    unary = -np.log(np.maximum(np.asarray(posterior, dtype=np.float64), 1e-300))
    labels = np.asarray(initial, dtype=np.int64).copy()
    classes = np.arange(1, posterior.shape[1] + 1, dtype=np.int64)
    order = np.argsort(np.asarray(ids, dtype=str), kind="mergesort")
    graph = spatial.tocsr()
    for _ in range(20):
        changed = False
        for i in order:
            nb = graph.indices[graph.indptr[i]:graph.indptr[i+1]]
            energy = unary[i] + float(beta) * np.asarray([(labels[nb] != c).sum() for c in classes])
            best = np.flatnonzero(np.isclose(energy, energy.min(), atol=1e-15, rtol=0))
            original_pos = int(labels[i] - 1)
            choice = original_pos if original_pos in best else int(best.min())
            new = int(classes[choice])
            changed |= new != labels[i]; labels[i] = new
        if not changed: break
    return labels


def run_head(head: Mapping[str, object], views: Mapping[str, np.ndarray],
             k: int, coords: np.ndarray, ids: Sequence[str],
             artifact_dir: Path | None = None) -> Tuple[np.ndarray, dict]:
    forbidden = {"labels", "ground_truth", "ari", "nmi", "evaluator_path"} & set(views)
    if forbidden:
        raise RuntimeError(f"label-bearing payload rejected: {sorted(forbidden)}")
    hid = str(head["id"])
    private1, private2, fused = (views["emb_latent_omics1"], views["emb_latent_omics2"], views["SpaLORA_fused"])
    aux = {"head_id": hid, "fallback": False}
    h00_result = None
    if hid == "H00_FUSED_PCA20_MCLUST_EEE":
        result = h00(fused, k); labels = result["labels"]; aux.update(result)
    elif hid == "H01_FUSED_DIRECT_MCLUST_EEE":
        result = mclust(row_normalize(fused), k, "EEE", 2020); labels = result["labels"]; aux.update(result)
    elif hid == "H02_FUSED_PCA20_MCLUST_BIC":
        allowed = list(head["allowed_covariance_models"])
        result = mclust(deterministic_pca(row_normalize(fused), 20), k, ro_str_vector(allowed), 2020)
        labels = result["labels"]; aux.update(result); aux["allowed_models"] = allowed
    elif hid == "H03_CONCAT_PRIVATE_PCA20_MCLUST_EEE":
        x = np.concatenate((row_normalize(private1) / math.sqrt(2), row_normalize(private2) / math.sqrt(2)), axis=1)
        result = mclust(deterministic_pca(x, 20), k, "EEE", 2020); labels = result["labels"]; aux.update(result)
        aux["block_scales"] = [1 / math.sqrt(2)] * 2
    elif hid == "H04_RELIABILITY_CONCAT3_PCA20_MCLUST_EEE":
        vv = [private1, private2, fused]; weights = reliability_weights(vv, 10, ids)
        x = np.concatenate([row_normalize(v) * math.sqrt(float(w)) for v, w in zip(vv, weights)], axis=1)
        result = mclust(deterministic_pca(x, 20), k, "EEE", 2020); labels = result["labels"]; aux.update(result)
        aux["reliability_weights"] = weights.tolist()
    elif hid in {"H05_EQUAL3_AFFINITY_SPECTRAL", "H06_AGREEMENT3_AFFINITY_SPECTRAL",
                 "H07_EQUAL3_AFFINITY_SPATIAL05", "H08_EQUAL3_AFFINITY_SPATIAL10"}:
        affinities = [self_tuning_affinity(v, 10, ids) for v in (private1, private2, fused)]
        mean = (affinities[0] + affinities[1] + affinities[2]) * (1 / 3)
        if hid.startswith("H06"):
            matrix = _agreement_affinity(affinities, mean, ids)
        elif hid.startswith("H07") or hid.startswith("H08"):
            beta = 0.05 if hid.startswith("H07") else 0.10
            spatial = row_stochastic(spatial_k6(coords, ids))
            matrix = mean * (1.0 - beta) + spatial * beta
            aux.update({"molecular_weight": 1.0 - beta, "spatial_weight": beta})
        else:
            matrix = mean
        labels = spectral(matrix, k)
        aux.update({"affinity_nnz": int(matrix.nnz), "affinity_sha256": sparse_sha(matrix)})
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            sp.save_npz(artifact_dir / "affinity.npz", matrix, compressed=True)
    elif hid == "H09_VIEW_PARTITION_COASSOCIATION":
        partitions = [mclust(row_normalize(v), k, "EEE", 2020)["labels"] for v in (private1, private2, fused)]
        supports = [binary_knn(row_normalize(v), 10, "euclidean", ids, "union") for v in (private1, private2, fused)]
        support = supports[0].maximum(supports[1]).maximum(supports[2]).tocsr()
        rows, cols = support.nonzero()
        values = np.mean(np.column_stack([p[rows] == p[cols] for p in partitions]), axis=1)
        matrix = sp.coo_matrix((values, (rows, cols)), shape=support.shape).tocsr(); matrix.eliminate_zeros()
        labels = spectral(matrix, k)
        aux.update({"affinity_nnz": int(matrix.nnz), "affinity_sha256": sparse_sha(matrix),
                    "partition_sha256": [array_sha(p) for p in partitions]})
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True); sp.save_npz(artifact_dir / "coassociation.npz", matrix, compressed=True)
    elif hid in {"H10_MCLUST_POSTERIOR_MRF05", "H11_MCLUST_POSTERIOR_MRF10"}:
        h00_result = h00(fused, k)
        beta = 0.05 if hid.startswith("H10") else 0.10
        if h00_result["posterior"].ndim != 2:
            raise RuntimeError("H00 posterior unavailable")
        labels = _icm(h00_result["posterior"], h00_result["labels"], spatial_k6(coords, ids), ids, beta)
        aux.update({"beta": beta, "h00_selected_model": h00_result["selected_model"],
                    "posterior_sha256": array_sha(h00_result["posterior"])})
    else:
        raise ValueError(f"unregistered head: {hid}")
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != len(fused) or len(np.unique(labels)) != int(k):
        raise RuntimeError(f"invalid cluster output for {hid}")
    for key in list(aux):
        if isinstance(aux[key], np.ndarray):
            del aux[key]
    aux["clusters_sha256"] = array_sha(labels)
    return labels, aux


def ro_str_vector(values: Sequence[str]):
    import rpy2.robjects as ro
    return ro.StrVector(list(values))


def file_row(path: Path) -> dict:
    return {"path": str(path), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def save_views(path: Path, views: Mapping[str, np.ndarray], ids: Sequence[str]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **{k: np.asarray(views[k]) for k in VIEW_KEYS})
    os.replace(tmp, path)
    obs_sha = observation_sha(ids)
    return {key: {"shape": list(np.asarray(views[key]).shape), "dtype": str(np.asarray(views[key]).dtype),
                  "canonical_array_sha256": array_sha(np.asarray(views[key])),
                  "ordered_observation_sha256": obs_sha}
            for key in VIEW_KEYS}


def load_views(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(VIEW_KEYS):
            raise RuntimeError("six-view archive contract mismatch")
        return {k: z[k] for k in VIEW_KEYS}


def runtime_resources(start: float) -> dict:
    return {
        "runtime_seconds": float(time.perf_counter() - start),
        "peak_gpu_allocated_mib": float(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0,
        "process_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }


def make_trainer(data, dataset: str, seed: int, device: torch.device) -> Night5ATrainer:
    return Night5ATrainer(data, DATASET_CFG[dataset], BASE_C04, int(seed), device, {}, 1e-12)


def forward_model(model, data: Mapping[str, object], device: torch.device) -> Dict[str, np.ndarray]:
    features1 = torch.as_tensor(data["features_omics1"], dtype=torch.float32, device=device)
    features2 = torch.as_tensor(data["features_omics2"], dtype=torch.float32, device=device)
    adj = tuple(data[k].to(device) for k in ("adj_spatial_omics1", "adj_feature_omics1", "adj_spatial_omics2", "adj_feature_omics2"))
    model.eval()
    with torch.no_grad():
        result = _Forward(features1, features2, adj)(model)
    return normalized_views(result)


__all__ = [
    "BASE_C04", "DATASET_CFG", "VIEW_KEYS", "array_sha", "atomic_json",
    "atomic_torch_save", "build_graph_data", "canonical_json_sha", "file_row",
    "forward_model", "h00", "load_graph_data", "load_views", "make_trainer",
    "mclust", "moran_scores", "observation_sha", "parse_registry", "run_head", "runtime_resources",
    "save_views", "sha256_file", "sparse_sha",
]
