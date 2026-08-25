#!/usr/bin/env python3
"""Build a label-free numeric human-placenta carrier and frozen start bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(); digest.update(value.dtype.str.encode()); digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def string_sha(values) -> str:
    return hashlib.sha256(b"\0".join(str(value).encode("utf-8") for value in values)).hexdigest()


def standardize(value: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(value, dtype=np.float64)).astype(np.float32)


def row_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return (value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-10)).astype(np.float32)


def coordinate_views(rna: np.ndarray, atac: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    reference, target = row_normalize(standardize(rna)), row_normalize(standardize(atac))
    # Estimate the orthogonal map in float64; float32 SVD leaves a ~1e-6
    # orthogonality residue that is avoidable and complicates exact replay.
    left, singular, right_t = np.linalg.svd(target.astype(np.float64).T @ reference.astype(np.float64), full_matrices=False)
    rotation = left @ right_t
    aligned = row_normalize(target @ rotation)
    fused = row_normalize(0.5 * (reference + aligned))
    diagnostics = {
        "alignment": "SAME_SPOT_ORTHOGONAL_PROCRUSTES",
        "rotation_sha256": array_sha(rotation.astype(np.float64)),
        "orthogonality_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(rotation.shape[0]))),
        "singular_values": [float(value) for value in singular],
        "pre_frobenius": float(np.linalg.norm(target - reference)),
        "post_frobenius": float(np.linalg.norm(aligned - reference)),
    }
    if diagnostics["orthogonality_error"] > 1e-8:
        raise RuntimeError("Procrustes rotation is not orthogonal")
    return reference, aligned, fused, diagnostics


def graph_from_coordinates(coordinates: np.ndarray, neighbours: int) -> sp.csr_matrix:
    model = NearestNeighbors(n_neighbors=min(neighbours + 1, len(coordinates)), metric="euclidean", n_jobs=1).fit(coordinates)
    _, index = model.kneighbors(coordinates)
    rows = np.repeat(np.arange(len(coordinates), dtype=np.int32), index.shape[1] - 1)
    cols = index[:, 1:].reshape(-1).astype(np.int32)
    graph = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(coordinates), len(coordinates)))
    graph = graph.maximum(graph.T).tocsr(); graph.setdiag(0); graph.eliminate_zeros(); graph.sort_indices()
    return graph


def transition(graph: sp.csr_matrix) -> sp.csr_matrix:
    degree = np.asarray(graph.sum(1)).ravel(); inverse = np.zeros_like(degree); inverse[degree > 0] = 1.0 / degree[degree > 0]
    return (sp.diags(inverse) @ graph).tocsr()


def exact_partition(partition: np.ndarray, k: int) -> np.ndarray:
    _, partition = np.unique(np.asarray(partition), return_inverse=True); partition = partition.astype(np.int32)
    if len(np.unique(partition)) != k: raise RuntimeError("start violated exact K")
    return partition


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter(); rna_path, atac_path = Path(args.rna), Path(args.atac)
    # Fail closed on the two governance fields only.  The authority artifact also
    # contains label counts for the evaluator audit, but preprocessing must not
    # traverse or consume those fields.
    raw_audit = json.loads(Path(args.authority_audit).read_text(encoding="utf-8"))
    allowed_audit = {key: raw_audit[key] for key in ("status", "authority_gap")}
    del raw_audit
    if allowed_audit["status"] != "PASS" or allowed_audit["authority_gap"]:
        raise RuntimeError("authority audit did not pass")
    rna, atac = ad.read_h5ad(rna_path), ad.read_h5ad(atac_path)
    ids_rna, ids_atac = np.asarray(rna.obs_names.astype(str)), np.asarray(atac.obs_names.astype(str))
    if not np.array_equal(ids_rna, ids_atac): raise RuntimeError("RNA/ATAC ordered IDs differ")
    rna_coordinates = np.asarray(rna.obsm["spatial"], dtype=np.float64) if "spatial" in rna.obsm else None
    atac_coordinates = np.asarray(atac.obsm["spatial"], dtype=np.float64) if "spatial" in atac.obsm else None
    if rna_coordinates is None or atac_coordinates is None or not np.array_equal(rna_coordinates, atac_coordinates):
        raise RuntimeError("RNA/ATAC spatial coordinates differ")
    coordinates = rna_coordinates.astype(np.float32)
    # Annotation columns are not indexed or used below. K is supplied as public protocol metadata.
    counts = sp.csr_matrix(rna.X, dtype=np.float64)
    if counts.shape != (1662, 36601) or counts.nnz == 0 or np.any(counts.data < 0) or not np.isfinite(counts.data).all():
        raise RuntimeError("RNA count matrix semantics failed")
    totals = np.asarray(counts.sum(1)).ravel()
    if np.any(totals <= 0): raise RuntimeError("RNA contains zero-total observations")
    logged = (sp.diags(10000.0 / totals) @ counts).tocsr(); logged.data = np.log1p(logged.data)
    detected = np.asarray((logged > 0).sum(0)).ravel()
    mean = np.asarray(logged.mean(0)).ravel(); second = np.asarray(logged.power(2).mean(0)).ravel(); variance = np.maximum(second - mean ** 2, 0.0)
    eligible = np.flatnonzero(detected >= 5)
    if len(eligible) < args.hvg: raise RuntimeError("too few eligible RNA features")
    hvg = eligible[np.argsort(variance[eligible], kind="mergesort")[-args.hvg:]]
    hvg.sort()
    rna_score = TruncatedSVD(n_components=args.dimension, algorithm="randomized", n_iter=7, random_state=0).fit_transform(logged[:, hvg])
    atac_x = sp.csr_matrix(atac.X, dtype=np.float64).toarray() if sp.issparse(atac.X) else np.asarray(atac.X, dtype=np.float64)
    if atac_x.shape != (1662, 63) or not np.isfinite(atac_x).all(): raise RuntimeError("ATAC-derived feature matrix semantics failed")
    atac_score = PCA(n_components=args.dimension, svd_solver="full", whiten=False).fit_transform(StandardScaler().fit_transform(atac_x))
    view1, view2, retained, alignment = coordinate_views(rna_score, atac_score)
    graphs = tuple(graph_from_coordinates(coordinates, value) for value in (4, 8, 18))
    low = row_normalize(0.65 * retained + 0.35 * np.asarray(transition(graphs[1]) @ retained))
    starts, names = [], []
    def add(name: str, partition: np.ndarray) -> None:
        starts.append(exact_partition(partition, args.k)); names.append(name)
    for seed in range(5): add(f"FUSED_KMEANS_S{seed}", KMeans(args.k, n_init=1, random_state=seed).fit_predict(retained))
    for seed in range(2): add(f"RNA_KMEANS_S{seed}", KMeans(args.k, n_init=1, random_state=seed).fit_predict(view1))
    for seed in range(2): add(f"ATAC_REGULATORY_KMEANS_S{seed}", KMeans(args.k, n_init=1, random_state=seed).fit_predict(view2))
    for seed in range(2): add(f"LOWPASS_FUSED_KMEANS_S{seed}", KMeans(args.k, n_init=1, random_state=seed).fit_predict(low))
    for seed in range(2): add(f"FUSED_GMM_DIAG_S{seed}", GaussianMixture(args.k, covariance_type="diag", n_init=1, random_state=seed, reg_covar=1e-5, max_iter=300).fit_predict(retained))
    numeric = {"ids": ids_rna.astype("U"), "coordinates": coordinates, "view1": view1, "view2": view2, "retained": retained,
               "start_names": np.asarray(names, dtype="U"), "start_partitions": np.stack(starts).astype(np.int32)}
    for index, graph in enumerate(graphs):
        numeric[f"graph{index}__data"] = graph.data.astype(np.float32); numeric[f"graph{index}__indices"] = graph.indices.astype(np.int32)
        numeric[f"graph{index}__indptr"] = graph.indptr.astype(np.int32); numeric[f"graph{index}__shape"] = np.asarray(graph.shape, dtype=np.int64)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(output, **numeric)
    with np.load(output, allow_pickle=False) as replay:
        if replay.files != list(numeric) or not all(np.array_equal(replay[key], value) for key, value in numeric.items()):
            raise RuntimeError("carrier save/reload mismatch")
    manifest = {
        "status": "FROZEN_BEFORE_PARTITION_EVALUATION", "schema": "night18d-placenta-numeric-carrier-v1",
        "source_rna_sha256": file_sha(rna_path), "source_atac_sha256": file_sha(atac_path), "authority_audit_sha256": file_sha(Path(args.authority_audit)),
        "carrier_sha256": file_sha(output), "ordered_ids_sha256": string_sha(ids_rna), "coordinates_sha256": array_sha(coordinates),
        "n": len(ids_rna), "k_public_protocol": args.k, "rna_input_shape": list(rna.shape), "atac_input_shape": list(atac.shape),
        "rna_view_shape": list(view1.shape), "atac_view_shape": list(view2.shape), "retained_shape": list(retained.shape),
        "rna_preprocessing": {"library_size": 10000, "transform": "log1p", "min_detected_cells": 5, "hvg": args.hvg, "svd_dimension": args.dimension, "svd_seed": 0},
        "rna_hvg_var_names_sha256": string_sha(np.asarray(rna.var_names.astype(str))[hvg]), "rna_hvg_indices_sha256": array_sha(hvg.astype(np.int64)),
        "atac_preprocessing": {"semantic_name": "official processed ATAC-derived TF-associated regulatory features", "standardize": True, "pca_dimension": args.dimension, "svd_solver": "full"},
        "alignment": alignment, "graph_scales_knn": [4, 8, 18], "graph_nnz": [int(graph.nnz) for graph in graphs],
        "start_bank_count": len(starts), "start_names": names, "start_partitions_sha256": array_sha(np.stack(starts).astype(np.int32)),
        "annotation_columns_accessed_by_preprocessing_computation": 0, "dense_n_by_n_created": 0,
        "authority_audit_fields_consumed": sorted(allowed_audit),
        "carrier_reload": "PASS", "wall_seconds": time.perf_counter() - started,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--rna", required=True); parser.add_argument("--atac", required=True); parser.add_argument("--authority-audit", required=True)
    parser.add_argument("--k", type=int, default=10); parser.add_argument("--hvg", type=int, default=3000); parser.add_argument("--dimension", type=int, default=30); parser.add_argument("--output", required=True); run(parser.parse_args())


if __name__ == "__main__": main()
