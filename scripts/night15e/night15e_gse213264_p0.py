#!/usr/bin/env python3
"""Build and replay the GSE213264 Human tonsil unlabeled real path."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15e_continuous_reliability_energy import (
    ContinuousEnergyConfig,
    continuous_reliability_energy,
    prepare_continuous_evidence,
)


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> np.ndarray:
    ids = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        next(handle)
        for line in handle:
            ids.append(line.split("\t", 1)[0])
    return np.asarray(ids, dtype=str)


def parse_coordinates(ids: Sequence[str]) -> np.ndarray:
    coordinates = []
    for identifier in ids:
        match = re.fullmatch(r"(\d+)x(\d+)", str(identifier))
        if match is None:
            raise ValueError(f"unparseable coordinate ID: {identifier}")
        coordinates.append((int(match.group(1)), int(match.group(2))))
    value = np.asarray(coordinates, dtype=np.float32)
    if len(np.unique(value, axis=0)) != len(value):
        raise ValueError("coordinate IDs are not unique")
    return value


def read_matrix(path: Path, ordered_ids: np.ndarray) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    # pandas 3 applies a scalar dtype to the index column as well; preserve
    # string spot IDs during parsing and cast only the numeric matrix below.
    frame = pd.read_csv(path, sep="\t", index_col=0, compression="gzip")
    observed_ids = frame.index.to_numpy(dtype=str)
    if len(np.unique(observed_ids)) != len(observed_ids):
        raise ValueError(f"duplicate spot ID in {path.name}")
    if set(observed_ids) != set(np.asarray(ordered_ids, dtype=str)):
        raise ValueError(f"spot ID set mismatch in {path.name}")
    frame = frame.loc[np.asarray(ordered_ids, dtype=str)]
    feature_ids = frame.columns.to_numpy(dtype=str)
    if len(np.unique(feature_ids)) != len(feature_ids):
        raise ValueError(f"duplicate feature ID in {path.name}")
    matrix = sp.csr_matrix(frame.to_numpy(dtype=np.float32, copy=True))
    del frame
    matrix.eliminate_zeros()
    return matrix, feature_ids, observed_ids


def reduced_rna(matrix: sp.csr_matrix, dim: int = 30) -> np.ndarray:
    value = normalize(matrix, norm="l1", axis=1, copy=True) * 1e4
    value.data = np.log1p(value.data)
    embedding = TruncatedSVD(n_components=int(dim), random_state=20260824).fit_transform(value)
    return StandardScaler().fit_transform(embedding).astype(np.float32)


def reduced_protein(matrix: sp.csr_matrix, dim: int = 30) -> np.ndarray:
    dense = np.log1p(matrix.toarray()).astype(np.float32)
    dense = StandardScaler().fit_transform(dense)
    embedding = PCA(n_components=min(int(dim), dense.shape[1]), svd_solver="full").fit_transform(dense)
    return StandardScaler().fit_transform(embedding).astype(np.float32)


def sparse_graph(coordinates: np.ndarray, neighbors: int = 6) -> sp.csr_matrix:
    model = NearestNeighbors(n_neighbors=int(neighbors) + 1, algorithm="kd_tree").fit(coordinates)
    _, indices = model.kneighbors(coordinates)
    rows = np.repeat(np.arange(len(coordinates), dtype=np.int32), int(neighbors))
    cols = indices[:, 1:].reshape(-1).astype(np.int32)
    graph = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(coordinates), len(coordinates)))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def save_csr(payload: Dict[str, np.ndarray], prefix: str, value: sp.csr_matrix) -> None:
    value = sp.csr_matrix(value)
    payload[f"{prefix}__data"] = value.data
    payload[f"{prefix}__indices"] = value.indices
    payload[f"{prefix}__indptr"] = value.indptr
    payload[f"{prefix}__shape"] = np.asarray(value.shape, dtype=np.int64)


def default_config() -> ContinuousEnergyConfig:
    return ContinuousEnergyConfig(
        beta=0.35,
        edge_floor=0.08,
        conflict_center=0.22,
        conflict_temperature=0.10,
        conflict_union_weight=0.55,
        conflict_penalty=0.25,
        mass_center=0.40,
        mass_temperature=0.12,
        neighbor_capacity=0.55,
        low_weight=0.65,
        twohop_weight=0.20,
        high_weight=0.50,
        unary_temperature=0.55,
        retained_bias=1.0,
        view_balance=0.0,
        trust_scale=1.2,
        trust_center=0.18,
        trust_temperature=0.12,
        move_threshold=0.015,
        move_fraction=0.06,
        sweeps=2,
    )


def build(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    rna_ids_raw = read_ids(args.rna)
    protein_ids_raw = read_ids(args.protein)
    if len(rna_ids_raw) != 2492 or len(protein_ids_raw) != 2492:
        raise ValueError("registered 2492-spot contract failed")
    same_set = set(rna_ids_raw) == set(protein_ids_raw)
    same_order = bool(np.array_equal(rna_ids_raw, protein_ids_raw))
    if not same_set:
        raise ValueError("RNA/protein spot-ID sets differ")
    ordered_ids = np.asarray(sorted(set(rna_ids_raw)), dtype=str)
    coordinates = parse_coordinates(ordered_ids)
    rna, gene_ids, _ = read_matrix(args.rna, ordered_ids)
    protein, target_ids, _ = read_matrix(args.protein, ordered_ids)
    if rna.shape != (2492, 28417) or protein.shape != (2492, 283):
        raise ValueError(f"registered matrix shape failed: {rna.shape}, {protein.shape}")
    view1 = reduced_rna(rna)
    view2 = reduced_protein(protein)
    concatenated = StandardScaler().fit_transform(np.column_stack((view1, view2)))
    retained = PCA(n_components=32, svd_solver="full").fit_transform(concatenated).astype(np.float32)
    retained = StandardScaler().fit_transform(retained).astype(np.float32)
    graph = sparse_graph(coordinates)
    engineering_k = int(args.engineering_k)
    initial = KMeans(n_clusters=engineering_k, random_state=20260824, n_init=20).fit_predict(retained).astype(np.int32)
    evidence = prepare_continuous_evidence(graph, retained, view1, view2)
    config = default_config()
    partition, diagnostics = continuous_reliability_energy(initial, engineering_k, evidence, config)
    sensitivity_k = 7
    initial_k7 = KMeans(n_clusters=sensitivity_k, random_state=20260824, n_init=20).fit_predict(retained).astype(np.int32)
    partition_k7, diagnostics_k7 = continuous_reliability_energy(
        initial_k7, sensitivity_k, evidence, config
    )
    args.output.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {
        "ids": ordered_ids,
        "coordinates": coordinates,
        "view1": view1,
        "view2": view2,
        "retained": retained,
        "initial": initial,
        "partition": partition,
        "initial_k7": initial_k7,
        "partition_k7": partition_k7,
        "engineering_k": np.asarray([engineering_k], dtype=np.int32),
    }
    save_csr(payload, "graph", graph)
    artifact = args.output / "gse213264_human_tonsil_p0.npz"
    np.savez_compressed(artifact, **payload)
    audit = {
        "status": "PASS",
        "accession": "GSE213264",
        "unit": "Human tonsil",
        "rna_accession": "GSM6578062",
        "protein_accession": "GSM6578071",
        "rna_shape": list(rna.shape),
        "protein_shape": list(protein.shape),
        "rna_dtype": str(rna.dtype),
        "protein_dtype": str(protein.dtype),
        "rna_nnz": int(rna.nnz),
        "protein_nnz": int(protein.nnz),
        "rna_feature_count": int(len(gene_ids)),
        "protein_target_count": int(len(target_ids)),
        "rna_raw_order_ids_sha256": sha256_array(rna_ids_raw),
        "protein_raw_order_ids_sha256": sha256_array(protein_ids_raw),
        "raw_order_equal": same_order,
        "spot_id_sets_equal": same_set,
        "explicit_string_id_alignment_performed": True,
        "ordered_ids_sha256": sha256_array(ordered_ids),
        "coordinate_shape": list(coordinates.shape),
        "coordinate_min": coordinates.min(axis=0).astype(int).tolist(),
        "coordinate_max": coordinates.max(axis=0).astype(int).tolist(),
        "unique_coordinate_count": int(len(np.unique(coordinates, axis=0))),
        "view1_shape": list(view1.shape),
        "view2_shape": list(view2.shape),
        "retained_shape": list(retained.shape),
        "graph_shape": list(graph.shape),
        "graph_nnz": int(graph.nnz),
        "engineering_k": engineering_k,
        "engineering_k_role": "public author-reported RNA cluster count used as known engineering K; no per-spot reference labels read",
        "engineering_k_provenance": "original Spatial-CITE-seq Human tonsil RNA analysis reports 8 author-derived unsupervised clusters",
        "protein_k_sensitivity_context": 7,
        "protein_k_sensitivity_context_role": "public author-reported protein cluster count; not ground truth",
        "initial_partition_sha256": sha256_array(initial),
        "partition_sha256": sha256_array(partition),
        "initial_k7_partition_sha256": sha256_array(initial_k7),
        "partition_k7_sha256": sha256_array(partition_k7),
        "cluster_sizes": np.bincount(partition, minlength=engineering_k).astype(int).tolist(),
        "cluster_sizes_k7": np.bincount(partition_k7, minlength=sensitivity_k).astype(int).tolist(),
        "config": asdict(config),
        "diagnostics": diagnostics,
        "diagnostics_k7": diagnostics_k7,
        "rna_file_sha256": sha256_file(args.rna),
        "protein_file_sha256": sha256_file(args.protein),
        "artifact_sha256": sha256_file(artifact),
        "artifact_path": str(artifact.resolve()),
        "reference_partition_status": "UNSUPPORTED_NO_MANUAL_OR_EXPERT_SPATIAL_DOMAIN_LABEL_FOUND",
        "author_clusters_interpretation": "author-algorithm-derived clusters are not manual/expert ground truth",
        "absolute_ari_nmi_computed": 0,
        "labels_read": 0,
        "dense_n_by_n_count": 0,
        "wall_seconds": time.perf_counter() - started,
    }
    (args.output / "gse213264_human_tonsil_p0.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output / "gse213264_human_tonsil_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "artifact": str(artifact), "partition_sha256": audit["partition_sha256"]}))


def load_csr(archive: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (archive[f"{prefix}__data"], archive[f"{prefix}__indices"], archive[f"{prefix}__indptr"]),
        shape=tuple(map(int, archive[f"{prefix}__shape"])),
    )


def replay(args: argparse.Namespace) -> None:
    registered = json.loads(args.audit.read_text(encoding="utf-8"))
    archive = np.load(args.artifact, allow_pickle=False)
    graph = load_csr(archive, "graph")
    evidence = prepare_continuous_evidence(
        graph, archive["retained"], archive["view1"], archive["view2"]
    )
    config = ContinuousEnergyConfig(**registered["config"])
    partition, diagnostics = continuous_reliability_energy(
        archive["initial"], int(archive["engineering_k"][0]), evidence, config
    )
    partition_k7, diagnostics_k7 = continuous_reliability_energy(
        archive["initial_k7"], 7, evidence, config
    )
    observed = sha256_array(partition)
    if observed != registered["partition_sha256"]:
        raise RuntimeError("fresh-process GSE213264 partition mismatch")
    observed_k7 = sha256_array(partition_k7)
    if observed_k7 != registered["partition_k7_sha256"]:
        raise RuntimeError("fresh-process GSE213264 K7 partition mismatch")
    result = {
        "status": "PASS",
        "artifact_sha256": sha256_file(args.artifact),
        "ordered_ids_sha256": sha256_array(archive["ids"]),
        "partition_sha256": observed,
        "partition_k7_sha256": observed_k7,
        "partition_exact": True,
        "shape_exact": [
            list(archive["view1"].shape) == registered["view1_shape"],
            list(archive["view2"].shape) == registered["view2_shape"],
            list(archive["retained"].shape) == registered["retained_shape"],
        ],
        "finite": bool(np.isfinite(archive["retained"]).all()),
        "diagnostics": diagnostics,
        "diagnostics_k7": diagnostics_k7,
        "labels_read": 0,
        "dense_n_by_n_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "partition_sha256": observed}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--rna", type=Path, required=True)
    build_parser.add_argument("--protein", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--engineering-k", type=int, default=8)
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--artifact", type=Path, required=True)
    replay_parser.add_argument("--audit", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
