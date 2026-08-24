#!/usr/bin/env python3
"""Build annotation-free numeric carriers for Night-16F producers.

The carrier is the only object read by formal candidate generation.  Public
reference annotations remain in separate evaluator inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from SpaLORA.night16e_tsre import partition_sha256
from scripts.night16e.human_hippocampus_producer import select_unlabeled_start
from scripts.night16e.night16e_producer import load_numeric_inputs


SEED = 20260824


def ordered_id_sha256(ids: np.ndarray) -> str:
    return hashlib.sha256(b"\0".join(str(x).encode("utf-8") for x in ids)).hexdigest()


def standardize(value: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(value, dtype=np.float32)).astype(np.float32)


def reduced_view(adata: ad.AnnData, components: int = 64) -> np.ndarray:
    selected = (
        np.asarray(adata.var["highly_variable"], dtype=bool)
        if "highly_variable" in adata.var
        else np.ones(adata.n_vars, dtype=bool)
    )
    matrix = sp.csr_matrix(adata[:, selected].X, dtype=np.float32)
    dimension = max(1, min(int(components), matrix.shape[0] - 1, matrix.shape[1] - 1))
    return standardize(
        TruncatedSVD(
            n_components=dimension,
            algorithm="randomized",
            n_iter=7,
            random_state=SEED,
        ).fit_transform(matrix)
    )


def spatial_graph(coordinates: np.ndarray, neighbours: int) -> sp.csr_matrix:
    n = len(coordinates)
    model = NearestNeighbors(n_neighbors=min(int(neighbours) + 1, n), algorithm="kd_tree")
    distances, indices = model.fit(coordinates).kneighbors(coordinates)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    weights = np.exp(-np.square(distances / max(bandwidth, 1e-8)))
    rows = np.repeat(np.arange(n, dtype=np.int32), indices.shape[1])
    graph = sp.csr_matrix((weights.ravel(), (rows, indices.ravel())), shape=(n, n))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def kmeans_starts(retained: np.ndarray, k: int) -> tuple[list[str], list[np.ndarray]]:
    identifiers: list[str] = []
    partitions: list[np.ndarray] = []
    for seed in range(5):
        value = KMeans(n_clusters=int(k), n_init=20, random_state=seed).fit_predict(retained)
        identifiers.append(f"KMEANS_RETAINED_S{seed}")
        partitions.append(np.asarray(value, dtype=np.int32))
    return identifiers, partitions


def save_carrier(
    output: Path,
    *,
    ids: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    retained: np.ndarray,
    graphs: tuple[sp.csr_matrix, ...],
    starts: list[np.ndarray],
    start_ids: list[str],
    metadata: dict[str, object],
) -> None:
    if len(graphs) != 3:
        raise ValueError("Night-16F requires exactly three registered graph scales")
    n = len(ids)
    if not (len(view1) == len(view2) == len(retained) == n):
        raise ValueError("carrier observation mismatch")
    if len(starts) != len(start_ids) or not starts:
        raise ValueError("start-bank mismatch")
    payload: dict[str, object] = {
        "ids": np.asarray(ids).astype("U"),
        "view1": np.asarray(view1, dtype=np.float32),
        "view2": np.asarray(view2, dtype=np.float32),
        "retained": np.asarray(retained, dtype=np.float32),
        "start_ids": np.asarray(start_ids).astype("U"),
        "start_partitions": np.stack(starts).astype(np.int32),
    }
    for index, graph in enumerate(graphs):
        graph = sp.csr_matrix(graph, dtype=np.float32)
        graph.sort_indices()
        payload[f"graph{index}__data"] = graph.data
        payload[f"graph{index}__indices"] = graph.indices
        payload[f"graph{index}__indptr"] = graph.indptr
        payload[f"graph{index}__shape"] = np.asarray(graph.shape, dtype=np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output)
    with np.load(output, allow_pickle=False) as replay:
        observed = [partition_sha256(x) for x in replay["start_partitions"]]
    expected = [partition_sha256(x) for x in starts]
    if observed != expected:
        raise RuntimeError("carrier start-bank reload mismatch")
    manifest = {
        **metadata,
        "schema": "night16f-annotation-free-numeric-carrier-v1",
        "n": int(n),
        "view1_shape": list(view1.shape),
        "view2_shape": list(view2.shape),
        "retained_shape": list(retained.shape),
        "graph_shapes_nnz": [
            {"shape": list(graph.shape), "nnz": int(graph.nnz)} for graph in graphs
        ],
        "ordered_id_sha256": ordered_id_sha256(ids),
        "start_bank": [
            {
                "start_id": identifier,
                "partition_sha256": digest,
                "cluster_sizes": [int(v) for v in np.bincount(partition)],
            }
            for identifier, digest, partition in zip(start_ids, expected, starts)
        ],
        "annotation_columns_in_carrier": [],
        "formal_producer_label_reads": 0,
        "artifact_reload": "PASS",
    }
    output.with_suffix(".carrier.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def build_from_kit(args: argparse.Namespace) -> None:
    inputs = load_numeric_inputs(
        Path(args.kit_root),
        Path(args.retained_root),
        Path(args.starts_root),
        args.data_id,
        args.lane,
        args.k,
    )
    seed_ids, seed_starts = kmeans_starts(inputs["retained"], args.k)
    start_ids = ["AUTHORITY_STRONG_START", *seed_ids]
    starts = [np.asarray(inputs["initial"], dtype=np.int32), *seed_starts]
    save_carrier(
        Path(args.output),
        ids=inputs["ids"],
        view1=inputs["view1"],
        view2=inputs["view2"],
        retained=inputs["retained"],
        graphs=inputs["graphs"],
        starts=starts,
        start_ids=start_ids,
        metadata={
            "source_kind": "night16d-local-compute-kit",
            "data_id": args.data_id,
            "lane": args.lane,
            "k": int(args.k),
            "start_provenance": (
                "AUTHORITY_STRONG_START is historical public-label-assisted benchmark HPO; "
                "KMEANS starts are annotation-free robustness starts"
            ),
            "retained_id": inputs["retained_id"],
            "source_archive_keys_accessed": inputs["accessed_keys"],
        },
    )


def build_from_h5ad(args: argparse.Namespace) -> None:
    rna = ad.read_h5ad(args.rna)
    atac = ad.read_h5ad(args.atac)
    rna_ids = np.asarray(rna.obs_names.astype(str))
    atac_ids = np.asarray(atac.obs_names.astype(str))
    if len(np.unique(rna_ids)) != len(rna_ids) or len(np.unique(atac_ids)) != len(atac_ids):
        raise ValueError("non-unique observation identifiers")
    lookup = {identifier: index for index, identifier in enumerate(atac_ids)}
    if set(rna_ids) != set(atac_ids):
        raise ValueError("RNA/ATAC identifier sets differ")
    order = np.asarray([lookup[identifier] for identifier in rna_ids], dtype=np.int64)
    atac = atac[order].copy()
    if not np.array_equal(np.asarray(atac.obs_names.astype(str)), rna_ids):
        raise RuntimeError("explicit identifier alignment failed")
    coordinates = np.asarray(rna.obsm[args.spatial_key], dtype=np.float64)
    atac_coordinates = np.asarray(atac.obsm[args.spatial_key], dtype=np.float64)
    if not np.allclose(coordinates, atac_coordinates, atol=float(args.coordinate_atol), rtol=0):
        raise ValueError("aligned RNA/ATAC coordinates differ")
    view1 = reduced_view(rna)
    view2 = reduced_view(atac)
    fused = standardize(np.concatenate((view1, view2), axis=1))
    retained_dimension = max(1, min(64, fused.shape[0] - 1, fused.shape[1]))
    retained = standardize(
        PCA(n_components=retained_dimension, svd_solver="full").fit_transform(fused)
    )
    graphs = tuple(spatial_graph(coordinates, value) for value in (4, 8, 18))
    medoid, medoid_rows = select_unlabeled_start(retained, view1, view2, args.k)
    seed_ids, seed_starts = kmeans_starts(retained, args.k)
    start_ids = ["UNLABELED_PARTITION_MEDOID", *seed_ids]
    starts = [np.asarray(medoid, dtype=np.int32), *seed_starts]
    save_carrier(
        Path(args.output),
        ids=rna_ids,
        view1=view1,
        view2=view2,
        retained=retained,
        graphs=graphs,
        starts=starts,
        start_ids=start_ids,
        metadata={
            "source_kind": "paired-h5ad-numeric-carrier",
            "data_id": args.data_id,
            "lane": args.lane,
            "k": int(args.k),
            "rna_input_shape": [int(rna.n_obs), int(rna.n_vars)],
            "atac_input_shape": [int(atac.n_obs), int(atac.n_vars)],
            "rna_source_obs_columns": [str(x) for x in rna.obs.columns],
            "atac_source_obs_columns": [str(x) for x in atac.obs.columns],
            "carrier_builder_note": (
                "source AnnData obs metadata were loaded during one-time carrier construction; "
                "formal producer reads only annotation-free numeric carrier"
            ),
            "start_provenance": "annotation-free partition-consensus medoid plus KMeans seeds",
            "unlabeled_start_selector_rows": medoid_rows,
            "spatial_shape": list(coordinates.shape),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)
    kit = sub.add_parser("kit")
    kit.add_argument("--kit-root", required=True)
    kit.add_argument("--retained-root", required=True)
    kit.add_argument("--starts-root", required=True)
    kit.add_argument("--data-id", required=True)
    kit.add_argument("--lane", required=True)
    kit.add_argument("--k", type=int, required=True)
    kit.add_argument("--output", required=True)
    h5ad = sub.add_parser("h5ad")
    h5ad.add_argument("--rna", required=True)
    h5ad.add_argument("--atac", required=True)
    h5ad.add_argument("--data-id", required=True)
    h5ad.add_argument("--lane", required=True)
    h5ad.add_argument("--k", type=int, required=True)
    h5ad.add_argument("--spatial-key", default="spatial")
    h5ad.add_argument("--coordinate-atol", type=float, default=0.0)
    h5ad.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.kind == "kit":
        build_from_kit(args)
    else:
        build_from_h5ad(args)


if __name__ == "__main__":
    main()
