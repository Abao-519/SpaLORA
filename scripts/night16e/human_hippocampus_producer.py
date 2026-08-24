#!/usr/bin/env python3
"""Label-free frozen-profile producer for the MultiGATE human hippocampus unit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from SpaLORA.night15f_multiscale_expansion import prepare_expansion_evidence
from SpaLORA.night16e_tsre import (
    direct_energy_control,
    partition_sha256,
    prepare_tsre_evidence,
    tsre_expansion,
)
from scripts.night16e.night16e_producer import config_sha256, load_config


SEED = 20260824


def ordered_id_sha256(ids: np.ndarray) -> str:
    payload = b"\0".join(str(value).encode("utf-8") for value in ids)
    return hashlib.sha256(payload).hexdigest()


def standardize(value: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(value, dtype=np.float32)).astype(np.float32)


def reduced_view(adata: ad.AnnData, components: int = 64) -> np.ndarray:
    selected = (
        np.asarray(adata.var["highly_variable"], dtype=bool)
        if "highly_variable" in adata.var
        else np.ones(adata.n_vars, dtype=bool)
    )
    matrix = adata[:, selected].X
    matrix = sp.csr_matrix(matrix, dtype=np.float32)
    dimension = min(int(components), matrix.shape[0] - 1, matrix.shape[1] - 1)
    value = TruncatedSVD(
        n_components=dimension,
        algorithm="randomized",
        n_iter=7,
        random_state=SEED,
    ).fit_transform(matrix)
    return standardize(value)


def sparse_spatial_graph(coordinates: np.ndarray, neighbours: int) -> sp.csr_matrix:
    n = len(coordinates)
    model = NearestNeighbors(n_neighbors=min(int(neighbours) + 1, n), algorithm="kd_tree")
    model.fit(coordinates)
    distances, indices = model.kneighbors(coordinates)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    weights = np.exp(-np.square(distances / max(bandwidth, 1e-8)))
    rows = np.repeat(np.arange(n, dtype=np.int32), indices.shape[1])
    graph = sp.csr_matrix((weights.reshape(-1), (rows, indices.reshape(-1))), shape=(n, n))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def select_unlabeled_start(
    retained: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    sources = {"fused": retained, "rna": view1, "atac": view2}
    candidates: list[tuple[str, np.ndarray]] = []
    for source_name, value in sources.items():
        for seed in (0, 1, 2, 3, 4):
            labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(value)
            candidates.append((f"KMEANS_{source_name}_S{seed}", labels.astype(np.int32)))
    for covariance in ("diag", "tied"):
        labels = GaussianMixture(
            n_components=k,
            covariance_type=covariance,
            reg_covar=1e-5,
            max_iter=300,
            random_state=0,
        ).fit_predict(retained)
        candidates.append((f"GMM_{covariance}_S0", labels.astype(np.int32)))
    threshold = max(5, int(np.ceil(0.01 * len(retained) / float(k))))
    rows: list[dict[str, object]] = []
    for index, (identifier, labels) in enumerate(candidates):
        centrality = float(
            np.mean(
                [
                    adjusted_rand_score(labels, other)
                    for other_index, (_, other) in enumerate(candidates)
                    if other_index != index
                ]
            )
        )
        sizes = np.bincount(labels, minlength=k)
        microcluster_penalty = max(0.0, float(threshold - sizes.min()) / float(threshold))
        rows.append(
            {
                "candidate_id": identifier,
                "centrality": centrality,
                "microcluster_penalty": microcluster_penalty,
                "selector_score": centrality - microcluster_penalty,
                "cluster_sizes": [int(x) for x in sizes],
                "partition_sha256": partition_sha256(labels),
            }
        )
    selected = sorted(rows, key=lambda row: (-row["selector_score"], row["candidate_id"]))[0]
    partition = dict(candidates)[str(selected["candidate_id"])]
    return partition, rows


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    rna = ad.read_h5ad(args.rna)
    atac = ad.read_h5ad(args.atac)
    rna_ids = np.asarray(rna.obs_names.astype(str))
    atac_ids = np.asarray(atac.obs_names.astype(str))
    if len(np.unique(rna_ids)) != len(rna_ids) or len(np.unique(atac_ids)) != len(atac_ids):
        raise ValueError("non-unique spot identifiers")
    atac_lookup = {identifier: index for index, identifier in enumerate(atac_ids)}
    if set(rna_ids) != set(atac_ids):
        raise ValueError("RNA/ATAC spot-ID sets differ")
    order = np.asarray([atac_lookup[identifier] for identifier in rna_ids], dtype=np.int64)
    if not np.array_equal(atac_ids[order], rna_ids):
        raise RuntimeError("explicit RNA/ATAC ID alignment failed")
    atac = atac[order].copy()
    coordinates_rna = np.asarray(rna.obsm["spatial"], dtype=np.float64)
    coordinates_atac = np.asarray(atac.obsm["spatial"], dtype=np.float64)
    if not np.array_equal(coordinates_rna, coordinates_atac):
        raise ValueError("aligned RNA/ATAC spatial coordinates differ")
    view1 = reduced_view(rna)
    view2 = reduced_view(atac)
    fused = standardize(np.concatenate((view1, view2), axis=1))
    retained = standardize(PCA(n_components=64, svd_solver="full").fit_transform(fused))
    graphs = tuple(sparse_spatial_graph(coordinates_rna, value) for value in (4, 8, 18))
    initial, start_bank = select_unlabeled_start(retained, view1, view2, args.k)

    registry = json.loads(Path(args.registry).read_text())
    candidates = [
        candidate
        for candidate in registry["candidates"]
        if candidate["variant"] in (
            "INPUT_STRONG_START",
            "TSRE_FULL",
            "NIGHT15F_DIRECT",
            "SUPPORT_MODULATION_ONLY",
            "BOUNDARY_OFF",
            "CONFLICT_PRIVATE_OFF",
            "REJECTED_MASS_STAY_OFF",
            "PURE_SUPPORT_POTTS_FULL",
        )
    ]
    evidence = prepare_tsre_evidence(prepare_expansion_evidence(graphs, retained, view1, view2))
    partitions: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_started = time.perf_counter()
        if candidate["variant"] == "INPUT_STRONG_START":
            partition = initial.copy()
            diagnostics: dict[str, object] = {"control": "UNLABELED_START_BANK_MEDOID"}
        elif candidate["variant"] == "NIGHT15F_DIRECT":
            partition, diagnostics = direct_energy_control(
                initial, args.k, evidence, load_config(candidate["config"])
            )
        else:
            partition, diagnostics = tsre_expansion(initial, args.k, evidence, load_config(candidate["config"]))
        if len(np.unique(partition)) != args.k:
            raise RuntimeError(f"candidate lost exact K: {candidate['candidate_id']}")
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "profile_id": candidate["profile_id"],
                "variant": candidate["variant"],
                "config_sha256": config_sha256(candidate.get("config")),
                "partition_index": len(partitions),
                "partition_sha256": partition_sha256(partition),
                "cluster_sizes_full": [int(x) for x in np.bincount(partition, minlength=args.k)],
                "changed_from_unlabeled_start": int(np.sum(partition != initial)),
                "diagnostics": diagnostics,
                "status": "PASS",
                "wall_seconds": float(time.perf_counter() - candidate_started),
            }
        )
        partitions.append(np.asarray(partition, dtype=np.int32))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, ids=rna_ids.astype("U"), partitions=np.stack(partitions))
    temporary.replace(output)
    with np.load(output, allow_pickle=False) as replay:
        for index, row in enumerate(rows):
            if partition_sha256(replay["partitions"][index]) != row["partition_sha256"]:
                raise RuntimeError("fresh artifact reload mismatch")
    manifest = {
        "schema": "night16e-human-hippocampus-label-free-producer-v1",
        "family": "RNA_CHROMATIN",
        "data_id": "MULTIGATE_HUMAN_HIPPOCAMPUS",
        "k": int(args.k),
        "known_k_source": "public manual hippocampus layer/white-matter annotation protocol",
        "n": int(len(rna_ids)),
        "rna_input_shape": [int(rna.n_obs), int(rna.n_vars)],
        "atac_input_shape": [int(atac.n_obs), int(atac.n_vars)],
        "view1_shape": list(view1.shape),
        "view2_shape": list(view2.shape),
        "retained_shape": list(retained.shape),
        "spatial_shape": list(coordinates_rna.shape),
        "graph_nnz": [int(graph.nnz) for graph in graphs],
        "ordered_id_sha256": ordered_id_sha256(rna_ids),
        "numeric_input_obs_columns": {
            "rna": [str(value) for value in rna.obs.columns],
            "atac": [str(value) for value in atac.obs.columns],
        },
        "annotation_like_input_columns_present": [
            str(value)
            for value in list(rna.obs.columns) + list(atac.obs.columns)
            if any(
                token in str(value).lower()
                for token in ("label", "cluster", "annotation", "cell_type", "ground_truth")
            )
        ],
        "explicit_spot_id_alignment": "PASS",
        "coordinate_alignment": "PASS",
        "start_selection": "mean partition-consensus ARI centrality minus microcluster penalty; lexical tie-break",
        "start_centrality_metric": (
            "pairwise adjusted Rand index between candidate partitions; no ground-truth reference"
        ),
        "selected_start": sorted(start_bank, key=lambda row: (-row["selector_score"], row["candidate_id"]))[0],
        "start_bank": start_bank,
        "rows": rows,
        "annotation_arrays_accessed": [],
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
        "artifact_reload": "PASS",
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    output.with_suffix(".producer.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", required=True)
    parser.add_argument("--atac", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
