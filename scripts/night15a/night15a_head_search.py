#!/usr/bin/env python3
"""Public-benchmark head search for one frozen Night-15A checkpoint.

Labels are consumed only after each partition is generated.  They may rank
cross-run endpoint configurations, but never enter model training or a cluster
fit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/night13b"), str(REPO / "scripts/night14b")]

import night13b_run as n13b  # noqa: E402
import night14b_head_search as n14heads  # noqa: E402
from SpaLORA.night14b_atac import anchored_majority_refine, diffuse, spatial_operator  # noqa: E402
from SpaLORA.night15a_mcdf import array_sha256, cluster_known_k, coordinate_features, row_l2  # noqa: E402


LABEL_POLICY = "PUBLIC_BENCHMARK_HPO_AFTER_PARTITION_ONLY"
FILTERS = (
    {"id": "NONE", "kind": "NONE"},
    {"id": "LOW_K12_B80_S2", "kind": "LOW", "graph_k": 12, "beta": 0.8, "steps": 2},
    {"id": "LOW_K18_B90_S3", "kind": "LOW", "graph_k": 18, "beta": 0.9, "steps": 3},
    {"id": "BILAT_K12_Q50_B85_S2", "kind": "BILATERAL", "graph_k": 12, "quantile": 0.5, "beta": 0.85, "steps": 2},
    {"id": "BILAT_K18_Q50_B85_S2", "kind": "BILATERAL", "graph_k": 18, "quantile": 0.5, "beta": 0.85, "steps": 2},
)
COORDINATES = (
    ("NONE", 0.0),
    ("LINEAR", 0.6),
    ("LINEAR", 1.2),
    ("LINEAR", 2.4),
    ("POLY2", 1.2),
)
ALGORITHMS = ("KMEANS", "GMM_DIAG", "GMM_SPHERICAL")


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def load_run(run_dir: Path) -> Tuple[dict, dict]:
    audit = json.loads((run_dir / "training_audit.json").read_text(encoding="utf-8"))
    reload_audit = json.loads((run_dir / "fresh_process_reload.json").read_text(encoding="utf-8"))
    if not reload_audit["all_numerically_close"]:
        raise RuntimeError("training checkpoint fresh-process replay exceeds tolerance")
    archive = np.load(run_dir / "views.npz", allow_pickle=False)
    views = {
        key: np.asarray(archive[key], dtype=np.float32)
        for key in ("z1", "z2", "base_fused", "mcdf")
    }
    views["ids"] = np.asarray(archive["ids"], dtype=str)
    views["coordinates"] = np.asarray(archive["coordinates"], dtype=np.float64)
    return audit, views


def source_filter_cache(views: Mapping[str, np.ndarray]) -> Dict[Tuple[str, str], np.ndarray]:
    coordinates = np.asarray(views["coordinates"])
    content_views = {
        "emb_latent_omics1": views["z1"],
        "emb_latent_omics2": views["z2"],
        "SpaLORA_fused": views["mcdf"],
    }
    cache = {}
    for spec in FILTERS:
        if spec["kind"] == "NONE":
            operator = None
        elif spec["kind"] == "LOW":
            operator = spatial_operator(coordinates, int(spec["graph_k"]))
        else:
            operator, _ = n14heads.anisotropic_operator(
                content_views,
                coordinates,
                int(spec["graph_k"]),
                float(spec["quantile"]),
            )
        for source in ("z1", "z2", "base_fused", "mcdf"):
            value = np.asarray(views[source], dtype=np.float32)
            if operator is not None:
                value = diffuse(value, operator, float(spec["beta"]), int(spec["steps"]))
            cache[(source, spec["id"])] = value
    return cache


def endpoint_matrix(
    value: np.ndarray,
    coordinates: np.ndarray,
    dimension: int,
    coordinate_basis: str,
    coordinate_weight: float,
) -> np.ndarray:
    source = row_l2(value)
    dimension = min(int(dimension), source.shape[0] - 1, source.shape[1])
    molecular = PCA(dimension, random_state=0, svd_solver="randomized").fit_transform(source)
    molecular = StandardScaler().fit_transform(molecular).astype(np.float32)
    if float(coordinate_weight) == 0.0:
        return molecular
    coord = coordinate_features(coordinates, coordinate_basis)
    return np.concatenate((molecular, float(coordinate_weight) * coord), axis=1).astype(np.float32)


def metric_row(payload: Mapping[str, object], partition: np.ndarray) -> dict:
    return n13b.partition_metrics(
        payload["labels"], payload["label_mask"], partition, payload["metric_graph"]
    )


def run(
    dataset: str,
    run_dir: Path,
    output: Path,
    endpoint_seeds,
    protocol: str = "PROJECT",
    screen_only: bool = False,
    quick_screen: bool = False,
    freeze_screen: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started_all = time.perf_counter()
    audit, views = load_run(run_dir)
    payload = n13b.base_payload(dataset)
    registered_ids = np.asarray(payload["ids"], dtype=str)
    if not np.array_equal(registered_ids, views["ids"]):
        raise RuntimeError("head-search observation order mismatch")
    if not np.array_equal(np.asarray(payload["coordinates"]), views["coordinates"]):
        raise RuntimeError("head-search coordinates mismatch")
    result_dataset = dataset
    label_semantics = "PROJECT_PUBLIC_GROUND_TRUTH"
    if protocol == "P22_3DOT_K18":
        if dataset != "P22":
            raise ValueError("P22_3DOT_K18 is only valid for the P22 lane")
        label_path = Path(
            "/root/autodl-fs/night15a_multimodal_contribution_and_score_stability_20260823/"
            "protocol_inputs/3dot_zenodo_15089427/3d-OT.h5ad"
        )
        carrier = ad.read_h5ad(str(label_path), backed="r")
        carrier_ids = carrier.obs_names.astype(str).to_numpy()
        if not np.array_equal(carrier_ids, registered_ids):
            raise RuntimeError("official 3d-OT K18 observation order mismatch")
        labels = carrier.obs["3d-OT"].astype(str).to_numpy()
        if int(pd.Series(labels).nunique()) != 18:
            raise RuntimeError("official 3d-OT label cardinality is not 18")
        payload = {
            **payload,
            "labels": labels,
            "label_mask": np.ones(len(labels), dtype=bool),
        }
        cluster_ks = (18,)
        result_dataset = "P22_3DOT_K18"
        label_semantics = "OFFICIAL_3DOT_SUPPLIED_18_STATE_ANNOTATION_NOT_K9_SPLIT"
    else:
        if dataset == "P22":
            cluster_ks = (9,)
        elif dataset == "MISAR_E15_5_S1":
            cluster_ks = (7, 12)
        else:
            cluster_ks = (int(payload["k"]),)
    filtered = source_filter_cache(views)
    active_filters = FILTERS if not (quick_screen or freeze_screen) else tuple(
        spec for spec in FILTERS
        if spec["id"] in (
            {"NONE", "LOW_K12_B80_S2"}
            if freeze_screen
            else {"NONE", "LOW_K12_B80_S2", "BILAT_K12_Q50_B85_S2"}
        )
    )
    active_coordinates = COORDINATES if not (quick_screen or freeze_screen) else (
        ("NONE", 0.0), ("LINEAR", 0.6), ("POLY2", 1.2)
    )
    active_algorithms = ALGORITHMS if not (quick_screen or freeze_screen) else ("KMEANS", "GMM_DIAG")
    active_dimensions = (16,) if freeze_screen else (16, 32)
    matrix_cache = {}
    screen_rows = []
    for cluster_k in cluster_ks:
        for source in ("z1", "z2", "base_fused", "mcdf"):
            for spec in active_filters:
                for dimension in active_dimensions:
                    for coordinate_basis, coordinate_weight in active_coordinates:
                        basis = "LINEAR" if coordinate_basis == "NONE" else coordinate_basis
                        key = (source, spec["id"], dimension, coordinate_basis, coordinate_weight)
                        if key not in matrix_cache:
                            matrix_cache[key] = endpoint_matrix(
                                filtered[(source, spec["id"])],
                                views["coordinates"],
                                dimension,
                                basis,
                                coordinate_weight,
                            )
                        matrix = matrix_cache[key]
                        for algorithm in active_algorithms:
                            started = time.perf_counter()
                            partition = cluster_known_k(
                                matrix,
                                cluster_k,
                                algorithm,
                                0,
                                10 if algorithm == "KMEANS" else 2,
                            )
                            screen_rows.append(
                                {
                                    "stage": "SCREEN",
                                    "dataset": result_dataset,
                                    "candidate_id": audit["candidate_id"],
                                    "mechanism_family": audit["mechanism_family"],
                                    "model_seed": audit["seed"],
                                    "cluster_k": cluster_k,
                                    "source_view": source,
                                    "filter_id": spec["id"],
                                    "pca_dimension": dimension,
                                    "coordinate_basis": coordinate_basis,
                                    "coordinate_weight": coordinate_weight,
                                    "algorithm": algorithm,
                                    "endpoint_seed": 0,
                                    "refinement_id": "NONE",
                                    "known_k_unsupervised": True,
                                    "labels_in_model_or_cluster_fit": False,
                                    "label_policy": LABEL_POLICY,
                                    "matrix_sha256": array_sha256(matrix),
                                    "partition_sha256": array_sha256(partition),
                                    "status": "PASS",
                                    "wall_seconds": time.perf_counter() - started,
                                    **metric_row(payload, partition),
                                }
                            )
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(output / "screen_ledger.csv", index=False)
    shortlist = (
        screen.sort_values(
            ["cluster_k", "absolute_ari", "absolute_nmi"],
            ascending=[True, False, False],
        )
        .groupby("cluster_k", as_index=False)
        .head(12)
    )
    shortlist.to_csv(output / "screen_shortlist.csv", index=False)
    if screen_only:
        atomic_json(
            output / "head_search_manifest.json",
            {
                "dataset": dataset,
                "result_dataset": result_dataset,
                "protocol": protocol,
                "label_semantics": label_semantics,
                "candidate_id": audit["candidate_id"],
                "model_seed": audit["seed"],
                "screen_rows": len(screen),
                "robust_rows": 0,
                "screen_only": True,
                "quick_screen": bool(quick_screen),
                "freeze_screen": bool(freeze_screen),
                "labels_in_model_or_cluster_fit": False,
                "public_labels_used_for_cross_run_hpo_and_post_partition_evaluation": True,
                "dense_n_by_n_count": 0,
                "wall_seconds": time.perf_counter() - started_all,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024.0,
            },
        )
        return
    robust_rows = []
    for _, item in shortlist.iterrows():
        key = (
            item.source_view,
            item.filter_id,
            int(item.pca_dimension),
            item.coordinate_basis,
            float(item.coordinate_weight),
        )
        matrix = matrix_cache[key]
        refinements = [("NONE", None)]
        for graph_k in (12, 18, 24):
            operator = spatial_operator(views["coordinates"], graph_k)
            for anchor in (0.1, 0.5):
                for iterations in (5, 10):
                    refinements.append(
                        ("MAJ_K%d_A%.1f_I%d" % (graph_k, anchor, iterations), operator)
                    )
        for endpoint_seed in endpoint_seeds:
            initial = cluster_known_k(
                matrix,
                int(item.cluster_k),
                item.algorithm,
                int(endpoint_seed),
                20 if item.algorithm == "KMEANS" else 3,
            )
            for refinement_id, operator in refinements:
                if operator is None:
                    partition = initial
                else:
                    parts = refinement_id.split("_")
                    anchor = float(parts[2][1:])
                    iterations = int(parts[3][1:])
                    partition = anchored_majority_refine(
                        initial, operator, int(item.cluster_k), anchor, iterations
                    )
                robust_rows.append(
                    {
                        **{
                            key_name: item[key_name]
                            for key_name in (
                                "dataset",
                                "candidate_id",
                                "mechanism_family",
                                "model_seed",
                                "cluster_k",
                                "source_view",
                                "filter_id",
                                "pca_dimension",
                                "coordinate_basis",
                                "coordinate_weight",
                                "algorithm",
                            )
                        },
                        "stage": "ROBUST_ENDPOINT",
                        "endpoint_seed": int(endpoint_seed),
                        "refinement_id": refinement_id,
                        "known_k_unsupervised": True,
                        "labels_in_model_or_cluster_fit": False,
                        "label_policy": LABEL_POLICY,
                        "matrix_sha256": array_sha256(matrix),
                        "partition_sha256": array_sha256(partition),
                        "status": "PASS",
                        **metric_row(payload, partition),
                    }
                )
    robust = pd.DataFrame(robust_rows)
    robust.to_csv(output / "robust_endpoint_ledger.csv", index=False)
    summary = (
        robust.groupby(
            [
                "dataset",
                "candidate_id",
                "model_seed",
                "cluster_k",
                "source_view",
                "filter_id",
                "pca_dimension",
                "coordinate_basis",
                "coordinate_weight",
                "algorithm",
                "refinement_id",
            ],
            as_index=False,
        )
        .agg(
            run_count=("status", "size"),
            ari_best=("absolute_ari", "max"),
            ari_median=("absolute_ari", "median"),
            ari_mean=("absolute_ari", "mean"),
            ari_min=("absolute_ari", "min"),
            nmi_best=("absolute_nmi", "max"),
            nmi_median=("absolute_nmi", "median"),
            nmi_mean=("absolute_nmi", "mean"),
            nmi_min=("absolute_nmi", "min"),
        )
        .sort_values(
            ["cluster_k", "ari_median", "nmi_median", "ari_best"],
            ascending=[True, False, False, False],
        )
    )
    summary.to_csv(output / "robust_endpoint_summary.csv", index=False)
    atomic_json(
        output / "head_search_manifest.json",
        {
            "dataset": dataset,
            "result_dataset": result_dataset,
            "protocol": protocol,
            "label_semantics": label_semantics,
            "candidate_id": audit["candidate_id"],
            "model_seed": audit["seed"],
            "screen_rows": len(screen),
            "robust_rows": len(robust),
            "endpoint_seeds": list(endpoint_seeds),
            "quick_screen": bool(quick_screen),
            "freeze_screen": bool(freeze_screen),
            "labels_in_model_or_cluster_fit": False,
            "public_labels_used_for_cross_run_hpo_and_post_partition_evaluation": True,
            "dense_n_by_n_count": 0,
            "wall_seconds": time.perf_counter() - started_all,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3", "P22", "MISAR_E15_5_S1"),
        required=True,
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--protocol", choices=("PROJECT", "P22_3DOT_K18"), default="PROJECT"
    )
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--quick-screen", action="store_true")
    parser.add_argument("--freeze-screen", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(item) for item in args.endpoint_seeds.split(","))
    run(
        args.dataset,
        Path(args.run_dir),
        Path(args.output),
        seeds,
        args.protocol,
        args.screen_only,
        args.quick_screen,
        args.freeze_screen,
    )


if __name__ == "__main__":
    main()
