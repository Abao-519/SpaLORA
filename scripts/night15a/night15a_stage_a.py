#!/usr/bin/env python3
"""Matched score-source controls for Night-15A.

Every control shares K, label mask, endpoint seed, clustering algorithm and
refinement configuration.  Public labels enter only ``metric_row`` after a
partition has been produced.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/night13b"), str(REPO / "scripts/night14b")]

import night13b_run as n13b  # noqa: E402
import night14b_head_search as n14heads  # noqa: E402
import night14b_formal as n14formal  # noqa: E402
from SpaLORA.night14b_atac import anchored_majority_refine, spatial_operator  # noqa: E402
from SpaLORA.night15a_mcdf import (  # noqa: E402
    array_sha256,
    cluster_known_k,
    matched_control_matrix,
)


ROOT = Path("/root/autodl-fs/night15a_multimodal_contribution_and_score_stability_20260823")
N14A_RUN = Path(
    "/root/autodl-fs/night14a_topology_conflict_sprint_20260823/"
    "formal/development_cycle3/C15_BAL_XREC_600_WEAK_ALIGN"
)
N14B_FORMAL = Path(
    "/root/autodl-fs/night14b_atac_score_acceleration_20260823/formal_frozen"
)
LABEL_POLICY = "KNOWN_K_UNSUPERVISED_PUBLIC_LABELS_EVALUATOR_ONLY"

PRIMARY_FORMAL_IDS = (
    "F30_P22_K9_MAX_ARI",
    "F32_MISAR_K7_MAX_ARI",
    "F33_MISAR_K12_MAX_ARI",
)
BASE_CONTROLS = (
    "COORDINATE_ONLY",
    "RNA_ONLY",
    "ATAC_ONLY",
    "RNA_PLUS_COORDINATES",
    "ATAC_PLUS_COORDINATES",
    "FUSED_FULL",
    "FUSED_WITHOUT_COORDINATES",
    "FUSED_WITHOUT_GRAPH_FILTER",
    "FUSED_WITHOUT_SPATIAL_REFINEMENT",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def load_views(dataset: str, model_seed: int) -> Dict[str, np.ndarray]:
    path = N14A_RUN / dataset / ("seed_%d" % int(model_seed)) / "roundtrip_expected.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    archive = np.load(path, allow_pickle=False)
    return {
        "emb_latent_omics1": np.asarray(archive["z1"], dtype=np.float32),
        "emb_latent_omics2": np.asarray(archive["z2"], dtype=np.float32),
        "SpaLORA_fused": np.asarray(archive["fused"], dtype=np.float32),
    }


def apply_filter(
    config: Mapping[str, object],
    views: Mapping[str, np.ndarray],
    payload: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    spec = config["filter"]
    if spec["kind"] == "BILATERAL":
        operator, _ = n14heads.anisotropic_operator(
            views,
            np.asarray(payload["coordinates"]),
            int(spec["graph_k"]),
            float(spec["quantile"]),
        )
    else:
        operator = spatial_operator(np.asarray(payload["coordinates"]), int(spec["graph_k"]))
    return {
        key: n14heads.diffuse(value, operator, float(spec["beta"]), int(spec["steps"]))
        for key, value in views.items()
    }


def refine_if_requested(
    partition: np.ndarray,
    control_id: str,
    config: Mapping[str, object],
    payload: Mapping[str, object],
) -> np.ndarray:
    if control_id == "FUSED_WITHOUT_SPATIAL_REFINEMENT":
        return np.asarray(partition, dtype=np.int64)
    spec = config["refinement"]
    return anchored_majority_refine(
        np.asarray(partition, dtype=np.int64),
        spatial_operator(np.asarray(payload["coordinates"]), int(spec["graph_k"])),
        int(config["cluster_k"]),
        float(spec["anchor"]),
        int(spec["iterations"]),
    )


def metrics(payload: Mapping[str, object], partition: np.ndarray) -> dict:
    return n13b.partition_metrics(
        payload["labels"], payload["label_mask"], partition, payload["metric_graph"]
    )


def selected_configs():
    lookup = {item["formal_id"]: item for item in n14formal.FORMAL_CONFIGS}
    return [lookup[item] for item in PRIMARY_FORMAL_IDS]


def run(output: Path, endpoint_seeds) -> None:
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    start_all = time.perf_counter()
    source_formal = pd.read_csv(N14B_FORMAL / "formal_absolute_metrics.csv")
    replay_checks = []
    for config in selected_configs():
        dataset = str(config["dataset"])
        payload = n13b.base_payload(dataset)
        model_seeds = (0,) if False else (0, 1, 2)
        for model_seed in model_seeds:
            unfiltered = load_views(dataset, model_seed)
            filtered = apply_filter(config, unfiltered, payload)
            controls = list(BASE_CONTROLS)
            for control_id in controls:
                # Geometry is independent of a model checkpoint; register it once.
                if control_id == "COORDINATE_ONLY" and model_seed != 0:
                    continue
                matrix = matched_control_matrix(
                    control_id,
                    filtered,
                    unfiltered,
                    np.asarray(payload["coordinates"]),
                    str(config["representation"]),
                    int(config["pca_dimension"]),
                    str(config["coordinate_basis"]),
                    float(config["coordinate_weight"]),
                )
                for endpoint_seed in endpoint_seeds:
                    started = time.perf_counter()
                    partition = cluster_known_k(
                        matrix,
                        int(config["cluster_k"]),
                        str(config["algorithm"]),
                        int(endpoint_seed),
                        int(config["algorithm_n_init"]),
                    )
                    partition = refine_if_requested(partition, control_id, config, payload)
                    row = {
                        "formal_id": config["formal_id"],
                        "dataset": dataset,
                        "cluster_k": int(config["cluster_k"]),
                        "control_id": control_id,
                        "model_seed": np.nan if control_id == "COORDINATE_ONLY" else model_seed,
                        "endpoint_seed": int(endpoint_seed),
                        "status": "PASS",
                        "known_k_unsupervised": True,
                        "labels_in_model_or_fit": False,
                        "label_policy": LABEL_POLICY,
                        "matrix_shape": "x".join(map(str, matrix.shape)),
                        "matrix_sha256": array_sha256(matrix),
                        "partition_sha256": array_sha256(partition),
                        "wall_seconds": time.perf_counter() - started,
                        "gpu_seconds": 0.0,
                        "peak_gpu_mib": 0.0,
                        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                        **metrics(payload, partition),
                    }
                    rows.append(row)
                    if control_id == "FUSED_FULL":
                        alias = dict(row)
                        alias["control_id"] = "NIGHT14B_FROZEN_BEST"
                        alias["reused_exact_alias_of"] = "FUSED_FULL"
                        alias["wall_seconds"] = 0.0
                        rows.append(alias)
                        if endpoint_seed in (0, 1, 2):
                            matched = source_formal[
                                (source_formal.formal_id == config["formal_id"])
                                & (source_formal.model_seed == model_seed)
                                & (source_formal.endpoint_seed == endpoint_seed)
                            ]
                            if len(matched) == 1:
                                expected = matched.iloc[0]
                                replay_checks.append(
                                    {
                                        "formal_id": config["formal_id"],
                                        "model_seed": model_seed,
                                        "endpoint_seed": endpoint_seed,
                                        "partition_sha_exact": bool(
                                            row["partition_sha256"]
                                            == expected.partition_sha256
                                        ),
                                        "ari_abs_error": float(
                                            abs(row["absolute_ari"] - expected.absolute_ari)
                                        ),
                                        "nmi_abs_error": float(
                                            abs(row["absolute_nmi"] - expected.absolute_nmi)
                                        ),
                                    }
                                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "score_source_ablation.csv", index=False)
    summary = (
        frame.groupby(["formal_id", "dataset", "cluster_k", "control_id"], as_index=False)
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
            ami_mean=("ami", "mean"),
            fmi_mean=("fmi", "mean"),
            morans_i_mean=("morans_i", "mean"),
            gearys_c_mean=("gearys_c", "mean"),
            wall_seconds=("wall_seconds", "sum"),
            peak_rss_mib=("peak_rss_mib", "max"),
        )
    )
    summary.to_csv(output / "score_source_summary.csv", index=False)
    deltas = []
    for config in selected_configs():
        lane = frame[frame.formal_id == config["formal_id"]]
        for model_seed in (0, 1, 2):
            for endpoint_seed in endpoint_seeds:
                get = lambda control: lane[
                    (lane.control_id == control)
                    & (lane.endpoint_seed == endpoint_seed)
                    & (
                        lane.model_seed.isna()
                        if control == "COORDINATE_ONLY"
                        else (lane.model_seed == model_seed)
                    )
                ].iloc[0]
                fused = get("FUSED_FULL")
                coordinate = get("COORDINATE_ONLY")
                rna = get("RNA_PLUS_COORDINATES")
                atac = get("ATAC_PLUS_COORDINATES")
                no_coord = get("FUSED_WITHOUT_COORDINATES")
                deltas.append(
                    {
                        "formal_id": config["formal_id"],
                        "dataset": config["dataset"],
                        "cluster_k": config["cluster_k"],
                        "model_seed": model_seed,
                        "endpoint_seed": endpoint_seed,
                        "fused_minus_coordinate_ari": fused.absolute_ari - coordinate.absolute_ari,
                        "fused_minus_coordinate_nmi": fused.absolute_nmi - coordinate.absolute_nmi,
                        "fused_minus_best_unimodal_ari": fused.absolute_ari
                        - max(rna.absolute_ari, atac.absolute_ari),
                        "fused_minus_best_unimodal_nmi": fused.absolute_nmi
                        - max(rna.absolute_nmi, atac.absolute_nmi),
                        "with_minus_without_coordinate_ari": fused.absolute_ari
                        - no_coord.absolute_ari,
                        "with_minus_without_coordinate_nmi": fused.absolute_nmi
                        - no_coord.absolute_nmi,
                    }
                )
    pd.DataFrame(deltas).to_csv(output / "matched_contribution_deltas.csv", index=False)
    atomic_json(
        output / "stage_a_manifest.json",
        {
            "row_count": len(frame),
            "pass_count": int((frame.status == "PASS").sum()),
            "failure_count": int((frame.status != "PASS").sum()),
            "endpoint_seeds": list(endpoint_seeds),
            "model_seeds": [0, 1, 2],
            "coordinate_only_registered_once_per_endpoint_seed": True,
            "night14b_best_alias_reuses_fused_partition": True,
            "night14b_replay_checks": replay_checks,
            "night14b_partition_replay_all_exact": bool(replay_checks)
            and all(item["partition_sha_exact"] for item in replay_checks),
            "night14b_metric_replay_tolerance": 1e-12,
            "night14b_metric_replay_all_within_tolerance": bool(replay_checks)
            and all(
                item["ari_abs_error"] <= 1e-12 and item["nmi_abs_error"] <= 1e-12
                for item in replay_checks
            ),
            "labels_in_model_or_clustering_fit": False,
            "public_labels_in_post_partition_evaluator": True,
            "wall_seconds": time.perf_counter() - start_all,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint-seeds", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()
    seeds = tuple(int(item) for item in args.endpoint_seeds.split(","))
    run(Path(args.output), seeds)


if __name__ == "__main__":
    main()
