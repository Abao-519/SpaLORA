#!/usr/bin/env python3
"""Frozen multi-seed replay for the Night-14B leading head configurations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/night13b"))
sys.path.insert(0, str(REPO / "scripts/night14b"))

import night13b_run as n13b  # noqa: E402
import night14b_head_search as heads  # noqa: E402
from SpaLORA.night14b_atac import (  # noqa: E402
    anchored_majority_refine, array_sha256, diffuse, spatial_operator,
)
from night14b_run import LABEL_POLICY, N14A_RUN, atomic_json, file_sha256, metric_row  # noqa: E402


FORMAL_CONFIGS = [
    {
        "formal_id": "F30_P22_K9_MAX_ARI", "dataset": "P22", "cluster_k": 9,
        "filter": {"kind": "BILATERAL", "graph_k": 18, "quantile": .50,
                   "beta": .85, "steps": 2},
        "representation": "FUSED", "pca_dimension": 32,
        "coordinate_basis": "LINEAR", "coordinate_weight": 1.20,
        "algorithm": "KMEANS", "algorithm_n_init": 20,
        "refinement": {"graph_k": 18, "anchor": .10, "iterations": 5},
    },
    {
        "formal_id": "F31_P22_K9_HIGH_NMI", "dataset": "P22", "cluster_k": 9,
        "filter": {"kind": "BILATERAL", "graph_k": 18, "quantile": .50,
                   "beta": .85, "steps": 2},
        "representation": "FUSED", "pca_dimension": 32,
        "coordinate_basis": "LINEAR", "coordinate_weight": 1.20,
        "algorithm": "GMM_DIAG", "algorithm_n_init": 3,
        "refinement": {"graph_k": 24, "anchor": .10, "iterations": 20},
    },
    {
        "formal_id": "F32_MISAR_K7_MAX_ARI", "dataset": "MISAR_E15_5_S1",
        "cluster_k": 7,
        "filter": {"kind": "FIXED_LOW", "graph_k": 12, "beta": .90,
                   "steps": 3},
        "representation": "EQUAL3", "pca_dimension": 32,
        "coordinate_basis": "POLY2", "coordinate_weight": 1.20,
        "algorithm": "GMM_DIAG", "algorithm_n_init": 3,
        "refinement": {"graph_k": 24, "anchor": .10, "iterations": 10},
    },
    {
        "formal_id": "F33_MISAR_K12_MAX_ARI", "dataset": "MISAR_E15_5_S1",
        "cluster_k": 12,
        "filter": {"kind": "FIXED_LOW", "graph_k": 18, "beta": .95,
                   "steps": 5},
        "representation": "PRIVATE2", "pca_dimension": 16,
        "coordinate_basis": "LINEAR", "coordinate_weight": 1.20,
        "algorithm": "GMM_SPHERICAL", "algorithm_n_init": 3,
        "refinement": {"graph_k": 12, "anchor": .00, "iterations": 10},
    },
]


def canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def load_views(dataset, model_seed):
    path = N14A_RUN / dataset / f"seed_{model_seed}" / "roundtrip_expected.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    value = np.load(path, allow_pickle=False)
    return {
        "emb_latent_omics1": np.asarray(value["z1"], dtype=np.float32),
        "emb_latent_omics2": np.asarray(value["z2"], dtype=np.float32),
        "SpaLORA_fused": np.asarray(value["fused"], dtype=np.float32),
    }, path


def apply_frozen_filter(config, views, payload):
    spec = config["filter"]
    if spec["kind"] == "BILATERAL":
        operator, _ = heads.anisotropic_operator(
            views, payload["coordinates"], int(spec["graph_k"]),
            float(spec["quantile"]),
        )
    else:
        operator = spatial_operator(payload["coordinates"], int(spec["graph_k"]))
    return {
        key: diffuse(value, operator, float(spec["beta"]), int(spec["steps"]))
        for key, value in views.items()
    }


def cluster(value, config, seed):
    algorithm = config["algorithm"]
    if algorithm == "KMEANS":
        return KMeans(
            int(config["cluster_k"]), random_state=seed,
            n_init=int(config["algorithm_n_init"]),
        ).fit_predict(value).astype(np.int64)
    covariance = algorithm.split("_", 1)[1].lower()
    return GaussianMixture(
        n_components=int(config["cluster_k"]), covariance_type=covariance,
        random_state=seed, n_init=int(config["algorithm_n_init"]),
        max_iter=300, reg_covar=1e-5,
    ).fit_predict(value).astype(np.int64)


def run_one(config, model_seed, endpoint_seed, payload=None):
    if payload is None:
        payload = n13b.base_payload(config["dataset"])
    views, source = load_views(config["dataset"], model_seed)
    filtered = apply_frozen_filter(config, views, payload)
    molecular = heads.molecular_matrix(
        filtered, config["representation"], int(config["pca_dimension"])
    )
    value = heads.augmented_matrix(
        molecular, payload["coordinates"], config["coordinate_basis"],
        float(config["coordinate_weight"]),
    )
    partition = cluster(value, config, endpoint_seed)
    refinement = config["refinement"]
    partition = anchored_majority_refine(
        partition,
        spatial_operator(payload["coordinates"], int(refinement["graph_k"])),
        int(config["cluster_k"]), float(refinement["anchor"]),
        int(refinement["iterations"]),
    )
    return partition, filtered, source


def formal(output):
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "frozen_formal_configs.json", {
        "config_count": len(FORMAL_CONFIGS), "configs": FORMAL_CONFIGS,
        "config_sha256": canonical_sha(FORMAL_CONFIGS),
        "model_seeds": [0, 1, 2], "endpoint_seeds": [0, 1, 2],
        "selection_source": "Night-14B stage3 and stage4 public-benchmark HPO",
        "labels_in_fit_gradient_or_checkpoint": False,
    })
    rows = []
    partitions = {}
    started_all = time.perf_counter()
    for config in FORMAL_CONFIGS:
        payload = n13b.base_payload(config["dataset"])
        for model_seed in (0, 1, 2):
            for endpoint_seed in (0, 1, 2):
                started = time.perf_counter()
                partition, filtered, source = run_one(
                    config, model_seed, endpoint_seed, payload
                )
                key = f"{config['formal_id']}__m{model_seed}__e{endpoint_seed}"
                partitions[key] = np.asarray(partition, dtype=np.int64)
                rows.append({
                    "formal_id": config["formal_id"], "dataset": config["dataset"],
                    "cluster_k": int(config["cluster_k"]),
                    "model_seed": model_seed, "endpoint_seed": endpoint_seed,
                    "status": "PASS", "label_policy": LABEL_POLICY,
                    "labels_in_fit_gradient_or_checkpoint": False,
                    "config_sha256": canonical_sha(config),
                    "source_view_path": str(source),
                    "source_view_sha256": file_sha256(source),
                    "embedding_sha256": array_sha256(filtered["SpaLORA_fused"]),
                    "partition_sha256": array_sha256(partitions[key]),
                    "wall_seconds": time.perf_counter() - started,
                    **metric_row(payload, partitions[key]),
                })
    np.savez_compressed(output / "formal_partitions.npz", **partitions)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "formal_absolute_metrics.csv", index=False)
    summary = frame.groupby(
        ["formal_id", "dataset", "cluster_k"], as_index=False
    ).agg(
        run_count=("status", "size"),
        ari_best=("absolute_ari", "max"), ari_median=("absolute_ari", "median"),
        ari_mean=("absolute_ari", "mean"), ari_min=("absolute_ari", "min"),
        nmi_best=("absolute_nmi", "max"), nmi_median=("absolute_nmi", "median"),
        nmi_mean=("absolute_nmi", "mean"), nmi_min=("absolute_nmi", "min"),
        wall_seconds=("wall_seconds", "sum"),
    )
    summary.to_csv(output / "formal_seed_summary.csv", index=False)
    subprocess.run([
        sys.executable, str(Path(__file__).resolve()), "replay",
        "--output", str(output),
    ], cwd=str(REPO), check=True)
    atomic_json(output / "formal_manifest.json", {
        "row_count": len(frame), "pass_count": int((frame.status == "PASS").sum()),
        "failure_count": int((frame.status == "FAIL").sum()),
        "wall_seconds": time.perf_counter() - started_all,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "partition_bundle_sha256": file_sha256(output / "formal_partitions.npz"),
        "fresh_process_replay": json.loads(
            (output / "fresh_process_replay.json").read_text()
        ),
    })


def replay(output):
    expected = np.load(output / "formal_partitions.npz", allow_pickle=False)
    rows = []
    for config in FORMAL_CONFIGS:
        payload = n13b.base_payload(config["dataset"])
        for model_seed in (0, 1, 2):
            for endpoint_seed in (0, 1, 2):
                key = f"{config['formal_id']}__m{model_seed}__e{endpoint_seed}"
                partition, _, _ = run_one(config, model_seed, endpoint_seed, payload)
                rows.append({
                    "key": key, "partition_exact": bool(np.array_equal(
                        partition.astype(np.int64), expected[key]
                    )),
                    "actual_sha256": array_sha256(partition.astype(np.int64)),
                    "expected_sha256": array_sha256(expected[key]),
                })
    atomic_json(output / "fresh_process_replay.json", {
        "fresh_process": True, "row_count": len(rows),
        "exact_count": sum(row["partition_exact"] for row in rows),
        "all_exact": all(row["partition_exact"] for row in rows),
        "rows": rows,
    })


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("formal", "replay"):
        item = sub.add_parser(name)
        item.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "formal":
        formal(Path(args.output))
    else:
        replay(Path(args.output))


if __name__ == "__main__":
    main()
