#!/usr/bin/env python3
"""Evaluate one frozen endpoint configuration on an unseen backbone seed."""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/night13b"), str(REPO / "scripts/night14b")]

import night13b_run as n13b  # noqa: E402
from SpaLORA.night14b_atac import anchored_majority_refine, spatial_operator  # noqa: E402
from SpaLORA.night15a_mcdf import array_sha256, cluster_known_k  # noqa: E402
import night15a_head_search as search  # noqa: E402


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def run(dataset: str, run_dir: Path, config_path: Path, output: Path, endpoint_seeds: tuple[int, ...]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started_all = time.perf_counter()
    audit, views = search.load_run(run_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["dataset"] != dataset:
        raise RuntimeError("frozen endpoint dataset mismatch")
    payload = n13b.base_payload(dataset)
    if not np.array_equal(np.asarray(payload["ids"], dtype=str), views["ids"]):
        raise RuntimeError("frozen endpoint observation order mismatch")
    cache = search.source_filter_cache(views)
    matrix = search.endpoint_matrix(
        cache[(config["source_view"], config["filter_id"])],
        views["coordinates"],
        int(config["pca_dimension"]),
        config["coordinate_basis"],
        float(config["coordinate_weight"]),
    )
    operator = None
    refinement = config.get("refinement_id", "NONE")
    if refinement != "NONE":
        parts = refinement.split("_")
        operator = spatial_operator(views["coordinates"], int(parts[1][1:]))
        anchor = float(parts[2][1:])
        iterations = int(parts[3][1:])
    rows = []
    partitions = []
    for endpoint_seed in endpoint_seeds:
        started = time.perf_counter()
        initial = cluster_known_k(
            matrix,
            int(config["cluster_k"]),
            config["algorithm"],
            int(endpoint_seed),
            20 if config["algorithm"] == "KMEANS" else 3,
        )
        partition = initial if operator is None else anchored_majority_refine(
            initial, operator, int(config["cluster_k"]), anchor, iterations
        )
        partitions.append(partition)
        rows.append({
            "dataset": dataset,
            "candidate_id": audit["candidate_id"],
            "model_seed": int(audit["seed"]),
            "endpoint_seed": int(endpoint_seed),
            "endpoint_role": "BASE_RANDOM_START",
            "medoid_source_endpoint_seed": np.nan,
            "cluster_k": int(config["cluster_k"]),
            "source_view": config["source_view"],
            "filter_id": config["filter_id"],
            "pca_dimension": int(config["pca_dimension"]),
            "coordinate_basis": config["coordinate_basis"],
            "coordinate_weight": float(config["coordinate_weight"]),
            "algorithm": config["algorithm"],
            "refinement_id": refinement,
            "known_k_unsupervised": True,
            "labels_in_model_or_cluster_fit": False,
            "partition_sha256": array_sha256(partition),
            "matrix_sha256": array_sha256(matrix),
            "status": "PASS",
            "wall_seconds": time.perf_counter() - started,
            **n13b.partition_metrics(
                payload["labels"], payload["label_mask"], partition, payload["metric_graph"]
            ),
        })
    # The medoid is selected only from agreement among candidate partitions. Public
    # labels are applied afterward, exactly as for each individual endpoint seed.
    agreement = np.eye(len(partitions), dtype=np.float64)
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            value = adjusted_rand_score(partitions[left], partitions[right])
            agreement[left, right] = agreement[right, left] = value
    medoid_index = int(np.argmax(agreement.mean(axis=1)))
    medoid_partition = partitions[medoid_index]
    template = rows[medoid_index].copy()
    template.update({
        "endpoint_seed": -1,
        "endpoint_role": "LABEL_FREE_PARTITION_MEDOID",
        "medoid_source_endpoint_seed": int(endpoint_seeds[medoid_index]),
        "partition_sha256": array_sha256(medoid_partition),
        "wall_seconds": 0.0,
        **n13b.partition_metrics(
            payload["labels"], payload["label_mask"], medoid_partition, payload["metric_graph"]
        ),
    })
    rows.append(template)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "frozen_endpoint_ledger.csv", index=False)
    atomic_json(output / "frozen_endpoint_manifest.json", {
        "dataset": dataset,
        "candidate_id": audit["candidate_id"],
        "model_seed": int(audit["seed"]),
        "frozen_config_path": str(config_path),
        "frozen_config": config,
        "endpoint_seeds": list(endpoint_seeds),
        "rows": len(frame),
        "base_endpoint_rows": len(endpoint_seeds),
        "label_free_partition_medoid_rows": 1,
        "partition_agreement_matrix_shape": list(agreement.shape),
        "labels_in_model_or_cluster_fit": False,
        "public_labels_used_after_partition_only": True,
        "dense_n_by_n_count": 0,
        "wall_seconds": time.perf_counter() - started_all,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("P22", "MISAR_E15_5_S1"), required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint-seeds", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()
    run(
        args.dataset,
        Path(args.run_dir),
        Path(args.config),
        Path(args.output),
        tuple(int(value) for value in args.endpoint_seeds.split(",")),
    )


if __name__ == "__main__":
    main()
