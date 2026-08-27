"""Generate locked Stage-B sparse exact-K partitions without opening references or teachers."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from SpaLORA.night23a_xbed import (
    STAGE_B_ARMS,
    array_sha,
    file_sha,
    load_csr,
    spectral_exact_k_partition,
    stage_b_arm_weights,
    validate_teacher,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--role", choices=("PRIMARY", "SECONDARY"), required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--core-source", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    feature_path, prediction_path, carrier_path = map(Path, (args.features, args.prediction, args.carrier))
    prediction_manifest = json.loads(Path(args.prediction_manifest).read_text(encoding="utf-8"))
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    if args.lane not in contract["lanes"] or int(contract["lanes"][args.lane]["k"]) != args.k:
        raise RuntimeError("lane/K contract mismatch")
    if file_sha(prediction_path) != prediction_manifest["prediction_artifact_sha256"]:
        raise RuntimeError("locked prediction artifact mismatch")
    if file_sha(feature_path) != prediction_manifest["heldout_feature_sha256"]:
        raise RuntimeError("locked feature artifact mismatch")
    with np.load(feature_path, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"]).astype("U")
        rows = np.asarray(archive["rows"], dtype=np.int32)
        cols = np.asarray(archive["cols"], dtype=np.int32)
        features = np.asarray(archive["features"], dtype=np.float32)
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {key: np.asarray(archive[key]) for key in archive.files}
        if not np.array_equal(ids, np.asarray(archive["ids"]).astype("U")):
            raise RuntimeError("feature/prediction ID mismatch")
        if not np.array_equal(rows, archive["rows"]) or not np.array_equal(cols, archive["cols"]):
            raise RuntimeError("feature/prediction edge mismatch")
    with np.load(carrier_path, allow_pickle=False) as archive:
        carrier_ids = np.asarray(archive["ids"]).astype("U")
        spatial = load_csr(archive, "graph0")
    if not np.array_equal(ids, carrier_ids):
        raise RuntimeError("feature/carrier ID mismatch")
    arm_weights = stage_b_arm_weights(features, prediction)
    partitions, ledger = [], []
    for arm in STAGE_B_ARMS:
        arm_started = time.perf_counter()
        weights = arm_weights[arm]
        partition = spectral_exact_k_partition(len(ids), rows, cols, weights, args.k, seed=23)
        structure = validate_teacher(partition, args.k, spatial)
        if not structure["exact_k"]:
            raise RuntimeError(f"exact-K failure: {arm}")
        partitions.append(partition)
        ledger.append(
            {
                "candidate_id": arm,
                "partition_index": len(partitions) - 1,
                "partition_sha256": array_sha(partition),
                "observed_k": structure["observed_k"],
                "cluster_sizes_full": structure["cluster_sizes"],
                "min_cluster_size_full": structure["min_cluster_size"],
                "internal_spatial_edges_per_cluster": structure["internal_spatial_edges_per_cluster"],
                "all_clusters_have_internal_spatial_edge": structure["all_clusters_have_internal_spatial_edge"],
                "edge_weight_sum": float(weights.sum()),
                "edge_weight_nonzero": int(np.count_nonzero(weights)),
                "edge_weight_min": float(weights.min()),
                "edge_weight_max": float(weights.max()),
                "wall_seconds": time.perf_counter() - arm_started,
            }
        )
    partition_matrix = np.stack(partitions).astype(np.int32)
    output_bank = Path(args.output_bank)
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_bank,
        ids=ids,
        candidate_ids=np.asarray(STAGE_B_ARMS, dtype="U"),
        partitions=partition_matrix,
        rows=rows,
        cols=cols,
    )
    manifest = {
        "schema": "night23a-stage-b-partitions-v1",
        "lane": args.lane,
        "role": args.role,
        "k": args.k,
        "n": len(ids),
        "edge_count": len(rows),
        "candidate_count": len(ledger),
        "candidate_ids": list(STAGE_B_ARMS),
        "rows": ledger,
        "bank_sha256": file_sha(output_bank),
        "partitions_sha256": array_sha(partition_matrix),
        "ids_sha256": array_sha(ids),
        "rows_sha256": array_sha(rows),
        "cols_sha256": array_sha(cols),
        "feature_sha256": file_sha(feature_path),
        "prediction_sha256": file_sha(prediction_path),
        "prediction_checkpoint_sha256": prediction_manifest["checkpoint_sha256"],
        "carrier_sha256": file_sha(carrier_path),
        "contract_sha256": file_sha(args.contract),
        "core_source_sha256": file_sha(args.core_source),
        "labels_read": 0,
        "teacher_files_read": 0,
        "heldout_teacher_generated": False,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    Path(args.output_manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("lane", "candidate_count", "bank_sha256", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
