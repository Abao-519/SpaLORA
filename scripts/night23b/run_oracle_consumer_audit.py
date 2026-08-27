"""Read locked Night-23A authorities and measure oracle consumer identifiability without benchmark labels."""
from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path

import numpy as np

from SpaLORA.night23a_xbed import file_sha
from SpaLORA.night23b_signed_bridge import (
    array_sha,
    carrier_preserving_signed_partition,
    positive_connectivity_audit,
    recovery_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--teacher-manifest", required=True)
    parser.add_argument("--teacher-partition", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-bank", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    contract = json.loads(Path(args.contract).read_text())
    lane_contract = contract["lanes"][args.lane]
    feature_manifest = json.loads(Path(args.feature_manifest).read_text())
    teacher_manifest = json.loads(Path(args.teacher_manifest).read_text())
    if file_sha(args.feature) != feature_manifest["artifact_sha256"]:
        raise RuntimeError("feature authority mismatch")
    if file_sha(args.teacher) != teacher_manifest["artifact_sha256"]:
        raise RuntimeError("teacher relation authority mismatch")
    if file_sha(args.teacher_partition) != teacher_manifest["teacher_file_sha256"]:
        raise RuntimeError("teacher partition file authority mismatch")
    with np.load(args.feature, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"]).astype("U")
        rows = np.asarray(archive["rows"], dtype=np.int32)
        cols = np.asarray(archive["cols"], dtype=np.int32)
        features = np.asarray(archive["features"], dtype=np.float32)
    with np.load(args.teacher, allow_pickle=False) as archive:
        relation = np.asarray(archive["relation"], dtype=np.uint8)
    with np.load(args.teacher_partition, allow_pickle=False) as archive:
        teacher_ids = np.asarray(archive["ids"]).astype("U")
        teacher_partition = np.asarray(archive["partition"], dtype=np.int32)
        teacher_candidate_id = str(np.asarray(archive["selected_candidate_id"]).item())
    with np.load(args.carrier, allow_pickle=False) as archive:
        carrier_ids = np.asarray(archive["ids"]).astype("U")
        retained = np.asarray(archive["retained"], dtype=np.float64)
    if (not np.array_equal(ids, carrier_ids) or not np.array_equal(ids, teacher_ids)
            or len(relation) != len(rows) or len(teacher_partition) != len(ids)):
        raise RuntimeError("oracle authority/shape mismatch")
    if array_sha(relation) != teacher_manifest["relation_sha256"]:
        raise RuntimeError("teacher relation array mismatch")
    if array_sha(teacher_partition) != teacher_manifest["teacher_partition_sha256"]:
        raise RuntimeError("teacher partition array mismatch")
    if teacher_candidate_id != teacher_manifest["teacher_candidate_id"]:
        raise RuntimeError("teacher candidate authority mismatch")
    k = int(lane_contract["k"])
    connectivity = positive_connectivity_audit(len(ids), rows, cols, relation, teacher_partition)
    spatial_index = feature_manifest["feature_names"].index("registered_spatial_edge")
    spatial_mask = features[:, spatial_index] > 0
    rows_out = []
    candidate_ids = []
    partitions = []
    zero = np.zeros(len(rows), dtype=np.float64)
    one_positive = relation.astype(np.float64)
    one_negative = (1 - relation).astype(np.float64)
    for mode, scale in [("CARRIER_ONLY", 0.0)]:
        partition, fused = carrier_preserving_signed_partition(retained, rows, cols, zero, zero, k, scale)
        candidate_id = mode
        candidate_ids.append(candidate_id); partitions.append(partition)
        rows_out.append({"lane": args.lane, "role": lane_contract["role"], "candidate_id": candidate_id, "mode": mode, "relation_scale": scale, **recovery_metrics(teacher_partition, partition, rows, cols, relation), "partition_sha256": array_sha(partition), "representation_sha256": array_sha(fused)})
    for scale in contract["consumer"]["relation_scale_candidates"]:
        for mode, negative in (("ORACLE_BINARY_POSITIVE", zero), ("ORACLE_SIGNED_RELATION", one_negative)):
            partition, fused = carrier_preserving_signed_partition(retained, rows, cols, one_positive, negative, k, float(scale))
            candidate_id = f"{mode}__S{float(scale):g}"
            candidate_ids.append(candidate_id); partitions.append(partition)
            rows_out.append({"lane": args.lane, "role": lane_contract["role"], "candidate_id": candidate_id, "mode": mode, "relation_scale": scale, **recovery_metrics(teacher_partition, partition, rows, cols, relation), "partition_sha256": array_sha(partition), "representation_sha256": array_sha(fused)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader(); writer.writerows(rows_out)
    output_bank = Path(args.output_bank)
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    partition_matrix = np.stack(partitions).astype(np.int32, copy=False)
    np.savez_compressed(output_bank, ids=ids, candidate_ids=np.asarray(candidate_ids, dtype="U"), partitions=partition_matrix)
    audit = {
        "schema": "night23b-oracle-consumer-audit-v1",
        "lane": args.lane,
        "k": k,
        "n": len(ids),
        "union_edge_count": len(rows),
        "teacher_same_edge_count": int(relation.sum()),
        "teacher_boundary_edge_count": int((1 - relation).sum()),
        "registered_spatial_edge_count": int(spatial_mask.sum()),
        "registered_spatial_same_count": int(np.sum(relation[spatial_mask] == 1)),
        "registered_spatial_boundary_count": int(np.sum(relation[spatial_mask] == 0)),
        "registered_spatial_coverage_by_union": 1.0,
        "connectivity": connectivity,
        "feature_sha256": file_sha(args.feature),
        "teacher_relation_sha256": file_sha(args.teacher),
        "teacher_partition_file_sha256": file_sha(args.teacher_partition),
        "teacher_partition_sha256": teacher_manifest["teacher_partition_sha256"],
        "teacher_candidate_id": teacher_candidate_id,
        "carrier_sha256": file_sha(args.carrier),
        "contract_sha256": file_sha(args.contract),
        "output_bank_sha256": file_sha(output_bank),
        "partition_matrix_sha256": array_sha(partition_matrix),
        "benchmark_reference_labels_read": 0,
        "oracle_is_diagnostic_only": True,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    output.with_suffix(".json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"lane": args.lane, "rows": len(rows_out), "connectivity": connectivity, "wall_seconds": audit["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
