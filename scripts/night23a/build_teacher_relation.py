"""Materialize permutation-invariant teacher edge relations separately from numeric features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from SpaLORA.night23a_xbed import array_sha, file_sha, load_csr, teacher_relations, validate_teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    spec = contract["lanes"][args.lane]
    feature_path = Path(args.features)
    feature_manifest = json.loads(feature_path.with_suffix(".json").read_text(encoding="utf-8"))
    if file_sha(feature_path) != feature_manifest["artifact_sha256"] or feature_manifest["teacher_files_read"] != 0:
        raise RuntimeError("feature authority failure")
    with np.load(feature_path, allow_pickle=False) as feature:
        ids = np.asarray(feature["ids"])
        rows = np.asarray(feature["rows"], dtype=np.int32)
        cols = np.asarray(feature["cols"], dtype=np.int32)
    teacher_path = Path(spec["teacher"])
    with np.load(teacher_path, allow_pickle=False) as teacher:
        if sorted(teacher.files) != ["ids", "partition", "selected_candidate_id"]:
            raise RuntimeError("unexpected Night16H teacher schema")
        teacher_ids = np.asarray(teacher["ids"])
        partition = np.asarray(teacher["partition"], dtype=np.int32)
        candidate_id = str(np.asarray(teacher["selected_candidate_id"]).item())
    if not np.array_equal(ids.astype(str), teacher_ids.astype(str)):
        raise RuntimeError("teacher/feature ordered ID mismatch")
    with np.load(spec["carrier"], allow_pickle=False) as carrier:
        spatial = load_csr(carrier, "graph0")
    relation = teacher_relations(partition, rows, cols)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, relation=relation)
    manifest = {
        "schema": "night23a-teacher-edge-relation-v1",
        "lane": args.lane,
        "teacher_path": str(teacher_path),
        "teacher_file_sha256": file_sha(teacher_path),
        "teacher_candidate_id": candidate_id,
        "teacher_partition_sha256": array_sha(partition),
        "teacher_validation": validate_teacher(partition, spec["k"], spatial),
        "feature_artifact_sha256": file_sha(feature_path),
        "feature_ids_sha256": array_sha(ids),
        "edge_rows_sha256": array_sha(rows),
        "edge_cols_sha256": array_sha(cols),
        "relation_sha256": array_sha(relation),
        "relation_count": int(len(relation)),
        "same_edge_count": int(relation.sum()),
        "boundary_edge_count": int(len(relation) - relation.sum()),
        "same_edge_fraction": float(relation.mean()),
        "teacher_cluster_ids_exported_to_model": False,
        "benchmark_reference_labels_read": 0,
        "artifact_sha256": file_sha(output),
    }
    if not all([
        manifest["teacher_validation"]["exact_k"],
        manifest["teacher_validation"]["no_singleton"],
        manifest["teacher_validation"]["all_clusters_have_internal_spatial_edge"],
    ]):
        raise RuntimeError("Night16H teacher structural authority failed")
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
