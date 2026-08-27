"""Build held-out-safe sparse edge features; this process never opens a teacher or label file."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from SpaLORA.night23a_xbed import array_sha, build_union_edge_features, file_sha, load_csr


ALLOW = {
    "ids", "view1", "view2", "retained",
    "graph0__data", "graph0__indices", "graph0__indptr", "graph0__shape",
    "graph1__data", "graph1__indices", "graph1__indptr", "graph1__shape",
    "graph2__data", "graph2__indices", "graph2__indptr", "graph2__shape",
    "start_ids", "start_partitions",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    spec = contract["lanes"][args.lane]
    carrier_path = Path(spec["carrier"])
    with np.load(carrier_path, allow_pickle=False) as carrier:
        discovered = sorted(carrier.files)
        unknown = sorted(set(discovered) - ALLOW)
        if unknown:
            raise RuntimeError(f"unknown carrier arrays: {unknown}")
        ids = np.asarray(carrier["ids"])
        view1 = np.asarray(carrier["view1"], dtype=np.float32)
        view2 = np.asarray(carrier["view2"], dtype=np.float32)
        retained = np.asarray(carrier["retained"], dtype=np.float32)
        spatial = load_csr(carrier, "graph0")
    result = build_union_edge_features(
        view1, view2, retained, spatial, k=int(contract["edge_feature_contract"]["knn_k"])
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=ids,
        rows=result["rows"],
        cols=result["cols"],
        features=result["features"],
        feature_names=result["feature_names"],
        spatial_score=result["spatial_score"],
        intersection_score=result["intersection_score"],
    )
    manifest = {
        "schema": "night23a-edge-features-v1",
        "lane": args.lane,
        "role": spec["role"],
        "k": spec["k"],
        "carrier_path": str(carrier_path),
        "carrier_sha256": file_sha(carrier_path),
        "contract_sha256": file_sha(contract_path),
        "discovered_carrier_keys": discovered,
        "accessed_carrier_keys": ["ids", "view1", "view2", "retained", "graph0__*"],
        "annotation_or_teacher_arrays_accessed": 0,
        "ids_shape": list(ids.shape),
        "view1_shape": list(view1.shape),
        "view2_shape": list(view2.shape),
        "retained_shape": list(retained.shape),
        "spatial_shape": list(spatial.shape),
        "spatial_nnz": int(spatial.nnz),
        "union_edge_count": result["edge_count"],
        "component_edge_counts": result["component_edge_counts"],
        "feature_shape": list(result["features"].shape),
        "feature_names": result["feature_names"].tolist(),
        "ids_sha256": array_sha(ids),
        "rows_sha256": array_sha(result["rows"]),
        "cols_sha256": array_sha(result["cols"]),
        "features_sha256": array_sha(result["features"]),
        "artifact_sha256": file_sha(output),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "labels_read": 0,
        "teacher_files_read": 0,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
