#!/usr/bin/env python3
"""Freeze Night-15D lane configs after the registered development search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lanes = {}
    for root in args.summary_roots:
        for path in root.glob("*__summary.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            best = value["best"]
            lanes[value["lane"]] = {
                "k": int(value["k"]),
                "config": json.loads(best["config_json"]),
                "absolute_ari": float(best["absolute_ari"]),
                "absolute_nmi": float(best["absolute_nmi"]),
                "ami": float(best["ami"]),
                "fmi": float(best["fmi"]),
                "delta_vs_night15c_ari": float(best["delta_ari"]),
                "delta_vs_night15c_nmi": float(best["delta_nmi"]),
                "authority_partition_sha256": value["authority_partition_sha256"],
                "partition_sha256": value["best_partition_sha256"],
                "changed_observations": int(best["changed_observations"]),
                "cluster_sizes": json.loads(best["cluster_sizes"]),
                "search_rows": int(value["run_rows"]),
                "dual_positive_rows": int(value["dual_positive_rows"]),
            }
    if len(lanes) != 9:
        raise RuntimeError(f"expected 9 frozen lanes, observed {len(lanes)}")
    result = {
        "status": "FROZEN_BEFORE_FRESH_PROCESS_REPLAY",
        "parent_commit": "44502a3ffc1193c0e91c0083f1db14e1828a4cf9",
        "selection_rule": "public development HPO; require delta_ari>1e-10 and delta_nmi>1e-10, then maximize ari+0.35*nmi",
        "initial_rule": "each finalist starts from its byte-exact Night-15C stable authority partition",
        "shared_core": "adaptive reliability-mass prototype energy",
        "dataset_name_reads_in_model_core": 0,
        "labels_in_model_or_energy": 0,
        "labels_used_for_known_k_hpo_and_evaluation": 1,
        "numerical_environment": {
            "feature_reduction": "full SVD",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1"
        },
        "lanes": dict(sorted(lanes.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "lanes": len(lanes), "output": str(args.output)}))


if __name__ == "__main__":
    main()
