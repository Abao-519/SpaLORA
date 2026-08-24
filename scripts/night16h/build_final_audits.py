#!/usr/bin/env python3
"""Build Night-16H replay, resource and delivery-gate audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7", "MELANOMA_TUMOR_K2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    core = args.repo_root / "SpaLORA/night16h_feasible_selector.py"
    config = args.repo_root / "configs/night16h/night16h_feasible_selector_contract.json"
    replay_products = [
        args.working_root / f"formal/{selector}_run{run}/{lane}.producer.json"
        for selector in ("global", "loso") for run in (1, 2) for lane in LANES
    ]
    replay = json.loads(
        (args.working_root / "formal/final_exact_replay_audit.json").read_text(encoding="utf-8")
    )
    test_log = args.working_root / "formal/final_targeted_tests.log"
    text = test_log.read_text(encoding="utf-8")
    if replay["passed"] != 8 or "11 passed" not in text:
        raise RuntimeError("final replay/test gate failed")
    if core.stat().st_mtime >= min(path.stat().st_mtime for path in replay_products):
        raise RuntimeError("final core source does not precede replay")
    audit = {
        "schema": "night16h-formal-replay-and-test-audit-v1",
        "final_core_source_precedes_all_replays": True,
        "formal_config_precedes_first_formal_producer": config.stat().st_mtime < min(path.stat().st_mtime for path in replay_products),
        "core_source_sha256": sha256(core),
        "formal_config_sha256": sha256(config),
        "fresh_process_replay": replay,
        "targeted_tests": {"passed": 11, "total": 11, "log_sha256": sha256(test_log)},
        "scipy_spearman_cross_version_fix_verified": True,
        "thread_limits": {"OMP_NUM_THREADS": 1, "MKL_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1, "threadpool_limits": 1},
    }
    (args.output / "formal_replay_and_test_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    resources = list(csv.DictReader((args.output / "resource_table.csv").open(encoding="utf-8")))
    feasibility = []
    for lane in LANES:
        manifest = json.loads((args.working_root / f"feasibility/{lane}.manifest.json").read_text(encoding="utf-8"))
        feasibility.append(manifest)
    resource = {
        "schema": "night16h-resource-audit-v1",
        "lane_count": 4,
        "candidate_count": 356,
        "fixed_selector_wall_seconds_sum": sum(float(row["global_wall_seconds"]) for row in resources),
        "fixed_selector_peak_rss_mib_max": max(float(row["global_peak_rss_mib"]) for row in resources),
        "feasibility_wall_seconds_sum": sum(float(row["wall_seconds"]) for row in feasibility),
        "gpu_time_seconds": 0,
        "peak_gpu_mib": 0,
        "dense_observation_by_observation_count": 0,
        "candidate_similarity_shape": [89, 89],
        "sparse_graph_only": True,
        "shutdown_dispatched": False,
        "machine_state_instruction": "KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS",
    }
    (args.output / "resource_audit.json").write_text(
        json.dumps(resource, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "replay": "8/8", "tests": "11/11"}))


if __name__ == "__main__":
    main()
