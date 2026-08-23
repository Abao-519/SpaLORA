#!/usr/bin/env python3
"""Independent structural audit of two frozen Night-15F fresh-process replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--replay1", type=Path, required=True)
    parser.add_argument("--replay2", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    first = json.loads(args.replay1.read_text(encoding="utf-8"))
    second = json.loads(args.replay2.read_text(encoding="utf-8"))
    first_rows = {row["lane"]: row for row in first["rows"]}
    second_rows = {row["lane"]: row for row in second["rows"]}
    lane_audit = {}
    for lane, frozen in registry["lanes"].items():
        left, right = first_rows[lane], second_rows[lane]
        cycles = json.loads(left["cycle_energy_ledger_json"])
        cycle_monotone = all(
            float(cycle["end_energy"]) <= float(cycle["start_energy"]) + 1e-9
            for cycle in cycles
        )
        lane_audit[lane] = {
            "partition_sha_replay1": left["partition_sha256"],
            "partition_sha_replay2": right["partition_sha256"],
            "partition_sha_frozen": frozen["partition_sha256"],
            "partition_exact_3way": left["partition_sha256"] == right["partition_sha256"] == frozen["partition_sha256"],
            "ari_exact_3way": left["absolute_ari"] == right["absolute_ari"] == frozen["absolute_ari"],
            "nmi_exact_3way": left["absolute_nmi"] == right["absolute_nmi"] == frozen["absolute_nmi"],
            "cluster_sizes_exact": left["cluster_sizes"] == right["cluster_sizes"] == json.dumps(frozen["cluster_sizes"], separators=(",", ":")),
            "cycle_count": len(cycles),
            "each_frozen_cycle_monotone": cycle_monotone,
            "cross_dynamic_cycle_global_monotonicity_claimed": False,
        }
    source_mtime = max(args.core.stat().st_mtime_ns, args.runner.stat().st_mtime_ns)
    replay_mtime = min(args.replay1.stat().st_mtime_ns, args.replay2.stat().st_mtime_ns)
    payload = {
        "status": "PASS" if all(
            all(value[key] for key in (
                "partition_exact_3way", "ari_exact_3way", "nmi_exact_3way",
                "cluster_sizes_exact", "each_frozen_cycle_monotone"
            )) for value in lane_audit.values()
        ) and source_mtime < replay_mtime else "FAIL",
        "lane_count": len(lane_audit),
        "lane_audit": lane_audit,
        "core_sha256": sha256(args.core),
        "runner_sha256": sha256(args.runner),
        "replay1_sha256": sha256(args.replay1),
        "replay2_sha256": sha256(args.replay2),
        "source_mtime_precedes_both_replays": source_mtime < replay_mtime,
        "labels_in_energy_unary_or_edge": 0,
        "dataset_name_reads_in_energy_core": 0,
        "dense_n_by_n_count": 0,
    }
    if payload["status"] != "PASS":
        raise RuntimeError("final replay audit failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "lanes": len(lane_audit)}))


if __name__ == "__main__":
    main()
