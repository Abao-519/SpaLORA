#!/usr/bin/env python3
"""Audit two fresh-process Night-16H selector replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for selector in ("global", "loso"):
        for lane in LANES:
            paths = [args.working_root / f"formal/{selector}_run{run}/{lane}.npz" for run in (1, 2)]
            manifests = [json.loads(path.with_suffix(".producer.json").read_text(encoding="utf-8")) for path in paths]
            with np.load(paths[0], allow_pickle=False) as left, np.load(paths[1], allow_pickle=False) as right:
                exact = (
                    np.array_equal(left["ids"], right["ids"])
                    and np.array_equal(left["partition"], right["partition"])
                    and str(left["selected_candidate_id"]) == str(right["selected_candidate_id"])
                )
            manifest_exact = all(
                manifests[0][key] == manifests[1][key]
                for key in (
                    "selected_candidate_id", "selected_partition_sha256", "feasibility_mode",
                    "selector", "config_sha256", "feature_sha256", "partition_bank_sha256",
                    "producer_label_reads", "dense_observation_by_observation_count", "thread_limit",
                )
            )
            rows.append({
                "selector": selector,
                "lane": lane,
                "partition_exact": exact,
                "manifest_semantics_exact": manifest_exact,
                "selected_candidate_id": manifests[0]["selected_candidate_id"],
                "partition_sha256": manifests[0]["selected_partition_sha256"],
                "artifact_run1_sha256": sha256(paths[0]),
                "artifact_run2_sha256": sha256(paths[1]),
                "status": "PASS" if exact and manifest_exact else "FAIL",
            })
    passed = sum(row["status"] == "PASS" for row in rows)
    result = {
        "schema": "night16h-fresh-process-exact-replay-audit-v1",
        "passed": passed,
        "total": len(rows),
        "rows": rows,
    }
    if passed != len(rows):
        raise RuntimeError("fresh-process replay mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "total": len(rows)}))


if __name__ == "__main__":
    main()
