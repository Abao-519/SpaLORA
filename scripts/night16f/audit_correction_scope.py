#!/usr/bin/env python3
"""Compare superseded thread-unpinned and corrected Night-16F artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def raw_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def evaluation_map(path: Path) -> dict[str, dict[str, str]]:
    return {row["candidate_id"]: row for row in csv.DictReader(path.open())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    work = Path(args.work_root)
    old = work / "superseded" / "thread_unpinned_formal"
    lane_specs = (
        ("P22_K9", "P22_K9", "development"),
        ("MISAR_K7", "MISAR_K7", "development"),
        ("HUMAN_HIPPOCAMPUS_K7", "HUMAN_HIPPOCAMPUS_K7", "development"),
        ("MELANOMA_TUMOR_K2", "MELANOMA_TUMOR_K2", "formal"),
    )
    rows = []
    for lane, carrier_name, phase in lane_specs:
        old_carrier = old / "carriers" / f"{carrier_name}.npz"
        new_carrier = work / "carriers" / f"{carrier_name}.npz"
        with np.load(old_carrier, allow_pickle=False) as before, np.load(
            new_carrier, allow_pickle=False
        ) as after:
            carrier_arrays = {}
            for key in ("ids", "view1", "view2", "retained", "start_partitions"):
                carrier_arrays[key] = {
                    "old_dtype": str(before[key].dtype),
                    "new_dtype": str(after[key].dtype),
                    "old_sha256": raw_sha(before[key]),
                    "new_sha256": raw_sha(after[key]),
                    "byte_exact": bool(np.array_equal(before[key], after[key])),
                }
            graphs = []
            for index in range(3):
                graph = {}
                for component in ("data", "indices", "indptr"):
                    key = f"graph{index}__{component}"
                    graph[f"{component}_old_dtype"] = str(before[key].dtype)
                    graph[f"{component}_new_dtype"] = str(after[key].dtype)
                    graph[f"{component}_old_sha256"] = raw_sha(before[key])
                    graph[f"{component}_new_sha256"] = raw_sha(after[key])
                    graph[f"{component}_byte_exact"] = bool(
                        np.array_equal(before[key], after[key])
                    )
                graph["scale_index"] = index
                graphs.append(graph)

        old_bank = old / phase / lane / "partitions.npz"
        new_bank = work / phase / lane / "partitions.npz"
        with np.load(old_bank, allow_pickle=False) as before, np.load(
            new_bank, allow_pickle=False
        ) as after:
            if not np.array_equal(before["ids"], after["ids"]):
                raise RuntimeError(f"{lane} ordered IDs changed")
            old_partitions = before["partitions"]
            new_partitions = after["partitions"]
            exact = int(
                sum(np.array_equal(x, y) for x, y in zip(old_partitions, new_partitions))
            )
        old_eval = evaluation_map(old / phase / lane / "evaluation.csv")
        new_eval = evaluation_map(work / phase / lane / "evaluation.csv")
        if set(old_eval) != set(new_eval):
            raise RuntimeError(f"{lane} candidate IDs changed")
        authority_id = next(
            key for key, row in new_eval.items() if row["arm"] == "INPUT_START" and int(row["start_index"]) == 0
        )
        direct_id = next(
            key for key, row in new_eval.items() if row["arm"] == "DIRECT_BASE" and int(row["start_index"]) == 0
        )
        new_manifest = json.loads(
            (work / phase / lane / "partitions.producer.json").read_text()
        )
        rows.append(
            {
                "lane": lane,
                "carrier_arrays": carrier_arrays,
                "graphs": graphs,
                "partition_count": int(len(new_partitions)),
                "old_new_partition_exact_count": exact,
                "old_new_partition_changed_count": int(len(new_partitions) - exact),
                "producer_thread_limit_old": None,
                "producer_thread_limit_new": new_manifest.get("deterministic_thread_limit"),
                "authority_metrics_old": {
                    "ari": float(old_eval[authority_id]["absolute_ari"]),
                    "nmi": float(old_eval[authority_id]["absolute_nmi"]),
                },
                "authority_metrics_new": {
                    "ari": float(new_eval[authority_id]["absolute_ari"]),
                    "nmi": float(new_eval[authority_id]["absolute_nmi"]),
                },
                "direct_metrics_old": {
                    "ari": float(old_eval[direct_id]["absolute_ari"]),
                    "nmi": float(old_eval[direct_id]["absolute_nmi"]),
                },
                "direct_metrics_new": {
                    "ari": float(new_eval[direct_id]["absolute_ari"]),
                    "nmi": float(new_eval[direct_id]["absolute_nmi"]),
                },
            }
        )
    output = {
        "schema": "night16f-engineering-correction-scope-audit-v1",
        "status": "PASS",
        "correction": "fixed BLAS thread boundary plus lossless graph dtype storage",
        "all_first_pass_outputs_superseded": True,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
