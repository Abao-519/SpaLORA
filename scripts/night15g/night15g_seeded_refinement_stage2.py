#!/usr/bin/env python3
"""Second-stage local search around the strongest morphology-seeded energy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15f_multiscale_expansion import (  # noqa: E402
    continuous_multiscale_expansion,
    prepare_expansion_evidence,
)
from scripts.night15f.night15f_solver_search import random_configs  # noqa: E402
from scripts.night15g.night15g_optional_view_search import (  # noqa: E402
    array_sha256,
    evaluate,
    load_lane,
    parse_base,
    selection_key,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--night15f-registry", type=Path, required=True)
    parser.add_argument("--stage1-summary", type=Path, required=True)
    parser.add_argument("--stage1-partition", type=Path, required=True)
    parser.add_argument("--direct-partition", type=Path, required=True)
    parser.add_argument("--lane", default="D1")
    parser.add_argument("--random-configs", type=int, default=800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "partitions").mkdir(exist_ok=True)

    data, graph, graphs, retained, authority, labels, mask, k, _ = load_lane(args, args.lane)
    stage1 = json.loads(args.stage1_summary.read_text(encoding="utf-8"))
    stage1_row = stage1["profiles"]["max_ari"]
    anchor = parse_base(json.loads(stage1_row["config_json"]))
    stage1_partition = np.load(args.stage1_partition, allow_pickle=False).astype(np.int32)
    direct_partition = np.load(args.direct_partition, allow_pickle=False).astype(np.int32)
    for name, partition in (("stage1", stage1_partition), ("direct", direct_partition)):
        if len(partition) != len(authority) or len(np.unique(partition)) != k:
            raise RuntimeError(f"{name} start failed shape/cardinality contract")
    if array_sha256(stage1_partition) != stage1_row["partition_sha256"]:
        raise RuntimeError("stage1 max-ARI partition hash mismatch")

    starts = {"DIRECT_START": direct_partition, "STAGE1_MAX_ARI": stage1_partition}
    evidence = prepare_expansion_evidence(
        graphs,
        retained,
        np.asarray(data["view1"]),
        np.asarray(data["view2"]),
    )
    configs = random_configs(
        anchor.local,
        args.random_configs,
        20260824 + 2909,
        anchor=anchor,
    )
    # Dense local grid over the two energy controls that created the Stage-1
    # jump.  This remains a numeric, dataset-agnostic formula; the public lane
    # is allowed to select its own values between completed runs.
    for beta in (0.015, 0.025, 0.035, 0.045, 0.055, 0.07, 0.09, 0.12, 0.18):
        for self_return in (0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.75, 1.0):
            for cycles in (1, 2):
                config = replace(
                    anchor,
                    pairwise_beta=beta,
                    self_return_strength=self_return,
                    expansion_cycles=cycles,
                )
                configs[f"GRID_B{beta:g}_S{self_return:g}_C{cycles}"] = config

    authority_metrics = evaluate(labels, mask, authority, graph)
    rows: list[dict] = []
    best: dict[str, tuple[dict, np.ndarray]] = {}

    def consider(row, partition):
        keys = {
            "balanced": selection_key,
            "max_ari": lambda value: (float(value["absolute_ari"]), float(value["absolute_nmi"])),
            "max_nmi": lambda value: (float(value["absolute_nmi"]), float(value["absolute_ari"])),
            "composite": lambda value: (
                float(value["absolute_ari"]) + 0.35 * float(value["absolute_nmi"]),
            ),
        }
        for profile, key in keys.items():
            if profile not in best or key(row) > key(best[profile][0]):
                best[profile] = (row, partition.copy())

    for start_name, initial in starts.items():
        for config_id, config in configs.items():
            started = time.perf_counter()
            try:
                partition, diagnostics = continuous_multiscale_expansion(initial, k, evidence, config)
                metrics = evaluate(labels, mask, partition, graph)
                row = {
                    "lane": args.lane,
                    "start": start_name,
                    "config_id": config_id,
                    "config_json": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
                    "status": "PASS",
                    "failure": "",
                    "partition_sha256": array_sha256(partition),
                    "absolute_ari": metrics["absolute_ari"],
                    "absolute_nmi": metrics["absolute_nmi"],
                    "delta_ari": metrics["absolute_ari"] - authority_metrics["absolute_ari"],
                    "delta_nmi": metrics["absolute_nmi"] - authority_metrics["absolute_nmi"],
                    "wall_seconds": time.perf_counter() - started,
                    **{key: value for key, value in metrics.items() if key not in {"absolute_ari", "absolute_nmi"}},
                    **diagnostics,
                }
                rows.append(row)
                if int(diagnostics["observed_cardinality"]) == k:
                    consider(row, partition)
            except Exception as error:
                rows.append(
                    {
                        "lane": args.lane,
                        "start": start_name,
                        "config_id": config_id,
                        "config_json": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
                        "status": "FAILED",
                        "failure": repr(error),
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
        write_csv(args.output / "all_run_ledger.partial.csv", rows)

    summary = {
        "status": "NIGHT15G_SEEDED_REFINEMENT_STAGE2_COMPLETE",
        "lane": args.lane,
        "candidate_rows": len(rows),
        "labels_in_energy": 0,
        "labels_in_cross_run_hpo_and_evaluation": 1,
        "profiles": {},
    }
    for profile, (row, partition) in best.items():
        np.save(args.output / "partitions" / f"{args.lane}__{profile}.npy", partition, allow_pickle=False)
        summary["profiles"][profile] = row
    write_csv(args.output / "all_run_ledger.csv", rows)
    (args.output / "search_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                profile: {
                    "ari": row["absolute_ari"],
                    "nmi": row["absolute_nmi"],
                    "start": row["start"],
                    "min_cluster_size": row["min_cluster_size"],
                }
                for profile, (row, _) in best.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
