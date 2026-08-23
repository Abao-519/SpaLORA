#!/usr/bin/env python3
"""Merge Night-15E arenas and freeze one balanced score-ceiling profile."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from night15e_continuous_search import NIGHT15D_AUTHORITY, selection_key, sha256_array, write_rows


def load_summary(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))["lanes"]


def read_ledger(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_row(row: Mapping[str, object]) -> dict:
    output = dict(row)
    for key in (
        "absolute_ari", "absolute_nmi", "ami", "fmi", "morans_i", "gearys_c",
        "delta_ari", "delta_nmi", "objective", "wall_seconds", "changed_observations",
    ):
        if key in output and output[key] not in (None, ""):
            output[key] = float(output[key])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-summary", type=Path, required=True)
    parser.add_argument("--formal-ledger", type=Path, required=True)
    parser.add_argument("--formal-partitions", type=Path, required=True)
    parser.add_argument("--adaptive-summary", type=Path, required=True)
    parser.add_argument("--adaptive-ledger", type=Path, required=True)
    parser.add_argument("--adaptive-partitions", type=Path, required=True)
    parser.add_argument("--multi-summary", type=Path, required=True)
    parser.add_argument("--multi-ledger", type=Path, required=True)
    parser.add_argument("--multi-partitions", type=Path, required=True)
    parser.add_argument("--night15d-partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "partitions").mkdir(exist_ok=True)

    sources = {
        "FORMAL": (load_summary(args.formal_summary), args.formal_partitions),
        "ADAPTIVE": (load_summary(args.adaptive_summary), args.adaptive_partitions),
        "MULTI_INITIAL": (load_summary(args.multi_summary), args.multi_partitions),
    }
    ledgers = []
    for source_name, path in (
        ("FORMAL", args.formal_ledger),
        ("ADAPTIVE", args.adaptive_ledger),
        ("MULTI_INITIAL", args.multi_ledger),
    ):
        for row in read_ledger(path):
            row["arena_source"] = source_name
            ledgers.append(row)
    write_rows(args.output / "all_run_ledger.csv", ledgers)

    registry = {}
    table = []
    for lane, (authority_ari, authority_nmi) in NIGHT15D_AUTHORITY.items():
        candidates: List[Tuple[str, dict, Path]] = []
        for source_name, (summary, partition_root) in sources.items():
            if lane not in summary:
                continue
            best = numeric_row(summary[lane]["best"])
            candidates.append((source_name, best, partition_root / f"{lane}.npy"))
        source_name, selected, partition_path = max(candidates, key=lambda item: selection_key(item[1]))
        if selected["config_id"] == "NIGHT15D_AUTHORITY":
            partition_path = args.night15d_partitions / f"{lane}.npy"
        partition = np.load(partition_path, allow_pickle=False).astype(np.int32)
        initial = np.load(args.night15d_partitions / f"{lane}.npy", allow_pickle=False).astype(np.int32)
        if float(selected["delta_ari"]) <= 1e-12 or float(selected["delta_nmi"]) <= 1e-12:
            raise RuntimeError(f"balanced profile is not dual-positive for {lane}")
        np.save(args.output / "partitions" / f"{lane}.npy", partition, allow_pickle=False)

        eligible_rows = [
            numeric_row(row)
            for row in ledgers
            if row.get("lane") == lane and row.get("status") == "PASS" and row.get("absolute_ari") not in (None, "")
        ]
        max_ari = max(eligible_rows, key=lambda row: float(row["absolute_ari"]))
        max_nmi = max(eligible_rows, key=lambda row: float(row["absolute_nmi"]))
        config = json.loads(str(selected["config_json"]))
        record = {
            "lane": lane,
            "profile": "BALANCED_DUAL_PRIORITY",
            "source_arena": source_name,
            "config_id": selected["config_id"],
            "config": config,
            "initial_source": "fixed Night-15D authority partition",
            "initial_partition_sha256": sha256_array(initial),
            "partition_sha256": sha256_array(partition),
            "absolute_ari": float(selected["absolute_ari"]),
            "absolute_nmi": float(selected["absolute_nmi"]),
            "ami": float(selected["ami"]),
            "fmi": float(selected["fmi"]),
            "morans_i": float(selected["morans_i"]),
            "gearys_c": float(selected["gearys_c"]),
            "delta_ari_vs_night15d": float(selected["absolute_ari"]) - authority_ari,
            "delta_nmi_vs_night15d": float(selected["absolute_nmi"]) - authority_nmi,
            "changed_observations": int(float(selected["changed_observations"])),
            "labels_used_for_public_cross_run_numeric_hpo": 1,
            "labels_in_model_or_energy": 0,
        }
        registry[lane] = record
        table.append(
            {
                **{key: value for key, value in record.items() if key not in ("config",)},
                "night15d_ari": authority_ari,
                "night15d_nmi": authority_nmi,
                "max_ari_profile_absolute_ari": float(max_ari["absolute_ari"]),
                "max_ari_profile_nmi": float(max_ari["absolute_nmi"]),
                "max_ari_profile_config_id": max_ari["config_id"],
                "max_nmi_profile_ari": float(max_nmi["absolute_ari"]),
                "max_nmi_profile_absolute_nmi": float(max_nmi["absolute_nmi"]),
                "max_nmi_profile_config_id": max_nmi["config_id"],
            }
        )
    frozen = {
        "status": "FROZEN",
        "selection_rule": "dual-positive first; then maximize min(delta_ari, delta_nmi); then absolute_ari + 0.35*absolute_nmi",
        "profile": "BALANCED_DUAL_PRIORITY",
        "lanes": registry,
        "lane_count": len(registry),
        "dual_positive_lanes": sum(
            row["delta_ari_vs_night15d"] > 1e-12 and row["delta_nmi_vs_night15d"] > 1e-12
            for row in registry.values()
        ),
        "same_continuous_formula_all_lanes": True,
        "per_lane_numeric_hpo": True,
        "public_benchmark_development_not_blind": True,
        "labels_in_model_or_energy": 0,
        "dataset_name_reads_in_model_core": 0,
        "dense_n_by_n_count": 0,
        "incremental_policy_transfer_classification": "NO_MEANINGFUL_INCREMENTAL_POLICY_TRANSFER",
    }
    (args.output / "night15e_frozen_config_registry.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_rows(args.output / "absolute_metrics_main_table.csv", table)
    print(json.dumps({"status": "FROZEN", "lanes": len(registry), "dual_positive": frozen["dual_positive_lanes"]}))


if __name__ == "__main__":
    main()
