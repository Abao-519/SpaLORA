#!/usr/bin/env python3
"""Merge Night-15F arenas and freeze balanced/max-ARI/max-NMI profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import ExpansionEnergyConfig
from scripts.night15f.night15f_solver_search import (
    NIGHT15E_AUTHORITY,
    authority_row,
    load_expansion_lane,
    run_one,
    sha256_array,
    write_rows,
)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["source_ledger"] = str(path)
    return rows


def number(row: Mapping[str, object], key: str, default: float = -math.inf) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def profile_key(row: Mapping[str, object], profile: str) -> tuple:
    if row.get("status") != "PASS":
        return (-1, -math.inf, -math.inf, -math.inf)
    da, dn = number(row, "delta_ari"), number(row, "delta_nmi")
    if profile == "BALANCED_DUAL_PRIORITY":
        return (int(da > 1e-12 and dn > 1e-12), min(da, dn), da + dn, number(row, "absolute_ari"))
    if profile == "MAX_ARI":
        return (1, number(row, "absolute_ari"), number(row, "absolute_nmi"), min(da, dn))
    if profile == "MAX_NMI":
        return (1, number(row, "absolute_nmi"), number(row, "absolute_ari"), min(da, dn))
    raise ValueError(profile)


def parse_config(row: Mapping[str, object]) -> ExpansionEnergyConfig | None:
    if row["config_id"] == "NIGHT15E_AUTHORITY":
        return None
    raw = json.loads(str(row["config_json"]))
    return ExpansionEnergyConfig(**{**raw, "local": ContinuousEnergyConfig(**raw["local"])})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, action="append", required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "partitions").mkdir(exist_ok=True)

    all_rows = []
    for path in args.ledger:
        all_rows.extend(read_rows(path))
    write_rows(args.output / "all_run_ledger.csv", all_rows)
    passed = [row for row in all_rows if row.get("status") == "PASS"]
    lanes = list(NIGHT15E_AUTHORITY)
    profiles = {}
    for profile in ("BALANCED_DUAL_PRIORITY", "MAX_ARI", "MAX_NMI"):
        profiles[profile] = {}
        for lane in lanes:
            eligible = [row for row in passed if row.get("lane") == lane]
            if not eligible:
                raise RuntimeError(f"no passing row for {lane}")
            profiles[profile][lane] = max(eligible, key=lambda row: profile_key(row, profile))

    registry = {
        "status": "FROZEN",
        "selection_rule": "dual-positive first; maximize min(delta_ari,delta_nmi), then sum of deltas",
        "profile": "BALANCED_DUAL_PRIORITY",
        "lanes": {},
        "lane_count": len(lanes),
        "public_label_cross_run_per_lane_numeric_hpo": True,
        "blind_or_automatic_selection_claimed": False,
        "labels_in_energy_unary_or_edge": 0,
        "dataset_name_reads_in_energy_core": 0,
        "dense_n_by_n_count": 0,
    }
    main_table = []
    for lane in lanes:
        selected = profiles["BALANCED_DUAL_PRIORITY"][lane]
        _, graph, labels, mask, k, initial, evidence = load_expansion_lane(
            args.kit, args.banks, args.partitions, lane
        )
        config = parse_config(selected)
        if config is None:
            rerun = authority_row(lane, graph, labels, mask, k, initial)
            partition = initial
        else:
            rerun, partition = run_one(
                lane, graph, labels, mask, k, initial, evidence,
                str(selected["config_id"]), config, "FREEZE_RERUN"
            )
        for metric in ("absolute_ari", "absolute_nmi"):
            if abs(number(rerun, metric) - number(selected, metric)) > 1e-12:
                raise RuntimeError(f"freeze {metric} mismatch for {lane}")
        if rerun["partition_sha256"] != selected["partition_sha256"]:
            raise RuntimeError(f"freeze partition mismatch for {lane}")
        np.save(args.output / "partitions" / f"{lane}.npy", partition, allow_pickle=False)
        base_ari, base_nmi = NIGHT15E_AUTHORITY[lane]
        lane_payload = {
            "lane": lane,
            "config_id": selected["config_id"],
            "config": None if config is None else json.loads(str(selected["config_json"])),
            "source_ledger": selected["source_ledger"],
            "initial_partition_sha256": sha256_array(initial),
            "partition_sha256": rerun["partition_sha256"],
            "absolute_ari": number(rerun, "absolute_ari"),
            "absolute_nmi": number(rerun, "absolute_nmi"),
            "ami": number(rerun, "ami"),
            "fmi": number(rerun, "fmi"),
            "morans_i": number(rerun, "morans_i"),
            "gearys_c": number(rerun, "gearys_c"),
            "delta_ari_vs_night15e": number(rerun, "absolute_ari") - base_ari,
            "delta_nmi_vs_night15e": number(rerun, "absolute_nmi") - base_nmi,
            "cluster_sizes": json.loads(str(rerun["cluster_sizes"])),
            "changed_observations": int(number(rerun, "changed_observations", 0)),
            "cycle_energy_ledger": json.loads(str(rerun.get("cycle_energy_ledger_json", "[]"))),
            "labels_used_for_public_cross_run_hpo_and_evaluation": 1,
            "labels_in_energy_unary_or_edge": 0,
        }
        registry["lanes"][lane] = lane_payload
        main_table.append({
            "lane": lane,
            "k": k,
            "total_observations": len(partition),
            "evaluated_observations": int(mask.sum()),
            "night15e_ari": base_ari,
            "night15e_nmi": base_nmi,
            **{key: lane_payload[key] for key in (
                "absolute_ari", "absolute_nmi", "ami", "fmi", "morans_i", "gearys_c",
                "delta_ari_vs_night15e", "delta_nmi_vs_night15e", "changed_observations"
            )},
            "cluster_sizes": json.dumps(lane_payload["cluster_sizes"], separators=(",", ":")),
            "config_id": selected["config_id"],
            "partition_sha256": rerun["partition_sha256"],
            "source_ledger": selected["source_ledger"],
            "wall_seconds_search_row": number(selected, "wall_seconds", 0.0),
        })

    registry["dual_positive_lane_count"] = sum(
        value["delta_ari_vs_night15e"] > 1e-12 and value["delta_nmi_vs_night15e"] > 1e-12
        for value in registry["lanes"].values()
    )
    (args.output / "night15f_frozen_config_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_rows(args.output / "absolute_metrics_main_table.csv", main_table)

    compact_profiles = {}
    for profile, by_lane in profiles.items():
        compact_profiles[profile] = {}
        for lane, row in by_lane.items():
            compact_profiles[profile][lane] = {
                "config_id": row["config_id"],
                "source_ledger": row["source_ledger"],
                "absolute_ari": number(row, "absolute_ari"),
                "absolute_nmi": number(row, "absolute_nmi"),
                "delta_ari": number(row, "delta_ari"),
                "delta_nmi": number(row, "delta_nmi"),
                "partition_sha256": row["partition_sha256"],
                "cluster_sizes": json.loads(str(row["cluster_sizes"])),
            }
    (args.output / "score_profiles.json").write_text(
        json.dumps({"status": "PASS", "profiles": compact_profiles}, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "FROZEN", "lanes": len(lanes),
                      "dual_positive": registry["dual_positive_lane_count"]}))


if __name__ == "__main__":
    main()
