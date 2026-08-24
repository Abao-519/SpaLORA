#!/usr/bin/env python3
"""Build the transparent per-lane Night-16E score-frontier board.

The inputs are already materialized producer artifacts followed by independent
evaluation CSVs. This script never feeds metrics back into the family-frozen
configuration; it only records label-assisted per-lane development ceilings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TOLERANCE = 1e-12


def build_replay_candidates(
    selected_candidate_id: str,
    registry_candidates: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Materialize one no-op plus the selected profile without duplicate IDs."""
    input_candidate = {
        "candidate_id": "INPUT_STRONG_START",
        "profile_id": "INPUT_STRONG_START",
        "base_id": "NONE",
        "variant": "INPUT_STRONG_START",
        "config": None,
    }
    if selected_candidate_id == "INPUT_STRONG_START":
        chosen = [input_candidate]
    else:
        selected_candidate = registry_candidates.get(selected_candidate_id)
        if selected_candidate is None:
            raise KeyError(f"selected candidate missing from registries: {selected_candidate_id}")
        chosen = [input_candidate]
        chosen.extend(
            candidate
            for candidate in registry_candidates.values()
            if candidate.get("profile_id") == selected_candidate.get("profile_id")
            and candidate.get("candidate_id") != "INPUT_STRONG_START"
        )
    identifiers = [str(candidate["candidate_id"]) for candidate in chosen]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate replay candidate IDs: {identifiers}")
    return chosen


def choose(rows: pd.DataFrame, mode: str, baseline: pd.Series) -> pd.Series:
    candidates = rows[(rows.status == "PASS") & (rows.variant.isin(["TSRE_FULL", "INPUT_STRONG_START"]))].copy()
    candidates["delta_ari"] = candidates.absolute_ari - float(baseline.absolute_ari)
    candidates["delta_nmi"] = candidates.absolute_nmi - float(baseline.absolute_nmi)
    candidates["dual_positive"] = (
        (candidates.delta_ari > TOLERANCE) & (candidates.delta_nmi > TOLERANCE)
    ).astype(int)
    candidates["minimum_delta"] = candidates[["delta_ari", "delta_nmi"]].min(axis=1)
    candidates["sum_delta"] = candidates.delta_ari + candidates.delta_nmi
    if mode == "MAX_ARI":
        order = ["absolute_ari", "absolute_nmi", "candidate_id"]
        ascending = [False, False, True]
    elif mode == "MAX_NMI":
        order = ["absolute_nmi", "absolute_ari", "candidate_id"]
        ascending = [False, False, True]
    elif mode == "BALANCED":
        order = ["dual_positive", "minimum_delta", "sum_delta", "candidate_id"]
        ascending = [False, False, False, True]
    else:
        raise ValueError(mode)
    return candidates.sort_values(order, ascending=ascending).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", action="append", required=True, help="lane=family=evaluation.csv")
    parser.add_argument("--registry", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    registry_candidates: dict[str, dict[str, object]] = {}
    for path in args.registry:
        registry = json.loads(Path(path).read_text())
        for candidate in registry["candidates"]:
            registry_candidates[str(candidate["candidate_id"])] = candidate

    ledgers: list[pd.DataFrame] = []
    selected_rows: list[dict[str, object]] = []
    replay_registries: dict[str, object] = {}
    for specification in args.lane:
        lane, family, csv_path = specification.split("=", 2)
        rows = pd.read_csv(csv_path)
        rows.insert(0, "lane", lane)
        rows.insert(1, "family", family)
        rows.insert(2, "board", "PER_LANE_LABEL_ASSISTED_SCORE_FRONTIER")
        ledgers.append(rows)
        baseline = rows[rows.candidate_id == "INPUT_STRONG_START"].iloc[0]
        selections: dict[str, pd.Series] = {}
        for mode in ("BALANCED", "MAX_ARI", "MAX_NMI"):
            value = choose(rows, mode, baseline)
            selections[mode] = value
            selected_rows.append(
                {
                    "lane": lane,
                    "family": family,
                    "profile": mode,
                    "candidate_id": value.candidate_id,
                    "profile_id": value.profile_id,
                    "variant": value.variant,
                    "absolute_ari": value.absolute_ari,
                    "absolute_nmi": value.absolute_nmi,
                    "delta_ari_vs_input_strong_start": value.absolute_ari - baseline.absolute_ari,
                    "delta_nmi_vs_input_strong_start": value.absolute_nmi - baseline.absolute_nmi,
                    "ami": value.ami,
                    "fmi": value.fmi,
                    "homogeneity": value.homogeneity,
                    "v_measure": value.v_measure,
                    "morans_i_macro": value.morans_i_macro,
                    "gearys_c_macro": value.gearys_c_macro,
                    "neighbor_agreement": value.neighbor_agreement,
                    "min_cluster_size_full": value.min_cluster_size_full,
                    "min_cluster_size_eval": value.min_cluster_size_eval,
                    "cluster_sizes_full": value.cluster_sizes_full,
                    "cluster_sizes_eval": value.cluster_sizes_eval,
                    "partition_sha256": value.partition_sha256,
                    "selection_uses_public_annotation": True,
                }
            )
        balanced = selections["BALANCED"]
        chosen = build_replay_candidates(str(balanced.candidate_id), registry_candidates)
        replay_registries[lane] = {
            "schema": "night16e-per-lane-score-frontier-frozen-v1",
            "family": family,
            "lane": lane,
            "board": "PER_LANE_LABEL_ASSISTED_SCORE_FRONTIER",
            "selection_rule": (
                "prefer ARI/NMI dual-positive; maximize min(delta ARI, delta NMI); "
                "then sum of deltas; then lexical candidate ID"
            ),
            "selected_balanced_candidate": str(balanced.candidate_id),
            "candidates": chosen,
        }

    pd.concat(ledgers, ignore_index=True).to_csv(
        output_root / "all_candidate_hpo_ledger.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(output_root / "score_frontier_profiles.csv", index=False)
    (output_root / "score_frontier_replay_registries.json").write_text(
        json.dumps(replay_registries, indent=2, sort_keys=True)
    )
    replay_root = output_root / "replay_registries"
    replay_root.mkdir(exist_ok=True)
    for lane, registry in replay_registries.items():
        (replay_root / f"{lane}.json").write_text(
            json.dumps(registry, indent=2, sort_keys=True)
        )


if __name__ == "__main__":
    main()
