#!/usr/bin/env python3
"""Mechanically freeze one Night-16E numeric profile per modality family."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd


def candidate_id(variant: str, config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return f"{variant}_{hashlib.sha256((variant + payload).encode()).hexdigest()[:14]}"


def row(variant: str, config: dict[str, object], base_id: str, profile_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id(variant, config),
        "profile_id": profile_id,
        "base_id": base_id,
        "variant": variant,
        "config": config,
    }


def select_two_study(first: pd.DataFrame, second: pd.DataFrame) -> str:
    merged = first.merge(second, on="candidate_id", suffixes=("_first", "_second"))
    base_first = first[first.candidate_id == "INPUT_STRONG_START"].iloc[0]
    base_second = second[second.candidate_id == "INPUT_STRONG_START"].iloc[0]
    value = merged[merged.variant_first == "TSRE_FULL"].copy()
    value["dari_first"] = value.absolute_ari_first - base_first.absolute_ari
    value["dnmi_first"] = value.absolute_nmi_first - base_first.absolute_nmi
    value["dari_second"] = value.absolute_ari_second - base_second.absolute_ari
    value["dnmi_second"] = value.absolute_nmi_second - base_second.absolute_nmi
    value["dual_count"] = (
        ((value.dari_first > 1e-12) & (value.dnmi_first > 1e-12)).astype(int)
        + ((value.dari_second > 1e-12) & (value.dnmi_second > 1e-12)).astype(int)
    )
    value["worst_dari"] = value[["dari_first", "dari_second"]].min(axis=1)
    value["mean_dari"] = value[["dari_first", "dari_second"]].mean(axis=1)
    value["mean_dnmi"] = value[["dnmi_first", "dnmi_second"]].mean(axis=1)
    value = value.sort_values(
        ["dual_count", "worst_dari", "mean_dari", "mean_dnmi", "candidate_id"],
        ascending=[False, False, False, False, True],
    )
    return str(value.iloc[0].candidate_id)


def select_chromatin(value: pd.DataFrame) -> str:
    baseline = value[value.candidate_id == "INPUT_STRONG_START"].iloc[0]
    candidates = value[value.variant == "TSRE_FULL"].copy()
    candidates["dari"] = candidates.absolute_ari - baseline.absolute_ari
    candidates["dnmi"] = candidates.absolute_nmi - baseline.absolute_nmi
    candidates["dual"] = ((candidates.dari > 1e-12) & (candidates.dnmi > 1e-12)).astype(int)
    candidates["minimum"] = candidates[["dari", "dnmi"]].min(axis=1)
    candidates["sum"] = candidates.dari + candidates.dnmi
    candidates = candidates.sort_values(
        ["dual", "minimum", "sum", "candidate_id"], ascending=[False, False, False, True]
    )
    return str(candidates.iloc[0].candidate_id)


def build(args: argparse.Namespace) -> None:
    screen = json.loads(Path(args.screen_registry).read_text())
    evaluations = [pd.read_csv(path) for path in args.discovery_evaluation]
    if len(evaluations) == 2:
        selected_id = select_two_study(evaluations[0], evaluations[1])
    elif len(evaluations) == 1:
        selected_id = select_chromatin(evaluations[0])
    else:
        raise ValueError("family freeze requires one or two discovery evaluations")
    selected = copy.deepcopy(next(x for x in screen["candidates"] if x["candidate_id"] == selected_id))
    full = copy.deepcopy(selected["config"])
    profile_id = f"FROZEN_{args.family}_{selected_id[-14:]}"
    candidates = [
        {
            "candidate_id": "INPUT_STRONG_START",
            "profile_id": profile_id,
            "base_id": "NONE",
            "variant": "INPUT_STRONG_START",
            "config": None,
        }
    ]
    direct = copy.deepcopy(full)
    direct.update(
        support_mix=0.0,
        boundary_strength=0.0,
        private_strength=0.0,
        relation_stay_strength=0.0,
    )
    candidates.append(row("NIGHT15F_DIRECT", direct, selected["base_id"], profile_id))
    candidates.append(row("TSRE_FULL", full, selected["base_id"], profile_id))
    support_only = copy.deepcopy(full)
    support_only.update(boundary_strength=0.0, private_strength=0.0, relation_stay_strength=0.0)
    candidates.append(row("SUPPORT_MODULATION_ONLY", support_only, selected["base_id"], profile_id))
    boundary_off = copy.deepcopy(full)
    boundary_off["boundary_strength"] = 0.0
    candidates.append(row("BOUNDARY_OFF", boundary_off, selected["base_id"], profile_id))
    private_off = copy.deepcopy(full)
    private_off["private_strength"] = 0.0
    candidates.append(row("CONFLICT_PRIVATE_OFF", private_off, selected["base_id"], profile_id))
    stay_off = copy.deepcopy(full)
    stay_off["relation_stay_strength"] = 0.0
    stay_off["base"]["self_return_strength"] = 0.0
    candidates.append(row("REJECTED_MASS_STAY_OFF", stay_off, selected["base_id"], profile_id))
    pure_support = copy.deepcopy(full)
    pure_support["support_mix"] = 1.0
    candidates.append(row("PURE_SUPPORT_POTTS_FULL", pure_support, selected["base_id"], profile_id))
    output = {
        "schema": "night16e-family-frozen-profile-v1",
        "family": args.family,
        "selected_screen_candidate": selected_id,
        "selected_base_id": selected["base_id"],
        "profile_id": profile_id,
        "selection_rule": (
            "dual-positive discovery-study count; worst study delta ARI; "
            "study-balanced mean delta ARI; mean delta NMI; lexical candidate id"
            if len(evaluations) == 2
            else "dual-positive; minimum ARI/NMI delta; sum of deltas; lexical candidate id"
        ),
        "discovery_evaluation_count": len(evaluations),
        "support_semantics": (
            "base Potts plus tri-state support modulation when support_mix<1; "
            "PURE_SUPPORT_POTTS_FULL is the matched pure-support semantic control"
        ),
        "candidates": candidates,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("RNA_PROTEIN", "RNA_CHROMATIN"), required=True)
    parser.add_argument("--screen-registry", required=True)
    parser.add_argument("--discovery-evaluation", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
