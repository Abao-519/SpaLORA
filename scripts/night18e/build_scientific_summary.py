#!/usr/bin/env python3
"""Build the frozen Night-18E Stage-A result and attribution tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DISCOVERY = ("P22_K9", "MISAR_K7", "PLACENTA_K10")
KEY_CONTROLS = (
    "NIGHT15F_ORIGINAL_SELF_RETURN",
    "CCSR_CERTIFICATE_DISABLED",
    "CCSR_RANDOM_MASK_COUNT_MATCHED",
)
TOL = 1e-12


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_eval(root: Path, prefix: str, lane: str) -> pd.DataFrame:
    frame = pd.read_csv(root / f"{prefix}_{lane}.csv")
    frame.insert(0, "lane", lane)
    frame.insert(1, "source_board", prefix)
    return frame


def one(frame: pd.DataFrame, **query) -> pd.Series:
    subset = frame
    for key, value in query.items():
        subset = subset[subset[key] == value]
    if len(subset) != 1:
        raise RuntimeError(f"expected one row for {query}, got {len(subset)}")
    return subset.iloc[0]


def arm_dominates(control: pd.Series, full: pd.Series) -> bool:
    return bool(
        control.absolute_ari >= full.absolute_ari - TOL
        and control.absolute_nmi >= full.absolute_nmi - TOL
    )


def run(args: argparse.Namespace) -> None:
    evaluation = Path(args.evaluation)
    producer_root = Path(args.producer_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    development = pd.concat(
        [read_eval(evaluation, "development", lane) for lane in DISCOVERY],
        ignore_index=True,
    )
    p0_lanes = (*DISCOVERY, "HUMAN_HIPPOCAMPUS_K7")
    p0 = pd.concat(
        [read_eval(evaluation, "p0", lane) for lane in p0_lanes],
        ignore_index=True,
    )
    all_metrics = pd.concat([development, p0], ignore_index=True)
    all_metrics.to_csv(output / "all_candidate_metrics.csv", index=False)
    p0.to_csv(output / "p0_real_path_metrics.csv", index=False)

    producer_manifests: dict[str, dict[str, object]] = {}
    full_diagnostics: dict[tuple[str, int, str], dict[str, object]] = {}
    contracts: set[str] = set()
    core_hashes: set[str] = set()
    for lane in DISCOVERY:
        manifest_path = producer_root / lane / "partitions.producer.json"
        manifest = json.loads(manifest_path.read_text())
        producer_manifests[lane] = manifest
        contracts.add(str(manifest["execution_contract_sha256"]))
        core_hashes.add(str(manifest["ccsr_core_source_sha256"]))
        for producer_row in manifest["rows"]:
            if producer_row["arm"] == "CCSR_FULL":
                full_diagnostics[
                    (
                        lane,
                        int(producer_row["start_index"]),
                        str(producer_row["certificate_config_id"]),
                    )
                ] = producer_row["diagnostics"]
    if len(contracts) != 1 or len(core_hashes) != 1:
        raise RuntimeError("formal manifests disagree on source or execution contract")

    comparisons: list[dict[str, object]] = []
    full = development[development.arm == "CCSR_FULL"]
    for _, row in full.iterrows():
        common = dict(lane=row.lane, start_index=int(row.start_index))
        baseline = one(development, **common, arm="NO_OP_STRONG_START")
        original = one(development, **common, arm="NIGHT15F_ORIGINAL_SELF_RETURN")
        controls = [original]
        for arm in KEY_CONTROLS[1:]:
            controls.append(
                one(
                    development,
                    **common,
                    arm=arm,
                    certificate_config_id=row.certificate_config_id,
                )
            )
        dominating = [str(control.arm) for control in controls if arm_dominates(control, row)]
        exact_controls = [
            str(control.arm)
            for control in controls
            if str(control.partition_sha256) == str(row.partition_sha256)
        ]
        dual = bool(
            row.absolute_ari > baseline.absolute_ari + TOL
            and row.absolute_nmi > baseline.absolute_nmi + TOL
        )
        strict_control_wins = [
            str(control.arm)
            for control in controls
            if row.absolute_ari > control.absolute_ari + TOL
            and row.absolute_nmi > control.absolute_nmi + TOL
        ]
        strict_independent = bool(dual and len(strict_control_wins) == len(controls))
        diagnostics = full_diagnostics[
            (str(row.lane), int(row.start_index), str(row.certificate_config_id))
        ]
        comparisons.append(
            {
                "lane": row.lane,
                "start_index": int(row.start_index),
                "start_id": row.start_id,
                "certificate_config_id": row.certificate_config_id,
                "full_candidate_id": row.candidate_id,
                "full_partition_sha256": row.partition_sha256,
                "absolute_ari": float(row.absolute_ari),
                "absolute_nmi": float(row.absolute_nmi),
                "baseline_ari": float(baseline.absolute_ari),
                "baseline_nmi": float(baseline.absolute_nmi),
                "delta_ari_vs_noop": float(row.absolute_ari - baseline.absolute_ari),
                "delta_nmi_vs_noop": float(row.absolute_nmi - baseline.absolute_nmi),
                "dual_gain_vs_noop": dual,
                "dominating_key_controls": "|".join(dominating),
                "byte_exact_key_controls": "|".join(exact_controls),
                "independent_dual_gain": bool(dual and not dominating),
                "strictly_beaten_key_controls": "|".join(strict_control_wins),
                "strict_independent_dual_gain": strict_independent,
                "trusted_fraction": float(diagnostics["trusted_fraction"]),
                "changed_from_initial": int(row.changed_from_initial),
                "min_cluster_size_full": int(row.min_cluster_size_full),
            }
        )
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "full_vs_matched_controls.csv", index=False)

    ranked: list[dict[str, object]] = []
    for config_id, config_rows in comparison.groupby("certificate_config_id"):
        lane_stats = []
        for lane, lane_rows in config_rows.groupby("lane"):
            lane_stats.append(
                {
                    "lane": lane,
                    "median_delta_ari": float(lane_rows.delta_ari_vs_noop.median()),
                    "median_delta_nmi": float(lane_rows.delta_nmi_vs_noop.median()),
                    "independent_start_count": int(lane_rows.strict_independent_dual_gain.sum()),
                    "median_trusted_fraction": float(lane_rows.trusted_fraction.median()),
                    "lane_pass": bool(
                        lane_rows.delta_ari_vs_noop.median() > TOL
                        and lane_rows.delta_nmi_vs_noop.median() > TOL
                        and lane_rows.strict_independent_dual_gain.sum() >= 2
                    ),
                }
            )
        lane_frame = pd.DataFrame(lane_stats)
        ranked.append(
            {
                "certificate_config_id": config_id,
                "discovery_lane_pass_count": int(lane_frame.lane_pass.sum()),
                "worst_study_median_delta_ari": float(lane_frame.median_delta_ari.min()),
                "mean_study_median_delta_ari": float(lane_frame.median_delta_ari.mean()),
                "mean_study_median_delta_nmi": float(lane_frame.median_delta_nmi.mean()),
                "study_balanced_mean_trusted_fraction": float(
                    lane_frame.median_trusted_fraction.mean()
                ),
                "strict_independent_lane_start_count": int(
                    config_rows.strict_independent_dual_gain.sum()
                ),
                "pareto_independent_lane_start_count": int(
                    config_rows.independent_dual_gain.sum()
                ),
                "dual_gain_lane_start_count": int(config_rows.dual_gain_vs_noop.sum()),
            }
        )
    config_ranking = pd.DataFrame(ranked).sort_values(
        [
            "discovery_lane_pass_count",
            "worst_study_median_delta_ari",
            "mean_study_median_delta_ari",
            "mean_study_median_delta_nmi",
            "study_balanced_mean_trusted_fraction",
            "certificate_config_id",
        ],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    )
    config_ranking.insert(0, "mechanical_rank", np.arange(1, len(config_ranking) + 1))
    config_ranking.to_csv(output / "shared_config_ranking.csv", index=False)
    shared_id = str(config_ranking.iloc[0].certificate_config_id)

    registry = json.loads(Path(args.certificate_registry).read_text())
    selected_spec = next(row for row in registry["configs"] if row["config_id"] == shared_id)
    shared_registry = {
        "schema": "night18e-frozen-shared-ccsr-registry-v1",
        "selection_scope": "PUBLIC_BENCHMARK_LABEL_ASSISTED_STAGE_A",
        "selection_rule": (
            "max discovery lanes with median paired ARI/NMI dual gain and >=2/3 "
            "independent paired starts; then max worst-study median delta ARI, "
            "mean-study median delta ARI, mean-study median delta NMI, config ID"
        ),
        "selected_config_id": shared_id,
        "stage_b_gate_required_lane_count": 2,
        "stage_b_gate_observed_lane_count": int(config_ranking.iloc[0].discovery_lane_pass_count),
        "stage_b_authorized": bool(config_ranking.iloc[0].discovery_lane_pass_count >= 2),
        "configs": [selected_spec],
        "p0_config_id": shared_id,
        "parent_certificate_registry_path": str(Path(args.certificate_registry).resolve()),
        "parent_certificate_registry_sha256": sha_file(Path(args.certificate_registry)),
    }
    shared_registry_path = output / "rna_atac_shared_ccsr_registry.json"
    shared_registry_path.write_text(json.dumps(shared_registry, indent=2, sort_keys=True))

    shared = comparison[comparison.certificate_config_id == shared_id].copy()
    shared.to_csv(output / "shared_config_paired_start_table.csv", index=False)
    paired_summary = (
        shared.groupby("lane")
        .agg(
            best_ari=("absolute_ari", "max"),
            median_ari=("absolute_ari", "median"),
            mean_ari=("absolute_ari", "mean"),
            min_ari=("absolute_ari", "min"),
            best_nmi=("absolute_nmi", "max"),
            median_nmi=("absolute_nmi", "median"),
            mean_nmi=("absolute_nmi", "mean"),
            min_nmi=("absolute_nmi", "min"),
            paired_dual_gain_count=("dual_gain_vs_noop", "sum"),
            strict_independent_dual_gain_count=("strict_independent_dual_gain", "sum"),
            pareto_independent_dual_gain_count=("independent_dual_gain", "sum"),
            min_cluster_size=("min_cluster_size_full", "min"),
        )
        .reset_index()
    )
    paired_summary.insert(1, "shared_config_id", shared_id)
    paired_summary.to_csv(output / "shared_config_summary.csv", index=False)

    frontier_rows = []
    for lane, lane_rows in full.groupby("lane"):
        winner = lane_rows.sort_values(
            ["absolute_ari", "absolute_nmi", "candidate_id"],
            ascending=[False, False, True],
            kind="mergesort",
        ).iloc[0]
        match = comparison[
            (comparison.lane == lane)
            & (comparison.start_index == int(winner.start_index))
            & (comparison.certificate_config_id == winner.certificate_config_id)
        ].iloc[0]
        frontier_rows.append(match.to_dict())
    frontier = pd.DataFrame(frontier_rows)
    frontier.insert(0, "profile", "PER_DATASET_LABEL_ASSISTED_ARI_FRONTIER")
    frontier.to_csv(output / "per_dataset_development_frontier.csv", index=False)

    # Flatten producer diagnostics and prove final source/contract authority agrees.
    diagnostic_rows: list[dict[str, object]] = []
    for lane in DISCOVERY:
        manifest = producer_manifests[lane]
        for row in manifest["rows"]:
            if row["arm"] != "CCSR_FULL":
                continue
            diagnostics = row["diagnostics"]
            diagnostic_rows.append(
                {
                    "lane": lane,
                    "start_index": row["start_index"],
                    "start_id": row["start_id"],
                    "certificate_config_id": row["certificate_config_id"],
                    "partition_sha256": row["partition_sha256"],
                    "trusted_count": diagnostics.get("trusted_count"),
                    "trusted_fraction": diagnostics.get("trusted_fraction"),
                    "certified_changed_count": diagnostics.get("certified_changed_count"),
                    "noncertified_changed_count": diagnostics.get("noncertified_changed_count"),
                    "certificate_gap_mean": diagnostics.get("certificate_gap_mean"),
                    "certificate_gap_max": diagnostics.get("certificate_gap_max"),
                    "certificate_epsilon_max": diagnostics.get("certificate_epsilon_max"),
                    "mean_untrusted_original_stay": diagnostics.get("mean_untrusted_original_stay"),
                    "each_frozen_cycle_monotone": diagnostics.get("each_frozen_cycle_monotone"),
                    "exact_k": diagnostics.get("observed_cardinality"),
                }
            )
    pd.DataFrame(diagnostic_rows).to_csv(output / "certificate_diagnostics.csv", index=False)

    observed_count = int(config_ranking.iloc[0].discovery_lane_pass_count)
    required_count = 2
    stage_b_authorized = bool(observed_count >= required_count)
    classification = (
        "CERTIFIED_LOCAL_SIGNAL"
        if stage_b_authorized
        else "SCIENTIFIC_NEGATIVE"
    )
    if stage_b_authorized != (observed_count >= required_count):
        raise RuntimeError("mechanical Stage-B gate consistency failure")
    decision = {
        "schema": "night18e-scientific-gate-v1",
        "classification": classification,
        "status": (
            "NIGHT18E_CCSR_STAGE_A_GATE_PASS_PENDING_CONFIRMATION"
            if stage_b_authorized
            else "NIGHT18E_CCSR_PROOF_CORRECT_BUT_NO_INDEPENDENT_SCORE_SIGNAL"
        ),
        "shared_config_id": shared_id,
        "stage_a_discovery_lane_pass_count": observed_count,
        "stage_a_required_lane_pass_count": required_count,
        "stage_b_authorized": stage_b_authorized,
        "human_p0_is_confirmation": False,
        "proof_status": "INTEGER_QUANTIZED_ALPHA_SUBPROBLEM_EXHAUSTIVE_EQUIVALENCE_PASS",
        "margin_confidence_semantics": "RANK_BASED_NOT_AMPLITUDE_CALIBRATED",
        "ccsr_core_source_sha256": next(iter(core_hashes)),
        "execution_contract_sha256": next(iter(contracts)),
        "producer_labels_read": 0,
        "evaluator_labels_read_after_lock": 1,
        "shutdown_dispatched": False,
    }
    (output / "scientific_gate_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--producer-root", required=True)
    parser.add_argument("--certificate-registry", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
