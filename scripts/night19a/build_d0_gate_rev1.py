#!/usr/bin/env python3
"""Mechanical, label-free Night-19A D0 identifiability gate, REV1.

REV1 implements the taskbook-level notion of persistent *operational* conflict.
The first update (step 1) is a registered zero-start ramp and is reported but is
not an operational checkpoint.  A checkpoint counts only when the direction is
conflicting and both gradients have a non-negligible relative magnitude.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DIAGNOSTIC_STEPS = (0, 1, 5, 20, 40)
ZERO_START_RAMP_STEPS = (0, 1)
OPERATIONAL_STEPS = (5, 20, 40)
CRITICAL_PAIRS = ("relation__anchor", "relation__consistency", "relation__variance")
COSINE_CONFLICT_THRESHOLD = -0.05
MIN_CONFLICT_CHECKPOINTS_PER_SEED = 2
MIN_GRADIENT_NORM_RATIO = 1e-3
MIN_STRATUM_COSINE_SPAN = 0.05
PRIMARY_LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ratio(metrics):
    norm_a = float(metrics["norm_a"])
    norm_b = float(metrics["norm_b"])
    return min(norm_a, norm_b) / max(norm_a, norm_b, 1e-30)


def evaluate_seed(manifest):
    rows = {int(row["completed_steps"]): row for row in manifest["trajectory"]}
    if set(rows) != set(DIAGNOSTIC_STEPS):
        raise ValueError("trajectory checkpoint contract mismatch")
    pair_summary = {}
    for pair in CRITICAL_PAIRS:
        directional_negative_steps = []
        operational_conflict_steps = []
        operational_ratios = []
        step_diagnostics = {}
        for step in (1,) + OPERATIONAL_STEPS:
            metrics = rows[step]["pair_metrics"][pair]
            cosine = metrics["cosine"]
            ratio = _ratio(metrics)
            directional_negative = cosine is not None and float(cosine) <= COSINE_CONFLICT_THRESHOLD
            operational = bool(
                step in OPERATIONAL_STEPS
                and directional_negative
                and ratio >= MIN_GRADIENT_NORM_RATIO
            )
            if directional_negative:
                directional_negative_steps.append(step)
            if operational:
                operational_conflict_steps.append(step)
                operational_ratios.append(ratio)
            step_diagnostics[str(step)] = {
                "cosine": cosine,
                "gradient_norm_ratio": ratio,
                "directional_negative": bool(directional_negative),
                "operational_conflict": operational,
                "zero_start_ramp_excluded": step == 1,
            }
        pair_summary[pair] = {
            "directional_negative_steps_including_ramp": directional_negative_steps,
            "operational_conflict_steps": operational_conflict_steps,
            "operational_conflict_checkpoint_count": len(operational_conflict_steps),
            "min_operational_gradient_norm_ratio": min(operational_ratios) if operational_ratios else None,
            "step_diagnostics": step_diagnostics,
            "seed_pair_pass": len(operational_conflict_steps) >= MIN_CONFLICT_CHECKPOINTS_PER_SEED,
        }
    stratum_spans = []
    stratum_defined_counts = []
    for step in OPERATIONAL_STEPS:
        values = [
            item["cosine"]
            for item in rows[step]["relation_stratum_vs_anchor"].values()
            if item["cosine"] is not None
        ]
        stratum_defined_counts.append(len(values))
        stratum_spans.append(max(values) - min(values) if len(values) >= 2 else None)
    evidence_heterogeneous = any(
        value is not None and float(value) >= MIN_STRATUM_COSINE_SPAN for value in stratum_spans
    )
    anchor_norms = [float(rows[step]["gradient_norms"]["anchor"]) for step in OPERATIONAL_STEPS]
    return {
        "pair_summary": pair_summary,
        "evidence_stratum_cosine_spans_operational": stratum_spans,
        "evidence_stratum_defined_counts_operational": stratum_defined_counts,
        "evidence_heterogeneous": evidence_heterogeneous,
        "anchor_gradient_norms_operational": anchor_norms,
        "anchor_identifiable_operational": any(value > 1e-12 for value in anchor_norms),
        "step0_and_step1_excluded_from_operational_gate": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "night19a-gradient-d0-formula-freeze-rev1":
        raise ValueError("REV1 contract schema mismatch")
    manifests = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.manifest]
    by_lane = {}
    for manifest in manifests:
        lane = manifest["lane"]
        seed = int(manifest["training_seed"])
        if seed in by_lane.setdefault(lane, {}):
            raise ValueError("duplicate lane/seed manifest")
        by_lane[lane][seed] = evaluate_seed(manifest)
    lane_results = []
    for lane in sorted(by_lane):
        seeds = by_lane[lane]
        if set(seeds) != {0, 1}:
            raise ValueError("D0 requires seeds 0 and 1 per lane")
        reproducible_pairs = [
            pair for pair in CRITICAL_PAIRS
            if all(seeds[seed]["pair_summary"][pair]["seed_pair_pass"] for seed in (0, 1))
        ]
        evidence_pass = all(
            seeds[seed]["evidence_heterogeneous"] and seeds[seed]["anchor_identifiable_operational"]
            for seed in (0, 1)
        )
        lane_manifest = next(item for item in manifests if item["lane"] == lane)
        support = lane_manifest["evidence_strata"]
        support_nonconstant = bool(
            int(support["support_unique_count"]) >= 3
            and all(int(value) > 0 for value in support["stratum_counts"])
        )
        lane_pass = bool(reproducible_pairs and evidence_pass and support_nonconstant)
        lane_results.append({
            "lane": lane,
            "primary": lane in PRIMARY_LANES,
            "lane_pass": lane_pass,
            "reproducible_critical_pairs": reproducible_pairs,
            "evidence_pass": evidence_pass,
            "support_nonconstant": support_nonconstant,
            "seed_diagnostics": {str(seed): seeds[seed] for seed in (0, 1)},
        })
    primary_pass = sum(int(row["lane_pass"]) for row in lane_results if row["primary"])
    stage_a_authorized = primary_pass >= 2
    decision = {
        "schema": "night19a-gradient-d0-gate-rev1",
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": file_sha256(contract_path),
        "supersedes": {
            "reason": "REV0 treated the zero-start step1 ramp as an operational magnitude requirement, structurally creating a false negative against the taskbook persistence semantics.",
            "rev0_gate_output_sha256": "e6cc76a26eaf243cd591eec2a909a9005a5137bff151c2844336d14e0378bdd6",
            "rev0_gate_source_sha256": "12c673312273b209c58ce09830cdf62bb5618adc5f2a74cdd52ecdafdc0a20c3",
            "classification": "SUPERSEDED_IMPLEMENTATION_SEMANTICS_INVALID",
        },
        "thresholds_frozen_before_rev1_rerun": {
            "diagnostic_steps": list(DIAGNOSTIC_STEPS),
            "zero_start_ramp_steps_excluded": list(ZERO_START_RAMP_STEPS),
            "operational_steps": list(OPERATIONAL_STEPS),
            "critical_pairs": list(CRITICAL_PAIRS),
            "cosine_conflict_threshold": COSINE_CONFLICT_THRESHOLD,
            "min_operational_conflict_checkpoints_per_seed": MIN_CONFLICT_CHECKPOINTS_PER_SEED,
            "min_gradient_norm_ratio": MIN_GRADIENT_NORM_RATIO,
            "min_stratum_cosine_span": MIN_STRATUM_COSINE_SPAN,
            "required_primary_lanes": 2,
        },
        "revision_scope": "HIGH_LEVEL_CONTRACT_INTERPRETATION_ONLY_NO_LABEL_OR_SCORE_READ",
        "zero_norm_cosine_semantics": "NULL_NA_NOT_ZERO",
        "primary_lane_pass_count": primary_pass,
        "required_primary_lane_pass_count": 2,
        "stage_a_authorized": stage_a_authorized,
        "terminal_if_not_authorized": "NO_OPERATIONALLY_IDENTIFIABLE_PERSISTENT_GRADIENT_CONFLICT_OBJECT",
        "lane_results": lane_results,
    }
    Path(args.output).write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
