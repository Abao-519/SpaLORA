#!/usr/bin/env python3
"""Mechanical, label-free Night-19A D0 identifiability gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


POST_ZERO_STEPS = (1, 5, 20, 40)
CRITICAL_PAIRS = ("relation__anchor", "relation__consistency", "relation__variance")
COSINE_CONFLICT_THRESHOLD = -0.05
MIN_CONFLICT_CHECKPOINTS_PER_SEED = 2
MIN_GRADIENT_NORM_RATIO = 1e-3
MIN_STRATUM_COSINE_SPAN = 0.05
PRIMARY_LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")


def evaluate_seed(manifest):
    rows = {int(row["completed_steps"]): row for row in manifest["trajectory"]}
    if set(rows) != {0, 1, 5, 20, 40}:
        raise ValueError("trajectory checkpoint contract mismatch")
    pair_summary = {}
    for pair in CRITICAL_PAIRS:
        conflicts = []
        ratios = []
        for step in POST_ZERO_STEPS:
            metrics = rows[step]["pair_metrics"][pair]
            cosine = metrics["cosine"]
            norm_a = float(metrics["norm_a"])
            norm_b = float(metrics["norm_b"])
            if cosine is not None and float(cosine) <= COSINE_CONFLICT_THRESHOLD:
                conflicts.append(step)
                ratios.append(min(norm_a, norm_b) / max(norm_a, norm_b, 1e-30))
        pair_summary[pair] = {
            "conflict_steps": conflicts,
            "conflict_checkpoint_count": len(conflicts),
            "min_conflicting_norm_ratio": min(ratios) if ratios else None,
            "seed_pair_pass": bool(
                len(conflicts) >= MIN_CONFLICT_CHECKPOINTS_PER_SEED
                and ratios
                and min(ratios) >= MIN_GRADIENT_NORM_RATIO
            ),
        }
    stratum_spans = []
    stratum_defined_counts = []
    for step in POST_ZERO_STEPS:
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
    anchor_norms = [float(rows[step]["gradient_norms"]["anchor"]) for step in POST_ZERO_STEPS]
    return {
        "pair_summary": pair_summary,
        "evidence_stratum_cosine_spans": stratum_spans,
        "evidence_stratum_defined_counts": stratum_defined_counts,
        "evidence_heterogeneous": evidence_heterogeneous,
        "anchor_gradient_norms_post_zero": anchor_norms,
        "anchor_identifiable_post_zero": any(value > 1e-12 for value in anchor_norms),
        "step0_excluded_from_persistence_gate": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
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
            seeds[seed]["evidence_heterogeneous"] and seeds[seed]["anchor_identifiable_post_zero"]
            for seed in (0, 1)
        )
        support = manifests[[item["lane"] for item in manifests].index(lane)]["evidence_strata"]
        support_nonconstant = bool(
            int(support["support_unique_count"]) >= 3
            and all(int(value) > 0 for value in support["stratum_counts"])
        )
        lane_pass = bool(reproducible_pairs and evidence_pass and support_nonconstant)
        lane_results.append(
            {
                "lane": lane,
                "primary": lane in PRIMARY_LANES,
                "lane_pass": lane_pass,
                "reproducible_critical_pairs": reproducible_pairs,
                "evidence_pass": evidence_pass,
                "support_nonconstant": support_nonconstant,
                "seed_diagnostics": {str(seed): seeds[seed] for seed in (0, 1)},
            }
        )
    primary_pass = sum(int(row["lane_pass"]) for row in lane_results if row["primary"])
    stage_a_authorized = primary_pass >= 2
    decision = {
        "schema": "night19a-gradient-d0-gate-v1",
        "thresholds_frozen_before_real_d0_evaluation": {
            "post_zero_steps": list(POST_ZERO_STEPS),
            "critical_pairs": list(CRITICAL_PAIRS),
            "cosine_conflict_threshold": COSINE_CONFLICT_THRESHOLD,
            "min_conflict_checkpoints_per_seed": MIN_CONFLICT_CHECKPOINTS_PER_SEED,
            "min_gradient_norm_ratio": MIN_GRADIENT_NORM_RATIO,
            "min_stratum_cosine_span": MIN_STRATUM_COSINE_SPAN,
            "required_primary_lanes": 2,
        },
        "step0_semantics": "ZERO_START_BOUNDARY_EXCLUDED_FROM_CONFLICT_PERSISTENCE",
        "zero_norm_cosine_semantics": "NULL_NA_NOT_ZERO",
        "primary_lane_pass_count": primary_pass,
        "required_primary_lane_pass_count": 2,
        "stage_a_authorized": stage_a_authorized,
        "terminal_if_not_authorized": "NO_IDENTIFIABLE_GRADIENT_CONFLICT_OBJECT",
        "lane_results": lane_results,
    }
    Path(args.output).write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

