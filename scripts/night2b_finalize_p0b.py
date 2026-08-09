#!/usr/bin/env python3
"""Create explicit stopped-run compact artifacts after the immutable P0B failure."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results/night2b"
REPORTS = REPO / "reports"


def write_csv(path: Path, fields, rows=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def max_abs(obj):
    values = []
    if isinstance(obj, dict):
        if isinstance(obj.get("max_absolute_difference"), (int, float)):
            values.append(float(obj["max_absolute_difference"]))
        for value in obj.values():
            values.extend(max_abs(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(max_abs(value))
    return values


def main():
    p0b = json.loads((REPORTS / "night2b_parity_locked.json").read_text(encoding="utf-8"))
    diagnosis = json.loads((REPORTS / "night2b_p0b_diagnosis.json").read_text(encoding="utf-8"))
    if p0b["p0b_pass"] is not False or p0b["factorial_authorized"] is not False:
        raise RuntimeError("Stopped-run finalizer requires the original P0B failure")
    if list((RESULTS / "raw").glob("*/*/seed_*/metrics.json")):
        raise AssertionError("A P0B failure must have zero factorial metrics")
    if list((RESULTS / "tutorial2022").glob("*/metrics.json")):
        raise AssertionError("A P0B failure must have zero tutorial metrics")

    p0b_rows = []
    for dataset in ("a1", "placenta", "p22"):
        item = p0b["datasets"][dataset]
        one = item["v3_one_adam_step"]
        trajectory = item["v3_five_step_trajectory"]
        p0b_rows.append({
            "dataset": dataset,
            "p0b_pass": item["pass"],
            "consumed_inputs_pass": item["consumed_inputs"]["pass"],
            "initial_state_pass": item["initial_model_state"]["pass"],
            "forward_pass": one["forward"]["pass"],
            "forward_max_abs_diff": max(max_abs(one["forward"])),
            "loss_pass": one["losses"]["pass"],
            "loss_max_abs_diff": max(max_abs(one["losses"])),
            "gradient_pass": one["gradients"]["pass"],
            "gradient_max_abs_diff": max(max_abs(one["gradients"])),
            "adam_parameter_pass": one["updated_parameters"]["pass"],
            "adam_parameter_max_abs_diff": max(max_abs(one["updated_parameters"])),
            "optimizer_state_pass": one["optimizer_state"]["pass"],
            "optimizer_state_max_abs_diff": max(max_abs(one["optimizer_state"])),
            "five_step_pass": trajectory["pass"],
            "five_step_max_parameter_diff": max(step["maximum_parameter_difference"] for step in trajectory["steps"]),
            "same_model_gpu_repeat_max_diff": diagnosis["datasets"][dataset]["gpu_same_model_repeat"]["maximum_output_difference"],
            "same_model_cpu_repeat_max_diff": diagnosis["datasets"][dataset]["cpu_same_model_repeat"]["maximum_output_difference"],
            "m_bad": item["consumed_inputs"]["m_bad"],
        })
    write_csv(RESULTS / "p0b_summary.csv", p0b_rows[0].keys(), p0b_rows)

    write_csv(
        RESULTS / "per_seed_metrics.csv",
        ("dataset", "variant", "seed", "ari", "nmi", "ami", "fmi", "hungarian_macro_f1", "spatial_neighbor_agreement", "spatial_cluster_moran_mean", "embedding_silhouette", "embedding_davies_bouldin", "total_seconds", "gpu_peak_allocated_mib", "run_status"),
    )
    write_csv(
        RESULTS / "summary.csv",
        ("dataset", "variant", "metric", "mean", "sample_sd", "n", "run_status"),
    )
    write_csv(
        RESULTS / "paired_deltas.csv",
        ("dataset", "variant", "seed", "metric", "value", "locked_v0_value", "paired_delta", "run_status"),
    )
    write_csv(
        RESULTS / "v3_replay_audit.csv",
        ("dataset", "seed", "partition_ari", "attention_max_abs_diff", "ari_delta", "nmi_delta", "cluster_count_equal", "p1_warning", "run_status"),
    )
    write_csv(
        RESULTS / "factorial_effects.csv",
        ("dataset", "metric", "legacy_gap", "scale_recovery", "shape_recovery", "replay_recovery", "interaction", "run_status"),
    )
    write_csv(
        RESULTS / "loss_components.csv",
        ("dataset", "variant", "seed", "epoch", "raw_rna_reconstruction", "weighted_rna_before_global_scale", "global_scale_multiplier", "final_rna_contribution", "raw_modality2_reconstruction", "final_modality2_contribution", "raw_corr1", "final_corr1_contribution", "raw_corr2", "final_corr2_contribution", "total_loss", "m_bad", "gene_weight_mean", "gene_weight_min", "gene_weight_max", "run_status"),
    )
    write_csv(
        RESULTS / "attention_summary.csv",
        ("dataset", "variant", "seed", "cross_omics_rna", "rna_spatial", "modality2_spatial", "delta_vs_locked_v0", "delta_vs_night1_legacy", "run_status"),
    )
    write_csv(
        RESULTS / "per_domain_f1.csv",
        ("dataset", "variant", "seed", "domain", "support", "hungarian_f1", "run_status"),
    )

    with (REPO / "results/night1/summary.csv").open(newline="", encoding="utf-8") as handle:
        legacy = {row["dataset"]: row for row in csv.DictReader(handle) if row["variant"] == "legacy_exact"}
    manuscript = {"a1": (0.2443, 0.3780), "placenta": (0.7226, 0.7408), "p22": (0.4541, 0.5747)}
    paper_rows = []
    for dataset in ("a1", "placenta", "p22"):
        paper_rows.append({
            "dataset": dataset,
            "manuscript_ari": manuscript[dataset][0],
            "manuscript_nmi": manuscript[dataset][1],
            "night1_legacy_mean_ari": legacy[dataset]["ari_mean"],
            "night1_legacy_mean_nmi": legacy[dataset]["nmi_mean"],
            "tutorial2022_ari": "",
            "tutorial2022_nmi": "",
            "environment": "SpaLORA_torch112 planned",
            "notes": "NOT RUN: P0B hard gate failed before all training; no seed search.",
        })
    write_csv(
        RESULTS / "paper_repro_audit.csv",
        ("dataset", "manuscript_ari", "manuscript_nmi", "night1_legacy_mean_ari", "night1_legacy_mean_nmi", "tutorial2022_ari", "tutorial2022_nmi", "environment", "notes"),
        paper_rows,
    )

    gate = {
        "status": "stopped_by_p0b_hard_gate",
        "p0b_pass": False,
        "factorial_authorized": False,
        "main_runs": {"planned_if_authorized": 75, "completed": 0, "failed": 0, "skipped_by_gate": 75},
        "tutorial_runs": {"planned_if_authorized": 3, "completed": 0, "failed": 0, "skipped_by_gate": 3},
        "p1_status": "not_evaluated",
        "s_w_i_a_status": "not_evaluated",
        "ground_truth_accessed_during_p0b": False,
        "ground_truth_used_for_setting_selection": False,
        "seed_search_performed": False,
        "asr_modified": False,
        "empty_factorial_csvs_are_intentional": True,
        "reason": "Independent frozen-model GPU forwards exceeded the preregistered 1e-7 P0B tolerance.",
    }
    (RESULTS / "gate_status.json").write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    completion = {
        "status": gate["status"],
        "parent_commit": "16f0cc43673617c73527110962b7ca115c59b4c6",
        "branch": "revision/q2-night2b-parity-locked-loss-audit-20260809",
        "final_commit_reference": "annotated tag night2b-p0b-final-20260809",
        "p0b_pass": False,
        "main_runs_completed": 0,
        "tutorial_runs_completed": 0,
        "protected_manifest": "/root/autodl-fs/night2b_preexisting_20260809/protected_before.sha256",
        "shutdown_command_required_last": "/usr/bin/shutdown",
    }
    (REPORTS / "night2b_completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
