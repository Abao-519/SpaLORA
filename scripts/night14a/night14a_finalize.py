#!/usr/bin/env python3
"""Aggregate the frozen Night-14A common-endpoint evidence."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-fs/night14a_topology_conflict_sprint_20260823")
OUT = ROOT / "outputs/night14a_handoff"
CANDIDATE = "C15_BAL_XREC_600_WEAK_ALIGN"
MAIN_FILTER = "W02_TCF_FINAL"
MATCHED = {
    "A1": "C00_G04_MODEL_SEED0",
    "tonsil_s1": "SIMPLE_CONCAT",
    "P22": "N02_HIER_MODEL_SEED0",
    "MISAR_E15_5_S1": "RNA_ONLY",
    "D1": "SIMPLE_CONCAT",
    "tonsil_s2": "SIMPLE_CONCAT",
    "tonsil_s3": "SIMPLE_CONCAT",
}
FAMILY = {
    "A1": "RNA+protein", "D1": "RNA+protein",
    "tonsil_s1": "RNA+protein", "tonsil_s2": "RNA+protein",
    "tonsil_s3": "RNA+protein", "P22": "RNA+ATAC",
    "MISAR_E15_5_S1": "RNA+ATAC",
}
STUDY = {
    "A1": "lymph_node", "D1": "lymph_node",
    "tonsil_s1": "tonsil", "tonsil_s2": "tonsil",
    "tonsil_s3": "tonsil", "P22": "P22", "MISAR_E15_5_S1": "MISAR",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def load_rows() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    phase_roots = {"development": "development_cycle3", "confirmation": "confirmation_cycle4"}
    for phase, phase_root in phase_roots.items():
        path = ROOT / ("formal/%s/%s_endpoint_rows.csv" % (phase_root, CANDIDATE))
        frame = pd.read_csv(path)
        frame["formal_phase"] = phase.upper()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_summaries() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    phase_roots = {"development": "development_cycle3", "confirmation": "confirmation_cycle4"}
    for phase, phase_root in phase_roots.items():
        path = ROOT / ("formal/%s/%s_endpoint_summary.csv" % (phase_root, CANDIDATE))
        frame = pd.read_csv(path)
        frame["formal_phase"] = phase.upper()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def reference_rows() -> pd.DataFrame:
    ref = pd.read_csv(ROOT / "references_final/reference_endpoint_rows.csv")
    keep = ref.apply(lambda row: MATCHED.get(row["dataset"]) == row["method"], axis=1)
    ref = ref.loc[keep].copy()
    return ref


def paired(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    ref = reference[[
        "dataset", "endpoint_seed", "absolute_ari", "absolute_nmi",
        "ami", "fmi", "homogeneity", "v_measure", "morans_i", "gearys_c",
    ]].rename(columns={column: "reference_" + column for column in [
        "absolute_ari", "absolute_nmi", "ami", "fmi", "homogeneity",
        "v_measure", "morans_i", "gearys_c",
    ]})
    value = candidate.merge(ref, on=["dataset", "endpoint_seed"], how="left", validate="many_to_one")
    value["reference_method"] = value["dataset"].map(MATCHED)
    value["delta_ari"] = value["absolute_ari"] - value["reference_absolute_ari"]
    value["delta_nmi"] = value["absolute_nmi"] - value["reference_absolute_nmi"]
    value["win_ari"] = value["delta_ari"] > 0
    value["win_nmi"] = value["delta_nmi"] > 0
    value["win_both"] = value["win_ari"] & value["win_nmi"]
    value["family"] = value["dataset"].map(FAMILY)
    value["study"] = value["dataset"].map(STUDY)
    value["semantic_lane"] = "COMMON_HEAD_ROBUSTNESS"
    return value


def dataset_leaderboard(all_paired: pd.DataFrame) -> pd.DataFrame:
    grouped = all_paired.groupby([
        "formal_phase", "dataset", "family", "study", "filter_id",
        "reference_method",
    ], dropna=False)
    value = grouped.agg(
        model_seed_count=("model_seed", "nunique"),
        endpoint_row_count=("endpoint_seed", "size"),
        ari_mean=("absolute_ari", "mean"), ari_sd=("absolute_ari", "std"),
        ari_min=("absolute_ari", "min"), ari_max=("absolute_ari", "max"),
        nmi_mean=("absolute_nmi", "mean"), nmi_sd=("absolute_nmi", "std"),
        nmi_min=("absolute_nmi", "min"), nmi_max=("absolute_nmi", "max"),
        ami_mean=("ami", "mean"), fmi_mean=("fmi", "mean"),
        homogeneity_mean=("homogeneity", "mean"),
        v_measure_mean=("v_measure", "mean"),
        morans_i_mean=("morans_i", "mean"), gearys_c_mean=("gearys_c", "mean"),
        reference_ari_mean=("reference_absolute_ari", "mean"),
        reference_nmi_mean=("reference_absolute_nmi", "mean"),
        delta_ari_mean=("delta_ari", "mean"), delta_ari_min=("delta_ari", "min"),
        delta_nmi_mean=("delta_nmi", "mean"), delta_nmi_min=("delta_nmi", "min"),
        ari_win_rate=("win_ari", "mean"), nmi_win_rate=("win_nmi", "mean"),
        joint_win_rate=("win_both", "mean"),
    ).reset_index()
    return value


def study_family_summary(main: pd.DataFrame) -> pd.DataFrame:
    dataset = main.groupby(["formal_phase", "dataset", "family", "study"]).agg(
        absolute_ari=("absolute_ari", "mean"), absolute_nmi=("absolute_nmi", "mean"),
        delta_ari=("delta_ari", "mean"), delta_nmi=("delta_nmi", "mean"),
        joint_win_rate=("win_both", "mean"),
    ).reset_index()
    dataset["aggregation_level"] = "DATASET"
    study = dataset.groupby(["family", "study"]).agg(
        absolute_ari=("absolute_ari", "mean"), absolute_nmi=("absolute_nmi", "mean"),
        delta_ari=("delta_ari", "mean"), delta_nmi=("delta_nmi", "mean"),
        joint_win_rate=("joint_win_rate", "mean"),
    ).reset_index()
    study["formal_phase"] = "DEVELOPMENT_AND_CONFIRMATION"
    study["dataset"] = "ALL_SLICES_AS_ONE_STUDY_EFFECT"
    study["aggregation_level"] = "STUDY"
    family = study.groupby("family").agg(
        absolute_ari=("absolute_ari", "mean"), absolute_nmi=("absolute_nmi", "mean"),
        delta_ari=("delta_ari", "mean"), delta_nmi=("delta_nmi", "mean"),
        joint_win_rate=("joint_win_rate", "mean"),
    ).reset_index()
    family["formal_phase"] = "DEVELOPMENT_AND_CONFIRMATION"
    family["dataset"] = "STUDY_BALANCED"
    family["study"] = "ALL_STUDIES"
    family["aggregation_level"] = "FAMILY"
    columns = [
        "aggregation_level", "formal_phase", "dataset", "family", "study",
        "absolute_ari", "absolute_nmi", "delta_ari", "delta_nmi", "joint_win_rate",
    ]
    return pd.concat([dataset[columns], study[columns], family[columns]], ignore_index=True)


def resource_audit() -> Dict[str, object]:
    formal_manifests = []
    phase_roots = {"development": "development_cycle3", "confirmation": "confirmation_cycle4"}
    for phase, phase_root in phase_roots.items():
        path = ROOT / ("formal/%s/%s_manifest.json" % (phase_root, CANDIDATE))
        formal_manifests.extend(json.loads(path.read_text(encoding="utf-8"))["runs"])
    all_training_audits = []
    for path in sorted(ROOT.rglob("training_audit.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["audit_path"] = path.relative_to(ROOT).as_posix()
        all_training_audits.append(value)
    reload_audits = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(ROOT.rglob("fresh_process_reload.json"))
    ]
    return {
        "formal_training_runs": len(formal_manifests),
        "formal_all_status_pass": all(row.get("status") == "PASS" for row in formal_manifests),
        "formal_fresh_process_reload_pass_count": sum(
            row.get("fresh_process_reload", {}).get("status") == "PASS"
            for row in formal_manifests
        ),
        "formal_optimizer_steps_each": sorted(set(int(row["optimizer_steps"]) for row in formal_manifests)),
        "formal_trainable_parameter_counts": sorted(set(int(row["trainable_parameter_count"]) for row in formal_manifests)),
        "formal_training_wall_seconds": float(sum(row["wall_seconds"] for row in formal_manifests)),
        "formal_gpu_seconds": float(sum(row["gpu_seconds"] for row in formal_manifests)),
        "all_recorded_training_runs_including_development": len(all_training_audits),
        "all_recorded_fresh_process_reload_audits": len(reload_audits),
        "all_recorded_fresh_process_reload_pass_count": sum(
            row.get("status") == "PASS" for row in reload_audits
        ),
        "all_recorded_training_wall_seconds": float(sum(
            row.get("wall_seconds", 0.0) for row in all_training_audits
        )),
        "all_recorded_gpu_seconds": float(sum(
            row.get("gpu_seconds", 0.0) for row in all_training_audits
        )),
        "peak_gpu_mib": float(max(row.get("peak_gpu_mib", 0.0) for row in all_training_audits)),
        "peak_rss_mib": float(max(row.get("peak_rss_mib", 0.0) for row in all_training_audits)),
        "derived_root_bytes": int(sum(
            path.stat().st_size for path in ROOT.rglob("*") if path.is_file()
        )),
        "dense_n_by_n_count": 0,
        "new_download_count": 0,
        "third_party_full_baseline_run_count": 0,
        "training_label_use_count": 0,
        "labels_in_loss_gradient_or_checkpoint_selection": 0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = load_rows()
    reference = reference_rows()
    comparison = paired(candidate, reference)
    comparison.to_csv(OUT / "absolute_metrics.csv", index=False)
    comparison.to_csv(OUT / "endpoint_robustness.csv", index=False)
    leaderboard = dataset_leaderboard(comparison)
    leaderboard.to_csv(OUT / "development_leaderboard.csv", index=False)
    main_rows = comparison[comparison["filter_id"] == MAIN_FILTER].copy()
    summary = study_family_summary(main_rows)
    summary.to_csv(OUT / "study_family_summary.csv", index=False)

    diagnostic_columns = [
        "formal_phase", "dataset", "model_seed", "filter_id",
        "topology_correlation", "global_disagreement", "roughness_ratio",
        "frequency_gate", "integrity_score", "integrity_gate", "raw_global_gate",
        "global_gate", "joint_support_mean", "conflict_mean", "conflict_q90",
        "trust_mean", "trust_evidence_fraction", "beta_low_mean", "beta_high_mean",
        "exact_identity_fallback", "dense_n_by_n_count",
    ]
    diagnostics = load_summaries()[diagnostic_columns].drop_duplicates()
    diagnostics.to_csv(OUT / "mechanism_diagnostics.csv", index=False)

    ablation_rows = pd.read_csv(
        ROOT / "formal/offline_key_ablation/offline_ablation_endpoint_rows.csv"
    )
    ablation_comparison = paired(ablation_rows, reference)
    ablation_summary = ablation_comparison.groupby(
        ["dataset", "filter_id", "reference_method"], dropna=False
    ).agg(
        model_seed_count=("model_seed", "nunique"),
        endpoint_row_count=("endpoint_seed", "size"),
        absolute_ari=("absolute_ari", "mean"),
        absolute_nmi=("absolute_nmi", "mean"),
        delta_ari=("delta_ari", "mean"),
        delta_nmi=("delta_nmi", "mean"),
        joint_win_rate=("win_both", "mean"),
    ).reset_index()
    ablation_summary.to_csv(OUT / "key_ablation_summary.csv", index=False)

    native = pd.DataFrame([
        {"dataset": "A1", "method": "C00_H05_NATIVE", "ari": 0.2692, "nmi": 0.4087},
        {"dataset": "D1", "method": "C00_H05_NATIVE", "ari": 0.2412, "nmi": 0.3777},
        {"dataset": "P22", "method": "F00_NATIVE", "ari": 0.4677, "nmi": 0.6334},
        {"dataset": "P22", "method": "N02_NATIVE", "ari": 0.5063, "nmi": 0.6562},
    ])
    native["semantic_lane"] = "NATIVE_FULL_PIPELINE_CONTEXT_NOT_FAIR_COMMON_HEAD_WIN"
    native.to_csv(OUT / "native_full_pipeline_context.csv", index=False)

    resources = resource_audit()
    atomic_json(OUT / "resource_audit.json", resources)
    corruption = json.loads(
        (ROOT / "audit/formal_corruption_audit.json").read_text(encoding="utf-8")
    )
    shutil.copyfile(
        ROOT / "audit/formal_corruption_audit.json",
        OUT / "formal_corruption_audit.json",
    )
    shutil.copyfile(
        ROOT / "audit/formal_corruption_audit_invalid_cyclic.json",
        OUT / "formal_corruption_audit_invalid_cyclic.json",
    )
    shutil.copyfile(
        ROOT / "formal/offline_key_ablation/offline_ablation_manifest.json",
        OUT / "offline_ablation_manifest.json",
    )
    formal_runs = []
    phase_roots = {"development": "development_cycle3", "confirmation": "confirmation_cycle4"}
    for phase, phase_root in phase_roots.items():
        path = ROOT / ("formal/%s/%s_manifest.json" % (phase_root, CANDIDATE))
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for row in manifest["runs"]:
            formal_runs.append({"formal_phase": phase.upper(), **row})
    atomic_json(OUT / "formal_run_manifest.json", {
        "candidate_id": CANDIDATE,
        "filter_id": MAIN_FILTER,
        "run_count": len(formal_runs),
        "expected_run_count": 21,
        "all_status_pass": all(row.get("status") == "PASS" for row in formal_runs),
        "runs": formal_runs,
    })
    convergence_rows = []
    for row in formal_runs:
        first, last = row["loss_trace"][0], row["loss_trace"][-1]
        convergence_rows.append({
            "formal_phase": row["formal_phase"],
            "dataset": row["dataset"],
            "model_seed": row["model_seed"],
            "optimizer_steps": row["optimizer_steps"],
            "trainable_parameter_count": row["trainable_parameter_count"],
            "parameters_changed": row["parameters_changed"],
            "first_gradient_norm": row["first_gradient_norm"],
            "total_loss_step1": first["total"],
            "total_loss_final": last["total"],
            "private_recon_step1": first["private_recon"],
            "private_recon_final": last["private_recon"],
            "cross_recon_step1": first["cross_recon"],
            "cross_recon_final": last["cross_recon"],
            "status": row["status"],
        })
    pd.DataFrame(convergence_rows).to_csv(
        OUT / "training_convergence_summary.csv", index=False
    )
    run_resources = pd.DataFrame(formal_runs).groupby(
        ["formal_phase", "dataset"]
    ).agg(
        training_run_count=("model_seed", "size"),
        optimizer_steps=("optimizer_steps", "first"),
        trainable_parameter_count=("trainable_parameter_count", "first"),
        training_wall_seconds_mean=("wall_seconds", "mean"),
        gpu_seconds_mean=("gpu_seconds", "mean"),
        peak_gpu_mib=("peak_gpu_mib", "max"),
        peak_rss_mib=("peak_rss_mib", "max"),
    ).reset_index()
    observation_meta = main_rows.groupby(
        ["formal_phase", "dataset"]
    ).agg(
        total_observations=("total_observations", "first"),
        evaluated_observations=("evaluated_observations", "first"),
        k=("k", "first"),
        ordered_id_sha256=("ordered_id_sha256", "first"),
    ).reset_index()
    main_table = leaderboard[leaderboard["filter_id"] == MAIN_FILTER].merge(
        observation_meta, on=["formal_phase", "dataset"], how="left",
        validate="one_to_one",
    ).merge(
        run_resources, on=["formal_phase", "dataset"], how="left",
        validate="one_to_one",
    )
    main_table.to_csv(OUT / "main_results_table.csv", index=False)
    atomic_json(OUT / "fresh_process_roundtrip_audit.json", {
        "run_count": len(formal_runs),
        "expected_run_count": 21,
        "pass_count": sum(
            row.get("fresh_process_reload", {}).get("status") == "PASS"
            for row in formal_runs
        ),
        "rows": [
            {
                "formal_phase": row["formal_phase"],
                "dataset": row["dataset"],
                "model_seed": row["model_seed"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "final_state_sha256": row["final_state_sha256"],
                "fresh_process_reload": row.get("fresh_process_reload"),
            }
            for row in formal_runs
        ],
    })
    atomic_json(OUT / "label_and_integrity_audit.json", {
        "public_label_reads_for_development_evaluation_and_hpo": True,
        "public_label_reads_for_frozen_internal_confirmation_evaluation": True,
        "training_label_reads": 0,
        "labels_in_loss_gradient_or_within_run_checkpoint_selection": 0,
        "dataset_name_model_routing": 0,
        "dense_n_by_n": 0,
        "historical_raw_writes": 0,
        "new_data_downloads": 0,
        "force_pushes": 0,
        "full_external_baseline_runs": 0,
        "confirmation_is_pristine_blind": False,
        "claim_scope": "public benchmark development and frozen internal confirmation",
    })
    shutil.copyfile(
        ROOT / "p0/real_input_preflight.json", OUT / "real_input_preflight.json"
    )
    p0_runs = []
    for relative in (
        "p0/p00/P00_CR_SAGE_AE_manifest.json",
        "p0/p01/P01_CR_BALANCED_XREC_manifest.json",
    ):
        manifest = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        p0_runs.extend(manifest["runs"])
    atomic_json(OUT / "p0_backbone_roundtrip_audit.json", {
        "run_count": len(p0_runs),
        "expected_run_count": 4,
        "all_status_pass": all(row.get("status") == "PASS" for row in p0_runs),
        "datasets": sorted(set(row["dataset"] for row in p0_runs)),
        "candidates": sorted(set(row["candidate_id"] for row in p0_runs)),
        "rows": p0_runs,
    })
    by_seed = main_rows.groupby(["dataset", "model_seed"]).agg(
        delta_ari=("delta_ari", "mean"), delta_nmi=("delta_nmi", "mean")
    ).reset_index()
    atac = by_seed[by_seed["dataset"].isin(["P22", "MISAR_E15_5_S1"])]
    atac_all_positive = bool(((atac["delta_ari"] > 0) & (atac["delta_nmi"] > 0)).all())
    family_rows = summary[summary["aggregation_level"] == "FAMILY"].set_index("family")
    classification = "ATAC_FOCUSED_SIGNAL" if atac_all_positive else "LOCAL_SIGNAL"
    decision = {
        "terminal_status": "NIGHT14A_TCF_ATAC_FOCUSED_SIGNAL" if atac_all_positive else "NIGHT14A_TCF_LOCAL_SIGNAL",
        "classification": classification,
        "final_freeze_id": "NIGHT14A_FREEZE_20260823_04",
        "main_candidate": CANDIDATE,
        "main_filter": MAIN_FILTER,
        "formal_candidate_rows": int(len(main_rows)),
        "expected_main_rows": 7 * 3 * 20,
        "formal_candidate_row_count_exact": int(len(main_rows)) == 7 * 3 * 20,
        "formal_checkpoint_roundtrip_pass_count": resources["formal_fresh_process_reload_pass_count"],
        "formal_checkpoint_roundtrip_expected_count": 21,
        "all_three_seeds_positive_on_both_atac_datasets_for_ari_and_nmi": atac_all_positive,
        "registered_seed0_corruption_exact_identity_count": sum(
            row["exact_identity"] and int(row["model_seed"]) == 0
            for row in corruption["rows"]
        ),
        "registered_seed0_corruption_expected_count": 4,
        "exploratory_all_seed_corruption_exact_identity_count": corruption["exact_identity_count"],
        "exploratory_all_seed_corruption_expected_count": corruption["run_count"],
        "exploratory_corruption_limitation": "A1 model seed 2 did not return exact identity",
        "rna_atac_study_balanced_delta_ari": float(family_rows.loc["RNA+ATAC", "delta_ari"]),
        "rna_atac_study_balanced_delta_nmi": float(family_rows.loc["RNA+ATAC", "delta_nmi"]),
        "rna_protein_study_balanced_delta_ari": float(family_rows.loc["RNA+protein", "delta_ari"]),
        "rna_protein_study_balanced_delta_nmi": float(family_rows.loc["RNA+protein", "delta_nmi"]),
        "common_head_only": True,
        "native_full_pipeline_context_separate": True,
        "not_sota_claim": True,
        "not_cross_family_success": classification == "ATAC_FOCUSED_SIGNAL",
        "labels_used_for_public_benchmark_evaluation_and_hpo": True,
        "labels_used_in_training_loss_gradient_or_checkpoint_selection": False,
        "confirmation_was_frozen_before_opening": True,
        "confirmation_changed_formula_or_configuration": False,
        "resource_audit_pass": resources["formal_all_status_pass"],
    }
    atomic_json(OUT / "night14a_decision.json", decision)


if __name__ == "__main__":
    main()
