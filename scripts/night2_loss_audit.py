#!/usr/bin/env python3
"""Finalize the Night-2 audit when the preregistered P0 gate blocks runs.

This command never trains a model.  It refuses to proceed unless the parity
report explicitly denies factorial authorization, then writes compact,
machine-readable stopped-run artifacts without fabricating observations.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "night2"
PARITY = REPO / "reports" / "night2_parity.json"
NIGHT1_PER_SEED = REPO / "results" / "night1" / "per_seed_metrics.csv"
NIGHT1_SUMMARY = REPO / "results" / "night1" / "summary.csv"
REFERENCE_VARIANTS = {"legacy_exact", "corrected_unweighted"}
DATASETS = ("a1", "placenta", "p22")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stopped_csv(name: str, fields) -> None:
    write_rows(RESULTS / name, list(fields) + ["run_status"], [])


def main() -> None:
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    if parity.get("p0_pass") is not False or parity.get("factorial_authorized") is not False:
        raise RuntimeError("This stopped-run finalizer is valid only after an explicit P0 failure")

    per_seed, per_seed_fields = read_rows(NIGHT1_PER_SEED)
    reference_per_seed = [
        {**row, "source": "night1_fdecb33_reference", "run_status": "existing_reference"}
        for row in per_seed
        if row["variant"] in REFERENCE_VARIANTS
    ]
    write_rows(
        RESULTS / "per_seed_metrics.csv",
        per_seed_fields + ["source", "run_status"],
        reference_per_seed,
    )

    summary, summary_fields = read_rows(NIGHT1_SUMMARY)
    reference_summary = [
        {**row, "source": "night1_fdecb33_reference", "run_status": "existing_reference"}
        for row in summary
        if row["variant"] in REFERENCE_VARIANTS
    ]
    write_rows(
        RESULTS / "summary.csv",
        summary_fields + ["source", "run_status"],
        reference_summary,
    )

    stopped_csv(
        "paired_deltas.csv",
        ("dataset", "variant", "seed", "metric", "value", "v0_value", "paired_delta"),
    )
    stopped_csv(
        "factorial_effects.csv",
        ("dataset", "metric", "scale_recovery", "shape_recovery", "replay_recovery", "interaction"),
    )
    stopped_csv(
        "loss_components.csv",
        (
            "dataset", "variant", "seed", "epoch", "raw_rna_reconstruction",
            "weighted_rna_before_global_scale", "final_rna_contribution",
            "raw_modality2_reconstruction", "final_modality2_contribution",
            "raw_corr1", "final_corr1_contribution", "raw_corr2",
            "final_corr2_contribution", "total_loss", "global_scale_multiplier",
            "m_bad", "gene_weight_mean", "gene_weight_min", "gene_weight_max",
        ),
    )
    stopped_csv(
        "attention_summary.csv",
        (
            "dataset", "variant", "seed", "cross_omics_rna_mean",
            "rna_spatial_feature_mean", "modality2_spatial_feature_mean",
            "delta_vs_v0", "delta_vs_legacy_exact",
        ),
    )
    stopped_csv(
        "per_domain_f1.csv",
        ("dataset", "variant", "seed", "domain", "support", "hungarian_f1"),
    )

    manuscript = {
        "a1": (0.2443, 0.3780),
        "p22": (0.4541, 0.5747),
        "placenta": (0.7226, 0.7408),
    }
    legacy = {row["dataset"]: row for row in summary if row["variant"] == "legacy_exact"}
    paper_rows = []
    for dataset in DATASETS:
        paper_rows.append(
            {
                "dataset": dataset,
                "manuscript_ari": manuscript[dataset][0],
                "manuscript_nmi": manuscript[dataset][1],
                "night1_legacy_mean_ari": legacy[dataset]["ari_mean"],
                "night1_legacy_mean_nmi": legacy[dataset]["nmi_mean"],
                "tutorial2022_ari": "",
                "tutorial2022_nmi": "",
                "notes": (
                    "NOT RUN: P0 hard stop fired before scientific runs; the user's stop rule "
                    "limited work to diagnosis/report/persistence. No seed search was performed."
                ),
            }
        )
    write_rows(
        RESULTS / "paper_repro_audit.csv",
        (
            "dataset", "manuscript_ari", "manuscript_nmi", "night1_legacy_mean_ari",
            "night1_legacy_mean_nmi", "tutorial2022_ari", "tutorial2022_nmi", "notes",
        ),
        paper_rows,
    )

    parity_rows = []
    for dataset in DATASETS:
        item = parity["datasets"][dataset]
        rna_graph = item["graphs"]["feature_omics1"]
        parity_rows.append(
            {
                "dataset": dataset,
                "dataset_pass": item["pass"],
                "observations": item["observation_ids"]["legacy_count"],
                "observation_ids_exact": item["observation_ids"]["exact_names_and_order"],
                "hvg_count": item["hvg_gene_names"]["legacy_count"],
                "hvg_names_order_exact": item["hvg_gene_names"]["exact_names_and_order"],
                "rna_scaled_max_abs_diff": item["rna_scaled_hvg_matrix"]["max_absolute_difference"],
                "rna_scaled_pass": item["rna_scaled_hvg_matrix"]["pass"],
                "rna_pca_max_abs_diff": item["rna_pca_features"]["max_absolute_difference"],
                "rna_pca_pass": item["rna_pca_features"]["pass"],
                "rna_graph_only_legacy": rna_graph["edge_set"]["only_first"],
                "rna_graph_only_corrected": rna_graph["edge_set"]["only_second"],
                "rna_graph_jaccard": rna_graph["edge_set"]["jaccard"],
                "rna_graph_pass": rna_graph["edge_set"]["pass"],
                "rna_adj_max_abs_diff": rna_graph["normalized_adjacency"]["max_absolute_difference"],
                "modality2_pass": item["modality2_features"]["pass"],
                "all_other_graphs_pass": all(
                    graph["edge_set"]["pass"] and graph["normalized_adjacency"]["pass"]
                    for name, graph in item["graphs"].items()
                    if name != "feature_omics1"
                ),
            }
        )
    write_rows(RESULTS / "p0_parity_summary.csv", parity_rows[0].keys(), parity_rows)

    diagnosis = {
        "p0_pass": False,
        "factorial_authorized": False,
        "classification": "unexplained_material_pretraining_parity_difference",
        "observed_chain": [
            "Observation IDs and ordered HVG names agree for all datasets.",
            "Scaled RNA matrices differ at float32-scale; P22 exceeds the declared allclose tolerance.",
            "The actually used randomized PCA features exceed the declared sign-aligned tolerance for all datasets.",
            "RNA feature-graph edge sets consequently differ for A1 and P22; normalized adjacency differs materially at affected entries.",
            "Modality-2, spatial graphs, and controlled legacy/corrected model forward mapping pass.",
        ],
        "root_cause_hypothesis": (
            "Different scale/select operation order and sparse/dense numerical paths introduce tiny RNA values; "
            "randomized PCA and near-tied correlation-neighbor selection amplify them. This is supported but not fully proven."
        ),
        "scientific_consequence": (
            "V3 could not be interpreted as a loss-only replay, so all V1-V4 training and tutorial reproduction "
            "were withheld under the preregistered hard stop."
        ),
        "new_factorial_runs_requested": 60,
        "new_factorial_runs_completed": 0,
        "new_factorial_runs_skipped": 60,
        "ground_truth_used_for_tuning": False,
        "seed_search_performed": False,
        "asr_modified": False,
    }
    (REPO / "reports" / "night2_p0_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True), encoding="utf-8"
    )

    status = {
        "status": "stopped_by_p0_hard_gate",
        "p0_pass": False,
        "factorial_authorized": False,
        "parent_commit": "fdecb33706ea1fe8429439813eef09e3ea931c86",
        "branch": "revision/q2-night2-loss-audit-20260808",
        "new_main_runs": {"expected_if_authorized": 60, "completed": 0, "skipped": 60},
        "tutorial2022_runs": {"planned": 3, "completed": 0, "skipped": 3},
        "compact_reference_rows": {
            "per_seed_metrics": len(reference_per_seed),
            "summary": len(reference_summary),
        },
        "night1_input_sha256": {
            str(NIGHT1_PER_SEED.relative_to(REPO)): sha256(NIGHT1_PER_SEED),
            str(NIGHT1_SUMMARY.relative_to(REPO)): sha256(NIGHT1_SUMMARY),
        },
        "empty_factorial_csvs_are_intentional": True,
        "reason": "P0 parity failed before training; only diagnosis, reporting, persistence, and shutdown are permitted.",
    }
    (RESULTS / "gate_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
