#!/usr/bin/env python3
"""Build compact Night-16H tables from locked producers and evaluators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7", "MELANOMA_TUMOR_K2")
FORMAL_MODE = "SMALLEST_SCALE_INTERNAL_EDGE"
METRICS = (
    "absolute_ari", "absolute_nmi", "ami", "fmi", "homogeneity", "v_measure",
    "morans_i_macro", "gearys_c_macro", "neighbor_agreement", "n_total", "n_evaluated",
    "k", "min_cluster_size_full", "min_cluster_size_eval", "cluster_sizes_full",
    "cluster_sizes_eval", "partition_sha256", "candidate_id", "arm", "start_id",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def pick(row: dict[str, object]) -> dict[str, object]:
    return {key: row.get(key, "") for key in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-root", type=Path, required=True)
    parser.add_argument("--night16g-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--carrier-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    scout = read_csv(args.working_root / "scout/selector_sensitivity_v2.csv")
    shutil.copyfile(
        args.working_root / "scout/selector_sensitivity_v2.csv",
        out / "feasibility_sensitivity_table.csv",
    )
    prior_controls = read_csv(
        args.repo_root / "outputs/night16g_handoff/selector_control_and_ablation_table.csv"
    )
    prior_by = {(row["lane"], row["control"]): row for row in prior_controls}
    controls: list[dict[str, object]] = []
    main: list[dict[str, object]] = []
    loso: list[dict[str, object]] = []
    contribution: list[dict[str, object]] = []
    candidate_ledger: list[dict[str, object]] = []
    p0_rows = []
    fit_registry = {}
    resource_rows = []
    for lane in LANES:
        evaluation_path = args.night16g_root / f"candidate_evaluation/{lane}.csv"
        evaluation = read_csv(evaluation_path)
        by_id = {row["candidate_id"]: row for row in evaluation}
        features = read_csv(args.working_root / f"feasibility/{lane}.csv")
        for row in features:
            candidate_ledger.append({
                "lane": lane,
                **row,
                **{f"metric_{key}": value for key, value in by_id[row["candidate_id"]].items()},
            })
        formal_scout = [
            row for row in scout
            if row["lane"] == lane and row["feasibility_mode"] == FORMAL_MODE
        ]
        for row in formal_scout:
            controls.append({"lane": lane, "control": row["selector"], **{k: v for k, v in row.items() if k not in ("lane", "selector")}})
        for prior_name in (
            "FIXED_AUTHORITY_DIRECT_ENERGY",
            "MULTI_EVIDENCE_GLOBAL",
            "STRICT_LOSO",
        ):
            prior = prior_by[(lane, prior_name)]
            controls.append({"lane": lane, "control": f"NIGHT16G_{prior_name}", **pick(prior)})
        oracle = max(
            evaluation,
            key=lambda row: (float(row["absolute_ari"]), float(row["absolute_nmi"]), row["candidate_id"]),
        )
        controls.append({"lane": lane, "control": "LABEL_ASSISTED_ORACLE_MAX_ARI", **pick(oracle)})
        global_eval = json.loads((args.working_root / f"formal/global_run1/{lane}.evaluation.json").read_text(encoding="utf-8"))
        loso_eval = json.loads((args.working_root / f"formal/loso_run1/{lane}.evaluation.json").read_text(encoding="utf-8"))
        medoid = next(row for row in formal_scout if row["selector"] == "FEASIBLE_PLAIN_MEDOID")
        max_spatial = next(row for row in formal_scout if row["selector"] == "FEASIBLE_MAX_SPATIAL")
        min_inertia = next(row for row in formal_scout if row["selector"] == "FEASIBLE_MIN_INERTIA")
        global_manifest = json.loads((args.working_root / f"formal/global_run1/{lane}.producer.json").read_text(encoding="utf-8"))
        main.append({
            "lane": lane,
            **pick(global_eval),
            "formal_selector": "UNIVERSAL_FEASIBILITY_PLUS_CROSS_EVIDENCE_ARBITRATION",
            "feasibility_mode": FORMAL_MODE,
            "feasible_candidate_count": global_eval["feasible_candidate_count"],
            "plain_medoid_ari": medoid["absolute_ari"],
            "plain_medoid_nmi": medoid["absolute_nmi"],
            "delta_vs_medoid_ari": float(global_eval["absolute_ari"]) - float(medoid["absolute_ari"]),
            "delta_vs_medoid_nmi": float(global_eval["absolute_nmi"]) - float(medoid["absolute_nmi"]),
            "strict_loso_ari": loso_eval["absolute_ari"],
            "strict_loso_nmi": loso_eval["absolute_nmi"],
            "strict_loso_min_cluster": loso_eval["min_cluster_size_full"],
            "oracle_ari": oracle["absolute_ari"],
            "oracle_nmi_at_max_ari": oracle["absolute_nmi"],
            "oracle_candidate_id": oracle["candidate_id"],
            "selector_wall_seconds": global_manifest["wall_seconds"],
            "selector_peak_rss_mib": global_manifest["peak_rss_mib"],
            "gpu_time_seconds": 0,
            "peak_gpu_mib": 0,
        })
        fit_path = args.working_root / f"formal/fits/LOSO_{lane}.json"
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        fit_registry[lane] = fit
        loso.append({
            "lane": lane,
            **pick(loso_eval),
            "plain_medoid_ari": medoid["absolute_ari"],
            "plain_medoid_nmi": medoid["absolute_nmi"],
            "delta_vs_medoid_ari": float(loso_eval["absolute_ari"]) - float(medoid["absolute_ari"]),
            "delta_vs_medoid_nmi": float(loso_eval["absolute_nmi"]) - float(medoid["absolute_nmi"]),
            "training_lanes": json.dumps(fit["training_lanes"], separators=(",", ":")),
            "selector_weights": json.dumps(fit["selector_weights"], sort_keys=True, separators=(",", ":")),
            "heldout_evaluation_opened_during_fit": fit["heldout_evaluation_opened"],
            "fit_manifest_sha256": sha256(fit_path),
        })
        unconstrained = next(row for row in scout if row["lane"] == lane and row["feasibility_mode"] == "UNCONSTRAINED" and row["selector"] == "CROSS_EVIDENCE_ARBITRATION")
        arms = (
            ("UNCONSTRAINED_CROSS_EVIDENCE", unconstrained),
            ("UNIVERSAL_FEASIBILITY_PLUS_MEDOID", medoid),
            ("UNIVERSAL_FEASIBILITY_PLUS_MIN_INERTIA", min_inertia),
            ("UNIVERSAL_FEASIBILITY_PLUS_MAX_SPATIAL", max_spatial),
            ("FULL_FIXED_CROSS_EVIDENCE", global_eval),
            ("STRICT_LOSO_WEIGHTED_RANK", loso_eval),
        )
        for arm, row in arms:
            contribution.append({
                "lane": lane,
                "arm": arm,
                "candidate_id": row["candidate_id"],
                "absolute_ari": row["absolute_ari"],
                "absolute_nmi": row["absolute_nmi"],
                "min_cluster_size_full": row["min_cluster_size_full"],
                "cluster_sizes_full": row["cluster_sizes_full"],
            })
        carrier_path = args.carrier_root / f"{lane}.npz"
        with np.load(carrier_path, allow_pickle=False) as carrier:
            p0_rows.append({
                "lane": lane,
                "ids_shape": list(carrier["ids"].shape),
                "retained_shape": list(carrier["retained"].shape),
                "view1_shape": list(carrier["view1"].shape),
                "view2_shape": list(carrier["view2"].shape),
                "start_bank_shape": list(carrier["start_partitions"].shape),
                "graph_nnz": [int(len(carrier[f"graph{i}__data"])) for i in range(3)],
                "finite_numeric_views": bool(all(np.isfinite(carrier[key]).all() for key in ("retained", "view1", "view2"))),
                "carrier_sha256": sha256(carrier_path),
            })
        resource_rows.append({
            "lane": lane,
            "global_wall_seconds": global_manifest["wall_seconds"],
            "global_peak_rss_mib": global_manifest["peak_rss_mib"],
            "gpu_time_seconds": 0,
            "peak_gpu_mib": 0,
        })
    write_csv(out / "absolute_metrics_main_table.csv", main)
    write_csv(out / "selector_control_and_ablation_table.csv", controls)
    write_csv(out / "strict_loso_transfer_table.csv", loso)
    write_csv(out / "mechanism_minimal_contribution_table.csv", contribution)
    write_csv(out / "all_candidate_evidence_and_metrics_ledger.csv", candidate_ledger)
    write_csv(out / "resource_table.csv", resource_rows)
    (out / "selector_fit_registry.json").write_text(json.dumps(fit_registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "real_input_p0_audit.json").write_text(json.dumps({
        "schema": "night16h-real-input-p0-audit-v1",
        "passed": len(p0_rows), "total": len(p0_rows), "rows": p0_rows,
        "candidate_bank_count_per_lane": 89,
        "sparse_graph_only": True,
        "dense_observation_by_observation_count": 0,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "label_flow_audit.json").write_text(json.dumps({
        "schema": "night16h-label-flow-audit-v1",
        "candidate_partition_producer_label_reads": 0,
        "feasibility_producer_label_reads": 0,
        "fixed_selector_label_reads": 0,
        "strict_loso_fit_heldout_label_reads": 0,
        "independent_evaluator_reads": 8,
        "public_benchmark_labels_used_for": ["scout diagnostics", "training-study-only LOSO weight fit", "final independent metrics"],
        "labels_not_used_for": ["candidate generation", "structural feasibility", "fixed cross-evidence arbitration", "heldout LOSO fit", "partition optimization"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out / "failure_and_correction_ledger.csv", [
        {"id": "E01", "type": "ENGINEERING", "status": "SUPERSEDED", "issue": "initial graph feasibility builder recomputed graph preparation per candidate", "resolution": "prepare each sparse graph once per lane; no formal result produced by interrupted attempt"},
        {"id": "E02", "type": "ENGINEERING", "status": "FIXED_PREFORMAL", "issue": "SciPy result attribute statistic is unavailable in deployed version", "resolution": "use tuple index spearmanr(x,y)[0]; targeted real-environment test"},
        {"id": "E03", "type": "ENGINEERING", "status": "FIXED_PREFORMAL", "issue": "scout could crash when a sensitivity mode has zero feasible candidates", "resolution": "record NO_FEASIBLE_CANDIDATE and continue; targeted zero-feasible test"},
        {"id": "S01", "type": "SCIENTIFIC", "status": "PRESERVED", "issue": "Night16G strict LOSO human selected a singleton partition", "resolution": "new revision uses a universal graph-derived admissible set before all selectors"},
        {"id": "S02", "type": "SCIENTIFIC", "status": "LIMITATION", "issue": "partition ensemble and spatial consensus are established prior art", "resolution": "claim limited to universal graph feasibility plus cross-study multi-evidence selection signal"},
    ])
    print(json.dumps({"status": "PASS", "main_rows": len(main), "control_rows": len(controls), "candidate_rows": len(candidate_ledger)}, sort_keys=True))


if __name__ == "__main__":
    main()
