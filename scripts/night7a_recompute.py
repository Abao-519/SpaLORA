#!/usr/bin/env python3
"""Independent table-level recomputation of Night-7A keys, deltas, gates, and rank."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night7a_consensus import CANDIDATE_ORDER, DATASETS, atomic_json  # noqa: E402

OUT = REPO / "outputs/night7a_handoff"
DUAL = set(CANDIDATE_ORDER[2:])
COMPLEXITY_ORDER = (
    "C01_G00_H05", "C00_G04_H05_CONFIRMED",
    "C02_DUAL_ARITHMETIC_MEAN", "C03_DUAL_ELEMENTWISE_MAX",
    "C04_DUAL_ELEMENTWISE_MIN", "C05_DUAL_HARMONIC_INTERSECTION",
    "C06_DUAL_ROW_STOCHASTIC_MEAN", "C08_SIX_VIEW_SUPPORT_MEDIAN",
    "C07_DUAL_LOCAL_RELIABILITY", "C10_DUAL_MEAN_SPATIAL05",
    "C11_DUAL_MEAN_SPATIAL10", "C09_DUAL_SPARSE_SNF10",
)
RANK = {candidate: index for index, candidate in enumerate(COMPLEXITY_ORDER)}


def spatial_failed(neighbor: float, moran: float, geary: float) -> bool:
    return bool((neighbor < -.03 and moran < -.03) or
                (geary > .03 and (neighbor < -.03 or moran < -.03)))


def close(left, right, tol=1e-12):
    return abs(float(left) - float(right)) <= tol


def main() -> None:
    metrics = pd.read_csv(OUT / "per_seed_metrics.csv")
    if metrics.duplicated(["dataset", "seed", "candidate_id"]).any():
        raise RuntimeError("duplicate metric primary keys")
    q_error = float(np.max(np.abs(metrics.q - (metrics.ari + metrics.nmi) / 2)))
    boundary_error = float(np.max(np.abs(metrics.boundary_disagreement -
                                         (1 - metrics.neighbor_agreement))))
    if q_error > 1e-12 or boundary_error > 1e-12:
        raise RuntimeError(f"metric formula mismatch Q={q_error}, boundary={boundary_error}")
    reference = metrics[metrics.candidate_id == "REFERENCE_G00_H00"].set_index(["dataset", "seed"])
    official_delta_frame = pd.read_csv(OUT / "paired_delta_vs_g00h00.csv")
    official_delta = official_delta_frame.set_index(
        ["dataset", "seed", "candidate_id"]
    )
    max_delta_error = 0.0
    for row in metrics[metrics.role == "candidate"].itertuples(index=False):
        ref = reference.loc[(row.dataset, int(row.seed))]
        official = official_delta.loc[(row.dataset, int(row.seed), row.candidate_id)]
        for metric in ("ari", "nmi", "q", "neighbor_agreement", "moran_i",
                       "geary_c", "boundary_disagreement"):
            observed = float(getattr(row, metric) - ref[metric])
            max_delta_error = max(max_delta_error, abs(observed - official[f"delta_{metric}"]))
    if max_delta_error > 1e-12:
        raise RuntimeError(f"paired delta mismatch: {max_delta_error}")
    summary_frame = pd.read_csv(OUT / "four_dataset_summary.csv")
    summary = summary_frame.set_index(["candidate_id", "dataset"])
    metric_names = ("ari", "nmi", "q", "neighbor_agreement", "moran_i",
                    "geary_c", "boundary_disagreement")
    lower_is_better = {"geary_c", "boundary_disagreement"}
    max_summary_error = 0.0
    wins_mismatches = []
    for candidate in CANDIDATE_ORDER:
        for dataset in DATASETS:
            values = official_delta_frame[
                (official_delta_frame.candidate_id == candidate) &
                (official_delta_frame.dataset == dataset)
            ]
            expected_n = 5 if dataset in {"a1", "tonsil"} else 10
            official = summary.loc[(candidate, dataset)]
            if len(values) != expected_n:
                if bool(official["complete"]):
                    raise RuntimeError(f"summary completeness mismatch {candidate}/{dataset}")
                continue
            for metric in metric_names:
                raw_delta = values[f"delta_{metric}"].to_numpy(dtype=float)
                checks = {
                    f"mean_{metric}": float(values[metric].mean()),
                    f"mean_delta_{metric}": float(raw_delta.mean()),
                    f"median_delta_{metric}": float(np.median(raw_delta)),
                    f"sd_delta_{metric}": float(np.std(raw_delta, ddof=1)),
                }
                for key, expected in checks.items():
                    max_summary_error = max(
                        max_summary_error, abs(float(official[key]) - expected)
                    )
                directional = -raw_delta if metric in lower_is_better else raw_delta
                expected_wins = int(np.sum(directional > 0))
                if int(official[f"wins_delta_{metric}"]) != expected_wins:
                    wins_mismatches.append(f"{candidate}/{dataset}/{metric}")
    if max_summary_error > 1e-12 or wins_mismatches:
        raise RuntimeError(
            f"summary/win direction mismatch: error={max_summary_error}; wins={wins_mismatches}"
        )
    macro = pd.read_csv(OUT / "dataset_balanced_macro_summary.csv").set_index("candidate_id")
    max_macro_error = 0.0
    for candidate in CANDIDATE_ORDER:
        group = summary_frame[
            (summary_frame.candidate_id == candidate) & summary_frame.complete
        ]
        official = macro.loc[candidate]
        if len(group) == 4:
            for metric in metric_names:
                max_macro_error = max(
                    max_macro_error,
                    abs(float(official[f"macro_mean_{metric}"]) -
                        float(group[f"mean_{metric}"].mean())),
                    abs(float(official[f"macro_mean_delta_{metric}"]) -
                        float(group[f"mean_delta_{metric}"].mean())),
                )
    if max_macro_error > 1e-12:
        raise RuntimeError(f"dataset-balanced macro mismatch: {max_macro_error}")
    official_gate = pd.read_csv(OUT / "candidate_gate_table.csv").set_index("candidate_id")
    gates = []
    c00 = summary.loc["C00_G04_H05_CONFIRMED"]
    c00_macro = float(c00.mean_delta_q.mean()); c00_worst = float(c00.mean_delta_q.min())
    for candidate in CANDIDATE_ORDER:
        group = official_delta_frame[
            official_delta_frame.candidate_id == candidate
        ].copy()
        complete = all(len(group[group.dataset == dataset]) == (5 if dataset in {"a1", "tonsil"} else 10)
                       for dataset in DATASETS)
        ds_rows = []
        if complete:
            for dataset in DATASETS:
                values = group[group.dataset == dataset]
                ds_rows.append({
                    "dataset": dataset,
                    "mean_delta_q": float(values.delta_q.mean()),
                    "median_delta_q": float(values.delta_q.median()),
                    "mean_delta_nmi": float(values.delta_nmi.mean()),
                    "mean_delta_ari": float(values.delta_ari.mean()),
                    "mean_q": float(values.q.mean()),
                    "q_wins": int((values.delta_q > 0).sum()),
                    "spatial_failed": spatial_failed(float(values.delta_neighbor_agreement.mean()),
                                                     float(values.delta_moran_i.mean()),
                                                     float(values.delta_geary_c.mean())),
                })
        ds = pd.DataFrame(ds_rows).set_index("dataset") if ds_rows else pd.DataFrame()
        q_gate = complete and bool((ds.mean_delta_q >= .005).all())
        nmi_gate = complete and bool((ds.mean_delta_nmi > 0).all())
        ari_count = int((ds.mean_delta_ari > 0).sum()) if complete else 0
        ari_floor = complete and bool((ds.mean_delta_ari >= -.005).all())
        wins_gate = complete and ds.loc["a1", "q_wins"] >= 4 and ds.loc["tonsil", "q_wins"] >= 4 and ds.loc["d1", "q_wins"] >= 7 and ds.loc["p22", "q_wins"] >= 7
        median_gate = complete and bool((ds.median_delta_q > 0).all())
        spatial_gate = complete and not bool(ds.spatial_failed.any())
        generalization = q_gate and nmi_gate and ari_count >= 3 and ari_floor and wins_gate and median_gate and spatial_gate
        macro = float(ds.mean_delta_q.mean()) if complete else np.nan
        worst = float(ds.mean_delta_q.min()) if complete else np.nan
        if complete:
            per_dataset_vs_c00 = [float(ds.loc[d, "mean_q"] - c00.loc[d, "mean_q"]) for d in DATASETS]
            option_a = macro >= c00_macro + .0075 and min(per_dataset_vs_c00) >= -.002
            option_b = worst >= c00_worst + .005 and macro >= c00_macro and min(per_dataset_vs_c00) >= -.002
        else:
            option_a = option_b = False
        complexity = (option_a or option_b) if candidate in DUAL else True
        total_wins = int(sum(ds.q_wins)) if complete else 0
        gates.append({"candidate_id": candidate, "complete": complete,
                      "generalization_gate_pass": generalization,
                      "complexity_gate_pass": complexity,
                      "eligible": generalization and complexity,
                      "worst_dataset_mean_delta_q": worst,
                      "dataset_balanced_macro_mean_delta_q": macro,
                      "total_paired_q_wins": total_wins,
                      "future_complexity_rank": RANK[candidate]})
        official = official_gate.loc[candidate]
        for key, value in (("complete", complete),
                           ("generalization_gate_pass", generalization),
                           ("complexity_gate_pass", complexity),
                           ("eligible", generalization and complexity)):
            if bool(official[key]) != bool(value):
                raise RuntimeError(f"gate mismatch {candidate}/{key}")
        for key, value in (("worst_dataset_mean_delta_q", worst),
                           ("dataset_balanced_macro_mean_delta_q", macro)):
            if complete and not close(official[key], value):
                raise RuntimeError(f"gate numeric mismatch {candidate}/{key}")
    gate = pd.DataFrame(gates)
    eligible = gate[gate.eligible].sort_values(
        ["worst_dataset_mean_delta_q", "dataset_balanced_macro_mean_delta_q",
         "total_paired_q_wins", "future_complexity_rank", "candidate_id"],
        ascending=[False, False, False, True, True],
    )
    selected = "C00_G04_H05_CONFIRMED"
    selectable = [
        row for row in eligible.itertuples(index=False)
        if row.candidate_id in DUAL or row.candidate_id == "C00_G04_H05_CONFIRMED"
    ]
    if selectable:
        selected = selectable[0].candidate_id
    decision = json.loads((OUT / "night7a_decision.json").read_text())
    if decision["selected_structure"] != selected:
        raise RuntimeError(f"independent rank mismatch {selected}/{decision['selected_structure']}")
    atomic_json(OUT / "independent_recompute.json", {
        "status": "PASS", "tolerance": 1e-12,
        "primary_keys_unique": True, "metric_rows": len(metrics),
        "q_formula_max_error": q_error,
        "boundary_direction_max_error": boundary_error,
        "paired_delta_max_error": max_delta_error,
        "summary_statistic_max_error": max_summary_error,
        "directional_wins_mismatches": wins_mismatches,
        "dataset_balanced_macro_max_error": max_macro_error,
        "gate_rows_recomputed": len(gates), "selected_structure": selected,
        "decision_match": True, "label_files_reopened": False,
    })
    print(json.dumps({"status": "PASS", "selected_structure": selected,
                      "max_delta_error": max_delta_error}, sort_keys=True))


if __name__ == "__main__":
    main()
