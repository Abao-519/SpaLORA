#!/usr/bin/env python3
"""Evaluate only fully locked Night-7B adapter stages and issue opaque promotions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night7a_consensus import atomic_json, sha256_file  # noqa: E402
from scripts.night7b_evaluate import (  # noqa: E402
    DATASETS, WEIGHTS, load_coordinates, load_labels, metric, read_clusters,
)

OUT = REPO / "outputs/night7b_handoff"
RAW = Path("/root/autodl-fs/night7b_score_rnd_20260818")
ADAPTER = RAW / "adapter_stage"
REG = REPO / "protocols/night7b/SpaLORA_Night7B_Adaptive_Relational_Fusion_Registry_2026-08-18.json"
METRICS = ("ari", "nmi", "q", "neighbor_agreement", "moran_i", "geary_c", "boundary_disagreement")


def canonical_json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def label_window(stage: str):
    labels, audit = {}, {}
    for dataset in DATASETS:
        ids, true, digest, keys = load_labels(dataset)
        coords = load_coordinates(dataset, ids)
        labels[dataset] = (ids, true, coords)
        audit[dataset] = {
            "authorized_role":"stage_%s_evaluator" % stage,
            "snapshot_sha256":digest, "snapshot_keys":keys,
            "used_for_training_or_transform":False,
        }
    return labels, audit


def reference_table() -> pd.DataFrame:
    frame = pd.read_csv(OUT / "H_per_seed_metrics.csv")
    if len(frame) != 540:
        raise RuntimeError("H reference metric cardinality mismatch")
    return frame


def summarize(frame: pd.DataFrame, reference: pd.DataFrame, config_ids: list[str],
              reference_id: str) -> pd.DataFrame:
    ref = reference[reference.config_id == reference_id].set_index(["dataset", "seed"])
    rows = []
    for order, config_id in enumerate(config_ids):
        group = frame[frame.config_id == config_id].copy()
        row = {"config_id":config_id, "registry_order":order,
               "success_cells":int(group.success.sum()), "failure_cells":int((~group.success).sum())}
        for dataset in DATASETS:
            dg = group[group.dataset == dataset].sort_values("seed")
            row[dataset + "_success"] = int(dg.success.sum())
            for name in METRICS:
                values = dg[name].dropna()
                complete = len(values) == len(dg) and len(dg) > 0
                row[dataset + "_mean_" + name] = float(values.mean()) if complete else np.nan
                if complete:
                    rv = np.asarray([ref.loc[(dataset, int(seed)), name] for seed in dg.seed], dtype=float)
                    delta = dg[name].to_numpy(dtype=float) - rv
                    row[dataset + "_mean_delta_" + name] = float(delta.mean())
                    row[dataset + "_wins_" + name] = int(np.sum(delta > 0))
                else:
                    row[dataset + "_mean_delta_" + name] = np.nan
                    row[dataset + "_wins_" + name] = 0
                if complete and len(values) > 1:
                    row[dataset + "_std_" + name] = float(values.std(ddof=1))
                else:
                    row[dataset + "_std_" + name] = 0.0 if complete else np.nan
        eligible = row["failure_cells"] == 0 and len(group) > 0
        row["complete_eligible"] = eligible
        row["priority_weighted_q"] = sum(WEIGHTS[d] * row[d + "_mean_q"] for d in DATASETS) if eligible else np.nan
        row["priority_weighted_delta_q"] = sum(WEIGHTS[d] * row[d + "_mean_delta_q"] for d in DATASETS) if eligible else np.nan
        row["priority_weighted_delta_ari"] = sum(WEIGHTS[d] * row[d + "_mean_delta_ari"] for d in DATASETS) if eligible else np.nan
        row["priority_weighted_delta_nmi"] = sum(WEIGHTS[d] * row[d + "_mean_delta_nmi"] for d in DATASETS) if eligible else np.nan
        row["balanced_macro_q"] = float(np.mean([row[d + "_mean_q"] for d in DATASETS])) if eligible else np.nan
        row["balanced_macro_delta_q"] = float(np.mean([row[d + "_mean_delta_q"] for d in DATASETS])) if eligible else np.nan
        row["human_lymph_equal_mean_delta_q"] = float((row["a1_mean_delta_q"] + row["d1_mean_delta_q"]) / 2) if eligible else np.nan
        row["worst_dataset_delta_q"] = min(row[d + "_mean_delta_q"] for d in DATASETS) if eligible else np.nan
        row["total_q_wins"] = sum(row[d + "_wins_q"] for d in DATASETS) if eligible else 0
        row["mean_runtime_seconds"] = float(group.training_runtime_seconds.mean()) if len(group) else np.nan
        row["mean_peak_gpu_mib"] = float(group.peak_gpu_mib.mean()) if len(group) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ranked(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.sort_values(
        ["complete_eligible", "priority_weighted_delta_q", "balanced_macro_delta_q",
         "worst_dataset_delta_q", "total_q_wins", "mean_runtime_seconds",
         "mean_peak_gpu_mib", "registry_order"],
        ascending=[False, False, False, False, False, True, True, True],
    )


def evaluate_stage(stage: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    locked_path = OUT / ("locked_%s_manifest.json" % stage)
    locked = json.loads(locked_path.read_text())
    if locked["status"] != "LOCKED_PRE_LABEL" or locked["label_access"]:
        raise RuntimeError("stage was not fully locked before label window")
    expected = 320 if stage == "R1" else len(locked["config_ids"]) * 22
    if len(locked["transforms"]) != expected:
        raise RuntimeError("locked stage transform cardinality mismatch")
    labels, audit = label_window(stage)
    training = {(x["recipe_id"], x["unit_id"]):x for x in locked["training_cells"]}
    rows = []
    for cell in locked["transforms"]:
        train = training[(cell["recipe_id"], cell["unit_id"])]
        tmanifest = train.get("training_manifest", {})
        row = {
            "stage":stage, "config_id":cell["config_id"], "recipe_id":cell["recipe_id"],
            "endpoint":cell["endpoint"], "head_id":cell["head_id"],
            "unit_id":cell["unit_id"], "dataset":cell["dataset"], "seed":cell["seed"],
            "status":cell["status"], "success":cell["status"] == "success",
            "training_runtime_seconds":float(tmanifest.get("runtime_seconds", np.nan)),
            "peak_gpu_mib":float(tmanifest.get("peak_gpu_mib", np.nan)),
        }
        if row["success"]:
            target = ADAPTER / stage / "formal" / cell["recipe_id"] / cell["unit_id"] / "attempt_001" / "transforms" / cell["endpoint"] / cell["head_id"]
            ids, true, coords = labels[cell["dataset"]]
            pred = read_clusters(target / "clusters.csv", ids)
            row.update(metric(true, pred, coords))
        else:
            row.update({name:np.nan for name in METRICS})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / ("%s_per_seed_metrics.csv" % stage), index=False)
    config_ids = list(dict.fromkeys(frame.config_id.tolist()))
    ref = reference_table()
    summary = ranked(summarize(frame, ref, config_ids, "H00"))
    summary.to_csv(OUT / ("%s_candidate_summary_vs_C00.csv" % stage), index=False)
    atomic_json(OUT / ("%s_label_window_audit.json" % stage), audit)
    return frame, summary, audit


def evaluate_r1() -> None:
    frame, summary, audit = evaluate_stage("R1")
    promoted = summary[summary.complete_eligible].head(4).config_id.tolist()
    if len(promoted) != 4:
        raise RuntimeError("R1 produced fewer than four complete configurations")
    registry = json.loads(REG.read_text())
    recipe_lookup = {x["id"]:x for x in registry["adapter_recipes"]}
    parameters = {}
    for config_id in promoted:
        recipe, endpoint, head = config_id.split("__")
        parameters[config_id] = canonical_json_sha({
            "recipe":recipe_lookup[recipe], "endpoint":endpoint, "head":head,
            "architecture":registry["adapter_architecture"], "loss_contract":registry["loss_contract"],
        })
    remaining = {x["id"]:x["seeds"][2:] for x in registry["dataset_matrix"]}
    contract = {
        "stage":"R1", "status":"PASS", "promoted_config_ids":promoted,
        "fixed_parameter_sha256":parameters, "remaining_seeds":remaining,
        "locked_R1_manifest_sha256":sha256_file(OUT / "locked_R1_manifest.json"),
        "aggregate_ranking_sha256":sha256_file(OUT / "R1_candidate_summary_vs_C00.csv"),
        "label_values_or_per_spot_metrics_in_message":False,
    }
    atomic_json(OUT / "R1_to_R2_contract.json", contract)


def spatial_gate(row: pd.Series) -> tuple[bool, dict]:
    details = {}
    for dataset in DATASETS:
        details[dataset] = {
            "neighbor":bool(row[dataset + "_mean_delta_neighbor_agreement"] >= -.01),
            "moran":bool(row[dataset + "_mean_delta_moran_i"] >= -.02),
            "geary":bool(row[dataset + "_mean_delta_geary_c"] <= .02),
            "boundary":bool(row[dataset + "_mean_delta_boundary_disagreement"] <= .01),
        }
    return all(all(v.values()) for v in details.values()), details


def final_gate(row: pd.Series) -> tuple[bool, dict]:
    spatial_ok, spatial = spatial_gate(row)
    tests = {
        "priority_weighted_delta_q_ge_0_018":bool(row.priority_weighted_delta_q >= .018),
        "human_lymph_equal_mean_delta_q_gt_0":bool((row.a1_mean_delta_q + row.d1_mean_delta_q) / 2 > 0),
        "p22_mean_delta_q_ge_0_03":bool(row.p22_mean_delta_q >= .03),
        "a1_delta_q_ge_0":bool(row.a1_mean_delta_q >= 0),
        "d1_delta_q_ge_0":bool(row.d1_mean_delta_q >= 0),
        "p22_delta_q_ge_0":bool(row.p22_mean_delta_q >= 0),
        "tonsil_delta_q_ge_minus_0_005":bool(row.tonsil_mean_delta_q >= -.005),
        "total_q_wins_ge_22":bool(row.total_q_wins >= 22),
        "a1_q_wins_ge_3":bool(row.a1_wins_q >= 3),
        "tonsil_q_wins_ge_3":bool(row.tonsil_wins_q >= 3),
        "d1_q_wins_ge_6":bool(row.d1_wins_q >= 6),
        "p22_q_wins_ge_7":bool(row.p22_wins_q >= 7),
        "spatial_protection":spatial_ok,
        "complete_eligible":bool(row.complete_eligible),
    }
    return all(tests.values()), {"tests":tests, "spatial_by_dataset":spatial}


def evaluate_r2() -> None:
    r2, _, audit = evaluate_stage("R2")
    r1 = pd.read_csv(OUT / "R1_per_seed_metrics.csv")
    promoted = json.loads((OUT / "R1_to_R2_contract.json").read_text())["promoted_config_ids"]
    full = pd.concat([r1[r1.config_id.isin(promoted)], r2], ignore_index=True)
    full = full.sort_values(["config_id", "dataset", "seed"])
    expected = sum(5 if d in ("a1", "tonsil") else 10 for d in DATASETS) * len(promoted)
    if len(full) != expected or full.duplicated(["config_id", "dataset", "seed"]).any():
        raise RuntimeError("R1+R2 full-seed merge mismatch")
    full.to_csv(OUT / "R2_full_per_seed_metrics.csv", index=False)
    ref = reference_table()
    vs_c00 = ranked(summarize(full, ref, promoted, "H00"))
    vs_c06 = ranked(summarize(full, ref, promoted, "H01"))
    gate_rows = []; passed = []
    for _, row in vs_c00.iterrows():
        ok, details = final_gate(row)
        gate_rows.append({"config_id":row.config_id, "pass":ok, **details})
        if ok: passed.append(row.config_id)
    vs_c00.to_csv(OUT / "R2_final_summary_vs_C00.csv", index=False)
    vs_c06.to_csv(OUT / "R2_final_summary_vs_C06.csv", index=False)
    atomic_json(OUT / "final_lock_gate_audit.json", {"candidates":gate_rows})
    if passed:
        status = "NIGHT7B_UNIFIED_SCORE_CANDIDATE_LOCKED"
    else:
        p22_frontier = any(float(row.p22_mean_delta_q) > 0 for _, row in vs_c06.iterrows()
                           if bool(row.complete_eligible))
        status = "NIGHT7B_P22_FRONTIER_ONLY_NO_UNIFIED_WINNER" if p22_frontier else "NIGHT7B_NO_NEW_SCORE_CANDIDATE"
    accuracy = vs_c00.sort_values(["priority_weighted_delta_q", "registry_order"], ascending=[False, True]).config_id.tolist()
    balanced = []
    for _, row in vs_c00.iterrows():
        ok, _ = spatial_gate(row)
        if ok and bool(row.complete_eligible): balanced.append(row.config_id)
    result = {
        "terminal_status":status, "unified_pass_config_ids":passed,
        "accuracy_frontier_order":accuracy, "balanced_frontier_order":balanced,
        "R2_lock_sha256":sha256_file(OUT / "locked_R2_manifest.json"),
        "R2_full_metrics_sha256":sha256_file(OUT / "R2_full_per_seed_metrics.csv"),
        "summary_vs_C00_sha256":sha256_file(OUT / "R2_final_summary_vs_C00.csv"),
        "summary_vs_C06_sha256":sha256_file(OUT / "R2_final_summary_vs_C06.csv"),
        "label_opened_only_after_R2_lock":True,
    }
    atomic_json(OUT / "night7b_final_candidate_lock.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("R1","R2")); args = parser.parse_args()
    if args.stage == "R1": evaluate_r1()
    else: evaluate_r2()


if __name__ == "__main__":
    main()
