#!/usr/bin/env python3
"""Post-lock Night-5A stage evaluator and deterministic funnel decision rules."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night3b_metrics import mean_one_vs_rest_geary, symmetric_knn_adjacency
from SpaLORA.night5a_rnd import load_registry, registry_contracts, sha256_file


CONFIG_PATH = REPO / "configs/night5a_metric_rnd.json"
METRICS = ("ari", "nmi", "spatial_neighbor_agreement", "spatial_cluster_moran_mean",
           "spatial_cluster_geary_mean", "runtime_seconds", "gpu_peak_allocated_mib")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def write_csv(path: Path, rows: list) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())


def raw_path(config, dataset, candidate, seed):
    return Path(config["paths"]["raw_runs"]) / dataset / candidate / ("seed_%d" % seed)


def load_locked_rows(output: Path, stage: str) -> list:
    path = output / ("%s_training_manifest.json" % stage.lower())
    complete = output / ("%s_training_complete.json" % stage.lower())
    payload = json.loads(path.read_text(encoding="utf-8"))
    completion = json.loads(complete.read_text(encoding="utf-8"))
    if not payload.get("locked_before_semantic_label_access"):
        raise RuntimeError("Stage was not locked before labels")
    if completion.get("training_manifest_sha256") != sha256_file(path):
        raise RuntimeError("Stage training manifest SHA mismatch")
    return payload["runs"]


def spatial_failed(rows: pd.DataFrame) -> bool:
    if rows.empty:
        return True
    neighbor_delta = float(rows["delta_spatial_neighbor_agreement"].mean())
    moran_delta = float(rows["delta_spatial_cluster_moran_mean"].mean())
    geary_delta = float(rows["delta_spatial_cluster_geary_mean"].mean())
    return bool((neighbor_delta < -0.03 and moran_delta < -0.03) or
                (geary_delta > 0.03 and (neighbor_delta < -0.03 or moran_delta < -0.03)))


def candidate_summary(per_seed: pd.DataFrame, candidate_id: str, seeds: list) -> dict:
    candidate = per_seed[(per_seed.candidate_id == candidate_id) & per_seed.seed.isin(seeds)]
    reference = per_seed[(per_seed.candidate_id == "C00_FULL_IGE") & per_seed.seed.isin(seeds)]
    expected = 2 * len(seeds)
    complete = len(candidate) == expected and len(reference) == expected
    if not complete:
        return {"candidate_id": candidate_id, "complete": False, "expected_rows": expected,
                "observed_rows": int(len(candidate)), "reason": "incomplete_or_failed"}
    merged = candidate.merge(reference, on=["dataset", "seed"], suffixes=("", "_reference"), validate="one_to_one")
    for metric in METRICS:
        merged["delta_" + metric] = merged[metric] - merged[metric + "_reference"]
    merged["q"] = (merged.ari + merged.nmi) / 2.0
    merged["q_reference"] = (merged.ari_reference + merged.nmi_reference) / 2.0
    merged["delta_q"] = merged.q - merged.q_reference
    dataset_delta_q = {dataset: float(group.delta_q.mean()) for dataset, group in merged.groupby("dataset")}
    cells_nonnegative = sum(float(merged.loc[merged.dataset == dataset, "delta_" + metric].mean()) >= 0
                            for dataset in ("a1", "placenta") for metric in ("ari", "nmi"))
    return {
        "candidate_id": candidate_id, "complete": True, "expected_rows": expected,
        "observed_rows": int(len(candidate)),
        "dev_macro_q": float(candidate.groupby("dataset").q.mean().mean()),
        "dev_macro_delta_q": float(np.mean(list(dataset_delta_q.values()))),
        "dev_macro_delta_ari": float(merged.groupby("dataset").delta_ari.mean().mean()),
        "dev_macro_delta_nmi": float(merged.groupby("dataset").delta_nmi.mean().mean()),
        "worst_dataset_delta_q": float(min(dataset_delta_q.values())),
        "dataset_delta_q": dataset_delta_q,
        "dataset_metric_mean_cells_nonnegative": int(cells_nonnegative),
        "paired_q_wins": int((merged.delta_q > 0).sum()),
        "paired_q_total": int(len(merged)),
        "spatial_protection_failed": spatial_failed(merged),
        "runtime_ratio": float(candidate.runtime_seconds.mean() / reference.runtime_seconds.mean()),
        "gpu_peak_ratio": float(candidate.gpu_peak_allocated_mib.max() / reference.gpu_peak_allocated_mib.max()),
        "runtime_seconds_mean": float(candidate.runtime_seconds.mean()),
        "per_seed_paired_delta_q": [
            {"dataset": row.dataset, "seed": int(row.seed), "delta_q": float(row.delta_q)}
            for row in merged.itertuples()
        ],
    }


def decide(stage: str, summary: list, contracts: dict) -> tuple:
    if stage == "R1":
        for row in summary:
            row["passes_numeric_gate"] = bool(row.get("complete") and
                row["dev_macro_delta_q"] >= 0.005 and row["worst_dataset_delta_q"] >= -0.02 and
                not row["spatial_protection_failed"])
        eligible = [row for row in summary if row["passes_numeric_gate"]]
        family_winners = []
        grouped = {}
        for row in eligible:
            family = contracts[row["candidate_id"]]["family"]
            slot = "simplified_base" if row["candidate_id"] == "C03_SIMPLE_BASE" else family
            grouped.setdefault(slot, []).append(row)
        for rows in grouped.values():
            rows.sort(key=lambda row: (-row["dev_macro_q"], -row["worst_dataset_delta_q"],
                                       row["runtime_seconds_mean"], row["candidate_id"]))
            family_winners.append(rows[0])
        family_winners.sort(key=lambda row: (-row["dev_macro_q"], -row["worst_dataset_delta_q"],
                                             row["runtime_seconds_mean"], row["candidate_id"]))
        advanced = [row["candidate_id"] for row in family_winners[:6]]
    elif stage == "R2":
        for row in summary:
            row["passes_numeric_gate"] = bool(row.get("complete") and
                row["dev_macro_delta_q"] >= 0.015 and row["worst_dataset_delta_q"] >= -0.01 and
                row["dataset_metric_mean_cells_nonnegative"] >= 3 and row["paired_q_wins"] >= 4 and
                not row["spatial_protection_failed"])
        eligible = [row for row in summary if row["passes_numeric_gate"]]
        eligible.sort(key=lambda row: (-row["worst_dataset_delta_q"], -row["dev_macro_delta_q"],
                                       -row["paired_q_wins"], row["runtime_seconds_mean"], row["candidate_id"]))
        advanced = [row["candidate_id"] for row in eligible[:3]]
    else:
        for row in summary:
            row["resource_qualified"] = bool(row.get("complete") and row["runtime_ratio"] <= 2.0 and row["gpu_peak_ratio"] <= 1.5)
            row["passes_numeric_gate"] = bool(row.get("complete") and
                row["dev_macro_delta_ari"] >= 0.01 and row["dev_macro_delta_nmi"] >= 0.01 and
                row["dev_macro_delta_q"] >= 0.02 and row["worst_dataset_delta_q"] >= -0.005 and
                row["paired_q_wins"] >= 7 and not row["spatial_protection_failed"])
            row["pareto_candidate"] = bool(row["passes_numeric_gate"] and not row["resource_qualified"])
        eligible = [row for row in summary if row["passes_numeric_gate"]]
        eligible.sort(key=lambda row: (-row["worst_dataset_delta_q"], -row["dev_macro_delta_q"],
                                       -row["paired_q_wins"], row["runtime_seconds_mean"], row["candidate_id"]))
        qualified = [row for row in eligible if row["resource_qualified"]]
        selected = eligible[:2]
        if selected and any(not row["resource_qualified"] for row in selected) and qualified and not any(
                row["resource_qualified"] for row in selected):
            selected[-1] = qualified[0]
        advanced = [row["candidate_id"] for row in selected]
    for row in summary:
        row["advanced"] = row["candidate_id"] in advanced
    return summary, advanced


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("R1", "R2", "R3"), required=True)
    args = parser.parse_args(); stage = args.stage
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    locked_rows = load_locked_rows(output, stage)
    registry = load_registry(REPO / config["candidate_registry"]); contracts = registry_contracts(registry)

    # Imports below are intentionally delayed until the stage manifest has been
    # locked and independently SHA-verified above.
    from SpaLORA.night1_evaluation import evaluate, load_evaluation_labels
    from scripts.night3a_evaluate import coordinates_for_ids

    metric_rows, failures = [], []
    by_dataset = {}
    for dataset in ("a1", "placenta"):
        success = next(row for row in locked_rows if row["dataset"] == dataset and row["status"] == "success")
        directory = Path(success["record_path"]).parent
        ids = pd.Index(pd.read_csv(directory / "observation_ids.csv")["observation_id"].astype(str))
        if ids.has_duplicates:
            raise RuntimeError("Duplicate observation ID")
        positions, labels = load_evaluation_labels(dataset, config["datasets"][dataset], ids)
        coordinates = coordinates_for_ids(config["datasets"][dataset], ids)
        by_dataset[dataset] = (ids, positions, labels, coordinates,
                               symmetric_knn_adjacency(coordinates, config["datasets"][dataset]["spatial_neighbors"]))
    for locked in locked_rows:
        if locked["status"] != "success":
            failures.append(locked); continue
        dataset, candidate_id, seed = locked["dataset"], locked["candidate_id"], int(locked["seed"])
        directory = Path(locked["record_path"]).parent
        ids, positions, labels, coordinates, graph = by_dataset[dataset]
        run_ids = pd.Index(pd.read_csv(directory / "observation_ids.csv")["observation_id"].astype(str))
        if not run_ids.equals(ids) or run_ids.has_duplicates:
            raise RuntimeError("Strict observation ID alignment failed")
        clusters = pd.read_csv(directory / "clusters.csv")["cluster"].to_numpy()
        with np.load(directory / "embedding.npz") as archive:
            embedding = np.asarray(archive["SpaLORA"], np.float32)
        metrics = evaluate(labels, clusters[positions], clusters, embedding, coordinates,
                           config["datasets"][dataset]["spatial_neighbors"])
        geary, _ = mean_one_vs_rest_geary(clusters, graph)
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        row = {"dataset": dataset, "candidate_id": candidate_id, "seed": seed, "stage_evaluated": stage,
               "ari": float(metrics["ari"]), "nmi": float(metrics["nmi"]),
               "q": float((metrics["ari"] + metrics["nmi"]) / 2.0),
               "spatial_neighbor_agreement": float(metrics["spatial_neighbor_agreement"]),
               "spatial_cluster_moran_mean": float(metrics["spatial_cluster_moran_mean"]),
               "spatial_cluster_geary_mean": float(geary),
               "runtime_seconds": float(manifest["timings"]["training_seconds"]),
               "gpu_peak_allocated_mib": float(manifest["resources"]["gpu_peak_allocated_mib"]),
               "run_manifest_sha256": sha256_file(directory / "run_manifest.json")}
        metric_rows.append(row)

    cumulative_path = output / "per_run_summary.csv"
    previous = pd.read_csv(cumulative_path).to_dict("records") if cumulative_path.exists() else []
    keyed = {(row["dataset"], row["candidate_id"], int(row["seed"])): row for row in previous}
    for row in metric_rows:
        keyed[(row["dataset"], row["candidate_id"], int(row["seed"]))] = row
    cumulative = list(keyed.values()); cumulative.sort(key=lambda r: (r["dataset"], r["candidate_id"], int(r["seed"])))
    write_csv(cumulative_path, cumulative)
    atomic_json(output / "per_run_summary.json", {"schema_version": 1, "rows": cumulative, "failures": failures})
    per_seed = pd.DataFrame(cumulative)
    stage_candidates = sorted({row["candidate_id"] for row in locked_rows if row["candidate_id"] != "C00_FULL_IGE"})
    seeds = {"R1": [0], "R2": [0, 1, 2], "R3": [0, 1, 2, 3, 4]}[stage]
    summary = [candidate_summary(per_seed, candidate_id, seeds) for candidate_id in stage_candidates]
    summary, advanced = decide(stage, summary, contracts)
    payload = {
        "schema_version": 1, "stage": stage, "decision_locked": True,
        "labels_read_only_after_training_manifest_lock": True,
        "candidates_evaluated": stage_candidates, "candidate_summaries": summary,
        "advanced_candidates": advanced, "failed_runs": failures,
        "parameter_tuning": False, "seed_search": False, "withheld_results_opened": False,
    }
    atomic_json(output / ("%s_decision.json" % stage.lower()), payload)
    write_csv(output / ("%s_paired_deltas.csv" % stage.lower()), summary)
    atomic_json(output / ("%s_semantic_label_access.json" % stage.lower()), {
        "stage": stage, "occurred": True, "after_training_manifest_lock": True,
        "development_datasets_only": ["a1", "placenta"], "withheld_results_opened": False,
    })
    if stage == "R3":
        status = "R3_CANDIDATES_READY_FOR_LOCKED_P22" if advanced else "NO_DEV_CANDIDATE"
        atomic_json(output / "selected_for_p22_lock.json", {
            "schema_version": 1, "status": status, "selected_candidates": advanced,
            "selection_source": "r3_decision.json", "p22_run": False, "d1_run": False,
            "night4b_run": False,
        })
    print("%s_EVALUATED candidates=%d advanced=%s" % (stage, len(stage_candidates), ",".join(advanced)), flush=True)


if __name__ == "__main__":
    main()
