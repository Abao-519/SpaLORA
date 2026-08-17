#!/usr/bin/env python3
"""One-window post-lock evaluator for the preregistered Night-6D factorial."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night1_evaluation import _mean_cluster_moran
from SpaLORA.night3af_cache import load_cache, sha256_file
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary, symmetric_knn_adjacency
from SpaLORA.night6d_firewall import guard_path
from SpaLORA.night6d_pipeline import GRAPHS, HEADS, atomic_json

OUT = REPO / "outputs/night6d_handoff"
REGISTRY = REPO / "protocols/night6d/SpaLORA_Night6D_Locked_D1_P22_Confirmation_Registry_2026-08-17.json"
BASE = {
    "d1": Path("/root/autodl-fs/night6d_cache_20260817/base/d1"),
    "p22": Path("/root/autodl-fs/night3af_p0d_builds_20260810/process_a/p22"),
}
DATASET_KEY = {"d1": "d1_human_lymph_node", "p22": "p22_mouse_brain"}
G00, G04 = list(GRAPHS)
H00, H05 = list(HEADS)
METRICS = ("ari", "nmi", "q", "neighbor_agreement", "moran_i", "geary_c", "boundary_disagreement")


def locked_inputs() -> tuple[dict, dict]:
    training_path = OUT / "locked_training_manifest.json"
    transform_path = OUT / "locked_transform_manifest.json"
    training = json.loads(training_path.read_text())
    transform = json.loads(transform_path.read_text())
    if training.get("status") != "LOCKED" or training.get("planned_units") != 40:
        raise RuntimeError("40-unit training manifest is not locked")
    if training.get("success_count") != 40 or len(training.get("runs", [])) != 40:
        raise RuntimeError("40 successful training units required before evaluation")
    if any(not r.get("checkpoint_round_trip_pass") or not r.get("h00_cluster_reload_exact")
           for r in training["runs"]):
        raise RuntimeError("40/40 checkpoint round-trip and H00 exact replay required")
    if transform.get("status") != "LOCKED" or not transform.get("locked_before_label_access"):
        raise RuntimeError("transform manifest was not total-locked before labels")
    if transform.get("planned_transforms") != 80 or transform.get("attempted_transforms") != 80:
        raise RuntimeError("80 terminal transforms required before evaluation")
    keys = [(x["dataset"], x["graph_id"], x["head_id"], int(x["seed"]))
            for x in transform["transforms"]]
    expected = [(d, g, h, s) for d in ("d1", "p22") for g in (G00, G04)
                for h in (H00, H05) for s in range(10)]
    if keys != expected or len(set(keys)) != 80:
        raise RuntimeError("fixed transform order or primary key drift")
    return training, transform


def load_labels_once(registry: dict, prepared: dict) -> tuple[dict, dict]:
    labels = {}
    audit = {}
    for dataset in ("d1", "p22"):
        cfg = registry["datasets"][DATASET_KEY[dataset]]
        gt = guard_path(cfg["ground_truth_path"], role="evaluator",
                        operation="parse_ground_truth", phase_locked=True,
                        audit_log=OUT / "firewall/evaluator_access.jsonl")
        if sha256_file(gt) != cfg["ground_truth_sha256"]:
            raise RuntimeError(f"{dataset} ground-truth SHA drift at authorized window")
        table = pd.read_csv(gt)
        required = [cfg["ground_truth_id_column"], cfg["ground_truth_label_column"]]
        if any(c not in table.columns for c in required):
            raise RuntimeError(f"{dataset} ground-truth schema mismatch")
        ids = table[required[0]].astype(str)
        true = table[required[1]]
        if not ids.is_unique or true.isna().any():
            raise RuntimeError(f"{dataset} ground-truth IDs/labels invalid")
        mapping = pd.Series(true.astype(str).to_numpy(), index=ids.to_numpy())
        obs = prepared[dataset].obs_names.astype(str)
        valid = obs.isin(mapping.index)
        positions = np.flatnonzero(valid)
        aligned = mapping.reindex(obs[valid])
        if aligned.isna().any() or positions.size != len(mapping):
            raise RuntimeError(f"{dataset} exact ground-truth intersection mismatch")
        if dataset == "d1" and positions.size != len(obs):
            raise RuntimeError("D1 requires complete ground-truth alignment")
        if aligned.nunique() != int(cfg["n_clusters"]):
            raise RuntimeError(f"{dataset} locked ontology K mismatch")
        labels[dataset] = (positions, aligned.to_numpy(dtype=str))
        audit[dataset] = {
            "ground_truth_path": str(gt), "ground_truth_sha256": sha256_file(gt),
            "rows": int(len(mapping)), "aligned_rows": int(positions.size),
            "unique_labels": int(aligned.nunique()), "known_k": int(cfg["n_clusters"]),
            "deserialized_into_memory": True,
            "explicitly_indexed_or_observed": True,
            "used_for_training_or_selection": False,
            "authorized_role": "evaluator",
        }
    return labels, audit


def metric_row(true: np.ndarray, pred_labeled: np.ndarray,
               pred_all: np.ndarray, coords: np.ndarray) -> dict:
    graph = symmetric_knn_adjacency(coords, 18)
    rows, cols = graph.nonzero()
    neighbor = float(np.mean(pred_all[rows] == pred_all[cols]))
    moran = float(_mean_cluster_moran(pred_all, graph))
    geary, _ = mean_one_vs_rest_geary(pred_all, graph)
    ari = float(adjusted_rand_score(true, pred_labeled))
    nmi = float(normalized_mutual_info_score(true, pred_labeled))
    return {"ari": ari, "nmi": nmi, "q": (ari + nmi) / 2.0,
            "neighbor_agreement": neighbor, "moran_i": moran,
            "geary_c": float(geary), "boundary_disagreement": 1.0 - neighbor}


def exact_sign_flip(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise RuntimeError("exact sign-flip requires ten finite paired deltas")
    observed = float(values.mean())
    means = np.empty(1024, dtype=np.float64)
    for i, signs in enumerate(itertools.product((-1.0, 1.0), repeat=10)):
        means[i] = float(np.mean(values * np.asarray(signs)))
    count = int(np.sum(means >= observed - 1e-15))
    return {"statistic": "mean_delta", "alternative": "gain_one_sided",
            "observed_mean": observed, "enumerations": 1024,
            "tail_count": count, "raw_p": count / 1024.0}


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda k: (raw[k], k))
    out = {}
    running = 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (m - rank) * float(raw[key]))
        out[key] = min(1.0, running)
    return out


def bootstrap(delta: pd.DataFrame, seed: int = 20260817, n: int = 100000) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(n, len(delta)))
    result = {"seed": seed, "resamples": n, "method": "paired percentile 95% CI"}
    for metric in ("delta_ari", "delta_nmi", "delta_q"):
        values = delta[metric].to_numpy(dtype=np.float64)
        sampled = values[indices].mean(axis=1)
        low, high = np.percentile(sampled, [2.5, 97.5])
        result[metric] = {"mean": float(values.mean()), "ci_lower": float(low),
                          "ci_upper": float(high)}
    return result


def delta_frame(metrics: pd.DataFrame, dataset: str, graph: str, head: str) -> pd.DataFrame:
    ref = metrics[(metrics.dataset == dataset) & (metrics.graph_id == G00) &
                  (metrics.head_id == H00)].sort_values("seed")
    cand = metrics[(metrics.dataset == dataset) & (metrics.graph_id == graph) &
                   (metrics.head_id == head)].sort_values("seed")
    if len(ref) != 10 or len(cand) != 10 or not np.array_equal(ref.seed, cand.seed):
        raise RuntimeError(f"incomplete paired contrast {dataset}/{graph}/{head}")
    rows = {"dataset": dataset, "seed": ref.seed.to_numpy(dtype=int)}
    for metric in METRICS:
        rows[f"delta_{metric.replace('_agreement', '').replace('_i', '').replace('_c', '').replace('_disagreement', '')}"] = \
            cand[metric].to_numpy(dtype=float) - ref[metric].to_numpy(dtype=float)
    # Use unambiguous canonical names irrespective of the compact string mapping above.
    rows.update({
        "delta_ari": cand.ari.to_numpy() - ref.ari.to_numpy(),
        "delta_nmi": cand.nmi.to_numpy() - ref.nmi.to_numpy(),
        "delta_q": cand.q.to_numpy() - ref.q.to_numpy(),
        "delta_neighbor": cand.neighbor_agreement.to_numpy() - ref.neighbor_agreement.to_numpy(),
        "delta_moran": cand.moran_i.to_numpy() - ref.moran_i.to_numpy(),
        "delta_geary": cand.geary_c.to_numpy() - ref.geary_c.to_numpy(),
        "delta_boundary": cand.boundary_disagreement.to_numpy() - ref.boundary_disagreement.to_numpy(),
    })
    return pd.DataFrame(rows)[["dataset", "seed", "delta_ari", "delta_nmi", "delta_q",
                               "delta_neighbor", "delta_moran", "delta_geary", "delta_boundary"]]


def contrast_summary(delta: pd.DataFrame) -> dict:
    exact = exact_sign_flip(delta.delta_q.to_numpy())
    boot = bootstrap(delta)
    return {
        "paired_seeds": delta.seed.astype(int).tolist(),
        "mean_delta_ari": float(delta.delta_ari.mean()),
        "mean_delta_nmi": float(delta.delta_nmi.mean()),
        "mean_delta_q": float(delta.delta_q.mean()),
        "median_delta_q": float(delta.delta_q.median()),
        "sample_sd_delta_q": float(delta.delta_q.std(ddof=1)),
        "ari_wins": int((delta.delta_ari > 0).sum()),
        "nmi_wins": int((delta.delta_nmi > 0).sum()),
        "q_wins": int((delta.delta_q > 0).sum()),
        "per_seed_deltas": delta.to_dict("records"),
        "exact_sign_flip": exact, "bootstrap": boot,
    }


def spatial_summary(delta: pd.DataFrame) -> dict:
    values = {k: float(delta[k].mean()) for k in
              ("delta_neighbor", "delta_moran", "delta_geary", "delta_boundary")}
    failed = ((values["delta_neighbor"] < -0.03 and values["delta_moran"] < -0.03) or
              (values["delta_geary"] > 0.03 and
               (values["delta_neighbor"] < -0.03 or values["delta_moran"] < -0.03)))
    return {**values, "spatial_gate_failed": bool(failed),
            "boundary_disagreement_role": "report_only"}


def primary_and_secondary(metrics: pd.DataFrame) -> tuple[dict, dict, dict, str]:
    primary = {"family": "two dataset-level primary exact tests", "datasets": {}}
    spatial = {"applied_per_dataset": True, "datasets": {}}
    raw_primary = {}
    for dataset in ("d1", "p22"):
        delta = delta_frame(metrics, dataset, G04, H05)
        summary = contrast_summary(delta)
        primary["datasets"][dataset] = summary
        raw_primary[dataset] = summary["exact_sign_flip"]["raw_p"]
        spatial["datasets"][dataset] = spatial_summary(delta)
    adjusted = holm(raw_primary)
    for dataset in ("d1", "p22"):
        row = primary["datasets"][dataset]
        row["holm_adjusted_p"] = adjusted[dataset]
        material = (row["mean_delta_ari"] > 0 and row["mean_delta_nmi"] > 0 and
                    row["mean_delta_q"] >= .01 and row["q_wins"] >= 7 and
                    row["holm_adjusted_p"] < .05 and
                    row["bootstrap"]["delta_q"]["ci_lower"] > 0)
        positive = row["mean_delta_ari"] > 0 and row["mean_delta_nmi"] > 0 and row["mean_delta_q"] > 0
        row["material_accuracy_confirmed"] = bool(material)
        row["dataset_conclusion"] = ("MATERIAL_ACCURACY_CONFIRMED" if material else
                                     "POSITIVE_BUT_INCONCLUSIVE" if positive else
                                     "NOT_CONFIRMED_OR_MIXED_METRICS")
    primary["holm_method"] = "step-down across D1 and P22 primary p-values"

    definitions = {
        "head_only": (G00, H05),
        "graph_only": (G04, H00),
    }
    secondary = {"family": "six secondary factorial exact tests", "contrasts": {}}
    secondary_raw = {}
    for dataset in ("d1", "p22"):
        for name, (graph, head) in definitions.items():
            delta = delta_frame(metrics, dataset, graph, head)
            key = f"{dataset}:{name}"
            secondary["contrasts"][key] = contrast_summary(delta)
            secondary_raw[key] = secondary["contrasts"][key]["exact_sign_flip"]["raw_p"]
        gh = delta_frame(metrics, dataset, G04, H05)
        g = delta_frame(metrics, dataset, G04, H00)
        h = delta_frame(metrics, dataset, G00, H05)
        interaction = gh.copy()
        for col in [c for c in gh.columns if c.startswith("delta_")]:
            interaction[col] = gh[col] - g[col] - h[col]
        key = f"{dataset}:interaction"
        secondary["contrasts"][key] = contrast_summary(interaction)
        secondary_raw[key] = secondary["contrasts"][key]["exact_sign_flip"]["raw_p"]
    secondary_adjusted = holm(secondary_raw)
    for key, value in secondary_adjusted.items():
        secondary["contrasts"][key]["holm_adjusted_p"] = value
    secondary["used_for_candidate_selection"] = False
    secondary["can_change_primary_method_identity"] = False

    conclusions = [primary["datasets"][d]["dataset_conclusion"] for d in ("d1", "p22")]
    if conclusions == ["MATERIAL_ACCURACY_CONFIRMED", "MATERIAL_ACCURACY_CONFIRMED"]:
        terminal = ("NIGHT6D_D1_P22_BALANCED_CONFIRMED" if
                    not any(spatial["datasets"][d]["spatial_gate_failed"] for d in ("d1", "p22")) else
                    "NIGHT6D_D1_P22_ACCURACY_CONFIRMED_WITH_SPATIAL_TRADEOFF")
    elif all(x == "NOT_CONFIRMED_OR_MIXED_METRICS" for x in conclusions):
        terminal = "NIGHT6D_LOCKED_CANDIDATE_NOT_CONFIRMED"
    else:
        terminal = "NIGHT6D_PARTIAL_OR_MIXED_EVIDENCE"
    return primary, secondary, spatial, terminal


def main() -> None:
    training, transform = locked_inputs()
    registry = json.loads(REGISTRY.read_text())
    prepared = {d: load_cache(path, sha256_file(path / "manifest.json")) for d, path in BASE.items()}
    labels, label_audit = load_labels_once(registry, prepared)
    rows = []
    training_by_key = {(r["dataset"], r["graph_id"], int(r["seed"])): r for r in training["runs"]}
    for item in transform["transforms"]:
        if item["status"] != "success":
            continue
        dataset = item["dataset"]
        table = pd.read_csv(Path(item["head_dir"]) / "clusters.csv")
        obs = prepared[dataset].obs_names.astype(str)
        if not np.array_equal(table["observation_id"].astype(str).to_numpy(), obs.to_numpy()):
            raise RuntimeError("cluster observation order mismatch")
        pred = table["cluster"].to_numpy(dtype=np.int64)
        pos, true = labels[dataset]
        observed = metric_row(true, pred[pos], pred, prepared[dataset].coordinates)
        train = training_by_key[(dataset, item["graph_id"], int(item["seed"]))]
        rows.append({"dataset": dataset, "graph_id": item["graph_id"],
                     "head_id": item["head_id"], "seed": int(item["seed"]),
                     **observed,
                     "training_runtime_seconds": train["runtime_seconds"],
                     "head_runtime_seconds": item["runtime_seconds"],
                     "effective_runtime_seconds": train["runtime_seconds"] + item["runtime_seconds"],
                     "peak_gpu_allocated_mib": train["peak_gpu_allocated_mib"],
                     "process_peak_rss_mib": max(train["process_peak_rss_mib"], item["process_peak_rss_mib"]),
                     "checkpoint_file_sha256": train["checkpoint_file_sha256"],
                     "cluster_file_sha256": item["cluster_file_sha256"]})
    metrics = pd.DataFrame(rows).sort_values(["dataset", "graph_id", "head_id", "seed"])
    metrics.to_csv(OUT / "d1_p22_per_seed_metrics.csv", index=False)

    primary_complete = len(metrics) == 80 and not metrics.duplicated(
        ["dataset", "graph_id", "head_id", "seed"]).any()
    if primary_complete:
        primary, secondary, spatial, terminal = primary_and_secondary(metrics)
    else:
        primary = {"status": "PRIMARY_CELLS_INCOMPLETE", "successful_cells": len(metrics)}
        secondary = {"status": "NOT_RUN_PRIMARY_INCOMPLETE"}
        spatial = {"status": "NOT_RUN_PRIMARY_INCOMPLETE"}
        terminal = "NIGHT6D_CONFIRMATION_INCOMPLETE_NUMERICAL"
    atomic_json(OUT / "primary_confirmatory_tests.json", primary)
    atomic_json(OUT / "secondary_factorial_tests.json", secondary)
    atomic_json(OUT / "spatial_protection.json", spatial)
    atomic_json(OUT / "label_window_audit.json", {
        "status": "PASS", "single_authorized_window": True,
        "training_manifest_sha256": sha256_file(OUT / "locked_training_manifest.json"),
        "transform_manifest_sha256": sha256_file(OUT / "locked_transform_manifest.json"),
        "training_locked_before_access": True, "checkpoint_roundtrips_before_access": "40/40",
        "transforms_terminal_before_access": "80/80", "datasets_opened_together": ["d1", "p22"],
        "datasets": label_audit, "labels_used_for_training_transform_or_selection": False,
        "return_to_training_or_clustering_after_access": False,
    })
    atomic_json(OUT / "night6d_decision.json", {
        "schema_version": 1, "terminal_status": terminal,
        "primary_method": f"{G04}/{H05}", "fresh_reference": f"{G00}/{H00}",
        "primary_complete": primary_complete, "candidate_selection_in_night6d": False,
        "parameter_tuning": False, "seed_search": False, "labels_opened_post_total_lock": True,
        "d1_role": "primary_held_out_within_study_confirmation",
        "p22_role": "cross_dataset_confirmation_not_pristine",
    })
    print(json.dumps({"terminal_status": terminal, "evaluated_cells": len(metrics)}, sort_keys=True))


if __name__ == "__main__":
    main()
