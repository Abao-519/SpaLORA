#!/usr/bin/env python3
"""Deterministic prespecified summaries and figures for authorized Night-2C runs."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import adjusted_rand_score


os.environ.setdefault("R_HOME", "/opt/R/4.0.3/lib/R")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night1_evaluation import load_evaluation_labels


DATASETS = ("a1", "placenta", "p22")
METRICS = ("ari", "nmi", "ami", "fmi", "homogeneity", "v_measure",
           "hungarian_macro_f1", "hungarian_weighted_f1", "hungarian_balanced_accuracy",
           "spatial_neighbor_agreement", "spatial_cluster_moran_mean",
           "embedding_silhouette", "embedding_davies_bouldin")
CONTRASTS = (
    "uniform_scale_effect", "legacy_shape_effect", "scale_at_legacy_shape",
    "legacy_shape_at_full_scale", "factorial_scale_main", "factorial_shape_main",
    "interaction", "asr_vs_uniform_same_scale", "asr_vs_legacy_same_scale",
)


def contrast_values(values: dict) -> dict:
    v0, v1, v2, v3, v4 = (float(values[i]) for i in range(5))
    return {
        "uniform_scale_effect": v1 - v0,
        "legacy_shape_effect": v2 - v0,
        "scale_at_legacy_shape": v3 - v2,
        "legacy_shape_at_full_scale": v3 - v1,
        "factorial_scale_main": 0.5 * ((v1 - v0) + (v3 - v2)),
        "factorial_shape_main": 0.5 * ((v2 - v0) + (v3 - v1)),
        "interaction": (v3 - v2) - (v1 - v0),
        "asr_vs_uniform_same_scale": v4 - v1,
        "asr_vs_legacy_same_scale": v4 - v3,
    }


def sign_flip_p(values) -> float:
    values = np.asarray(values, dtype=float)
    if values.shape != (5,):
        raise ValueError("exact sign-flip test requires the five fixed seeds")
    observed = abs(float(values.mean()))
    permuted = [abs(float(np.mean(values * np.asarray(signs))))
                for signs in product((-1.0, 1.0), repeat=5)]
    return float(sum(value >= observed - 1e-15 for value in permuted) / 32.0)


def bootstrap_ci(values, seed: int, replicates: int) -> tuple:
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    samples = values[rng.randint(0, values.size, size=(replicates, values.size))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def statistics(values, bootstrap_seed, bootstrap_replicates) -> dict:
    values = np.asarray(values, dtype=float)
    low, high = bootstrap_ci(values, bootstrap_seed, bootstrap_replicates)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
            "median": float(np.median(values)), "minimum": float(values.min()),
            "maximum": float(values.max()), "bootstrap_95_low": low,
            "bootstrap_95_high": high, "n": int(values.size)}


def write_csv(path: Path, rows: list, fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError("fields required for empty CSV")
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def variant_code(variant: str) -> int:
    return {"locked_unweighted": 0, "locked_uniform_legacy_scale": 1,
            "locked_legacy_shape_normalized": 2, "locked_legacy_loss_replay": 3,
            "locked_asr_hvg_legacy_scale_diagnostic": 4}[variant]


def collect_main(config):
    expected = {(dataset, variant, seed) for dataset in DATASETS
                for variant in config["variants"] for seed in config["seeds"]}
    found = {}
    for path in (REPO / "results/night2c/raw").glob("*/*/seed_*/metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (payload["dataset"], payload["variant"], int(payload["seed"]))
        if key in found:
            raise AssertionError("duplicate main result %r" % (key,))
        if not all((path.parent / name).is_file() for name in
                   ("clusters.csv", "embedding.npz", "attention.npz", "loss_components.csv", "observation_ids.csv")):
            raise AssertionError("incomplete main result %s" % path.parent)
        found[key] = (path, payload)
    if set(found) != expected:
        raise AssertionError("main result keys differ: missing=%r extra=%r" % (sorted(expected-set(found)), sorted(set(found)-expected)))
    return found


def collect_seed_tables(config, found):
    per_seed = []; loss_rows = []; attention_rows = []; domain_rows = []
    for (dataset, variant, seed), (path, payload) in sorted(found.items()):
        row = {"dataset": dataset, "variant": variant, "variant_code": "V%d" % variant_code(variant),
               "seed": seed, "run_status": "complete"}
        for metric in METRICS:
            value = payload["metrics"].get(metric)
            row[metric] = value
            if value is not None and not math.isfinite(float(value)):
                raise AssertionError("nonfinite metric %s %r" % (metric, (dataset, variant, seed)))
        row["total_seconds"] = payload["timings"]["total_seconds"]
        row["gpu_peak_allocated_mib"] = payload["memory"]["gpu_peak_allocated_mib"]
        per_seed.append(row)
        with (path.parent / "loss_components.csv").open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                loss_rows.append({"dataset": dataset, "variant": variant, "seed": seed, **item})
        attention_rows.append({"dataset": dataset, "variant": variant, "seed": seed,
                               **payload["final_attention_means"]})
        metrics = payload["metrics"]
        ids = pd.read_csv(path.parent / "observation_ids.csv")["observation_id"].astype(str)
        _, labels = load_evaluation_labels(dataset, config["datasets"][dataset], pd.Index(ids))
        supports = pd.Series(labels.astype(str)).value_counts().to_dict()
        for domain, f1 in metrics["hungarian_per_domain_f1"].items():
            domain_rows.append({"dataset": dataset, "variant": variant, "seed": seed,
                                "domain": domain, "support": int(supports.get(str(domain), 0)),
                                "hungarian_f1": f1})
    return per_seed, loss_rows, attention_rows, domain_rows


def summary_tables(config, per_seed):
    lookup = {(r["dataset"], variant_code(r["variant"]), r["seed"]): r for r in per_seed}
    summaries = []; raw_deltas = []; effects = []
    for dataset in DATASETS:
        for variant in config["variants"]:
            rows = [lookup[(dataset, variant_code(variant), seed)] for seed in config["seeds"]]
            for metric in METRICS:
                values = [r[metric] for r in rows]
                summaries.append({"dataset": dataset, "variant": variant, "metric": metric,
                                  **statistics(values, config["bootstrap_seed"], config["bootstrap_replicates"])})
        for metric in METRICS:
            by_contrast = {name: [] for name in CONTRASTS}
            for seed in config["seeds"]:
                values = {index: lookup[(dataset, index, seed)][metric] for index in range(5)}
                contrasts = contrast_values(values)
                for name, value in contrasts.items():
                    by_contrast[name].append(value)
                    raw_deltas.append({"dataset": dataset, "metric": metric, "contrast": name,
                                       "seed": seed, "paired_delta": value})
            for name in CONTRASTS:
                values = by_contrast[name]
                effects.append({"dataset": dataset, "metric": metric, "contrast": name,
                                **statistics(values, config["bootstrap_seed"], config["bootstrap_replicates"]),
                                "exact_32_sign_flip_two_sided_p": sign_flip_p(values)})
    return summaries, raw_deltas, effects


def v3_replay(config, found):
    root = Path(config["night1_results_root"]) / "raw"
    rows = []
    for dataset in DATASETS:
        for seed in config["seeds"]:
            new_path, new = found[(dataset, "locked_legacy_loss_replay", seed)]
            old_path = root / dataset / "legacy_exact" / ("seed_%d" % seed)
            old = json.loads((old_path / "metrics.json").read_text(encoding="utf-8"))
            new_clusters = pd.read_csv(new_path.parent / "clusters.csv")["cluster"].to_numpy()
            old_clusters = pd.read_csv(old_path / "clusters.csv")["cluster"].to_numpy()
            row = {"dataset": dataset, "seed": seed,
                   "partition_agreement_ari": adjusted_rand_score(old_clusters, new_clusters)}
            for metric in METRICS:
                row[metric + "_difference"] = float(new["metrics"][metric]) - float(old["metrics"][metric])
                row[metric + "_absolute_difference"] = abs(row[metric + "_difference"])
            if (old_path / "embedding.npz").exists():
                a = np.load(old_path / "embedding.npz")["SpaLORA"]
                b = np.load(new_path.parent / "embedding.npz")["SpaLORA"]
                row["embedding_relative_l2"] = float(np.linalg.norm(a-b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-12))
            else:
                row["embedding_relative_l2"] = ""
            row["individual_warning"] = row["ari_absolute_difference"] > 0.02 or row["nmi_absolute_difference"] > 0.02
            rows.append(row)
    for dataset in DATASETS:
        subset = [r for r in rows if r["dataset"] == dataset]
        ari_mean = abs(float(np.mean([r["ari_difference"] for r in subset])))
        nmi_mean = abs(float(np.mean([r["nmi_difference"] for r in subset])))
        for row in subset:
            row["dataset_mean_ari_difference"] = float(np.mean([r["ari_difference"] for r in subset]))
            row["dataset_mean_nmi_difference"] = float(np.mean([r["nmi_difference"] for r in subset]))
            row["mean_warning"] = ari_mean > 0.01 or nmi_mean > 0.01
    return rows


def row_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def linear_cka(x, y):
    x = x - x.mean(axis=0, keepdims=True); y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, "fro") ** 2
    return float(cross / max(np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro"), 1e-12))


def technical_replicates(config, found):
    rows = []
    for variant in ("locked_unweighted", "locked_legacy_loss_replay"):
        main_path, main_payload = found[("placenta", variant, 0)]
        paths = [main_path.parent] + [REPO / "results/night2c/technical/placenta" / variant / "seed_0" / ("repeat_%d" % repeat)
                                     for repeat in (1, 2)]
        payloads = [main_payload] + [json.loads((path / "metrics.json").read_text(encoding="utf-8")) for path in paths[1:]]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            ca = pd.read_csv(paths[i] / "clusters.csv")["cluster"].to_numpy()
            cb = pd.read_csv(paths[j] / "clusters.csv")["cluster"].to_numpy()
            ea = row_normalize(np.load(paths[i] / "embedding.npz")["SpaLORA"])
            eb = row_normalize(np.load(paths[j] / "embedding.npz")["SpaLORA"])
            rotation, _ = orthogonal_procrustes(ea, eb)
            procrustes = float(np.linalg.norm(ea @ rotation - eb) / max(np.linalg.norm(eb), 1e-12))
            loss_a = pd.read_csv(paths[i] / "loss_components.csv"); loss_b = pd.read_csv(paths[j] / "loss_components.csv")
            loss_range = float(np.max(np.abs(loss_a["total_loss"].to_numpy() - loss_b["total_loss"].to_numpy())))
            att_a = np.load(paths[i] / "attention.npz"); att_b = np.load(paths[j] / "attention.npz")
            attention_range = max(float(np.max(np.abs(att_a[name] - att_b[name]))) for name in ("alpha", "alpha_omics1", "alpha_omics2"))
            aris = [p["metrics"]["ari"] for p in payloads]; nmis = [p["metrics"]["nmi"] for p in payloads]
            rows.append({"dataset": "placenta", "variant": variant, "repeat_a": i, "repeat_b": j,
                         "partition_agreement_ari": adjusted_rand_score(ca, cb),
                         "embedding_relative_l2": float(np.linalg.norm(ea-eb) / max(np.linalg.norm(ea), np.linalg.norm(eb), 1e-12)),
                         "orthogonal_procrustes_residual": procrustes, "linear_cka": linear_cka(ea, eb),
                         "ari_range_across_three": float(max(aris)-min(aris)),
                         "nmi_range_across_three": float(max(nmis)-min(nmis)),
                         "attention_pair_max_abs_difference": attention_range,
                         "loss_trajectory_pair_max_abs_difference": loss_range})
    return rows


def paper_repro(config, found):
    manuscript = {"a1": (0.2443, 0.3780), "placenta": (0.7226, 0.7408), "p22": (0.4541, 0.5747)}
    rows = []
    for dataset in DATASETS:
        tutorial = json.loads((REPO / "results/night2c/tutorial2022" / dataset / "metrics.json").read_text(encoding="utf-8"))
        v3 = [found[(dataset, "locked_legacy_loss_replay", seed)][1]["metrics"] for seed in config["seeds"]]
        rows.append({"dataset": dataset, "manuscript_ari": manuscript[dataset][0],
                     "manuscript_nmi": manuscript[dataset][1],
                     "night2c_v3_mean_ari": float(np.mean([x["ari"] for x in v3])),
                     "night2c_v3_mean_nmi": float(np.mean([x["nmi"] for x in v3])),
                     "tutorial2022_ari": tutorial["metrics"]["ari"],
                     "tutorial2022_nmi": tutorial["metrics"]["nmi"],
                     "notes": "fixed five model seeds; tutorial is separate exact seed-2022 replay"})
    return rows


def save_figure(fig, stem):
    figures = REPO / "results/night2c/figures"; figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(figures / (stem + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figures(per_seed, effects, losses, attention, replay, technical):
    frame = pd.DataFrame(per_seed)
    for metric in ("ari", "nmi"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
        for ax, dataset in zip(axes, DATASETS):
            subset = frame[frame.dataset == dataset]
            for seed in sorted(subset.seed.unique()):
                s = subset[subset.seed == seed].sort_values("variant_code")
                ax.plot(range(5), s[metric], marker="o", alpha=.65)
            ax.set_title(dataset); ax.set_xticks(range(5), ["V0", "V1", "V2", "V3", "V4"]); ax.set_ylabel(metric.upper())
        fig.suptitle("Paired %s; n=5 fixed model seeds (lines), summaries use sample SD and bootstrap CI" % metric.upper())
        save_figure(fig, "paired_seed_%s" % metric)
    e = pd.DataFrame(effects); e = e[e.metric.isin(["ari", "nmi"])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    selected = ["factorial_scale_main", "factorial_shape_main", "interaction"]
    for row, metric in enumerate(("ari", "nmi")):
        for col, dataset in enumerate(DATASETS):
            s = e[(e.metric == metric) & (e.dataset == dataset) & e.contrast.isin(selected)].set_index("contrast").loc[selected]
            axes[row, col].bar(range(3), s["mean"], yerr=s["sample_sd"], capsize=3)
            axes[row, col].axhline(0, color="black", linewidth=.7); axes[row, col].set_xticks(range(3), ["scale", "shape", "interaction"], rotation=20)
            axes[row, col].set_title("%s %s" % (dataset, metric.upper()))
    fig.suptitle("Factorial effects; n=5 fixed model seeds; bars=mean, error=sample SD (not CI)")
    save_figure(fig, "factorial_effects")
    l = pd.DataFrame(losses)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, dataset in zip(axes, DATASETS):
        s = l[l.dataset == dataset].copy(); s["epoch"] = s.epoch.astype(int)
        for variant, g in s.groupby("variant"):
            mean = g.groupby("epoch")[["final_rna_contribution", "final_modality2_contribution"]].mean()
            ax.plot(mean.index, mean.final_rna_contribution, label=variant.replace("locked_", "") + " RNA", alpha=.7)
            ax.plot(mean.index, mean.final_modality2_contribution, linestyle="--", alpha=.5)
        ax.set_title(dataset); ax.set_xlabel("epoch checkpoint")
    axes[0].set_ylabel("loss contribution"); axes[-1].legend(fontsize=5)
    fig.suptitle("RNA (solid) / modality-2 (dashed); n=5 fixed model seeds; means only")
    save_figure(fig, "loss_contribution_trajectories")
    a = pd.DataFrame(attention)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, dataset in zip(axes, DATASETS):
        s = a[a.dataset == dataset]
        for name in ("cross_omics_rna", "rna_spatial", "modality2_spatial"):
            means = s.groupby("variant")[name].mean().reindex(sorted(s.variant.unique(), key=variant_code))
            ax.plot(range(5), means, marker="o", label=name)
        ax.set_title(dataset); ax.set_xticks(range(5), ["V0", "V1", "V2", "V3", "V4"])
    axes[-1].legend(fontsize=6); fig.suptitle("Final attention means; n=5 fixed model seeds")
    save_figure(fig, "attention_trajectories")
    r = pd.DataFrame(replay)
    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset in DATASETS:
        s = r[r.dataset == dataset]
        ax.plot(s.seed, s.partition_agreement_ari, marker="o", label=dataset)
    ax.set_ylim(-.05, 1.05); ax.set_xlabel("fixed model seed"); ax.set_ylabel("partition agreement ARI"); ax.legend()
    ax.set_title("Night-2C V3 versus Night-1 legacy_exact; n=5 fixed model seeds")
    save_figure(fig, "v3_replay_agreement")
    t = pd.DataFrame(technical)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for ax, metric in zip(axes, ("partition_agreement_ari", "orthogonal_procrustes_residual")):
        for variant, group in t.groupby("variant"):
            ax.scatter([variant_code(variant)] * len(group), group[metric], label=variant)
        ax.set_xticks([0, 3], ["V0", "V3"]); ax.set_ylabel(metric)
    fig.suptitle("Placenta seed-0 technical stability; 3 fresh-process executions per variant")
    save_figure(fig, "placenta_technical_stability")


def main() -> None:
    config = json.loads((REPO / "configs/night2c_numerical_equivalence_factorial.json").read_text(encoding="utf-8"))
    gate = json.loads((REPO / "results/night2c/gate_status.json").read_text(encoding="utf-8"))
    if gate.get("factorial_authorized") is not True:
        raise RuntimeError("summarizer cannot run after P0C failure")
    found = collect_main(config)
    if len(list((REPO / "results/night2c/tutorial2022").glob("*/metrics.json"))) != 3:
        raise AssertionError("tutorial count is not 3/3")
    if len(list((REPO / "results/night2c/technical").glob("*/*/seed_*/repeat_*/metrics.json"))) != 4:
        raise AssertionError("technical count is not 4/4")
    per_seed, losses, attention, domains = collect_seed_tables(config, found)
    summaries, deltas, effects = summary_tables(config, per_seed)
    replay = v3_replay(config, found)
    technical = technical_replicates(config, found)
    paper = paper_repro(config, found)
    results = REPO / "results/night2c"
    write_csv(results / "per_seed_metrics.csv", per_seed)
    write_csv(results / "summary.csv", summaries)
    write_csv(results / "paired_deltas.csv", deltas)
    write_csv(results / "factorial_effects.csv", effects)
    write_csv(results / "loss_components.csv", losses)
    write_csv(results / "attention_summary.csv", attention)
    write_csv(results / "per_domain_f1.csv", domains)
    write_csv(results / "v3_replay_audit.csv", replay)
    write_csv(results / "placenta_seed0_technical_replicates.csv", technical)
    write_csv(results / "paper_repro_audit.csv", paper)
    figures(per_seed, effects, losses, attention, replay, technical)
    print(json.dumps({"main": len(found), "tutorial": 3, "technical": 4,
                      "factorial_effect_rows": len(effects), "v3_replay_rows": len(replay)}, sort_keys=True))


if __name__ == "__main__":
    main()
