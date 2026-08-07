#!/usr/bin/env python3
"""Validate, summarize, and plot the completed Night 1 sweep."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANTS = ["legacy_exact", "corrected_unweighted", "abundance_only", "asr_hvg", "asr_rescue"]
DATASETS = ["a1", "placenta", "p22"]
DISPLAY = {"a1": "Lymph node A1", "placenta": "Placenta", "p22": "P22 mouse brain"}
COLORS = {
    "legacy_exact": "#6b7280",
    "corrected_unweighted": "#2563eb",
    "abundance_only": "#f59e0b",
    "asr_hvg": "#10b981",
    "asr_rescue": "#8b5cf6",
}


def normalize_gene_tables(repo: Path, config: dict) -> None:
    rename = {
        "mean_xlog": "mean_log_abundance",
        "abundance_score_a": "A_score",
        "moran_i": "morans_I",
        "spatial_rank_s": "S_score",
        "reliability_r": "R_score",
        "q_asr": "Q_score",
        "weight": "final_weight",
        "is_rescued": "is_asr_rescued",
    }
    required = [
        "gene",
        "mean_log_abundance",
        "detection_count",
        "detection_rate",
        "morans_I",
        "A_score",
        "S_score",
        "R_score",
        "Q_score",
        "final_weight",
        "is_hvg",
        "is_asr_rescued",
    ]
    metrics = {
        (item["dataset"], item["variant"], item["seed"]): item
        for item in [json.load(open(p)) for p in glob.glob(str(repo / "results/night1/raw/*/*/seed_*/metrics.json"))]
    }
    for path_string in glob.glob(str(repo / "results/night1/interpretability/*_gene_scores.csv")):
        path = Path(path_string)
        dataset = next(name for name in DATASETS if path.name.startswith(name + "_"))
        variant = path.name[len(dataset) + 1 : -len("_gene_scores.csv")]
        table = pd.read_csv(path).rename(columns=rename)
        n_obs = metrics[(dataset, variant, 0)]["n_observations_trained"]
        table["detection_rate"] = table["detection_count"] / n_obs
        if not set(required).issubset(table.columns):
            raise AssertionError("Missing required gene columns in %s" % path)
        for name in ("A_score", "S_score", "R_score", "Q_score", "detection_rate"):
            if not np.isfinite(table[name]).all() or not table[name].between(0, 1).all():
                raise AssertionError("Invalid %s bounds in %s" % (name, path))
        if not table["final_weight"].between(1, 2).all():
            raise AssertionError("Invalid weight bounds in %s" % path)
        table.to_csv(path, index=False)


def load_metrics(repo: Path):
    paths = sorted(glob.glob(str(repo / "results/night1/raw/*/*/seed_*/metrics.json")))
    failures = sorted(glob.glob(str(repo / "results/night1/raw/*/*/seed_*/failure.json")))
    if failures:
        raise AssertionError("Failure artifacts exist: %r" % failures)
    if len(paths) != 75:
        raise AssertionError("Expected 75 metrics files, found %d" % len(paths))
    rows = []
    per_domain = []
    attention_rows = []
    for path_string in paths:
        path = Path(path_string)
        item = json.loads(path.read_text(encoding="utf-8"))
        metrics = item["metrics"]
        row = {
            "dataset": item["dataset"],
            "variant": item["variant"],
            "seed": item["seed"],
            "n_observations_trained": item["n_observations_trained"],
            "n_observations_evaluated": item["n_observations_evaluated"],
        }
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                row[name] = value
        for name, value in item["timings"].items():
            row[name] = value
        for name, value in item["memory"].items():
            row[name] = value
        rows.append(row)
        for domain, value in metrics["hungarian_per_domain_f1"].items():
            per_domain.append({**{k: row[k] for k in ("dataset", "variant", "seed")}, "domain": domain, "f1": value})

        attention = np.load(path.parent / "attention.npz")
        for key in ("alpha_omics1", "alpha_omics2", "alpha"):
            values = attention[key]
            if values.shape != (item["n_observations_trained"], 2):
                raise AssertionError("Unexpected attention shape in %s" % path.parent)
            if np.max(np.abs(values.sum(axis=1) - 1.0)) > 1e-5:
                raise AssertionError("Attention rows do not sum to one in %s" % path.parent)
            attention_rows.append(
                {
                    "dataset": row["dataset"],
                    "variant": row["variant"],
                    "seed": row["seed"],
                    "attention": key,
                    "alternative_0_mean": float(values[:, 0].mean()),
                    "alternative_1_mean": float(values[:, 1].mean()),
                    "alternative_0_sd_across_locations": float(values[:, 0].std(ddof=1)),
                    "max_row_sum_deviation": float(np.max(np.abs(values.sum(axis=1) - 1.0))),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(per_domain), pd.DataFrame(attention_rows)


def write_summaries(repo: Path, per_seed: pd.DataFrame, per_domain: pd.DataFrame, attention: pd.DataFrame):
    results = repo / "results" / "night1"
    per_seed["dataset"] = pd.Categorical(per_seed["dataset"], DATASETS, ordered=True)
    per_seed["variant"] = pd.Categorical(per_seed["variant"], VARIANTS, ordered=True)
    per_seed = per_seed.sort_values(["dataset", "variant", "seed"])
    per_seed.to_csv(results / "per_seed_metrics.csv", index=False)
    per_domain.to_csv(results / "per_domain_f1.csv", index=False)
    attention.to_csv(results / "attention_per_seed.csv", index=False)

    numeric = [c for c in per_seed.columns if c not in ("dataset", "variant", "seed")]
    grouped = per_seed.groupby(["dataset", "variant"], observed=True)[numeric].agg(["mean", "std"])
    grouped.columns = ["%s_%s" % pair for pair in grouped.columns]
    summary = grouped.reset_index()
    summary.to_csv(results / "summary.csv", index=False)

    baseline = summary[summary["variant"] == "corrected_unweighted"].set_index("dataset")
    delta_rows = []
    for _, row in summary.iterrows():
        base = baseline.loc[row["dataset"]]
        output = {"dataset": row["dataset"], "variant": row["variant"]}
        for metric in (
            "ari",
            "nmi",
            "ami",
            "fmi",
            "hungarian_macro_f1",
            "spatial_neighbor_agreement",
        ):
            output[metric + "_delta_vs_corrected_unweighted"] = row[metric + "_mean"] - base[metric + "_mean"]
        delta_rows.append(output)
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(results / "delta_vs_corrected_unweighted.csv", index=False)

    att_summary = attention.groupby(["dataset", "variant", "attention"], as_index=False).agg(
        alternative_0_mean=("alternative_0_mean", "mean"),
        alternative_0_seed_sd=("alternative_0_mean", "std"),
        alternative_1_mean=("alternative_1_mean", "mean"),
        alternative_1_seed_sd=("alternative_1_mean", "std"),
        max_row_sum_deviation=("max_row_sum_deviation", "max"),
    )
    att_summary.to_csv(results / "attention_summary.csv", index=False)
    return summary, deltas, att_summary


def write_gene_diagnostics(repo: Path, config: dict):
    interpretation = repo / "results" / "night1" / "interpretability"
    stats = []
    top = []
    for dataset in DATASETS:
        for variant in ("abundance_only", "asr_hvg", "asr_rescue"):
            path = interpretation / ("%s_%s_gene_scores.csv" % (dataset, variant))
            table = pd.read_csv(path)
            stats.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "n_genes_scored": len(table),
                    "n_selected": int(table["is_selected"].sum()),
                    "n_rescued_non_hvg": int(table["is_asr_rescued"].sum()),
                    **{
                        "%s_%s" % (column, stat): value
                        for column in ("A_score", "S_score", "R_score", "Q_score", "final_weight")
                        for stat, value in table[column].agg(["min", "mean", "max"]).items()
                    },
                }
            )
        rescue = pd.read_csv(interpretation / ("%s_asr_rescue_gene_scores.csv" % dataset))
        selected = rescue[rescue["is_asr_rescued"]].sort_values(["Q_score", "gene"], ascending=[False, True]).head(20)
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            top.append(
                {
                    "dataset": dataset,
                    "rank": rank,
                    "gene": row["gene"],
                    "Q_score": row["Q_score"],
                    "final_weight": row["final_weight"],
                    "A_score": row["A_score"],
                    "S_score": row["S_score"],
                    "R_score": row["R_score"],
                }
            )
    stats_frame = pd.DataFrame(stats)
    top_frame = pd.DataFrame(top)
    stats_frame.to_csv(repo / "results/night1/gene_score_summary.csv", index=False)
    top_frame.to_csv(repo / "results/night1/top20_rescued_genes.csv", index=False)
    return stats_frame, top_frame


def make_plots(repo: Path, config: dict, summary: pd.DataFrame, deltas: pd.DataFrame, attention: pd.DataFrame):
    figure_dir = repo / "figures" / "night1"
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(DATASETS))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for idx, variant in enumerate(VARIANTS):
        part = summary[summary["variant"] == variant].set_index("dataset").loc[DATASETS]
        ax.bar(
            x + (idx - 2) * width,
            part["ari_mean"],
            width,
            yerr=part["ari_std"],
            capsize=2,
            color=COLORS[variant],
            label=variant,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in DATASETS])
    ax.set_ylabel("ARI, mean ± SD (5 seeds)")
    ax.set_title("Night 1 preregistered comparison")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "ari_mean_sd.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    new_variants = ["abundance_only", "asr_hvg", "asr_rescue"]
    for ax, dataset in zip(axes, DATASETS):
        part = deltas[(deltas["dataset"] == dataset) & (deltas["variant"].isin(new_variants))]
        ax.bar(
            np.arange(3),
            part.set_index("variant").loc[new_variants, "ari_delta_vs_corrected_unweighted"],
            color=[COLORS[v] for v in new_variants],
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(0.02, color="#059669", linestyle="--", linewidth=0.8)
        ax.axhline(-0.02, color="#dc2626", linestyle="--", linewidth=0.8)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(["A only", "ASR-HVG", "ASR-rescue"], rotation=25, ha="right")
        ax.set_title(DISPLAY[dataset])
    axes[0].set_ylabel("Mean ARI delta vs corrected unweighted")
    fig.tight_layout()
    fig.savefig(figure_dir / "ari_delta_vs_corrected.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, dataset in zip(axes, DATASETS):
        path = repo / "results/night1/interpretability" / ("%s_asr_rescue_gene_scores.csv" % dataset)
        table = pd.read_csv(path)
        ax.hist(table.loc[table["is_hvg"], "Q_score"], bins=40, alpha=0.6, label="HVG")
        ax.hist(table.loc[table["is_asr_rescued"], "Q_score"], bins=40, alpha=0.7, label="rescued non-HVG")
        ax.set_title(DISPLAY[dataset])
        ax.set_xlabel("Q = A × S × R")
        ax.set_ylabel("Gene count")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "asr_score_distributions.png", dpi=180)
    plt.close(fig)

    for dataset in DATASETS:
        cfg = config["datasets"][dataset]
        source = ad.read_h5ad(cfg["rna"], backed="r")
        coordinates = np.asarray(source.obsm["spatial"])
        ids = source.obs_names.astype(str)
        source.file.close()
        clusters = pd.read_csv(
            repo / "results/night1/raw" / dataset / "asr_rescue" / "seed_0" / "clusters.csv"
        )
        cluster_ids = pd.Index(clusters["observation_id"].astype(str))
        positions = pd.Index(ids).get_indexer(cluster_ids)
        if np.any(positions < 0) or not cluster_ids.is_unique:
            raise AssertionError("Diagnostic spatial plot ID mismatch for %s" % dataset)
        coordinates = coordinates[positions]
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=clusters["cluster"], s=8, cmap="tab20")
        ax.set_title("%s: asr_rescue seed 0" % DISPLAY[dataset])
        ax.set_xlabel("spatial 1")
        ax.set_ylabel("spatial 2")
        ax.set_aspect("equal", adjustable="datalim")
        fig.colorbar(scatter, ax=ax, label="predicted cluster", shrink=0.8)
        fig.tight_layout()
        fig.savefig(figure_dir / ("%s_asr_rescue_spatial.png" % dataset), dpi=180)
        plt.close(fig)

    cross = attention[attention["attention"] == "alpha"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        part = cross[cross["dataset"] == dataset].set_index("variant").loc[VARIANTS]
        ax.bar(np.arange(5), part["alternative_0_mean"], color=[COLORS[v] for v in VARIANTS])
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(["legacy", "unweighted", "A", "ASR", "rescue"], rotation=25, ha="right")
        ax.set_title(DISPLAY[dataset])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Mean cross-omics attention, alternative 0")
    fig.tight_layout()
    fig.savefig(figure_dir / "attention_cross_omics.png", dpi=180)
    plt.close(fig)


def main():
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / "configs/night1.json").read_text(encoding="utf-8"))
    normalize_gene_tables(repo, config)
    per_seed, per_domain, attention = load_metrics(repo)
    summary, deltas, attention_summary = write_summaries(repo, per_seed, per_domain, attention)
    write_gene_diagnostics(repo, config)
    make_plots(repo, config, summary, deltas, attention_summary)
    print("summarized", len(per_seed), "runs into", repo / "results/night1/summary.csv")


if __name__ == "__main__":
    main()
