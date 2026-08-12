#!/usr/bin/env python3
"""Repair Night-3B evidence semantics and redraw figures without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night4a_evidence import METRIC_DIRECTIONS, corrected_count


ABLATIONS = (
    "DROP_RNA_RECON", "DROP_MOD2_RECON", "DROP_CORR1", "DROP_CORR2",
    "UNIFORM_WITHIN", "UNIFORM_CROSS", "UNIFORM_ALL",
)
DATASETS = ("a1", "placenta", "p22")
CHANNELS = (
    "cross_omics_rna_attention", "cross_omics_modality2_attention",
    "rna_spatial_attention", "rna_feature_attention",
    "modality2_spatial_attention", "modality2_feature_attention",
)
LOSSES = ("rna_recon", "mod2_recon", "corr1", "corr2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def save_figure(fig, figure_dir: Path, name: str, records: list, sources: list[Path]) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle(fig._suptitle.get_text() if fig._suptitle else "", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = figure_dir / f"{name}.png"
    pdf = figure_dir / f"{name}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", metadata={"Title": name, "Creator": "SpaLORA Night-4A locked-evidence redraw"})
    width, height = fig.get_size_inches()
    plt.close(fig)
    records.append({
        "figure": name,
        "png": str(png), "pdf": str(pdf),
        "png_sha256": sha256(png), "pdf_sha256": sha256(pdf),
        "source_files": [{"path": str(p), "sha256": sha256(p)} for p in sources],
        "script_sha256": sha256(Path(__file__)),
        "generation_command": f"python {Path(__file__).name} --source <locked-night3b> --output <night4a>",
        "size_inches": [float(width), float(height)],
        "font": "DejaVu Sans",
        "structural_status": "PASS",
        "visual_qa_status": "PENDING_RENDERED_REVIEW",
    })


def recompute_and_validate(source: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    per_seed_path = source / "per_seed_metrics.csv"
    paired_path = source / "paired_ablation_deltas.csv"
    summary_path = source / "summary.csv"
    support_path = source / "component_support_matrix.csv"
    # round_trip is required to recover the exact binary64 values serialized by
    # the locked evaluator.  The default fast parser changes one runtime mean.
    per_seed = pd.read_csv(per_seed_path, float_precision="round_trip")
    paired = pd.read_csv(paired_path, float_precision="round_trip")
    summary = pd.read_csv(summary_path, float_precision="round_trip")
    support = pd.read_csv(support_path, float_precision="round_trip")

    keys = per_seed[["dataset", "variant", "seed"]]
    expected_keys = {(d, v, s) for d in DATASETS for v in ("FULL_IGE",) + ABLATIONS for s in range(5)}
    actual_keys = set(map(tuple, keys.itertuples(index=False, name=None)))
    if len(per_seed) != 120 or not keys.duplicated().sum() == 0 or actual_keys != expected_keys:
        raise AssertionError("locked per-seed key coverage is not exactly 120/120")

    stat_diffs = []
    stat_ulp_units = []
    for row in summary.itertuples(index=False):
        values = per_seed[(per_seed.dataset == row.dataset) & (per_seed.variant == row.variant)][row.metric].to_numpy(float)
        expected = {
            "mean": np.mean(values), "sd": np.std(values, ddof=1),
            "median": np.median(values), "min": np.min(values), "max": np.max(values),
        }
        for field, calculated in expected.items():
            observed = float(getattr(row, field))
            difference = abs(observed - float(calculated))
            scale = max(abs(observed), abs(float(calculated)))
            ulp = float(np.spacing(scale)) if scale else float(np.nextafter(0.0, 1.0))
            stat_diffs.append(difference)
            stat_ulp_units.append(difference / ulp)
    max_summary_diff = float(max(stat_diffs))
    max_summary_ulp = float(max(stat_ulp_units))
    if max_summary_diff != 0.0:
        raise AssertionError(f"summary recomputation drift: abs={max_summary_diff} ulp={max_summary_ulp}")

    seed_rows = paired[paired.row_type == "seed"]
    delta_diffs = []
    for ablation in ABLATIONS:
        full = per_seed[per_seed.variant == "FULL_IGE"].set_index(["dataset", "seed"])
        abl = per_seed[per_seed.variant == ablation].set_index(["dataset", "seed"])
        locked = seed_rows[seed_rows.ablation == ablation].set_index(["dataset", "seed"])
        for metric in METRIC_DIRECTIONS:
            calc = full[metric] - abl[metric]
            observed = locked[f"{metric}_full_minus_ablation"]
            delta_diffs.extend(np.abs(calc.to_numpy(float) - observed.to_numpy(float)))
    max_seed_delta_diff = float(max(delta_diffs))
    if max_seed_delta_diff != 0.0:
        raise AssertionError(f"paired seed delta drift: {max_seed_delta_diff}")

    simplification = bool(support["simplification_dominance"].astype(bool).any())
    loss_supported = int(((support.classification == "SUPPORTED") & support.ablation.str.startswith("DROP_")).sum())
    attention_supported = bool(((support.classification == "SUPPORTED") & support.ablation.str.startswith("UNIFORM_")).any())
    recommendation = (
        "SIMPLIFY_CANDIDATE" if simplification
        else "KEEP_FULL" if attention_supported and loss_supported >= 2
        else "MIXED_EVIDENCE"
    )
    if recommendation != "MIXED_EVIDENCE":
        raise AssertionError(f"locked method decision changed: {recommendation}")

    corrected_rows = []
    for ablation in ABLATIONS:
        ablation_seeds = seed_rows[seed_rows.ablation == ablation]
        for metric in METRIC_DIRECTIONS:
            column = f"{metric}_full_minus_ablation"
            for dataset in DATASETS:
                values = ablation_seeds[ablation_seeds.dataset == dataset][column].to_numpy(float)
                row = {"ablation": ablation, "dataset": dataset, "metric": metric, "scope": "five_seed"}
                row.update(corrected_count(values, metric))
                row.update({"delta_definition": "FULL_IGE - ABLATION", "delta_mean": float(np.mean(values)), "delta_sd": float(np.std(values, ddof=1))})
                corrected_rows.append(row)
            values = ablation_seeds[column].to_numpy(float)
            row = {"ablation": ablation, "dataset": "ALL", "metric": metric, "scope": "fifteen_seed"}
            row.update(corrected_count(values, metric))
            row.update({"delta_definition": "FULL_IGE - ABLATION", "delta_mean": float(np.mean(values)), "delta_sd": float(np.std(values, ddof=1))})
            corrected_rows.append(row)
    corrected = pd.DataFrame(corrected_rows)
    corrected.to_csv(out / "night3b_corrected_win_counts.csv", index=False)

    registry = {
        metric: {
            "direction": "higher_is_better" if spec.higher_is_better else "lower_is_better",
            "delta_definition": "FULL_IGE - ABLATION", "full_win_rule": spec.full_win_rule,
        }
        for metric, spec in METRIC_DIRECTIONS.items()
    }
    atomic_json(out / "metric_direction_registry.json", registry)
    report = {
        "per_seed_rows": len(per_seed), "unique_primary_keys": len(actual_keys),
        "summary_rows": len(summary), "max_summary_absolute_difference": max_summary_diff,
        "max_summary_ulp_distance": max_summary_ulp,
        "summary_acceptance_rule": "exact binary64 equality after pandas round_trip CSV parsing; no tolerance",
        "paired_seed_rows": len(seed_rows), "max_seed_delta_absolute_difference": max_seed_delta_diff,
        "original_method_recommendation": "MIXED_EVIDENCE",
        "recomputed_method_recommendation": "MIXED_EVIDENCE",
        "decision_changed": False,
        "original_artifacts_modified": False,
        "source_sha256": {p.name: sha256(p) for p in (per_seed_path, paired_path, summary_path, support_path)},
    }
    atomic_json(out / "night3b_exact_diff_report.json", report)
    return per_seed, paired, report


def forest_plot(paired: pd.DataFrame, out: Path, records: list, sources: list[Path]):
    ds = paired[paired.row_type == "dataset_summary"]
    labels = [f"{a} | {d}" for a in ABLATIONS for d in DATASETS]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
    for axis, metric in zip(axes, ("ari", "nmi")):
        means, low, high = [], [], []
        for a in ABLATIONS:
            for d in DATASETS:
                row = ds[(ds.ablation == a) & (ds.dataset == d)].iloc[0]
                means.append(row[f"{metric}_full_minus_ablation_mean"])
                low.append(row[f"{metric}_bootstrap_ci_low"])
                high.append(row[f"{metric}_bootstrap_ci_high"])
        means = np.asarray(means); low = np.asarray(low); high = np.asarray(high)
        axis.errorbar(means, y, xerr=np.vstack([means-low, high-means]), fmt="o", ms=4, capsize=2)
        axis.axvline(0, color="black", lw=.8)
        axis.set_title(f"{metric.upper()} paired delta")
        axis.set_xlabel("Delta = FULL_IGE - ablation (higher is better)")
        axis.grid(axis="y", alpha=.18)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Night-3B paired five-seed ablation effects (descriptive 95% bootstrap CI)")
    save_figure(fig, out, "night3b_ablation_forest_publication_candidate", records, sources)


def heatmap_with_uncertainty(per_seed: pd.DataFrame, out: Path, records: list, sources: list[Path], ablations, name, title):
    metrics = ("ari", "nmi")
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(ablations) * .8)))
    for axis, metric in zip(axes, metrics):
        matrix = np.zeros((len(ablations), len(DATASETS)))
        sd = np.zeros_like(matrix)
        for i, a in enumerate(ablations):
            for j, d in enumerate(DATASETS):
                full = per_seed[(per_seed.dataset == d) & (per_seed.variant == "FULL_IGE")].sort_values("seed")[metric].to_numpy()
                abl = per_seed[(per_seed.dataset == d) & (per_seed.variant == a)].sort_values("seed")[metric].to_numpy()
                delta = full - abl
                matrix[i, j] = delta.mean(); sd[i, j] = delta.std(ddof=1)
        lim = max(abs(matrix.min()), abs(matrix.max()), .01)
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-lim, vmax=lim, aspect="auto")
        axis.set_xticks(range(3)); axis.set_xticklabels(DATASETS)
        axis.set_yticks(range(len(ablations))); axis.set_yticklabels(ablations)
        axis.set_title(f"{metric.upper()} mean +/- seed SD")
        for i in range(len(ablations)):
            for j in range(3):
                axis.text(j, i, f"{matrix[i,j]:+.3f}\n+/-{sd[i,j]:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, shrink=.75, label="FULL_IGE - ablation")
    fig.suptitle(title + "; five seeds are optimization repeats")
    save_figure(fig, out, name, records, sources)


def p22_heterogeneity(per_seed: pd.DataFrame, out: Path, records: list, sources: list[Path]):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for axis, metric, direction in zip(axes, ("ari", "nmi", "spatial_cluster_geary_mean"), ("higher", "higher", "lower")):
        arrays = []
        for a in ABLATIONS:
            full = per_seed[(per_seed.dataset == "p22") & (per_seed.variant == "FULL_IGE")].sort_values("seed")[metric].to_numpy()
            abl = per_seed[(per_seed.dataset == "p22") & (per_seed.variant == a)].sort_values("seed")[metric].to_numpy()
            arrays.append(full - abl)
        axis.boxplot(arrays, labels=ABLATIONS, showfliers=False)
        for x, values in enumerate(arrays, start=1):
            axis.scatter(np.full(5, x), values, s=16, color="black", alpha=.7, zorder=3)
        axis.axhline(0, color="black", lw=.8)
        axis.set_title(f"{metric}\n({direction} is better for raw metric)")
        axis.tick_params(axis="x", rotation=55, labelsize=7)
        axis.set_ylabel("Delta = FULL_IGE - ablation")
    fig.suptitle("P22 seed heterogeneity: all seven preregistered ablations, all five seeds")
    save_figure(fig, out, "p22_all_ablation_seed_heterogeneity", records, sources)


def attention_distribution(source: Path, out: Path, records: list):
    path = source / "attention_spot_summary.csv"
    table = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    labels = [c.replace("_attention", "").replace("_", "\n") for c in CHANNELS]
    for axis, dataset in zip(axes, DATASETS):
        arrays = [table[(table.dataset == dataset) & (table.channel == c)].sort_values("seed")["mean"].to_numpy() for c in CHANNELS]
        axis.boxplot(arrays, labels=labels, showfliers=False)
        for x, values in enumerate(arrays, start=1): axis.scatter(np.full(len(values), x), values, s=20, color="black")
        axis.set_title(dataset); axis.tick_params(axis="x", labelsize=7)
        axis.set_xlabel("six named attention channels")
    axes[0].set_ylabel("spot-level attention mean per seed")
    fig.suptitle("Attention distribution summaries; five seeds are optimization repeats")
    save_figure(fig, out, "attention_six_channel_distribution", records, [path])


def domain_plot(source: Path, out: Path, records: list):
    path = source / "attention_domain_association.csv"
    table = pd.read_csv(path)
    channel = "cross_omics_rna_attention"
    fig, axes = plt.subplots(3, 1, figsize=(15, 13))
    for axis, dataset in zip(axes, DATASETS):
        subset = table[(table.dataset == dataset) & (table.channel == channel)]
        domains = sorted(subset.domain.unique())
        arrays = [subset[subset.domain == d].sort_values("seed").domain_mean.to_numpy() for d in domains]
        axis.boxplot(arrays, labels=domains, showfliers=False)
        for x, values in enumerate(arrays, start=1): axis.scatter(np.full(len(values), x), values, s=12, color="black")
        axis.set_title(dataset); axis.tick_params(axis="x", rotation=40, labelsize=7)
        axis.set_ylabel("domain mean attention")
    fig.suptitle("Cross-omics RNA attention by true annotated domain; all five optimization seeds")
    save_figure(fig, out, "attention_domain_association_named", records, [path])


def gradient_plot(source: Path, out: Path, records: list):
    path = source / "gradient_influence_trajectories.csv"
    table = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    colors = dict(zip(LOSSES, plt.cm.tab10.colors[:4]))
    for axis, dataset in zip(axes, DATASETS):
        ds = table[table.dataset == dataset]
        for loss in LOSSES:
            col = f"{loss}_weighted_gradient_share"
            pivot = ds.pivot(index="fraction_of_training", columns="seed", values=col).sort_index()
            x = pivot.index.to_numpy(float); mean = pivot.mean(axis=1).to_numpy(); sd = pivot.std(axis=1, ddof=1).to_numpy()
            axis.fill_between(x, mean-sd, mean+sd, color=colors[loss], alpha=.16)
            axis.plot(x, mean, color=colors[loss], label=loss)
            for seed in pivot.columns: axis.plot(x, pivot[seed], color=colors[loss], lw=.35, alpha=.18)
            axis.scatter(x, mean, color=colors[loss], s=8)
        axis.set_title(dataset); axis.set_xlabel("fraction of training")
    axes[0].set_ylabel("weighted-gradient share")
    axes[-1].legend(fontsize=8)
    fig.suptitle("IGE weighted-gradient trajectories: mean +/- seed SD with individual seed traces")
    save_figure(fig, out, "ige_gradient_trajectories_with_seed_uncertainty", records, [path])


def spatial_tradeoff(per_seed: pd.DataFrame, out: Path, records: list, sources: list[Path]):
    metrics = (
        ("ari", "ARI (higher better)"), ("spatial_neighbor_agreement", "Neighbor agreement (higher better)"),
        ("spatial_cluster_moran_mean", "Moran I (higher better)"), ("spatial_cluster_geary_mean", "Geary C (lower better)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    colors = dict(zip(DATASETS, plt.cm.Set1.colors[:3]))
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        for j, ablation in enumerate(ABLATIONS):
            for k, dataset in enumerate(DATASETS):
                full = per_seed[(per_seed.dataset == dataset) & (per_seed.variant == "FULL_IGE")].sort_values("seed")[metric].to_numpy()
                abl = per_seed[(per_seed.dataset == dataset) & (per_seed.variant == ablation)].sort_values("seed")[metric].to_numpy()
                delta = full - abl
                x = j + (k-1)*.20
                axis.scatter(np.full(5, x), delta, s=13, color=colors[dataset], alpha=.55)
                axis.errorbar(x, delta.mean(), yerr=delta.std(ddof=1), fmt="o", color=colors[dataset], capsize=2)
        axis.axhline(0, color="black", lw=.7)
        axis.set_xticks(range(7)); axis.set_xticklabels(ABLATIONS, rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("Delta = FULL_IGE - ablation"); axis.set_title(label)
    handles = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=colors[d], label=d) for d in DATASETS]
    axes[0,0].legend(handles=handles)
    fig.suptitle("Paired spatial trade-offs with all five optimization seeds")
    save_figure(fig, out, "night3b_spatial_tradeoff_four_metrics", records, sources)


def p22_negative_boundary(source: Path, per_seed: pd.DataFrame, out: Path, records: list):
    attention_path = source / "attention_spot_summary.csv"
    attention = pd.read_csv(attention_path)
    p22 = attention[(attention.dataset == "p22") & (attention.channel == "cross_omics_rna_attention")].sort_values("seed")
    full = per_seed[(per_seed.dataset == "p22") & (per_seed.variant == "FULL_IGE")].sort_values("seed")
    uniform = per_seed[(per_seed.dataset == "p22") & (per_seed.variant == "UNIFORM_CROSS")].sort_values("seed")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(5)
    axes[0].plot(x, p22["mean"], marker="o", label="RNA attention mean")
    axes[0].plot(x, p22["normalized_two_way_entropy"], marker="s", label="binary entropy")
    axes[0].plot(x, p22["extreme_below_005_or_above_095_fraction"], marker="^", label="extreme fraction")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"seed {s}" for s in range(5)])
    axes[0].set_ylim(0, 1.03); axes[0].legend(fontsize=8); axes[0].set_title("Learned cross attention is near-saturated")
    for metric, marker in (("ari", "o"), ("nmi", "s"), ("spatial_neighbor_agreement", "^"), ("spatial_cluster_moran_mean", "D"), ("spatial_cluster_geary_mean", "x")):
        axes[1].plot(x, full[metric].to_numpy() - uniform[metric].to_numpy(), marker=marker, label=metric)
    axes[1].axhline(0, color="black", lw=.8); axes[1].set_xticks(x); axes[1].set_xticklabels([f"seed {s}" for s in range(5)])
    axes[1].set_ylabel("Delta = FULL_IGE - UNIFORM_CROSS"); axes[1].legend(fontsize=7)
    axes[1].set_title("Uniform cross is favored across accuracy/spatial directions")
    fig.suptitle("P22 negative boundary: stable extreme routing does not imply functional benefit")
    save_figure(fig, out, "p22_cross_attention_negative_boundary", records, [attention_path, source / "per_seed_metrics.csv"])


def a1_map(source: Path, out: Path, records: list):
    config = json.loads((REPO / "configs/night3b_ablation_interpretability.json").read_text(encoding="utf-8"))
    from SpaLORA.night1_evaluation import load_evaluation_labels
    from scripts.night3a_evaluate import coordinates_for_ids
    ids_path = source / "runs/a1/FULL_IGE/seed_0/observation_ids.csv"
    ids = pd.Index(pd.read_csv(ids_path)["observation_id"].astype(str))
    positions, labels = load_evaluation_labels("a1", config["datasets"]["a1"], ids)
    coordinates = coordinates_for_ids(config["datasets"]["a1"], ids)
    labels = np.asarray(labels).astype(str)
    domains = sorted(np.unique(labels)); domain_index = {d: i for i, d in enumerate(domains)}
    gt = np.asarray([domain_index[x] for x in labels])
    variants = ("FULL_IGE", "UNIFORM_WITHIN", "UNIFORM_CROSS", "UNIFORM_ALL")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))
    axes[0].scatter(coordinates[positions,0], coordinates[positions,1], c=gt, s=7, cmap="tab20", vmin=0, vmax=19)
    axes[0].set_title("Ground truth")
    for axis, variant in zip(axes[1:], variants):
        clusters = pd.read_csv(source / f"runs/a1/{variant}/seed_0/clusters.csv").cluster.to_numpy()
        observed = clusters[positions]
        unique = sorted(np.unique(observed)); table = np.zeros((len(unique), len(domains)), dtype=int)
        for i, cluster in enumerate(unique):
            for j in range(len(domains)): table[i,j] = int(np.sum((observed == cluster) & (gt == j)))
        rows, cols = linear_sum_assignment(-table)
        mapping = {unique[r]: int(c) for r, c in zip(rows, cols)}
        next_color = len(domains)
        colors = []
        for cluster in clusters:
            if cluster not in mapping:
                mapping[cluster] = next_color; next_color += 1
            colors.append(mapping[cluster])
        axis.scatter(coordinates[:,0], coordinates[:,1], c=colors, s=7, cmap="tab20", vmin=0, vmax=19)
        axis.set_title(f"{variant}\nseed 0")
    for axis in axes: axis.set_xticks([]); axis.set_yticks([]); axis.invert_yaxis()
    fig.suptitle("A1 maps; prediction colors Hungarian-aligned to GT for visualization only")
    save_figure(fig, out, "a1_ground_truth_and_hungarian_aligned_predictions", records, [ids_path, source / "runs/a1/FULL_IGE/seed_0/clusters.csv"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_seed, paired, diff = recompute_and_validate(source, output)
    figure_dir = output / "figures"; records = []
    table_sources = [source / "per_seed_metrics.csv", source / "paired_ablation_deltas.csv"]
    forest_plot(paired, figure_dir, records, [source / "paired_ablation_deltas.csv"])
    heatmap_with_uncertainty(per_seed, figure_dir, records, table_sources, ABLATIONS[:4], "loss_ablation_heatmap_with_seed_uncertainty", "Loss ablation paired deltas")
    heatmap_with_uncertainty(per_seed, figure_dir, records, table_sources, ABLATIONS[4:], "attention_ablation_heatmap_with_seed_uncertainty", "Attention ablation paired deltas")
    p22_heterogeneity(per_seed, figure_dir, records, table_sources)
    attention_distribution(source, figure_dir, records)
    domain_plot(source, figure_dir, records)
    gradient_plot(source, figure_dir, records)
    spatial_tradeoff(per_seed, figure_dir, records, table_sources)
    p22_negative_boundary(source, per_seed, figure_dir, records)
    a1_map(source, figure_dir, records)
    atomic_json(output / "figure_qa_manifest.json", {
        "schema_version": 1, "publication_ready": False,
        "status": "PENDING_RENDERED_PDF_VISUAL_QA", "figures": records,
        "required_checks": ["labels", "overlap", "crop", "legend_mapping", "color_semantics", "pdf_render"],
    })
    captions = [
        f"Figure {i+1}. {row['figure']}. Generated only from locked Night-3B evidence; five seeds are optimization repeats."
        for i, row in enumerate(records)
    ]
    (output / "figure_caption_drafts.md").write_text("# Night-4A figure caption drafts\n\n" + "\n\n".join(captions) + "\n", encoding="utf-8")
    atomic_json(output / "evidence_repair_completion.json", {
        "passed": True, "night3b_original_results_changed": False,
        "night3b_decision_changed": False, "method_recommendation": "MIXED_EVIDENCE",
        "corrected_win_count_rows": 7 * 6 * 4, "figure_pairs": len(records),
        "formal_benchmark_run": False, "exact_diff_report": diff,
    })
    print(f"EVIDENCE_REPAIR_PASS corrected_rows={7*6*4} figure_pairs={len(records)} decision=MIXED_EVIDENCE")


if __name__ == "__main__":
    main()
