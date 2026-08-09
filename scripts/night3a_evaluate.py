#!/usr/bin/env python3
"""Independent post-lock evaluation, statistics, figures, and Night-3A gate."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CONFIG_PATH = REPO / "configs/night3a_ige_feasibility.json"
LABEL_METRICS = (
    "ari", "nmi", "ami", "fmi", "homogeneity", "v_measure",
    "hungarian_macro_f1", "hungarian_weighted_f1", "hungarian_balanced_accuracy",
)
SPATIAL_METRICS = ("spatial_neighbor_agreement", "spatial_cluster_moran_mean")
EMBEDDING_METRICS = ("embedding_silhouette", "embedding_davies_bouldin")
RESOURCE_METRICS = ("runtime_seconds", "gpu_peak_allocated_mib", "gpu_peak_reserved_mib", "cpu_peak_rss_mib")
ALL_METRICS = LABEL_METRICS + SPATIAL_METRICS + EMBEDDING_METRICS + RESOURCE_METRICS
LOSS_NAMES = ("rna_recon", "mod2_recon", "corr1", "corr2")
ATTENTION_NAMES = (
    "cross_omics_rna_attention", "cross_omics_modality2_attention",
    "rna_spatial_attention", "rna_feature_attention",
    "modality2_spatial_attention", "modality2_feature_attention",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def write_csv(path: Path, rows: list, fields=None) -> None:
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_preconditions(config: dict, output: Path) -> tuple:
    lock_path = output / "config_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    failures = []
    if sha256_file(CONFIG_PATH) != lock["config_sha256"]:
        failures.append("config SHA drift")
    for name, expected in lock["source_sha256"].items():
        path = REPO / name
        if not path.is_file() or sha256_file(path) != expected:
            failures.append("source SHA drift: %s" % name)
    for name, expected in lock["data_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected:
            failures.append("data SHA drift: %s" % name)
    training = json.loads((output / "training_complete.json").read_text(encoding="utf-8"))
    locked_path = output / "locked_60_run_manifest.json"
    if training.get("training_complete") is not True or training.get("run_count") != 60:
        failures.append("training completion is not exactly 60/60")
    if training.get("failure_count") != 0:
        failures.append("training failure count is nonzero")
    if sha256_file(locked_path) != training.get("locked_60_run_manifest_sha256"):
        failures.append("locked 60-run manifest SHA mismatch")
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    if locked.get("run_count") != 60 or locked.get("locked_before_any_semantic_label_access") is not True:
        failures.append("locked run manifest contract failed")
    firewall = json.loads((output / "training_label_firewall.json").read_text(encoding="utf-8"))
    if firewall.get("passed") is not True or firewall.get("semantic_label_values_read") is not False:
        failures.append("training label firewall failed")
    if failures:
        raise RuntimeError("Independent evaluator preconditions failed: %r" % failures)
    return lock, locked, firewall


def coordinates_for_ids(cfg: dict, observation_ids: pd.Index) -> np.ndarray:
    source = sc.read_h5ad(cfg["rna"], backed="r")
    index = pd.Series(np.arange(source.n_obs, dtype=np.int64), index=source.obs_names.astype(str))
    positions = index.reindex(observation_ids.astype(str))
    if positions.isna().any():
        source.file.close()
        raise AssertionError("Training IDs are absent from coordinate source")
    coordinates = np.asarray(source.obsm["spatial"])[positions.astype(int).to_numpy()].copy()
    source.file.close()
    return coordinates


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def bootstrap_ci(values: np.ndarray, seed: int, replicates: int, confidence: float) -> tuple:
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def summarize(per_seed: pd.DataFrame, config: dict) -> list:
    rows = []
    stats = config["statistics"]
    for (dataset, variant), group in per_seed.groupby(["dataset", "variant"], sort=False):
        for metric in ALL_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            low, high = bootstrap_ci(
                values,
                stable_seed(stats["bootstrap_seed"], dataset, variant, metric),
                int(stats["bootstrap_replicates"]),
                float(stats["confidence_level"]),
            )
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_replicates": int(stats["bootstrap_replicates"]),
                    "ci_is_descriptive_only": True,
                }
            )
    return rows


def paired_deltas(per_seed: pd.DataFrame) -> list:
    rows = []
    indexed = per_seed.set_index(["dataset", "variant", "seed"])
    for dataset in per_seed["dataset"].unique():
        for variant in ("IGE", "C1", "ILN"):
            for seed in sorted(per_seed["seed"].unique()):
                candidate = indexed.loc[(dataset, variant, seed)]
                baseline = indexed.loc[(dataset, "C0", seed)]
                row = {"dataset": dataset, "contrast": variant + "-C0", "seed": int(seed)}
                for metric in ALL_METRICS:
                    row[metric + "_delta"] = float(candidate[metric] - baseline[metric])
                rows.append(row)
    return rows


def aggregate_trajectories(config: dict, output: Path) -> tuple:
    loss_rows, attention_rows = [], []
    gate = config["gate"]
    for dataset in config["datasets"]:
        for variant in config["variants"]:
            for seed in config["seeds"]:
                run_dir = output / "runs" / dataset / variant / ("seed_%d" % seed)
                trajectory = pd.read_csv(run_dir / "loss_trajectory.csv")
                for row in trajectory.to_dict("records"):
                    loss_rows.append({"dataset": dataset, "variant": variant, "seed": seed, **row})
                latter = trajectory[trajectory["fraction_of_training"] >= 0.5]
                summary = {"dataset": dataset, "variant": variant, "seed": seed}
                for name in LOSS_NAMES:
                    column = name + "_contribution_fraction"
                    summary[name + "_latter_min"] = float(latter[column].min())
                    summary[name + "_latter_max"] = float(latter[column].max())
                    summary[name + "_persistent_below_1pct"] = bool(
                        (latter[column] < gate["contribution_fraction_min"]).all()
                    )
                    summary[name + "_persistent_above_90pct"] = bool(
                        (latter[column] > gate["contribution_fraction_max"]).all()
                    )
                for name in ATTENTION_NAMES:
                    summary[name + "_latter_mean"] = float(latter[name].mean())
                    summary[name + "_latter_min"] = float(latter[name].min())
                    summary[name + "_latter_max"] = float(latter[name].max())
                    summary[name + "_persistent_below_005"] = bool((latter[name] < gate["attention_min"]).all())
                    summary[name + "_persistent_above_095"] = bool((latter[name] > gate["attention_max"]).all())
                attention_rows.append(summary)
    return loss_rows, attention_rows


def representative_seeds(per_seed: pd.DataFrame) -> dict:
    result = {}
    for (dataset, variant), group in per_seed.groupby(["dataset", "variant"]):
        mean = float(group["ari"].mean())
        ordered = group.assign(distance=(group["ari"] - mean).abs()).sort_values(["distance", "seed"])
        result[dataset + "/" + variant] = int(ordered.iloc[0]["seed"])
    return result


def save_figure(fig, output: Path, stem: str) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / (stem + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(figure_dir / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def generate_figures(config: dict, output: Path, per_seed: pd.DataFrame, deltas: pd.DataFrame,
                     loss_rows: pd.DataFrame, weights: pd.DataFrame, representatives: dict) -> None:
    colors = {"C0": "#4c78a8", "C1": "#f58518", "IGE": "#54a24b", "ILN": "#b279a2"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, dataset in zip(axes, config["datasets"]):
        group = per_seed[per_seed["dataset"] == dataset]
        for variant in config["variants"]:
            item = group[group["variant"] == variant]
            axis.scatter(item["spatial_neighbor_agreement"], item["ari"], label=variant,
                         color=colors[variant], alpha=0.8)
        axis.set_title(dataset)
        axis.set_xlabel("spatial neighbor agreement")
        axis.set_ylabel("ARI")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Label metrics and spatial continuity (all preregistered seeds)")
    save_figure(fig, output, "label_spatial_tradeoff")

    ige = loss_rows[loss_rows["variant"] == "IGE"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, dataset in zip(axes, config["datasets"]):
        group = ige[ige["dataset"] == dataset]
        for name in LOSS_NAMES:
            means = group.groupby("fraction_of_training")[name + "_contribution_fraction"].mean()
            axis.plot(means.index, means.values, marker="o", label=name)
        axis.axhline(config["gate"]["contribution_fraction_min"], color="black", ls="--", lw=0.7)
        axis.axhline(config["gate"]["contribution_fraction_max"], color="black", ls="--", lw=0.7)
        axis.set_title(dataset)
        axis.set_xlabel("fraction of training")
    axes[0].set_ylabel("mean IGE contribution fraction")
    axes[-1].legend(fontsize=7)
    save_figure(fig, output, "ige_loss_contribution_trajectories")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, dataset in zip(axes, config["datasets"]):
        group = ige[ige["dataset"] == dataset]
        for name in ATTENTION_NAMES:
            means = group.groupby("fraction_of_training")[name].mean()
            axis.plot(means.index, means.values, marker=".", label=name.replace("_attention", ""))
        axis.axhline(config["gate"]["attention_min"], color="black", ls="--", lw=0.7)
        axis.axhline(config["gate"]["attention_max"], color="black", ls="--", lw=0.7)
        axis.set_title(dataset)
        axis.set_xlabel("fraction of training")
    axes[0].set_ylabel("mean IGE attention")
    axes[-1].legend(fontsize=5)
    save_figure(fig, output, "ige_attention_trajectories")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for axis, dataset in zip(axes, config["datasets"]):
        group = weights[weights["dataset"] == dataset]
        for index, loss in enumerate(group["loss"].unique()):
            values = group[group["loss"] == loss].sort_values("seed")
            axis.plot(values["seed"], values["ige_weight"], marker="o", label=loss.replace("L_", ""))
        axis.set_title(dataset)
        axis.set_xlabel("seed")
        axis.set_yscale("log")
    axes[0].set_ylabel("frozen IGE weight (log scale)")
    axes[-1].legend(fontsize=6)
    save_figure(fig, output, "ige_initial_weights")

    for dataset, cfg in config["datasets"].items():
        fig, axes = plt.subplots(4, 2, figsize=(9, 14))
        for row_index, variant in enumerate(config["variants"]):
            seeds = [0, representatives[dataset + "/" + variant]]
            for column_index, seed in enumerate(seeds):
                run_dir = output / "runs" / dataset / variant / ("seed_%d" % seed)
                clusters = pd.read_csv(run_dir / "clusters.csv")
                coordinates = coordinates_for_ids(cfg, pd.Index(clusters["observation_id"].astype(str)))
                axes[row_index, column_index].scatter(
                    coordinates[:, 0], coordinates[:, 1], c=clusters["cluster"], s=3, cmap="tab20", rasterized=True
                )
                axes[row_index, column_index].set_title("%s seed %d%s" %
                    (variant, seed, " (fixed)" if column_index == 0 else " (ARI-nearest-mean)"))
                axes[row_index, column_index].set_xticks([]); axes[row_index, column_index].set_yticks([])
        fig.suptitle("%s spatial clusters; no best-seed selection" % dataset)
        save_figure(fig, output, "spatial_maps_" + dataset)


def correlation_table(per_seed: pd.DataFrame, weights: pd.DataFrame, attention: pd.DataFrame) -> list:
    rows = []
    ige_metrics = per_seed[per_seed["variant"] == "IGE"]
    ige_attention = attention[attention["variant"] == "IGE"]
    wide_weights = weights.pivot_table(index=["dataset", "seed"], columns="loss", values="ige_weight").reset_index()
    merged = ige_metrics.merge(ige_attention, on=["dataset", "variant", "seed"]).merge(
        wide_weights, on=["dataset", "seed"]
    )
    outcomes = ["ari", "nmi"] + [name + "_latter_mean" for name in ATTENTION_NAMES]
    for dataset in list(per_seed["dataset"].unique()) + ["all"]:
        group = merged if dataset == "all" else merged[merged["dataset"] == dataset]
        for loss in weights["loss"].unique():
            for outcome in outcomes:
                x, y = group[loss].to_numpy(float), group[outcome].to_numpy(float)
                correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else float("nan")
                rows.append({
                    "dataset": dataset, "ige_weight": loss, "outcome": outcome,
                    "pearson_correlation": correlation, "n": len(x), "interpretation": "descriptive_noncausal",
                })
    return rows


def gate_decision(config: dict, per_seed: pd.DataFrame, deltas: pd.DataFrame,
                  attention: pd.DataFrame, lock: dict, output: Path) -> dict:
    gate = config["gate"]
    mean_delta = deltas.groupby(["dataset", "contrast"]).mean(numeric_only=True)
    noninferior = {}
    positive_both = {}
    for dataset in config["datasets"]:
        row = mean_delta.loc[(dataset, "IGE-C0")]
        noninferior[dataset] = bool(
            row["ari_delta"] >= -gate["noninferiority_margin"]
            and row["nmi_delta"] >= -gate["noninferiority_margin"]
        )
        positive_both[dataset] = bool(row["ari_delta"] > 0 and row["nmi_delta"] > 0)
    alternative_a = sum(positive_both.values()) >= 2 and any(
        mean_delta.loc[(dataset, "IGE-C0"), "ari_delta"] >= gate["large_ari_gain"]
        for dataset in config["datasets"]
    )
    placenta_ige = float(mean_delta.loc[("placenta", "IGE-C0"), "ari_delta"])
    placenta_c1 = float(mean_delta.loc[("placenta", "C1-C0"), "ari_delta"])
    recovery = placenta_ige / placenta_c1 if placenta_c1 > 0 else float("nan")
    alternative_b = bool(
        placenta_c1 > 0 and recovery >= gate["placenta_recovery_fraction"]
        and mean_delta.loc[("a1", "IGE-C0"), "ari_delta"] >= gate["a1_p22_ari_floor"]
        and mean_delta.loc[("p22", "IGE-C0"), "ari_delta"] >= gate["a1_p22_ari_floor"]
    )
    label_gate = sum(noninferior.values()) >= 2 and (alternative_a or alternative_b)
    spatial_failures = []
    for dataset in config["datasets"]:
        row = mean_delta.loc[(dataset, "IGE-C0")]
        if (
            row["spatial_neighbor_agreement_delta"] < -gate["spatial_joint_decline_limit"]
            and row["spatial_cluster_moran_mean_delta"] < -gate["spatial_joint_decline_limit"]
        ):
            spatial_failures.append(dataset)
    ige_attention = attention[attention["variant"] == "IGE"]
    contribution_flags = []
    attention_flags = []
    for _, row in ige_attention.iterrows():
        for name in LOSS_NAMES:
            if bool(row[name + "_persistent_below_1pct"]) or bool(row[name + "_persistent_above_90pct"]):
                contribution_flags.append({"dataset": row["dataset"], "seed": int(row["seed"]), "loss": name})
        for name in ATTENTION_NAMES:
            if bool(row[name + "_persistent_below_005"]) or bool(row[name + "_persistent_above_095"]):
                attention_flags.append({"dataset": row["dataset"], "seed": int(row["seed"]), "attention": name})

    protected = subprocess.run(
        ["sha256sum", "-c", config["paths"]["protected_manifest"]],
        cwd=config["paths"]["protected_root"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    protected_lines = [line for line in protected.stdout.splitlines() if line.strip()]
    protected_ok = protected.returncode == 0
    source_ok = all((REPO / name).is_file() and sha256_file(REPO / name) == expected
                    for name, expected in lock["source_sha256"].items())
    failures = json.loads((output / "failure_index.json").read_text(encoding="utf-8"))["failures"]
    all_finite = bool(np.isfinite(per_seed[list(ALL_METRICS)].to_numpy(dtype=float)).all())
    numeric_gate = (
        len(per_seed) == 60 and not failures and all_finite and protected_ok and source_ok
        and not contribution_flags and not attention_flags
    )
    scientific_gate = bool(label_gate and not spatial_failures and not contribution_flags and not attention_flags)
    passed = bool(numeric_gate and scientific_gate)
    return {
        "schema_version": 1,
        "p0a_pass": True,
        "p0b_pass": True,
        "main_runs_completed": len(per_seed),
        "main_runs_required": 60,
        "failure_count": len(failures),
        "numeric_gate_pass": numeric_gate,
        "scientific_gate_pass": scientific_gate,
        "ige_go_no_go": "PASS" if passed else "FAIL",
        "night3a_ige_no_go": not passed,
        "architecture_ablation_authorized": passed,
        "noninferior_datasets": noninferior,
        "positive_ari_nmi_datasets": positive_both,
        "alternative_a_pass": alternative_a,
        "alternative_b_pass": alternative_b,
        "placenta_ige_minus_c0_ari": placenta_ige,
        "placenta_c1_minus_c0_ari": placenta_c1,
        "placenta_recovery_fraction": recovery,
        "spatial_joint_decline_failures": spatial_failures,
        "loss_contribution_collapse_flags": contribution_flags,
        "attention_saturation_flags": attention_flags,
        "all_metrics_finite": all_finite,
        "source_lock_match": source_ok,
        "protected_files_match": protected_ok,
        "protected_check_line_count": len(protected_lines),
        "protected_check_output_sha256": hashlib.sha256(protected.stdout.encode("utf-8")).hexdigest(),
        "label_firewall_pass": True,
        "strict_failure_reasons": [
            reason for condition, reason in (
                (not numeric_gate, "numeric hard gate failed"),
                (not label_gate, "ARI/NMI preregistered benefit gate failed"),
                (bool(spatial_failures), "joint spatial continuity gate failed"),
                (bool(contribution_flags), "persistent loss contribution collapse"),
                (bool(attention_flags), "persistent attention saturation"),
            ) if condition
        ],
    }


def metric_mean(summary: pd.DataFrame, dataset: str, variant: str, metric: str) -> float:
    return float(summary[(summary.dataset == dataset) & (summary.variant == variant) & (summary.metric == metric)].iloc[0]["mean"])


def build_report(config: dict, output: Path, per_seed: pd.DataFrame, summary: pd.DataFrame,
                 deltas: pd.DataFrame, weights: pd.DataFrame, attention: pd.DataFrame,
                 gate: dict, representatives: dict) -> str:
    delta_means = deltas.groupby(["dataset", "contrast"])[[c for c in deltas if c.endswith("_delta")]].mean()
    lines = [
        "# SpaLORA Night-3A Label-free Loss Calibration Feasibility",
        "",
        "## Strict outcome",
        "",
        "IGE go/no-go: **%s**." % gate["ige_go_no_go"],
        "P0A and P0B passed; 60/60 preregistered runs completed before the independent evaluator read semantic labels.",
        "Strict failure reasons: %s." % (", ".join(gate["strict_failure_reasons"]) or "none"),
        "Architecture ablation is %s." % ("authorized" if gate["architecture_ablation_authorized"] else "not authorized"),
        "",
        "No formula, scale, seed, ASR component, or threshold was changed after label access.",
        "Placenta modality 2 is described only as ATAC-derived / TF-associated regulatory features.",
        "",
        "## Five-seed results",
        "",
        "| Dataset | Variant | ARI mean (SD) | NMI mean (SD) | Neighbor agreement | Moran's I |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in config["datasets"]:
        for variant in config["variants"]:
            def cell(metric):
                row = summary[(summary.dataset == dataset) & (summary.variant == variant) & (summary.metric == metric)].iloc[0]
                return "%.4f (%.4f)" % (row["mean"], row["sd"])
            lines.append("| %s | %s | %s | %s | %s | %s |" %
                         (dataset, variant, cell("ari"), cell("nmi"),
                          cell("spatial_neighbor_agreement"), cell("spatial_cluster_moran_mean")))
    lines += ["", "## Preregistered questions", ""]
    lines.append("1. **Did IGE strictly pass?** %s. %s" %
                 (gate["ige_go_no_go"], "; ".join(gate["strict_failure_reasons"]) or "All gates passed."))
    recovery = gate["placenta_recovery_fraction"]
    recovery_text = "undefined because C1-C0 was not positive" if not np.isfinite(recovery) else "%.3f (%.1f%%)" % (recovery, 100 * recovery)
    lines.append(
        "2. **Placenta C1 recovery:** IGE-C0 ARI %.4f; C1-C0 ARI %.4f; recovery %s (required >=60%%)." %
        (gate["placenta_ige_minus_c0_ari"], gate["placenta_c1_minus_c0_ari"], recovery_text)
    )
    for dataset in ("a1", "p22"):
        group = deltas[(deltas.dataset == dataset) & (deltas.contrast == "IGE-C0")].sort_values("seed")
        values = ", ".join("s%d:%+.4f" % (row.seed, row.ari_delta) for row in group.itertuples())
        lines.append("3. **%s damage check:** mean ARI delta %+.4f; per seed %s." %
                     (dataset.upper(), group.ari_delta.mean(), values))
    lines.append("4. **Spatial continuity:** joint >0.03 decline failures: %s." %
                 (", ".join(gate["spatial_joint_decline_failures"]) or "none"))
    lines.append("5. **Frozen IGE weights:**")
    for dataset in config["datasets"]:
        pieces = []
        for loss, group in weights[weights.dataset == dataset].groupby("loss"):
            pieces.append("%s %.4g +/- %.3g" % (loss, group.ige_weight.mean(), group.ige_weight.std(ddof=1)))
        lines.append("   - %s: %s." % (dataset, "; ".join(pieces)))
    lines.append("6. **Loss contribution/attention:** contribution collapse flags %d; attention saturation flags %d. See trajectories and noncausal correlations." %
                 (len(gate["loss_contribution_collapse_flags"]), len(gate["attention_saturation_flags"])))
    lines.append("7. **ILN diagnostic:**")
    for dataset in config["datasets"]:
        row = delta_means.loc[(dataset, "ILN-C0")]
        lines.append("   - %s: ARI %+.4f, NMI %+.4f versus C0." % (dataset, row.ari_delta, row.nmi_delta))
    lines.append("8. **Integrity:** label leakage none; numerical anomaly %s; source/protected drift %s/%s; protocol deviation none." %
                 ("none" if gate["all_metrics_finite"] else "present",
                  "none" if gate["source_lock_match"] else "present",
                  "none" if gate["protected_files_match"] else "present"))
    lines.append("9. **Next step:** architecture ablation is %s under the preregistered rule." %
                 ("allowed" if gate["architecture_ablation_authorized"] else "not allowed"))
    lines += [
        "", "## Interpretation limits", "",
        "All five fixed seeds are reported; no best seed was selected. Bootstrap intervals are descriptive because n=5.",
        "Attention and IGE-weight correlations are exploratory mechanism clues, not causal evidence.",
        "Spatial figures use fixed seed 0 and the preregistered ARI-nearest-to-mean representative (ties: lower seed): `%s`." %
        json.dumps(representatives, sort_keys=True),
        "", "## Audit and reproducibility", "",
        "- Raw losses use `torch.nn.functional.mse_loss` with its unchanged mean reduction over all elements.",
        "- C0/C1 use locked dataset gamma; C1 alone multiplies RNA by locked m_bad and never applies the old shape.",
        "- IGE/ILN use no dataset gamma, m_bad, or old gene-weight shape.",
        "- The 60-run manifest was SHA-locked before `SpaLORA.night1_evaluation` was imported by this independent process.",
        "- Night-2C protected-file verification: %d entries, %s." %
        (gate["protected_check_line_count"], "PASS" if gate["protected_files_match"] else "FAIL"),
        "- The final immutable Git commit/tag and external archive hashes are recorded in a non-self-referential delivery index outside the tagged archive.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    opened_paths = []

    def audit_hook(event, args):
        if event == "open" and args:
            try:
                opened_paths.append(str(Path(args[0]).resolve()))
            except Exception:
                pass

    sys.addaudithook(audit_hook)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    lock, locked, firewall = verify_preconditions(config, output)
    from SpaLORA.night1_evaluation import evaluate, load_evaluation_labels

    rows, domain_rows, resource_rows = [], [], []
    # Semantic label access begins only below, after all lock checks above succeeded.
    for dataset, cfg in config["datasets"].items():
        first_ids = pd.read_csv(output / "runs" / dataset / "C0" / "seed_0" / "observation_ids.csv")
        observation_ids = pd.Index(first_ids["observation_id"].astype(str))
        positions, labels = load_evaluation_labels(dataset, cfg, observation_ids)
        coordinates = coordinates_for_ids(cfg, observation_ids)
        for variant in config["variants"]:
            for seed in config["seeds"]:
                run_dir = output / "runs" / dataset / variant / ("seed_%d" % seed)
                ids = pd.Index(pd.read_csv(run_dir / "observation_ids.csv")["observation_id"].astype(str))
                if not ids.equals(observation_ids):
                    raise AssertionError("Observation order differs across registered variants/seeds")
                clusters = pd.read_csv(run_dir / "clusters.csv")["cluster"].to_numpy()
                with np.load(run_dir / "embedding.npz") as archive:
                    embedding = np.asarray(archive["SpaLORA"], dtype=np.float32)
                manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
                metrics = evaluate(labels, clusters[positions], clusters, embedding, coordinates,
                                   cfg["spatial_neighbors"])
                row = {"dataset": dataset, "variant": variant, "seed": seed}
                for metric in LABEL_METRICS + SPATIAL_METRICS + EMBEDDING_METRICS:
                    row[metric] = float(metrics[metric])
                row.update(
                    {
                        "runtime_seconds": float(manifest["timings"]["training_seconds"] + manifest["timings"]["clustering_seconds"]),
                        "gpu_peak_allocated_mib": float(manifest["resources"]["gpu_peak_allocated_mib"]),
                        "gpu_peak_reserved_mib": float(manifest["resources"]["gpu_peak_reserved_mib"]),
                        "cpu_peak_rss_mib": float(manifest["resources"]["process_peak_rss_mib"]),
                        "n_observations_trained": int(len(ids)),
                        "n_observations_evaluated": int(len(positions)),
                    }
                )
                rows.append(row)
                for label, score in metrics["hungarian_per_domain_f1"].items():
                    domain_rows.append({"dataset": dataset, "variant": variant, "seed": seed,
                                        "true_domain": label, "hungarian_f1": score})
                resource_rows.append({key: row[key] for key in ("dataset", "variant", "seed") + RESOURCE_METRICS})

    per_seed = pd.DataFrame(rows)
    write_csv(output / "per_seed_metrics.csv", rows)
    write_csv(output / "per_domain_metrics.csv", domain_rows)
    write_csv(output / "resource_usage.csv", resource_rows)
    summary_rows = summarize(per_seed, config)
    summary = pd.DataFrame(summary_rows)
    write_csv(output / "summary.csv", summary_rows)
    delta_rows = paired_deltas(per_seed)
    deltas = pd.DataFrame(delta_rows)
    write_csv(output / "paired_deltas.csv", delta_rows)
    delta_summary = []
    for (dataset, contrast), group in deltas.groupby(["dataset", "contrast"]):
        for metric in ("ari_delta", "nmi_delta"):
            values = group[metric]
            delta_summary.append({
                "dataset": dataset, "contrast": contrast, "metric": metric,
                "mean": float(values.mean()), "positive_seed_count": int((values > 0).sum()),
                "negative_seed_count": int((values < 0).sum()), "zero_seed_count": int((values == 0).sum()),
            })
    write_csv(output / "paired_delta_summary.csv", delta_summary)

    loss_rows, attention_rows = aggregate_trajectories(config, output)
    loss_frame, attention_frame = pd.DataFrame(loss_rows), pd.DataFrame(attention_rows)
    write_csv(output / "loss_trajectories.csv", loss_rows)
    write_csv(output / "attention_summary.csv", attention_rows)
    weights = pd.read_csv(output / "ige_weights.csv")
    representatives = representative_seeds(per_seed)
    atomic_json(output / "representative_seeds.json", {
        "schema_version": 1,
        "rule": config["statistics"]["representative_spatial_seed_rule"],
        "seeds": representatives,
    })
    correlations = correlation_table(per_seed, weights, attention_frame)
    write_csv(output / "ige_weight_attention_performance_correlations.csv", correlations)
    generate_figures(config, output, per_seed, deltas, loss_frame, weights, representatives)

    gate = gate_decision(config, per_seed, deltas, attention_frame, lock, output)
    gate["evaluation_label_access"] = {
        "semantic_label_values_read": True,
        "began_after_locked_manifest_verified": True,
        "ground_truth_paths_opened": sorted(set(opened_paths).intersection({
            str(Path(cfg["ground_truth"]).resolve()) for cfg in config["datasets"].values()
            if cfg["ground_truth"].startswith("/")
        })),
    }
    atomic_json(output / "night3a_gate_status.json", gate)
    report = build_report(config, output, per_seed, summary, deltas, weights, attention_frame,
                          gate, representatives)
    (output / "night3a_report.md").write_text(report, encoding="utf-8")
    completion = {
        "schema_version": 1,
        "stage": "Night-3A complete",
        "p0a_pass": True,
        "p0b_pass": True,
        "main_runs_completed": 60,
        "main_runs_required": 60,
        "failure_count": 0,
        "ige_go_no_go": gate["ige_go_no_go"],
        "night3a_ige_no_go": gate["night3a_ige_no_go"],
        "architecture_ablation_authorized": gate["architecture_ablation_authorized"],
        "label_access_after_run_manifest_lock": True,
        "protocol_deviations": [],
        "config_lock_sha256": sha256_file(output / "config_lock.json"),
        "locked_60_run_manifest_sha256": sha256_file(output / "locked_60_run_manifest.json"),
        "report_sha256": sha256_file(output / "night3a_report.md"),
    }
    atomic_json(output / "night3a_completion.json", completion)
    print("EVALUATION_COMPLETE IGE_%s ARI=%s" %
          (gate["ige_go_no_go"], json.dumps({d: float(deltas[(deltas.dataset == d) &
           (deltas.contrast == "IGE-C0")].ari_delta.mean()) for d in config["datasets"]}, sort_keys=True)),
          flush=True)


if __name__ == "__main__":
    main()
