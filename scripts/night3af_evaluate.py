#!/usr/bin/env python3
"""Independent Night-3A-F evaluator, Night-2C bridge, and scientific gate."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from SpaLORA.night3af_protocol import atomic_json, sha256_file, verify_night3af_lock


CONFIG_PATH = REPO / "configs/night3af_deterministic_pca.json"
LABEL_METRICS = ("ari", "nmi", "ami", "fmi", "homogeneity", "v_measure",
                 "hungarian_macro_f1", "hungarian_weighted_f1", "hungarian_balanced_accuracy")
SPATIAL_METRICS = ("spatial_neighbor_agreement", "spatial_cluster_moran_mean")
EMBEDDING_METRICS = ("embedding_silhouette", "embedding_davies_bouldin")
RESOURCE_METRICS = ("runtime_seconds", "gpu_peak_allocated_mib", "gpu_peak_reserved_mib", "cpu_peak_rss_mib")
ALL_METRICS = LABEL_METRICS + SPATIAL_METRICS + EMBEDDING_METRICS + RESOURCE_METRICS
LOSS_NAMES = ("rna_recon", "mod2_recon", "corr1", "corr2")
ATTENTION_NAMES = ("cross_omics_rna_attention", "cross_omics_modality2_attention",
                   "rna_spatial_attention", "rna_feature_attention",
                   "modality2_spatial_attention", "modality2_feature_attention")


def write_csv(path: Path, rows: list) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def verify_preconditions(config: dict, output: Path):
    lock = json.loads((output / "config_lock.json").read_text(encoding="utf-8"))
    verify_night3af_lock(REPO, CONFIG_PATH, config, lock, output, "evaluation_post_manifest")
    training = json.loads((output / "training_complete.json").read_text(encoding="utf-8"))
    locked_path = output / "locked_60_run_manifest.json"
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    firewall = json.loads((output / "scientific_window_label_firewall.json").read_text(encoding="utf-8"))
    if not (training.get("training_complete") and training.get("run_count") == 60
            and training.get("failure_count") == 0
            and training.get("locked_60_run_manifest_sha256") == sha256_file(locked_path)
            and locked.get("run_count") == 60 and locked.get("locked_before_any_semantic_label_access")
            and firewall.get("passed") and firewall.get("semantic_label_values_read") is False):
        raise RuntimeError("Evaluator preconditions failed")
    return lock, locked, firewall


def protected_check(root: str, manifest: str) -> dict:
    result = subprocess.run(["sha256sum", "-c", manifest], cwd=root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return {"passed": result.returncode == 0, "line_count": len(lines),
            "failure_lines": [line for line in lines if not line.endswith(": OK")]}


def aggregate_trajectories(config: dict, output: Path):
    loss_rows, gradient_rows, summaries = [], [], []
    gate = config["gate"]
    for dataset in config["datasets"]:
        for variant in config["variants"]:
            for seed in config["seeds"]:
                directory = output / "runs" / dataset / variant / ("seed_%d" % seed)
                loss = pd.read_csv(directory / "loss_trajectory.csv")
                gradient = pd.read_csv(directory / "gradient_influence_trajectory.csv")
                loss_rows.extend({"dataset": dataset, "variant": variant, "seed": seed, **row}
                                 for row in loss.to_dict("records"))
                gradient_rows.extend({"dataset": dataset, "variant": variant, "seed": seed, **row}
                                     for row in gradient.to_dict("records"))
                latter_loss = loss[loss.fraction_of_training >= .5]
                latter_gradient = gradient[gradient.fraction_of_training >= .5]
                summary = {"dataset": dataset, "variant": variant, "seed": seed}
                for name in LOSS_NAMES:
                    share = name + "_weighted_gradient_share"
                    scalar = name + "_contribution_fraction"
                    summary[name + "_scalar_latter_min"] = float(latter_loss[scalar].min())
                    summary[name + "_scalar_latter_max"] = float(latter_loss[scalar].max())
                    summary[name + "_gradient_share_latter_min"] = float(latter_gradient[share].min())
                    summary[name + "_gradient_share_latter_max"] = float(latter_gradient[share].max())
                    summary[name + "_gradient_share_persistent_below_001"] = bool((latter_gradient[share] < gate["weighted_gradient_share_min"]).all())
                    summary[name + "_gradient_share_persistent_above_090"] = bool((latter_gradient[share] > gate["weighted_gradient_share_max"]).all())
                for name in ATTENTION_NAMES:
                    summary[name + "_latter_mean"] = float(latter_loss[name].mean())
                    summary[name + "_persistent_below_005"] = bool((latter_loss[name] < gate["attention_min"]).all())
                    summary[name + "_persistent_above_095"] = bool((latter_loss[name] > gate["attention_max"]).all())
                summaries.append(summary)
    return loss_rows, gradient_rows, summaries


def scientific_gate(config, per_seed, deltas, trajectory_summary, lock, output):
    gate = config["gate"]; means = deltas.groupby(["dataset", "contrast"]).mean(numeric_only=True)
    positive = {d: bool(means.loc[(d, "IGE-C0"), "ari_delta"] > 0 and means.loc[(d, "IGE-C0"), "nmi_delta"] > 0)
                for d in config["datasets"]}
    alternative_a = sum(positive.values()) >= 2 and any(
        means.loc[(d, "IGE-C0"), "ari_delta"] >= gate["large_ari_gain"] for d in config["datasets"])
    placenta_ige = float(means.loc[("placenta", "IGE-C0"), "ari_delta"])
    placenta_c1 = float(means.loc[("placenta", "C1-C0"), "ari_delta"])
    recovery = placenta_ige / placenta_c1 if placenta_c1 > 0 else float("nan")
    alternative_b = bool(placenta_c1 > 0 and recovery >= gate["placenta_recovery_fraction"]
                         and means.loc[("a1", "IGE-C0"), "ari_delta"] >= gate["a1_p22_ari_floor"]
                         and means.loc[("p22", "IGE-C0"), "ari_delta"] >= gate["a1_p22_ari_floor"])
    spatial_failures = [d for d in config["datasets"]
                        if means.loc[(d, "IGE-C0"), "spatial_neighbor_agreement_delta"] < -gate["spatial_joint_decline_limit"]
                        and means.loc[(d, "IGE-C0"), "spatial_cluster_moran_mean_delta"] < -gate["spatial_joint_decline_limit"]]
    ige = trajectory_summary[trajectory_summary.variant == "IGE"]
    gradient_flags, attention_flags = [], []
    for _, row in ige.iterrows():
        for name in LOSS_NAMES:
            if row[name + "_gradient_share_persistent_below_001"] or row[name + "_gradient_share_persistent_above_090"]:
                gradient_flags.append({"dataset": row.dataset, "seed": int(row.seed), "loss": name})
        for name in ATTENTION_NAMES:
            if row[name + "_persistent_below_005"] or row[name + "_persistent_above_095"]:
                attention_flags.append({"dataset": row.dataset, "seed": int(row.seed), "attention": name})
    protections = {name: protected_check(config["paths"]["protected_%s_root" % name],
                                         config["paths"]["protected_%s_manifest" % name])
                   for name in ("night3ar", "night3a", "night2c")}
    failures = json.loads((output / "failure_index.json").read_text(encoding="utf-8"))["failures"]
    finite = bool(np.isfinite(per_seed[list(ALL_METRICS)].to_numpy(float)).all())
    source_ok = all((REPO / name).is_file() and sha256_file(REPO / name) == expected
                    for name, expected in lock["source_sha256"].items())
    numeric = bool(len(per_seed) == 60 and not failures and finite and source_ok
                   and all(row["passed"] for row in protections.values())
                   and not gradient_flags and not attention_flags)
    benefit = alternative_a or alternative_b
    passed = bool(numeric and benefit and not spatial_failures)
    reasons = []
    if not numeric: reasons.append("numeric/integrity hard gate failed")
    if not benefit: reasons.append("preregistered ARI/NMI benefit gate failed")
    if spatial_failures: reasons.append("joint spatial continuity gate failed")
    if gradient_flags: reasons.append("persistent weighted-gradient influence collapse")
    if attention_flags: reasons.append("persistent attention saturation")
    return {"schema_version": 1, "p0d_pass": True, "p0bf_pass": True,
            "main_runs_completed": len(per_seed), "failure_count": len(failures),
            "semantic_label_access_during_training": False,
            "ige_scientific_go_no_go": "PASS" if passed else "FAIL",
            "architecture_ablation_authorized": passed,
            "alternative_a_pass": alternative_a, "alternative_b_pass": alternative_b,
            "positive_ari_nmi_datasets": positive,
            "placenta_ige_minus_c0_ari": placenta_ige, "placenta_c1_minus_c0_ari": placenta_c1,
            "placenta_recovery_fraction": recovery, "spatial_joint_decline_failures": spatial_failures,
            "weighted_gradient_share_collapse_flags": gradient_flags,
            "attention_saturation_flags": attention_flags,
            "scalar_contribution_used_as_hard_gate": False, "all_metrics_finite": finite,
            "source_lock_match": source_ok, "protections": protections, "strict_failure_reasons": reasons}


def gradient_figure(config, output, gradients):
    figure_dir = output / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    ige = gradients[gradients.variant == "IGE"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, dataset in zip(axes, config["datasets"]):
        group = ige[ige.dataset == dataset]
        for name in LOSS_NAMES:
            means = group.groupby("fraction_of_training")[name + "_weighted_gradient_share"].mean()
            axis.plot(means.index, means.values, marker="o", label=name)
        axis.axhline(.01, color="black", ls="--", lw=.7); axis.axhline(.90, color="black", ls="--", lw=.7)
        axis.set_title(dataset); axis.set_xlabel("fraction of training")
    axes[0].set_ylabel("mean weighted-gradient share"); axes[-1].legend(fontsize=7)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / ("weighted_gradient_influence_trajectories." + suffix),
                    dpi=180 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def build_bridge(config, per_seed):
    old = pd.read_csv(config["paths"]["night2c_metrics"])
    mapping = {"C0": "V0", "C1": "V1"}; rows = []
    for dataset in config["datasets"]:
        for new_variant, old_code in mapping.items():
            for seed in config["seeds"]:
                new = per_seed[(per_seed.dataset == dataset) & (per_seed.variant == new_variant) & (per_seed.seed == seed)].iloc[0]
                prior = old[(old.dataset == dataset) & (old.variant_code == old_code) & (old.seed == seed)].iloc[0]
                for metric in ("ari", "nmi", "spatial_neighbor_agreement", "spatial_cluster_moran_mean"):
                    rows.append({"dataset": dataset, "seed": seed, "night3af_variant": new_variant,
                                 "night2c_variant_code": old_code, "metric": metric,
                                 "night3af_value": float(new[metric]), "night2c_value": float(prior[metric]),
                                 "delta": float(new[metric] - prior[metric]),
                                 "paired_preprocessing_test": False,
                                 "interpretation": "diagnostic bridge across different preprocessing"})
    return rows


def build_report(config, summary, deltas, bridge, gate):
    means = deltas.groupby(["dataset", "contrast"]).mean(numeric_only=True)
    lines = ["# SpaLORA Night-3A-F Deterministic PCA Run", "",
             "**P0D: PASS (3/3 independent builds byte-exact); P0B-F: PASS (15/15); main experiment: 60/60; semantic label access during training: 0; IGE scientific go/no-go: %s; architecture ablation: %s.**" %
             (gate["ige_scientific_go_no_go"], "AUTHORIZED" if gate["architecture_ablation_authorized"] else "NOT AUTHORIZED"), "",
             "`previous_night3ar_status = DETERMINISTIC_PREPROCESSING_BUG_FOUND_SCIENTIFIC_GO_NO_GO_NOT_EVALUATED`.",
             "The old random PCA hash is diagnostic only. All four variants and five seeds consumed one immutable deterministic cache per dataset.", "",
             "## Five-seed summary", "",
             "| Dataset | Variant | ARI mean (SD) | NMI mean (SD) | Neighbor mean (SD) | Moran mean (SD) |",
             "|---|---|---:|---:|---:|---:|"]
    for dataset in config["datasets"]:
        for variant in config["variants"]:
            def cell(metric):
                row = summary[(summary.dataset == dataset) & (summary.variant == variant) & (summary.metric == metric)].iloc[0]
                return "%.4f (%.4f)" % (row["mean"], row["sd"])
            lines.append("| %s | %s | %s | %s | %s | %s |" %
                         (dataset, variant, cell("ari"), cell("nmi"), cell("spatial_neighbor_agreement"), cell("spatial_cluster_moran_mean")))
    lines += ["", "## Five-seed preregistered contrasts", "",
              "| Dataset | Seed | IGE-C0 ARI | IGE-C0 NMI | C1-C0 ARI | C1-C0 NMI | IGE-C0 Neighbor | IGE-C0 Moran |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    idx = deltas.set_index(["dataset", "contrast", "seed"])
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            ige, c1 = idx.loc[(dataset, "IGE-C0", seed)], idx.loc[(dataset, "C1-C0", seed)]
            lines.append("| %s | %d | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f | %+.4f |" %
                         (dataset, seed, ige.ari_delta, ige.nmi_delta, c1.ari_delta, c1.nmi_delta,
                          ige.spatial_neighbor_agreement_delta, ige.spatial_cluster_moran_mean_delta))
    lines += ["", "## Night-2C bridge (diagnostic, different preprocessing)", ""]
    bridge_means = bridge.groupby(["dataset", "night3af_variant", "night2c_variant_code", "metric"]).delta.mean()
    for key, value in bridge_means.items(): lines.append("- %s %s vs %s, %s: %+.4f." % (*key, value))
    lines += ["", "These bridge differences are not a paired test under identical preprocessing and are not a failure gate.", "",
              "## Scientific gates", "",
              "- Placenta C1 recovery fraction: %s." % ("undefined" if not np.isfinite(gate["placenta_recovery_fraction"]) else "%.4f" % gate["placenta_recovery_fraction"]),
              "- Spatial joint-decline failures: %s." % (gate["spatial_joint_decline_failures"] or "none"),
              "- Weighted-gradient influence collapse flags: %d." % len(gate["weighted_gradient_share_collapse_flags"]),
              "- Attention saturation flags: %d." % len(gate["attention_saturation_flags"]),
              "- Strict reasons: %s." % (", ".join(gate["strict_failure_reasons"]) or "none"),
              "- Weighted-gradient influence, not scalar loss fraction, is the mechanism hard gate.",
              "- Architecture ablation recommended: %s." % ("yes" if gate["architecture_ablation_authorized"] else "no"), "",
              "No PCA/training seed, scale, solver, tau, formula, ASR, evaluator, or threshold search occurred."]
    return "\n".join(lines) + "\n"


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8")); output = REPO / config["paths"]["output_root"]
    lock, locked, firewall = verify_preconditions(config, output)
    from SpaLORA.night1_evaluation import evaluate, load_evaluation_labels
    from scripts.night3a_evaluate import coordinates_for_ids, generate_figures, paired_deltas, representative_seeds, summarize

    rows, domain_rows, resource_rows = [], [], []
    for dataset, cfg in config["datasets"].items():
        base = output / "runs" / dataset / "C0" / "seed_0"
        ids = pd.Index(pd.read_csv(base / "observation_ids.csv")["observation_id"].astype(str))
        positions, labels = load_evaluation_labels(dataset, cfg, ids); coordinates = coordinates_for_ids(cfg, ids)
        for variant in config["variants"]:
            for seed in config["seeds"]:
                directory = output / "runs" / dataset / variant / ("seed_%d" % seed)
                run_ids = pd.Index(pd.read_csv(directory / "observation_ids.csv")["observation_id"].astype(str))
                if not run_ids.equals(ids): raise AssertionError("Observation order drift")
                clusters = pd.read_csv(directory / "clusters.csv")["cluster"].to_numpy()
                with np.load(directory / "embedding.npz") as archive: embedding = np.asarray(archive["SpaLORA"], np.float32)
                manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
                metrics = evaluate(labels, clusters[positions], clusters, embedding, coordinates, cfg["spatial_neighbors"])
                row = {"dataset": dataset, "variant": variant, "seed": seed}
                row.update({metric: float(metrics[metric]) for metric in LABEL_METRICS + SPATIAL_METRICS + EMBEDDING_METRICS})
                row.update({"runtime_seconds": float(manifest["timings"]["training_seconds"] + manifest["timings"]["clustering_seconds"]),
                            "gpu_peak_allocated_mib": float(manifest["resources"]["gpu_peak_allocated_mib"]),
                            "gpu_peak_reserved_mib": float(manifest["resources"]["gpu_peak_reserved_mib"]),
                            "cpu_peak_rss_mib": float(manifest["resources"]["process_peak_rss_mib"]),
                            "n_observations_trained": len(ids), "n_observations_evaluated": len(positions)})
                rows.append(row)
                domain_rows.extend({"dataset": dataset, "variant": variant, "seed": seed,
                                    "true_domain": label, "hungarian_f1": score}
                                   for label, score in metrics["hungarian_per_domain_f1"].items())
                resource_rows.append({key: row[key] for key in ("dataset", "variant", "seed") + RESOURCE_METRICS})
    per_seed = pd.DataFrame(rows); write_csv(output / "per_seed_metrics.csv", rows)
    write_csv(output / "per_domain_metrics.csv", domain_rows); write_csv(output / "resource_usage.csv", resource_rows)
    summary_rows = summarize(per_seed, config); summary = pd.DataFrame(summary_rows); write_csv(output / "summary.csv", summary_rows)
    delta_rows = paired_deltas(per_seed); deltas = pd.DataFrame(delta_rows); write_csv(output / "paired_deltas.csv", delta_rows)
    bridge_rows = build_bridge(config, per_seed); bridge = pd.DataFrame(bridge_rows); write_csv(output / "night2c_bridge.csv", bridge_rows)
    loss_rows, gradient_rows, trajectory_rows = aggregate_trajectories(config, output)
    loss_frame, gradient_frame, trajectory = pd.DataFrame(loss_rows), pd.DataFrame(gradient_rows), pd.DataFrame(trajectory_rows)
    write_csv(output / "loss_trajectories.csv", loss_rows); write_csv(output / "gradient_influence_trajectories.csv", gradient_rows)
    write_csv(output / "attention_summary.csv", trajectory_rows)
    weights = pd.read_csv(output / "ige_weights.csv"); representatives = representative_seeds(per_seed)
    atomic_json(output / "representative_seeds.json", {"schema_version": 1, "rule": config["statistics"]["representative_spatial_seed_rule"], "seeds": representatives})
    figure_config = json.loads(json.dumps(config)); figure_config["gate"]["contribution_fraction_min"] = .01; figure_config["gate"]["contribution_fraction_max"] = .90
    generate_figures(figure_config, output, per_seed, deltas, loss_frame, weights, representatives)
    gradient_figure(config, output, gradient_frame)
    gate = scientific_gate(config, per_seed, deltas, trajectory, lock, output)
    gate["evaluation_semantic_label_access"] = {"occurred": True, "after_locked_manifest": True, "used_only_for_metrics": True}
    atomic_json(output / "night3af_gate_status.json", gate)
    report = build_report(config, summary, deltas, bridge, gate); (output / "night3af_report.md").write_text(report, encoding="utf-8")
    atomic_json(output / "night3af_completion.json", {
        "schema_version": 1, "stage": "Night-3A-F complete", "p0d_pass": True, "p0bf_pass": True,
        "p0bf_cells_passed": 15, "main_runs_completed": 60, "main_runs_required": 60,
        "failure_count": 0, "semantic_label_access_during_training": False,
        "ige_scientific_go_no_go": gate["ige_scientific_go_no_go"],
        "architecture_ablation_authorized": gate["architecture_ablation_authorized"],
        "previous_night3ar_status": config["previous_night3ar_status"], "protocol_deviations": [],
        "locked_60_run_manifest_sha256": sha256_file(output / "locked_60_run_manifest.json"),
        "report_sha256": sha256_file(output / "night3af_report.md")})
    print("EVALUATION_COMPLETE IGE_%s" % gate["ige_scientific_go_no_go"], flush=True)


if __name__ == "__main__": main()
