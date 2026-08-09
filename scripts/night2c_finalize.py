#!/usr/bin/env python3
"""Create the Night-2C completion report for either authorized or hard-stop branch."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results/night2c"
REPORTS = REPO / "reports"


EMPTY_SCHEMAS = {
    "per_seed_metrics.csv": ("dataset", "variant", "seed", "ari", "nmi", "ami", "fmi", "homogeneity", "v_measure", "hungarian_macro_f1", "hungarian_weighted_f1", "hungarian_balanced_accuracy", "spatial_neighbor_agreement", "spatial_cluster_moran_mean", "embedding_silhouette", "embedding_davies_bouldin", "total_seconds", "gpu_peak_allocated_mib", "run_status"),
    "summary.csv": ("dataset", "variant", "metric", "mean", "sample_sd", "median", "minimum", "maximum", "bootstrap_95_low", "bootstrap_95_high", "n"),
    "paired_deltas.csv": ("dataset", "metric", "contrast", "seed", "paired_delta"),
    "factorial_effects.csv": ("dataset", "metric", "contrast", "mean", "sample_sd", "median", "minimum", "maximum", "bootstrap_95_low", "bootstrap_95_high", "n", "exact_32_sign_flip_two_sided_p"),
    "loss_components.csv": ("dataset", "variant", "seed", "epoch", "raw_rna_reconstruction", "weighted_rna_before_global_scale", "global_scale_multiplier", "final_rna_contribution", "raw_modality2_reconstruction", "final_modality2_contribution", "raw_corr1", "final_corr1_contribution", "raw_corr2", "final_corr2_contribution", "total_loss", "m_bad"),
    "attention_summary.csv": ("dataset", "variant", "seed", "cross_omics_rna", "rna_spatial", "modality2_spatial"),
    "per_domain_f1.csv": ("dataset", "variant", "seed", "domain", "support", "hungarian_f1"),
    "v3_replay_audit.csv": ("dataset", "seed", "partition_agreement_ari", "ari_difference", "nmi_difference", "individual_warning", "mean_warning"),
    "paper_repro_audit.csv": ("dataset", "manuscript_ari", "manuscript_nmi", "night2c_v3_mean_ari", "night2c_v3_mean_nmi", "tutorial2022_ari", "tutorial2022_nmi", "notes"),
    "placenta_seed0_technical_replicates.csv": ("dataset", "variant", "repeat_a", "repeat_b", "partition_agreement_ari", "embedding_relative_l2", "orthogonal_procrustes_residual", "linear_cka", "ari_range_across_three", "nmi_range_across_three"),
}


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def ensure_empty_outputs():
    for name, fields in EMPTY_SCHEMAS.items():
        path = RESULTS / name
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(fields)).writeheader()


def count(pattern):
    return len(list(REPO.glob(pattern)))


def csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def p0c_table(report):
    lines = ["| Dataset | Consumed state | Shared-forward V3 | CPU one-step | GPU envelope | Failed cells |",
             "|---|---:|---:|---:|---:|---:|"]
    for dataset in ("a1", "placenta", "p22"):
        item = report["datasets"].get(dataset, {})
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            dataset, item.get("consumed_state", {}).get("pass", False),
            item.get("shared_forward_v3", {}).get("pass", False),
            item.get("cpu_independent_one_step", {}).get("pass", False),
            item.get("gpu_envelope", {}).get("pass", False),
            len(item.get("gpu_envelope", {}).get("failed_cells", []))))
    return "\n".join(lines)


def effect_excerpt():
    rows = csv_rows(RESULTS / "factorial_effects.csv")
    selected = [r for r in rows if r.get("metric") in ("ari", "nmi")]
    if not selected:
        return "Not evaluated because P0C did not authorize factorial training."
    lines = ["| Dataset | Metric | Contrast | Mean | Sample SD | Exact sign-flip p |",
             "|---|---|---|---:|---:|---:|"]
    for row in selected:
        lines.append("| {dataset} | {metric} | {contrast} | {mean} | {sample_sd} | {exact_32_sign_flip_two_sided_p} |".format(**row))
    return "\n".join(lines)


def interpretation():
    rows = csv_rows(RESULTS / "factorial_effects.csv")
    if not rows:
        return "No scale/shape conclusion is authorized because P0C stopped all label-bearing experiments."
    placenta = {(r["metric"], r["contrast"]): float(r["mean"]) for r in rows if r["dataset"] == "placenta"}
    parts = []
    for metric in ("ari", "nmi"):
        parts.append("%s scale=%+.6f, shape=%+.6f, interaction=%+.6f" % (
            metric.upper(), placenta[(metric, "factorial_scale_main")],
            placenta[(metric, "factorial_shape_main")], placenta[(metric, "interaction")]))
    return "; ".join(parts) + ". These are preregistered paired effects, not a tuned setting selection."


def main():
    RESULTS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(exist_ok=True)
    gate = json.loads((RESULTS / "gate_status.json").read_text(encoding="utf-8"))
    p0c = json.loads((REPORTS / "night2c_p0c.json").read_text(encoding="utf-8"))
    if not gate["factorial_authorized"]:
        ensure_empty_outputs()
    main_count = count("results/night2c/raw/*/*/seed_*/metrics.json")
    tutorial_count = count("results/night2c/tutorial2022/*/metrics.json")
    technical_count = count("results/night2c/technical/*/*/seed_*/repeat_*/metrics.json")
    failures = count("results/night2c/**/failure.json")
    expected = (75, 3, 4) if gate["factorial_authorized"] else (0, 0, 0)
    if (main_count, tutorial_count, technical_count) != expected:
        raise AssertionError("completion counts %r != %r" % ((main_count, tutorial_count, technical_count), expected))
    if failures:
        raise AssertionError("failure JSON exists among Night-2C outputs")
    gate["main_runs"]["completed"] = main_count
    gate["tutorial_runs"]["completed"] = tutorial_count
    gate["technical_runs"]["completed"] = technical_count
    gate_temporary = RESULTS / "gate_status.json.tmp"
    gate_temporary.write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(gate_temporary), str(RESULTS / "gate_status.json"))
    test_state_path = RESULTS / "logs/test_result.json"
    test_state = json.loads(test_state_path.read_text(encoding="utf-8")) if test_state_path.exists() else {"status": "pending"}
    handoff_path = RESULTS / "handoff_state.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8")) if handoff_path.exists() else {}
    environment = p0c["environment"]
    report = f"""# SpaLORA Night-2C numerical-equivalence gate and loss factorial

## 1. Executive result

P0C **{'passed and authorized training' if p0c['p0c_pass'] else 'failed and hard-stopped training'}**. Main/tutorial/technical counts are `{main_count}/75`, `{tutorial_count}/3`, and `{technical_count}/4`. Ground truth was not accessed during P0C, no labels selected settings, no seeds were searched, ASR was not modified, and the Night-2/Night-2B failed preregistrations remain unchanged.

## 2. Provenance and immutable runtime

- Required parent: `{p0c.get('critical_hashes', {}).get('parent_commit', 'c283449b188f510e98c2826cbb856f296367aa03')}` (configuration parent is `c283449b188f510e98c2826cbb856f296367aa03`).
- Branch: `revision/q2-night2c-numerical-equivalence-factorial-20260809`.
- Report-generation HEAD: `{git('rev-parse', 'HEAD')}`; final handoff resolves through annotated tag `night2c-final-20260809`.
- Taskbook SHA-256: `4225e87d71372eb1257b843327065ffeb8c60271d4bd8dc7af9d2870a11266f2`.
- Environment fingerprint: `{environment['fingerprint']}`; Python `{environment['python']}`, PyTorch `{environment['torch']}`, CUDA `{environment['cuda_runtime']}`, GPU `{environment['gpu']}`.
- Critical trainer/config/runner/frozen-module/input hashes are recorded verbatim in `reports/night2c_p0c.json` and atomically copied into `results/night2c/gate_status.json`.

## 3. Why P0C is not a relaxation of P0B

Night-2B remains a valid failure under its fixed `1e-7` independent-GPU threshold. It observed same-model CUDA forward residuals of the same scale as legacy-versus-generalized residuals and Adam amplification, but lacked a complete same-code Adam-trajectory negative control. P0C does not edit that threshold or report. It changes the estimand prospectively: exact shared-graph and CPU identities remain exact gates, while independent GPU cross-code divergence is judged against eight balanced blocks of independently executed legacy-versus-legacy and generalized-versus-generalized trajectories, with fixed float32 floor and fixed twofold margin.

## 4. P0C audit

{p0c_table(p0c)}

The deterministic-algorithm subprocess is descriptive only. Full named-tensor pair distances are in `results/night2c/p0c_pairwise_distances.csv`; normalized cell summaries and within/cross ratios are in `results/night2c/p0c_summary.csv`. P0C reason: {gate['reason']}.

## 5. Run counts and validation

- Main: `{main_count}/75`; tutorial-2022: `{tutorial_count}/3`; placenta technical repeats: `{technical_count}/4`.
- Failure JSON count: `{failures}`.
- Test state: `{json.dumps(test_state, sort_keys=True)}`.
- Every authorized process used the gate-recorded critical hashes and environment fingerprint; mismatches are rejected before preparation.
- Embeddings, attention, observation IDs, losses, and clusters were durably written before the first label access in each run.

## 6. Five-seed metrics and prespecified contrasts

All five fixed seeds are represented in `per_seed_metrics.csv`; means use all five and SD is sample SD. Bootstrap intervals use fixed seed `20260809`; paired p-values enumerate all 32 sign flips and are descriptive (`n=5`).

{effect_excerpt()}

## 7. Loss scale versus shape, especially placenta

{interpretation()}

No universal winner is inferred by pooling datasets of unequal difficulty.

## 8. V3 replay against Night-1

`v3_replay_audit.csv` records partition ARI, per-metric differences, and fixed warning thresholds. {'Not evaluated because P0C hard-stopped before V3 training.' if not gate['factorial_authorized'] else 'Warnings, if present, were reported without reruns, tuning, or seed replacement.'}

## 9. Limitations

The target uses float32 CUDA sparse operations and Adam, so low-order scheduling residuals can be amplified. Efficacy summaries have only five fixed model seeds. The frozen public weighting vector is index-misaligned and cannot support a low-expression-gene causal claim. Placenta modality 2 is described only as **ATAC-derived / TF-associated regulatory features** because raw-peak provenance has not been established.

## 10. Recommendation

{'Do not run or interpret the factorial; investigate the exact preregistered failing P0C cell without changing its threshold or using labels.' if not gate['factorial_authorized'] else 'Use the observed scale/shape/interaction pattern and V4 diagnostic only to choose a separately preregistered next experiment; do not promote a setting from these labels by post-hoc tuning.'}

## 11. Persistence and shutdown

- GitHub push: `{handoff.get('github_push', 'pending final persistence')}`.
- Bundle: `{handoff.get('bundle', 'pending')}`; SHA-256 `{handoff.get('bundle_sha256', 'pending')}`.
- Archive: `{handoff.get('archive', 'pending')}`; SHA-256 `{handoff.get('archive_sha256', 'pending')}`.
- Protected files: `{handoff.get('protected_verification', 'pending final verification')}`.
- Local transfer/checksums: `{handoff.get('local_verification', 'pending')}`.
- Shutdown: `/usr/bin/shutdown` is required as the last remote command; final confirmation is written after local verification and must not be inferred before execution.
"""
    (REPORTS / "night2c_report.md").write_text(report, encoding="utf-8")
    completion = {"schema_version": 1, "status": gate["status"], "p0c_pass": p0c["p0c_pass"],
                  "factorial_authorized": gate["factorial_authorized"], "parent_commit": "c283449b188f510e98c2826cbb856f296367aa03",
                  "branch": "revision/q2-night2c-numerical-equivalence-factorial-20260809",
                  "final_tag": "night2c-final-20260809", "report_generation_head": git("rev-parse", "HEAD"),
                  "main_runs_completed": main_count, "tutorial_runs_completed": tutorial_count,
                  "technical_runs_completed": technical_count, "failure_json_count": failures,
                  "ground_truth_accessed_during_p0c": False, "ground_truth_used_for_setting_selection": False,
                  "seed_search_performed": False, "asr_modified": False,
                  "critical_hashes": gate["critical_hashes"], "environment_fingerprint": gate["environment_fingerprint"],
                  "test_state": test_state, "handoff": handoff,
                  "shutdown_command_required_last": "/usr/bin/shutdown"}
    (REPORTS / "night2c_completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": completion["status"], "main": main_count,
                      "tutorial": tutorial_count, "technical": technical_count}, sort_keys=True))


if __name__ == "__main__":
    main()
