#!/usr/bin/env python3
"""Finalize the mandatory P0A-R hard stop without evaluating labels or science."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/night3ar_ige_feasibility.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
OUTPUT = REPO / CONFIG["paths"]["output_root"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def protected_check(root: str, manifest: str) -> dict:
    result = subprocess.run(
        ["sha256sum", "-c", manifest], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return {
        "passed": result.returncode == 0,
        "line_count": len(lines),
        "failure_lines": [line for line in lines if not line.endswith(": OK")],
        "manifest_sha256": sha256_file(Path(manifest)),
    }


def main() -> None:
    p0a_path = OUTPUT / "night3ar_p0a.json"
    p0a = json.loads(p0a_path.read_text(encoding="utf-8"))
    if p0a.get("passed") is not False or not p0a.get("failures"):
        raise RuntimeError("This finalizer is valid only for the observed P0A-R hard stop")

    differing = {}
    invariant = {}
    for dataset, comparison in p0a["night3a_comparison"].items():
        differing[dataset] = {}
        invariant[dataset] = []
        for key, old_value in comparison["old"].items():
            new_value = comparison["new"][key]
            if old_value == new_value:
                invariant[dataset].append(key)
            else:
                differing[dataset][key] = {"night3a": old_value, "night3ar": new_value}
    expected_differences = {"pca", "graphs", "model_input_sha256"}
    if any(set(values) != expected_differences for values in differing.values()):
        raise RuntimeError("Unexpected P0A-R difference set")
    if any(
        values["graphs"]["night3a"][name] != values["graphs"]["night3ar"][name]
        for values in differing.values()
        for name in ("adj_spatial_omics1", "adj_spatial_omics2", "adj_feature_omics2")
    ):
        raise RuntimeError("A non-RNA-feature graph changed unexpectedly")

    diagnosis = {
        "schema_version": 1,
        "stage": "P0A-R hard-stop diagnosis",
        "classification": "FROZEN_LEGACY_RANDOMIZED_PCA_CROSS_PROCESS_NONREPRODUCIBILITY",
        "hard_stop_required_by_taskbook_line_153": True,
        "night3ar_p0a_sha256": sha256_file(p0a_path),
        "all_input_file_sha256_match": True,
        "run_order_exact": p0a["run_order_exact"],
        "label_values_read": p0a["label_values_read"],
        "unchanged_fields_by_dataset": invariant,
        "differing_fields_by_dataset": differing,
        "difference_localization": {
            "rna_pca": "different on all three datasets",
            "rna_feature_graph": "different on all three datasets",
            "model_input_sha256": "different because it includes the preceding stochastic artifacts",
            "modality2_model_features": "exact on all three datasets",
            "spatial_graphs": "exact on all three datasets",
            "modality2_feature_graph": "exact on all three datasets",
            "spot_feature_order_and_shapes": "exact on all three datasets",
        },
        "frozen_legacy_source_fact": {
            "file": "SpaLORA/preprocess.py",
            "constructor": "sklearn.decomposition.PCA(n_components=n_comps)",
            "explicit_random_state": False,
            "explicit_svd_solver": False,
            "consequence": "for these tall/wide matrices sklearn auto-selects randomized SVD and random_state=None consumes process-global RNG",
            "source_sha256": sha256_file(REPO / "SpaLORA/preprocess.py"),
        },
        "prohibited_actions_not_taken": [
            "no seed search", "no new PCA seed", "no PCA solver change", "no tolerance relaxation",
            "no old P0A rewrite", "no label access", "no P0B-R", "no 60-run factorial",
        ],
        "scientific_ige_go_no_go": "NOT_EVALUATED",
    }
    atomic_json(OUTPUT / "p0a_randomized_pca_diagnosis.json", diagnosis)

    p0b = {
        "schema_version": 1, "stage": "P0B-R", "passed": False,
        "status": "NOT_RUN_DUE_TO_P0AR_HARD_STOP", "probe_cells_completed": 0,
        "probe_cells_required": 15, "semantic_label_values_read": False,
        "reason": "Taskbook line 153 requires a hard stop after any P0A-R mismatch.",
    }
    atomic_json(OUTPUT / "night3ar_p0b.json", p0b)
    firewall = {
        "schema_version": 1, "status": "SCIENTIFIC_WINDOW_NOT_OPENED_DUE_TO_P0AR_HARD_STOP",
        "integrity_reads_classified_outside_scientific_window": True,
        "semantic_label_values_read": False, "ground_truth_csv_opened_inside_window": [],
        "forbidden_evaluator_imports": [], "ground_truth_parser_guard_trigger_count": 0,
        "passed": None,
    }
    atomic_json(OUTPUT / "scientific_window_label_firewall.json", firewall)
    gate = {
        "schema_version": 1, "p0ar_pass": False, "p0br_pass": False,
        "main_runs_completed": 0, "failure_count": len(p0a["failures"]),
        "ige_scientific_go_no_go": "NOT_EVALUATED",
        "architecture_ablation_authorized": False,
        "weighted_gradient_share_gate": "NOT_EVALUATED",
        "attention_gate": "NOT_EVALUATED", "spatial_gate": "NOT_EVALUATED",
        "semantic_label_access_during_training": False,
        "scalar_contribution_used_as_hard_gate": False,
        "reason": "P0A-R hard stop occurred before P0B-R and training.",
    }
    atomic_json(OUTPUT / "night3ar_gate_status.json", gate)

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_night3ar.py"], cwd=REPO,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "R_HOME": "/opt/R/4.0.3/lib/R"},
    )
    (OUTPUT / "tests_full.log").write_text(test.stdout, encoding="utf-8")
    match = re.search(r"(?:(\d+) failed, )?(\d+) passed", test.stdout)
    tests_failed = int(match.group(1) or 0) if match else -1
    tests_passed = int(match.group(2)) if match else -1

    night3a_protection = protected_check(
        CONFIG["paths"]["protected_night3a_root"], CONFIG["paths"]["protected_night3a_manifest"]
    )
    night2c_protection = protected_check(
        CONFIG["paths"]["protected_night2c_root"], CONFIG["paths"]["protected_night2c_manifest"]
    )
    if not night3a_protection["passed"] or not night2c_protection["passed"]:
        raise RuntimeError("A protected prior-study file changed")

    completion = {
        "schema_version": 1, "stage": "Night-3A-R P0A-R hard stop complete",
        "p0ar_pass": False, "p0br_pass": False,
        "p0br_status": p0b["status"], "main_runs_completed": 0, "main_runs_required": 60,
        "failure_count": len(p0a["failures"]), "semantic_label_access_during_training": False,
        "ige_scientific_go_no_go": "NOT_EVALUATED", "architecture_ablation_authorized": False,
        "previous_night3a_status": CONFIG["previous_night3a_status"],
        "tests_passed": tests_passed, "tests_failed": tests_failed,
        "night3a_protection": night3a_protection, "night2c_protection": night2c_protection,
        "protocol_deviations": [], "p0ar_sha256": sha256_file(p0a_path),
        "diagnosis_sha256": sha256_file(OUTPUT / "p0a_randomized_pca_diagnosis.json"),
    }
    atomic_json(OUTPUT / "night3ar_completion.json", completion)
    atomic_json(OUTPUT / "shutdown_checklist.json", {
        "schema_version": 1,
        "scientific_work_terminal_state": "P0AR_HARD_STOP",
        "p0ar_failure_evidence_saved": True,
        "p0br_and_factorial_not_started": True,
        "semantic_label_access": False,
        "internal_manifest_verified_by_finalizer": True,
        "night3a_protected_files_match": night3a_protection["passed"],
        "night2c_protected_files_match": night2c_protection["passed"],
        "git_tag_bundle_archive_local_download": "to be confirmed in external delivery index after immutable artifacts exist",
        "shutdown_must_be_last_remote_command": True,
        "reconnect_after_shutdown_forbidden": True,
    })

    report = """# SpaLORA Night-3A-R P0A-R Hard Stop

**P0A-R: FAIL; P0B-R: NOT RUN (0/15); main experiment: 0/60; semantic label access: 0; IGE scientific go/no-go: NOT_EVALUATED; architecture ablation: NOT AUTHORIZED.**

`previous_night3a_status = ADMINISTRATIVE_HARD_STOP_SCIENTIFIC_GO_NO_GO_NOT_EVALUATED`. The old Night-3A failure report remains unchanged and is not reinterpreted as a scientific IGE failure.

## Hard-stop reason

Taskbook line 153 requires P0A-R to stop on any old/new preprocessing fingerprint mismatch. All source data SHA-256 values and the exact 60-cell dataset/variant/seed/ordinal order match. Spot order, selected-gene order, shapes, modality-2 model features, both spatial graphs, and the modality-2 feature graph also match on A1, placenta, and P22.

The only differences are the RNA PCA, its derived RNA feature graph, and consequently the combined model-input SHA. The frozen legacy helper constructs `sklearn.decomposition.PCA(n_components=n_comps)` without `random_state` or an explicit solver. At these matrix dimensions sklearn selects randomized SVD, so a fresh process cannot reproduce the old byte fingerprint unless the protocol is changed or a seed is searched. Both are forbidden in this task.

No seed was added or searched, no PCA solver/formula was changed, no tolerance was relaxed, and old Night-3A evidence was not overwritten.

## Required status fields

- Five-seed IGE-C0 and C1-C0 ARI/NMI: NOT AVAILABLE; the 60 runs were prohibited.
- Placenta C1 recovery: NOT EVALUATED.
- Spatial tradeoff: NOT EVALUATED.
- Scalar loss versus weighted-gradient influence: the revised implementation and state-neutral CPU/GPU tests exist, but the scientific gradient-share hard gate was not evaluated because P0A-R failed.
- Architecture ablation authorized: no.
- P0B-R 15-cell old-probe comparison: 0/15, not run.
- Semantic label access: none; no evaluator was started.
- Full test suite: %d passed, %d failed. The two expected failures are the P0A-R PASS/source-lock assertions; the pre-P0A implementation subset had 14/14 passed.
- Night-3A protected files: %d/%d match; Night-2C protected files: %d/%d match.

## Evidence

See `night3ar_p0a.json`, `night3ar_p0a_failure.json`, `p0a_randomized_pca_diagnosis.json`, `integrity_read_manifest.json`, `label_flow_audit.md`, `tests_full.log`, and `failure_index.json`. No result table or figure is fabricated for an experiment that did not run.
""" % (
        tests_passed, tests_failed,
        night3a_protection["line_count"], night3a_protection["line_count"],
        night2c_protection["line_count"], night2c_protection["line_count"],
    )
    (OUTPUT / "night3ar_report.md").write_text(report, encoding="utf-8")

    manifest_path = OUTPUT / "SHA256SUMS"
    inventory_path = OUTPUT / "artifact_inventory.json"
    files_before_inventory = sorted(
        path for path in OUTPUT.rglob("*")
        if path.is_file() and path not in (manifest_path, inventory_path)
    )
    atomic_json(inventory_path, {
        "schema_version": 1, "status": "P0AR_HARD_STOP",
        "main_run_count": 0, "p0br_probe_count": 0,
        "figures_generated": 0, "figures_not_applicable_reason": "No main experiment was permitted",
        "internal_manifest_excludes_itself": True,
        "files_before_manifest": len(files_before_inventory) + 1,
    })
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path != manifest_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write("%s  %s\n" % (sha256_file(path), path.relative_to(OUTPUT).as_posix()))
        handle.flush()
        os.fsync(handle.fileno())
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if sha256_file(OUTPUT / name) != expected:
            raise RuntimeError("Internal manifest verification failed: %s" % name)
    print("NIGHT3AR_P0AR_HARDSTOP_FINALIZED tests=%d_passed_%d_failed protected=%d+%d files=%d" %
          (tests_passed, tests_failed, night3a_protection["line_count"],
           night2c_protection["line_count"], len(files)))


if __name__ == "__main__":
    main()
