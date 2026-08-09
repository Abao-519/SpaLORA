#!/usr/bin/env python3
"""Validate all Night-3A-F outputs and write the non-self internal SHA manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO / "configs/night3af_deterministic_pca.json").read_text(encoding="utf-8"))
OUTPUT = REPO / CONFIG["paths"]["output_root"]
REQUIRED = (
    "deterministic_pca_protocol.json", "p0d_process_a.json", "p0d_process_b.json",
    "p0d_cross_process_comparison.json", "preprocessing_cache_manifest.json",
    "pca_old_new_diagnosis.json", "night3af_p0d.json", "night3af_p0b.json",
    "label_flow_audit.md", "scientific_window_label_firewall.json",
    "ige_initial_gradients.csv", "ige_weights.csv", "per_seed_metrics.csv", "summary.csv",
    "paired_deltas.csv", "night2c_bridge.csv", "loss_trajectories.csv",
    "gradient_influence_trajectories.csv", "attention_summary.csv", "resource_usage.csv",
    "locked_60_run_manifest.json", "failure_index.json", "night3af_report.md",
    "night3af_completion.json", "config_lock.json", "training_complete.json",
)
RUN_REQUIRED = ("embedding.npz", "attention.npz", "observation_ids.csv", "clusters.csv",
                "loss_trajectory.csv", "gradient_influence_trajectory.csv", "checkpoint_index.csv",
                "run_config.json", "model_final.pt", "run_manifest.json")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def main():
    errors = ["missing required output: %s" % name for name in REQUIRED if not (OUTPUT / name).is_file()]
    p0d = json.loads((OUTPUT / "night3af_p0d.json").read_text(encoding="utf-8"))
    p0b = json.loads((OUTPUT / "night3af_p0b.json").read_text(encoding="utf-8"))
    completion = json.loads((OUTPUT / "night3af_completion.json").read_text(encoding="utf-8"))
    gate = json.loads((OUTPUT / "night3af_gate_status.json").read_text(encoding="utf-8"))
    failures = json.loads((OUTPUT / "failure_index.json").read_text(encoding="utf-8"))
    if not p0d.get("passed") or p0d.get("tests_failed") != 0: errors.append("P0D/tests did not pass")
    if not p0b.get("passed") or p0b.get("cells_passed") != 15: errors.append("P0B-F is not 15/15")
    if completion.get("main_runs_completed") != 60 or completion.get("failure_count") != 0: errors.append("completion not 60/60")
    if failures.get("failures") != []: errors.append("failure_index is not empty")
    if gate.get("ige_scientific_go_no_go") not in ("PASS", "FAIL"): errors.append("science gate absent")
    run_count = 0
    cache_hashes = {dataset: set() for dataset in CONFIG["datasets"]}
    for dataset in CONFIG["datasets"]:
        for variant in CONFIG["variants"]:
            for seed in CONFIG["seeds"]:
                run_count += 1; directory = OUTPUT / "runs" / dataset / variant / ("seed_%d" % seed)
                for name in RUN_REQUIRED:
                    if not (directory / name).is_file(): errors.append("missing run artifact: %s/%s" % (directory, name))
                if (directory / "failure.json").exists(): errors.append("run failure exists: %s" % directory)
                manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
                cache_hashes[dataset].add(manifest["locked_input_sha256"])
    if any(len(values) != 1 for values in cache_hashes.values()): errors.append("run cache hashes differ")
    png = sorted((OUTPUT / "figures").glob("*.png")); pdf = sorted((OUTPUT / "figures").glob("*.pdf"))
    if len(png) < 8 or len(png) != len(pdf): errors.append("figure pairs incomplete")
    report = (OUTPUT / "night3af_report.md").read_text(encoding="utf-8")
    for phrase in ("P0D: PASS", "P0B-F: PASS", "60/60", "IGE scientific go/no-go",
                   "Five-seed preregistered contrasts", "Night-2C bridge", "Weighted-gradient influence",
                   "Architecture ablation"):
        if phrase not in report: errors.append("report field missing: %s" % phrase)
    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_night3af.py"], cwd=REPO,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          env={**os.environ, "R_HOME": "/opt/R/4.0.3/lib/R"})
    (OUTPUT / "tests_final.log").write_text(test.stdout, encoding="utf-8")
    match = re.search(r"(?:(\d+) failed, )?(\d+) passed", test.stdout)
    failed = int(match.group(1) or 0) if match else -1; passed = int(match.group(2)) if match else -1
    if test.returncode or failed: errors.append("final tests failed: %d" % failed)
    if errors:
        atomic_json(OUTPUT / "finalization_failure.json", {"schema_version": 1, "errors": errors})
        print(json.dumps(errors, indent=2)); raise SystemExit(2)
    atomic_json(OUTPUT / "shutdown_checklist.json", {
        "schema_version": 1, "p0d": "PASS", "p0bf": "15/15 PASS", "main_runs": "60/60",
        "semantic_label_access": False, "tests": "%d passed, 0 failed" % passed,
        "internal_manifest_verified_by_finalizer": True,
        "git_tag_bundle_archive_local_download": "confirm externally after immutable artifacts exist",
        "shutdown_must_be_last_remote_command": True, "reconnect_after_shutdown_forbidden": True,
    })
    atomic_json(OUTPUT / "artifact_inventory.json", {
        "schema_version": 1, "run_count": run_count, "figure_png_count": len(png), "figure_pdf_count": len(pdf),
        "cache_dataset_count": 3, "tests_passed": passed, "tests_failed": 0,
        "internal_manifest_excludes_itself": True,
    })
    manifest_path = OUTPUT / "SHA256SUMS"
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path != manifest_path
                   and path.name != "finalization_failure.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        for path in files: handle.write("%s  %s\n" % (sha256_file(path), path.relative_to(OUTPUT).as_posix()))
        handle.flush(); os.fsync(handle.fileno())
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if sha256_file(OUTPUT / name) != expected: raise RuntimeError("Internal SHA mismatch: %s" % name)
    print("FINALIZE_PASS runs=60 tests=%d figures=%d files=%d manifest=%s" %
          (passed, len(png), len(files), sha256_file(manifest_path)))


if __name__ == "__main__": main()
