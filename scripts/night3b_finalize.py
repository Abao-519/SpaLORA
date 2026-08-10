#!/usr/bin/env python3
"""Validate and inventory the complete Night-3B scientific handoff."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night3b_protocol import atomic_json, sha256_file, verify_night3b_lock


CONFIG_PATH = REPO / "configs/night3b_ablation_interpretability.json"
REQUIRED = (
    "night3b_report.md", "night3b_completion.json", "night3b_gate_status.json",
    "p0_arch.json", "locked_120_run_manifest.json", "scientific_window_label_firewall.json",
    "per_seed_metrics.csv", "summary.csv", "paired_ablation_deltas.csv",
    "component_support_matrix.csv", "per_domain_metrics.csv", "geary_metrics.csv",
    "ige_coefficients.csv", "gradient_influence_trajectories.csv",
    "attention_spot_summary.csv", "attention_seed_stability.csv",
    "attention_qc_correlations.csv", "attention_domain_association.csv",
    "resource_usage.csv", "failure_index.json", "protocol_deviations.json",
    "tests_final.log",
)
FIGURES = (
    "ablation_ari_nmi_forest", "ablation_spatial_tradeoff",
    "loss_term_ablation_heatmap", "attention_ablation_heatmap",
    "ige_coefficients_by_dataset", "weighted_gradient_share_trajectories",
    "attention_distribution_by_dataset", "attention_entropy_and_stability",
    "attention_qc_correlations", "attention_domain_association",
    "a1_boundary_tradeoff_maps", "p22_seed_heterogeneity",
)


def protected_check(root: str, manifest: str, expected: int) -> dict:
    result = subprocess.run(["sha256sum", "-c", manifest], cwd=root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    ok = sum(line.endswith(": OK") for line in lines)
    failures = [line for line in lines if not line.endswith(": OK")]
    if result.returncode != 0 or ok != expected or failures:
        raise RuntimeError("Protection failed %s: %d/%d" % (root, ok, expected))
    return {"root": root, "manifest": manifest, "ok": ok, "expected": expected, "passed": True}


def run_tests(output: Path) -> tuple[int, str]:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_night3b.py"], cwd=str(REPO), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    path = output / "tests_final.log"
    path.write_text(result.stdout, encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.flush(); os.fsync(handle.fileno())
    return result.returncode, result.stdout


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    lock = json.loads((output / "config_lock.json").read_text(encoding="utf-8"))
    verify_night3b_lock(REPO, CONFIG_PATH, config, lock, output, "finalize")
    code, test_output = run_tests(output)
    if code != 0:
        raise RuntimeError("Final tests failed")
    for name in REQUIRED:
        if not (output / name).is_file():
            raise RuntimeError("Missing required deliverable: %s" % name)
    for name in FIGURES:
        for suffix in ("png", "pdf"):
            if not (output / "figures" / (name + "." + suffix)).is_file():
                raise RuntimeError("Missing required figure: %s.%s" % (name, suffix))
    failures = list(output.rglob("failure.json"))
    manifests = list((output / "runs").glob("*/*/seed_*/run_manifest.json"))
    if failures or len(manifests) != 120:
        raise RuntimeError("Run completeness failure: manifests=%d failures=%d" % (len(manifests), len(failures)))
    gate = json.loads((output / "night3b_gate_status.json").read_text(encoding="utf-8"))
    completion = json.loads((output / "night3b_completion.json").read_text(encoding="utf-8"))
    if not (gate.get("p0_arch_pass") and gate.get("main_runs_completed") == 120
            and gate.get("failure_count") == 0 and gate.get("full_ige_replay_passed") == 15
            and completion.get("main_runs_completed") == 120):
        raise RuntimeError("Scientific completion/gate status is incomplete")

    protections = {
        "night3af": protected_check(config["paths"]["protected_night3af_root"],
                                     config["paths"]["protected_night3af_manifest"], 709),
        "night3ar": protected_check(config["paths"]["protected_night3ar_root"],
                                     config["paths"]["protected_night3ar_manifest"], 211),
        "night3a": protected_check(config["paths"]["protected_night3a_root"],
                                    config["paths"]["protected_night3a_manifest"], 198),
        "night2c": protected_check(config["paths"]["protected_night2c_root"],
                                    config["paths"]["protected_night2c_manifest"], 913),
    }
    atomic_json(output / "shutdown_checklist.json", {
        "schema_version": 1, "scientific_results_complete": True,
        "p0_arch": "PASS", "variant_probes": "24/24", "main_runs": "120/120",
        "failure_json_count": 0, "semantic_label_access_during_training": False,
        "full_ige_replay": "15/15 PASS", "tests": test_output.strip().splitlines()[-1],
        "historical_protections": protections,
        "internal_sha256sums_validation": "performed after this checklist is fsynced",
        "git_commit_tag_pending": True, "bundle_archive_pending": True,
        "local_dual_verification_pending": True,
        "required_last_remote_command": "/usr/bin/shutdown",
        "reconnect_after_shutdown_forbidden": True,
    })
    inventory = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            inventory.append({
                "path": str(path.relative_to(output)), "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            })
    atomic_json(output / "artifact_inventory.json", {
        "schema_version": 1, "file_count_excluding_inventory_and_sha256sums": len(inventory),
        "files": inventory,
    })
    # Recompute after inventory creation so the inventory itself is protected.
    files = [path for path in sorted(output.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
    sha_path = output / "SHA256SUMS"
    with sha_path.open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            handle.write("%s  %s\n" % (sha256_file(path), path.relative_to(output)))
        handle.flush(); os.fsync(handle.fileno())
    check = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=str(output), text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    ok = sum(line.endswith(": OK") for line in check.stdout.splitlines())
    if check.returncode != 0 or ok != len(files):
        raise RuntimeError("Internal SHA256SUMS validation failed")
    print("FINALIZE_PASS runs=120 tests=%s figures=24 files=%d" %
          (test_output.strip().splitlines()[-1], len(files)), flush=True)


if __name__ == "__main__":
    main()
