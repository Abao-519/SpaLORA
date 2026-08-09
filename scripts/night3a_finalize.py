#!/usr/bin/env python3
"""Validate Night-3A deliverables and create an internal non-self checksum manifest."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO / "configs/night3a_ige_feasibility.json").read_text(encoding="utf-8"))
OUTPUT = REPO / CONFIG["paths"]["output_root"]


REQUIRED = (
    "night3a_report.md", "night3a_completion.json", "night3a_environment.json",
    "night3a_p0a.json", "night3a_p0b.json", "night3a_gate_status.json",
    "config_lock.json", "data_manifest.csv", "ige_initial_gradients.csv",
    "ige_weights.csv", "per_seed_metrics.csv", "summary.csv", "paired_deltas.csv",
    "loss_trajectories.csv", "attention_summary.csv", "resource_usage.csv",
    "failure_index.json", "locked_60_run_manifest.json", "training_complete.json",
    "training_label_firewall.json", "representative_seeds.json",
)
RUN_REQUIRED = (
    "embedding.npz", "attention.npz", "observation_ids.csv", "clusters.csv",
    "loss_trajectory.csv", "checkpoint_index.csv", "run_config.json",
    "model_final.pt", "run_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def main() -> None:
    errors = []
    for name in REQUIRED:
        if not (OUTPUT / name).is_file():
            errors.append("missing required output: %s" % name)
    for name in ("night3a_completion.json", "night3a_gate_status.json", "night3a_p0a.json", "night3a_p0b.json"):
        try:
            json.loads((OUTPUT / name).read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append("invalid JSON %s: %r" % (name, exc))
    completion = json.loads((OUTPUT / "night3a_completion.json").read_text(encoding="utf-8"))
    if completion.get("main_runs_completed") != 60 or completion.get("failure_count") != 0:
        errors.append("completion is not 60/60 with zero failures")
    failures = json.loads((OUTPUT / "failure_index.json").read_text(encoding="utf-8"))
    if failures.get("failures") != []:
        errors.append("failure_index is not empty")
    run_count = 0
    for dataset in CONFIG["datasets"]:
        for variant in CONFIG["variants"]:
            for seed in CONFIG["seeds"]:
                run_count += 1
                run_dir = OUTPUT / "runs" / dataset / variant / ("seed_%d" % seed)
                for name in RUN_REQUIRED:
                    if not (run_dir / name).is_file():
                        errors.append("missing run artifact: %s/%s" % (run_dir.relative_to(REPO), name))
                if (run_dir / "failure.json").exists():
                    errors.append("run failure exists: %s" % run_dir.relative_to(REPO))
    figure_png = sorted((OUTPUT / "figures").glob("*.png"))
    figure_pdf = sorted((OUTPUT / "figures").glob("*.pdf"))
    if len(figure_png) < 7 or len(figure_png) != len(figure_pdf):
        errors.append("figure PNG/PDF pairs incomplete")
    report = (OUTPUT / "night3a_report.md").read_text(encoding="utf-8")
    for phrase in (
        "Did IGE strictly pass?", "Placenta C1 recovery", "Spatial continuity",
        "Frozen IGE weights", "ILN diagnostic", "Next step",
    ):
        if phrase not in report:
            errors.append("report question missing: %s" % phrase)
    if errors:
        atomic_json(OUTPUT / "finalization_failure.json", {"schema_version": 1, "errors": errors})
        print(json.dumps(errors, indent=2))
        raise SystemExit(2)

    inventory = {
        "schema_version": 1,
        "run_count": run_count,
        "required_run_files_per_run": len(RUN_REQUIRED),
        "figure_png_count": len(figure_png),
        "figure_pdf_count": len(figure_pdf),
        "internal_manifest_excludes_itself": True,
        "external_delivery_index_required_to_avoid_archive_self_reference": True,
    }
    atomic_json(OUTPUT / "artifact_inventory.json", inventory)
    manifest_path = OUTPUT / "SHA256SUMS"
    files = sorted(
        path for path in OUTPUT.rglob("*")
        if path.is_file() and path != manifest_path and path.name != "finalization_failure.json"
    )
    lines = ["%s  %s" % (sha256_file(path), path.relative_to(OUTPUT).as_posix()) for path in files]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        expected, name = line.split("  ", 1)
        if sha256_file(OUTPUT / name) != expected:
            raise AssertionError("Internal SHA verification failed: %s" % name)
    print("FINALIZE_PASS runs=60 files=%d manifest=%s" %
          (len(files), sha256_file(manifest_path)))


if __name__ == "__main__":
    main()
