#!/usr/bin/env python3
"""Validate Night-3A-R deliverables and write the non-self internal manifest."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO / "configs/night3ar_ige_feasibility.json").read_text(encoding="utf-8"))
OUTPUT = REPO / CONFIG["paths"]["output_root"]

REQUIRED = (
    "protocol_amendment.json", "integrity_read_manifest.json", "label_flow_audit.md",
    "night3ar_report.md", "night3ar_completion.json", "night3ar_environment.json",
    "night3ar_p0a.json", "night3ar_p0b.json", "night3ar_gate_status.json",
    "config_lock.json", "data_manifest.csv", "ige_initial_gradients.csv",
    "ige_weights.csv", "per_seed_metrics.csv", "summary.csv", "paired_deltas.csv",
    "paired_delta_summary.csv", "loss_trajectories.csv",
    "gradient_influence_trajectories.csv", "attention_summary.csv",
    "resource_usage.csv", "failure_index.json", "locked_60_run_manifest.json",
    "training_complete.json", "scientific_window_label_firewall.json",
    "representative_seeds.json", "preregistered_run_order.json",
)
RUN_REQUIRED = (
    "embedding.npz", "attention.npz", "observation_ids.csv", "clusters.csv",
    "loss_trajectory.csv", "gradient_influence_trajectory.csv", "checkpoint_index.csv",
    "run_config.json", "model_final.pt", "run_manifest.json",
)


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


def main() -> None:
    errors = []
    for name in REQUIRED:
        if not (OUTPUT / name).is_file():
            errors.append("missing required output: %s" % name)
    json_names = (
        "night3ar_completion.json", "night3ar_gate_status.json", "night3ar_p0a.json",
        "night3ar_p0b.json", "scientific_window_label_firewall.json",
        "integrity_read_manifest.json", "locked_60_run_manifest.json",
    )
    parsed = {}
    for name in json_names:
        try:
            parsed[name] = json.loads((OUTPUT / name).read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append("invalid JSON %s: %r" % (name, exc))
    completion = parsed.get("night3ar_completion.json", {})
    if completion.get("main_runs_completed") != 60 or completion.get("failure_count") != 0:
        errors.append("completion is not 60/60 with zero failures")
    if not parsed.get("night3ar_p0a.json", {}).get("passed"):
        errors.append("P0A-R did not pass")
    if not parsed.get("night3ar_p0b.json", {}).get("passed"):
        errors.append("P0B-R did not pass")
    if not parsed.get("scientific_window_label_firewall.json", {}).get("passed"):
        errors.append("scientific-window label firewall did not pass")
    integrity = parsed.get("integrity_read_manifest.json", {})
    if not integrity.get("all_reads_match") or integrity.get("semantic_content_returned") is not False:
        errors.append("integrity byte-read audit is incomplete")
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
    if len(figure_png) < 8 or len(figure_png) != len(figure_pdf):
        errors.append("figure PNG/PDF pairs incomplete")
    report = (OUTPUT / "night3ar_report.md").read_text(encoding="utf-8")
    for phrase in (
        "P0A-R", "P0B-R", "60/60", "semantic label access", "IGE scientific go/no-go",
        "Five-seed IGE-C0 and C1-C0", "Placenta C1 recovery", "Spatial tradeoff",
        "Scalar loss versus weighted-gradient influence", "Architecture ablation authorized",
    ):
        if phrase not in report:
            errors.append("report homepage field missing: %s" % phrase)
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
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    for line in lines:
        expected, name = line.split("  ", 1)
        if sha256_file(OUTPUT / name) != expected:
            raise AssertionError("Internal SHA verification failed: %s" % name)
    print("FINALIZE_PASS runs=60 files=%d manifest=%s" %
          (len(files), sha256_file(manifest_path)))


if __name__ == "__main__":
    main()
