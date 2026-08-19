#!/usr/bin/env python3
"""Resume only the fixed Stage-W transforms after an infrastructure disconnect.

The 48 scientific training cells are immutable and are verified from their
existing manifests/checkpoints.  No training command is available here.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from SpaLORA.night6c_pipeline import array_sha  # noqa: E402
from SpaLORA.night7a_consensus import atomic_json, sha256_file  # noqa: E402
from scripts.night7c_replay_recovery_stage_w import (  # noqa: E402
    CANDIDATES,
    OUT,
    RAW,
    pilot_units,
    sampler,
    transform_cell,
)

FORMAL = RAW / "stage_w/formal"
INCIDENT = RAW / "invalid_attempts/stage_w_transform_attempt1_infrastructure_disconnect"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def load_verified_training() -> tuple[list[dict], list[dict]]:
    units = pilot_units()
    rows: list[dict] = []
    for candidate in CANDIDATES:
        for unit in units:
            cell = FORMAL / candidate / unit["unit_id"] / "attempt_001/cell_manifest.json"
            require(cell.is_file(), f"missing immutable training cell manifest: {cell}")
            row = json.loads(cell.read_text())
            require(row.get("status") == "success", f"training cell is not success: {cell}")
            worker = cell.parent / "worker"
            tm = worker / "training_manifest.json"
            reload = worker / "reload_forward_audit.json"
            checkpoint = worker / "model_final.pt"
            require(tm.is_file() and reload.is_file() and checkpoint.is_file(),
                    f"training evidence missing: {cell.parent}")
            require(sha256_file(tm) == row.get("training_manifest_sha256"),
                    f"training manifest SHA mismatch: {cell}")
            require(sha256_file(reload) == row.get("reload_audit_sha256"),
                    f"reload audit SHA mismatch: {cell}")
            tm_value = json.loads(tm.read_text())
            require(sha256_file(checkpoint) == tm_value.get("checkpoint_sha256"),
                    f"checkpoint file SHA mismatch: {cell}")
            embedding = worker / "embedding.npy"
            require(embedding.is_file(), f"embedding missing: {cell}")
            import numpy as np
            require(array_sha(np.load(embedding, allow_pickle=False)) == tm_value.get("embedding_sha256"),
                    f"embedding canonical SHA mismatch: {cell}")
            rows.append(row)
    require(len(rows) == 48, "immutable training cardinality is not 48")
    return units, rows


def archive_interrupted_empty_target() -> dict:
    source = FORMAL / "W00_FILTER75/u000/attempt_001/transform"
    require(source.is_dir(), "expected interrupted transform directory is missing")
    require(not any(source.iterdir()), "interrupted transform directory is not empty")
    require(not INCIDENT.exists(), "infrastructure incident archive already exists")
    archived = INCIDENT / "stage_w/formal/W00_FILTER75/u000/attempt_001/transform"
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(archived))
    evidence = {
        "schema_version": 1,
        "classification": "infrastructure_disconnect_before_transform_output",
        "preserved": True,
        "candidate_id": "W00_FILTER75",
        "unit_id": "u000",
        "scientific_training_added": 0,
        "scientific_transform_completed": 0,
        "label_access": False,
        "retry_classification": "infrastructure_correction_not_scientific_retry",
        "previous_physical_invocation_count": 1,
        "previous_output_file_count": 0,
        "last_verified_pid": 26013,
        "last_verified_pid_state": "Rl+",
        "last_verified_pid_elapsed": "04:01:51",
        "last_verified_pid_cpu_time": "03:44:06",
        "last_verified_pid_rss_kib": 597424,
        "last_verified_at": "2026-08-19T06:27:14+08:00",
        "shutdown_command_sent": False,
        "autodl_api_called": False,
        "archive_path": str(archived),
    }
    atomic_json(INCIDENT / "incident_manifest.json", evidence)
    return evidence


def main() -> None:
    require(not (OUT / "weighted_mnn_training_manifest.json").exists(),
            "overall Stage-W training manifest already exists")
    require(not (OUT / "weighted_mnn_transform_manifest.json").exists(),
            "overall Stage-W transform manifest already exists")
    units, training = load_verified_training()
    incident = archive_interrupted_empty_target()
    lookup = {(x["candidate_id"], x["unit_id"]): x for x in training}
    stop = threading.Event()
    resource = RAW / "stage_w_resume_resource.csv"
    thread = threading.Thread(target=sampler, args=(stop, resource, os.getpid()), daemon=True)
    thread.start()
    transforms: list[dict] = []
    start = time.perf_counter()
    try:
        for candidate in CANDIDATES:
            for unit in units:
                transforms.append(transform_cell(lookup[(candidate, unit["unit_id"])], unit))
    finally:
        stop.set()
        thread.join(timeout=3)
    require(len(transforms) == 48, "resumed transform cardinality mismatch")
    train_manifest = {
        "schema_version": 1,
        "status": "LOCKED_PRE_LABEL",
        "label_access": False,
        "planned_training": 48,
        "training_attempts": 48,
        "successful_training": 48,
        "failed_training": 0,
        "scientific_retry": 0,
        "fallback_count": 0,
        "reconstructed_from_immutable_completed_cells": True,
        "new_training_during_resume": 0,
        "training_cells": training,
    }
    transform_manifest = {
        "schema_version": 1,
        "status": "LOCKED_PRE_LABEL",
        "label_access": False,
        "planned_transforms": 48,
        "transform_attempts": 48,
        "physical_transform_invocations": 49,
        "infrastructure_aborted_invocations": 1,
        "infrastructure_corrections": 1,
        "successful_transforms": sum(x["status"] == "success" for x in transforms),
        "failed_transforms": sum(x["status"] != "success" for x in transforms),
        "scientific_retry": 0,
        "fallback_count": 0,
        "runtime_seconds": time.perf_counter() - start,
        "interrupted_attempt": incident,
        "transforms": transforms,
    }
    atomic_json(OUT / "weighted_mnn_training_manifest.json", train_manifest)
    atomic_json(OUT / "weighted_mnn_transform_manifest.json", transform_manifest)
    print(json.dumps({
        "status": "LOCKED_PRE_LABEL",
        "new_training": 0,
        "training_cells_reused": 48,
        "transform_attempts": 48,
        "physical_transform_invocations": 49,
        "transform_success": transform_manifest["successful_transforms"],
        "runtime_seconds": transform_manifest["runtime_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
