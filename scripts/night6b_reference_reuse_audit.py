#!/usr/bin/env python3
"""Fail-closed read-only audit for the required Night-5 C04/B01 replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path("/root/autodl-fs/SpaLORA-night6b")
OUT = REPO / "outputs/night6b_handoff"
SOURCE = Path("/root/autodl-fs/night5a_raw_runs_20260813/a1/C04_SHRINK25")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


seeds = []
artifact_errors = []
for seed in range(5):
    directory = SOURCE / f"seed_{seed}"
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = sorted(p.name for p in directory.iterdir() if p.is_file())
    checkpoint_files = [name for name in files if name.endswith((".pt", ".pth", ".ckpt")) or "checkpoint" in name.lower() or "model_final" in name.lower()]
    checks = []
    for name, expected in sorted(manifest["artifact_sha256"].items()):
        actual = sha(directory / name)
        checks.append({"name": name, "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected})
        if actual != expected:
            artifact_errors.append({"seed": seed, "name": name, "expected": expected, "actual": actual})
    seeds.append({
        "seed": seed,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha(manifest_path),
        "declared_artifacts": sorted(manifest["artifact_sha256"]),
        "all_files": files,
        "checkpoint_or_model_state_files": checkpoint_files,
        "final_state_sha256_only": manifest.get("final_state_sha256"),
        "artifact_checks": checks,
    })

runner = REPO / "scripts/night5a_runner.py"
source = runner.read_text()
payload = {
    "status": "FAIL_REQUIRED_VALID_CHECKPOINT_ABSENT",
    "terminal_status": "IMPLEMENTATION_SEMANTICS_INVALID",
    "hard_gate": "P0-SEMANTIC 5.2 and section 6 A1 reference reuse",
    "source_candidate": "Night-5 C04_SHRINK25 / B01_C04_SHRINK25",
    "source_root": str(SOURCE),
    "seeds": seeds,
    "artifact_hash_mismatches": artifact_errors,
    "night5a_runner_sha256": sha(runner),
    "night5a_runner_required_excludes_model_state": "model_final.pt" not in source and "torch.save" not in source,
    "valid_checkpoint_available": False,
    "forward_replay_executed": False,
    "private_views_derived": False,
    "retraining_performed": False,
    "night3af_weights_substituted": False,
    "night6a_invalid_weights_used": False,
    "formal_scientific_training_units": 0,
    "head_transforms": 0,
    "decision": "Stop before formal Night-6B training; rebuilding or substituting weights is forbidden by the authority taskbook.",
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "reference_reuse_and_multiview_parity.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": payload["status"], "terminal_status": payload["terminal_status"], "seeds": len(seeds), "artifact_hash_mismatches": len(artifact_errors)}, sort_keys=True))
raise SystemExit(2)

