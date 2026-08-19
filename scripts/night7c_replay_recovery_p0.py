#!/usr/bin/env python3
"""Fail-closed P0 authority audit for Night-7C replay portability recovery."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import night7c_p0_authority as old_p0  # noqa: E402

RAW = Path("/root/autodl-fs/night7c_replay_recovery_20260818")
OUT = REPO / "outputs/night7c_replay_recovery_handoff"
PROTO = REPO / "protocols/night7c_replay_recovery"
PARENT = "e34567db5ace4f0fcdd2526cfb94a84fc9148020"
PARENT_TAG = "night7c-final-20260818"
BRANCH = "revision/q2-night7c-replay-portability-recovery-20260818"

EXPECTED = {
    "SpaLORA_Night7C_Replay_Portability_Recovery_Planning_Index_2026-08-18.json": "a321232ebb2246ab6767385616aabcbb19a191c537f32a4a985e0eccd241e706",
    "SpaLORA_Night7C_Replay_Portability_Independent_Audit_2026-08-18.md": "beda9ee591cab487c3cca46d795cf4a368a2a10103a77763ee3a571e7410613f",
    "SpaLORA_Night7C_Replay_Portability_Recovery_Registry_2026-08-18.json": "f10f7d1301b00aa1314d98c2d22de2a463a26135ad5d96dce31f952f2437c004",
    "SpaLORA_Night7C_Replay_Portability_Recovery_Taskbook_2026-08-18.md": "7cb371738d82ec2f30af4c40f84e4e250a151b879657f5d46b7d8d3426171955",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def recovery_authority() -> dict:
    rows = []
    for name, expected in EXPECTED.items():
        path = PROTO / name
        actual = sha(path)
        require(actual == expected, f"recovery authority SHA mismatch: {name}")
        rows.append({"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size})
    registry = json.loads((PROTO / "SpaLORA_Night7C_Replay_Portability_Recovery_Registry_2026-08-18.json").read_text())
    require(registry["parent_commit"] == PARENT and registry["parent_tag"] == PARENT_TAG,
            "recovery parent mismatch")
    return {"files": rows, "registry_protocol_id": registry["protocol_id"]}


def git_audit() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    head = git("rev-parse", "HEAD")
    old_peel = git("rev-parse", PARENT_TAG + "^{}")
    branch = git("branch", "--show-current")
    require(head == PARENT, "recovery branch did not start at immutable Night-7C final commit")
    require(old_peel == PARENT, "old Night-7C final tag moved")
    require(branch == BRANCH, "recovery branch mismatch")
    return {"head": head, "old_tag_peel": old_peel, "branch": branch}


def old_invalid_audit() -> dict:
    old = REPO / "outputs/night7c_handoff"
    report = json.loads((old / "p1_semantic_contract.json").read_text())
    budget = json.loads((old / "budget_audit.json").read_text())
    labels = json.loads((old / "label_window_audit.json").read_text())
    require(report["terminal_status"] == "IMPLEMENTATION_SEMANTICS_INVALID", "old terminal changed")
    require(report["formal_training"] == 0 and report["formal_transforms"] == 0,
            "old scientific counts are not 0/0")
    require(labels.get("label_access") is False, "old label window was opened")
    return {"terminal_status": report["terminal_status"], "formal_training": 0,
            "formal_transforms": 0, "label_access": False,
            "budget_sha256": sha(old / "budget_audit.json")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Reuse the already reviewed Night-7C P0 implementation for immutable Night-7B
    # artifacts, GPU evidence, source manifests and environment.  Only Git/authority
    # are replaced because this is a new branch and a protocol revision.
    report = {
        "schema_version": 1,
        "status": "PASS",
        "label_access": False,
        "formal_training": 0,
        "formal_transforms": 0,
        "recovery_authority": recovery_authority(),
        "git": git_audit(),
        "old_invalid_run": old_invalid_audit(),
        "night7b_compact": old_p0.compact_audit(),
        "night7b_source": old_p0.source_audit(),
        "night7b_training_and_specialists": old_p0.training_and_specialist_audit(),
        "environment": old_p0.current_environment(),
        "firewall_static": old_p0.static_firewall_audit(),
        "new_raw_root": str(RAW),
        "invalid_partial_p1_reused": False,
    }
    target = OUT / "p0_recovery_authority.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(target),
                      "training_cells": report["night7b_training_and_specialists"]["training_cells"],
                      "specialist_transforms": report["night7b_training_and_specialists"]["specialist_transforms"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
