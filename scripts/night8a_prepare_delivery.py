#!/usr/bin/env python3
"""Write pre-tag Git and shutdown contracts for the final indexed commit."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/night8a_handoff"
BRANCH = "revision/q2-night8a-mfspc-rnd-20260820"
BASE = "e34567db5ace4f0fcdd2526cfb94a84fc9148020"
BASE_TAG = "night7c-final-20260818"
PROTECTION_TAG = "baseline/pre-night8a-mfspc-rnd-20260820"
FINAL_TAG = "night8a-final-20260820"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def atomic(name: str, payload: object) -> None:
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("branch mismatch")
    head = git("rev-parse", "HEAD")
    if git("rev-parse", BASE_TAG + "^{}") != BASE or git("rev-parse", PROTECTION_TAG + "^{}") != BASE:
        raise RuntimeError("base/protection tag mismatch")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", BASE, head], cwd=REPO)
    remote_line = git("ls-remote", "origin", "refs/heads/" + BRANCH)
    remote_head = remote_line.split()[0] if remote_line else None
    if remote_head != head:
        raise RuntimeError("remote branch does not match pre-delivery commit")
    final_remote = git("ls-remote", "origin", "refs/tags/" + FINAL_TAG)
    if final_remote:
        raise RuntimeError("final tag already exists before delivery-index commit")
    atomic("git_audit.json", {
        "status": "PASS_PRE_FINAL_INDEX", "branch": BRANCH, "base_commit": BASE, "base_tag": BASE_TAG,
        "protection_tag": PROTECTION_TAG, "protection_tag_peel": git("rev-parse", PROTECTION_TAG + "^{}"),
        "pre_delivery_index_commit": head, "remote_branch_commit": remote_head,
        "final_tag_absent_before_final_commit": True, "force_push_used": False,
        "final_binding_rule": "final tag is created once after the delivery-index commit and verified in compact_delivery_index.json",
    })
    atomic("shutdown_contract.json", {
        "status": "PLANNED", "connected_server": True, "api_used": False,
        "same_persistent_ssh_session_required": True, "final_remote_command": "/usr/bin/shutdown",
        "remote_reconnect_after_dispatch_forbidden": True,
        "dispatch_occurs_only_after_git_tag_push_compact_copy_and_windows_sha_verification": True,
    })
    print(json.dumps({"status": "PASS", "pre_delivery_index_commit": head, "remote_branch_commit": remote_head}, sort_keys=True))


if __name__ == "__main__":
    main()
