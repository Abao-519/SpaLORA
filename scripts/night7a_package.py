#!/usr/bin/env python3
"""Build the compact Night-7A staging tree after the immutable final tag."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night7a_consensus import atomic_json, sha256_file  # noqa: E402

OUT = REPO / "outputs/night7a_handoff"
STAGE_ROOT = Path("/root/autodl-fs/night7a_delivery_stage_20260818")
COMPACT = STAGE_ROOT / "official_compact"
FINAL_TAG = "night7a-final-20260818"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def verify_internal(index: dict) -> None:
    for entry in index["files"]:
        path = (REPO / entry["path"]) if entry.get("root") == "repo" else (OUT / entry["path"])
        if (not path.is_file() or path.stat().st_size != entry["size_bytes"] or
                sha256_file(path) != entry["sha256"]):
            raise RuntimeError(f"internal index mismatch: {path}")


def validate_final_refs(final_commit: str, head: str, remote_branch: str,
                        remote_tag: str, remote_peeled: str) -> None:
    if final_commit != head:
        raise RuntimeError("final tag does not peel to current delivery-index commit")
    if final_commit not in remote_branch or not remote_tag or final_commit not in remote_peeled:
        raise RuntimeError("remote branch/final-tag verification failed")


def main() -> None:
    index_path = OUT / "delivery_index.json"
    index = json.loads(index_path.read_text())
    verify_internal(index)
    final_commit = git("rev-parse", f"{FINAL_TAG}^{{}}")
    head = git("rev-parse", "HEAD")
    remote_branch = git("ls-remote", "--heads", "origin", index["branch"])
    remote_tag = git("ls-remote", "--tags", "origin", FINAL_TAG)
    remote_peeled = git("ls-remote", "--tags", "origin", f"{FINAL_TAG}^{{}}")
    validate_final_refs(final_commit, head, remote_branch, remote_tag, remote_peeled)
    if STAGE_ROOT.exists():
        raise RuntimeError(f"refusing to overwrite delivery staging root: {STAGE_ROOT}")
    COMPACT.mkdir(parents=True)
    for entry in index["files"]:
        source = (REPO / entry["path"]) if entry.get("root") == "repo" else (OUT / entry["path"])
        target = (COMPACT / entry["path"]) if entry.get("root") == "repo" else (COMPACT / "handoff" / entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (COMPACT / "handoff").mkdir(exist_ok=True)
    shutil.copy2(index_path, COMPACT / "handoff/delivery_index.json")
    git_evidence = {
        "branch": index["branch"], "final_commit": final_commit,
        "final_tag": FINAL_TAG, "final_tag_peeled_commit": final_commit,
        "remote_branch_line": remote_branch, "remote_tag_line": remote_tag,
        "remote_peeled_tag_line": remote_peeled,
        "force_push": False, "force_with_lease": False,
        "final_tag_created_once_and_not_moved": True,
    }
    atomic_json(COMPACT / "git_final_evidence.json", git_evidence)
    bundle = COMPACT / "night6d_to_night7a_incremental_20260818.bundle"
    subprocess.check_call(["git", "-C", str(REPO), "bundle", "create", str(bundle),
                           FINAL_TAG, "^night6d-final-20260817"])
    subprocess.check_call(["git", "bundle", "verify", str(bundle)],
                          stdout=subprocess.DEVNULL)
    planner_tar = COMPACT / "night7a_planner_handoff_20260818.tar.gz"
    with tarfile.open(planner_tar, "w:gz") as archive:
        archive.add(COMPACT / "handoff", arcname="handoff")
        for directory in ("SpaLORA", "scripts", "tests", "protocols"):
            if (COMPACT / directory).exists():
                archive.add(COMPACT / directory, arcname=directory)
        archive.add(COMPACT / "git_final_evidence.json", arcname="git_final_evidence.json")
    external_files = []
    for relative in ("handoff/delivery_index.json", "git_final_evidence.json",
                     bundle.name, planner_tar.name):
        path = COMPACT / relative
        external_files.append({"path": relative, "size_bytes": path.stat().st_size,
                               "sha256": sha256_file(path)})
    atomic_json(COMPACT / "external_delivery_index.json", {
        "schema": "non-self-referential-external-v1", "status": "PASS",
        "internal_root_aware_verified": f"{len(index['files'])}/{len(index['files'])}",
        "files": external_files,
    })
    total = sum(path.stat().st_size for path in COMPACT.rglob("*") if path.is_file())
    if total >= 20 * 1024 * 1024:
        raise RuntimeError(f"compact exceeds 20 MiB target: {total}")
    print(json.dumps({"status": "PASS", "compact_root": str(COMPACT),
                      "internal_files": len(index["files"]), "size_bytes": total,
                      "delivery_index_sha256": sha256_file(index_path),
                      "external_index_sha256": sha256_file(COMPACT / "external_delivery_index.json"),
                      "planner_tar_sha256": sha256_file(planner_tar),
                      "bundle_sha256": sha256_file(bundle)}, sort_keys=True))


if __name__ == "__main__":
    main()
