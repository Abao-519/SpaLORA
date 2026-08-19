#!/usr/bin/env python3
"""Create and audit the under-20-MiB Night-8A official compact handoff."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RAW = Path("/root/autodl-fs/night8a_raw_runs_20260820")
DEST = RAW / "official_compact"
BRANCH = "revision/q2-night8a-mfspc-rnd-20260820"
TAG = "night8a-final-20260820"
BASE_TAG = "night7c-final-20260818"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def copy(relative: str) -> None:
    source = REPO / relative
    destination = DEST / relative
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    if DEST.exists():
        raise RuntimeError("official compact destination already exists")
    final_commit = subprocess.check_output(["git", "rev-parse", BRANCH], cwd=REPO, text=True).strip()
    tag_commit = subprocess.check_output(["git", "rev-parse", TAG + "^{}"], cwd=REPO, text=True).strip()
    if final_commit != tag_commit:
        raise RuntimeError("final branch/tag peel mismatch")
    DEST.mkdir(parents=True)
    for relative in ["outputs/night8a_handoff", "protocols/night8a", "SpaLORA/night8a_mfspc.py"]:
        copy(relative)
    for folder, pattern in (("scripts", "night8a*.py"), ("tests", "test_night8a*.py")):
        for source in sorted((REPO / folder).glob(pattern)):
            destination = DEST / folder / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    bundle = DEST / "git/night7c_to_night8a_20260820.bundle"
    bundle.parent.mkdir(parents=True)
    subprocess.check_call(["git", "bundle", "create", str(bundle), "^" + BASE_TAG, BRANCH, TAG], cwd=REPO)

    tar_path = DEST / "night8a_planner_handoff_20260820.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        for relative in ("outputs/night8a_handoff", "protocols/night8a", "SpaLORA/night8a_mfspc.py", "scripts", "tests"):
            source = DEST / relative
            if source.exists():
                archive.add(source, arcname=relative, recursive=True)

    index = DEST / "compact_delivery_index.json"
    files = []
    for path in sorted(x for x in DEST.rglob("*") if x.is_file() and x != index):
        files.append({"path": path.relative_to(DEST).as_posix(), "sha256": sha(path),
                      "size_bytes": path.stat().st_size})
    root = hashlib.sha256("\n".join(x["sha256"] for x in sorted(files, key=lambda x: x["path"])).encode()).hexdigest()
    payload = {
        "schema_version": 1, "branch": BRANCH, "commit": final_commit, "final_tag": TAG,
        "final_tag_peel": tag_commit, "file_count": len(files), "files": files,
        "root_rule": "sha256(newline_join(file_sha256_in_lexical_path_order))", "root_sha256": root,
        "raw_files_included": False,
    }
    index.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(path.stat().st_size for path in DEST.rglob("*") if path.is_file())
    if total >= 20 * 1024 * 1024:
        raise RuntimeError(f"compact exceeds 20 MiB: {total}")
    print(json.dumps({"index_sha256": sha(index), "root_sha256": root, "file_count": len(files),
                      "total_size_bytes": total, "bundle_sha256": sha(bundle),
                      "planner_handoff_sha256": sha(tar_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
