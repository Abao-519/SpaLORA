#!/usr/bin/env python3
"""Build, verify, and stage the non-self-referential Night-6D compact."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/night6d_handoff"
INDEX = OUT / "handoff/delivery_index.json"
BRANCH = "revision/q2-night6d-locked-d1-p22-confirmation-20260817"
TAG = "night6d-final-20260817"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row(path: Path, relative: str, root: str | None = None) -> dict:
    value = {"path": relative.replace("\\", "/"), "size_bytes": path.stat().st_size,
             "sha256": sha256_file(path)}
    if root is not None:
        value["root"] = root
    return value


def selected_repo_files() -> list[Path]:
    patterns = [
        "SpaLORA/night6d*.py", "scripts/night6d*.py", "tests/test_night6d*.py",
        "protocols/night6d/*", "SpaLORA/night6c_pipeline.py", "scripts/night6c_train.py",
        "scripts/night6c_reload.py", "scripts/night6c_transform.py", "SpaLORA/night5a_rnd.py",
        "SpaLORA/night3af_cache.py", "SpaLORA/night3a_ige.py", "SpaLORA/night1_pipeline.py",
        "SpaLORA/night1_evaluation.py", "SpaLORA/night3b_metrics.py",
    ]
    files = []
    for pattern in patterns:
        files.extend(REPO.glob(pattern))
    return sorted(set(p for p in files if p.is_file()))


def build() -> dict:
    decision = json.loads((OUT / "night6d_decision.json").read_text())
    outputs = [p for p in OUT.rglob("*") if p.is_file() and p != INDEX and
               "__pycache__" not in p.parts]
    entries = [row(p, p.relative_to(OUT).as_posix()) for p in sorted(outputs)]
    entries.extend(row(p, p.relative_to(REPO).as_posix(), "repo") for p in selected_repo_files())
    entries.sort(key=lambda x: (x.get("root", "handoff"), x["path"]))
    index = {
        "schema": "non-self-referential-v1", "terminal_status": decision["terminal_status"],
        "branch": BRANCH, "planned_final_tag": TAG,
        "internal_output_root": "outputs/night6d_handoff", "repo_root_marker": "root=repo",
        "files": entries,
    }
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n",
                     encoding="utf-8")
    return index


def verify(index: dict, output_root: Path = OUT, repo_root: Path = REPO) -> None:
    seen = set()
    for item in index["files"]:
        root = item.get("root", "handoff")
        key = (root, item["path"])
        if key in seen:
            raise RuntimeError(f"duplicate delivery key: {key}")
        seen.add(key)
        path = (repo_root if root == "repo" else output_root) / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"delivery verification failed: {path}")


def stage(index: dict, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"refusing to reuse stage: {destination}")
    destination.mkdir(parents=True)
    for item in index["files"]:
        base = REPO if item.get("root", "handoff") == "repo" else OUT
        target = destination / (item["path"] if item.get("root") == "repo" else
                                Path("handoff") / item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / item["path"], target)
    target_index = destination / "handoff/delivery_index.json"
    target_index.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INDEX, target_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify", "stage"))
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    index = build() if args.command == "build" else json.loads(INDEX.read_text())
    verify(index)
    if args.command == "stage":
        if args.destination is None:
            parser.error("--destination required")
        stage(index, args.destination)
    print(json.dumps({"status": "PASS", "entries": len(index["files"]),
                      "delivery_index": str(INDEX),
                      "delivery_index_sha256": sha256_file(INDEX),
                      "staged_to": str(args.destination) if args.command == "stage" else None}, sort_keys=True))


if __name__ == "__main__":
    main()
