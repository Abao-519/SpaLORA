#!/usr/bin/env python3
"""Build and verify the non-self-referential Night-6C compact delivery index."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/night6c_handoff"
INDEX = OUT / "handoff/delivery_index.json"
BRANCH = "revision/q2-night6c-clean-baseline-graph-rescue-20260817"
TAG = "night6c-final-20260817"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path: Path, relative: str, root: str | None = None) -> dict:
    item = {
        "path": relative.replace("\\", "/"),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if root is not None:
        item["root"] = root
    return item


def selected_repo_files() -> list[Path]:
    files: list[Path] = []
    files.extend(sorted((REPO / "SpaLORA").glob("night6c*.py")))
    files.extend(sorted((REPO / "scripts").glob("night6c*.py")))
    files.append(REPO / "tests/test_night6c_semantic.py")
    files.extend(sorted((REPO / "protocols/night6c").glob("*")))
    return [p for p in files if p.is_file()]


def build_index() -> dict:
    terminal = json.loads((OUT / "night6c_decision.json").read_text())["terminal_status"]
    output_files = [p for p in OUT.rglob("*") if p.is_file() and p != INDEX and
                    "__pycache__" not in p.parts]
    entries = [record(path, path.relative_to(OUT).as_posix()) for path in sorted(output_files)]
    entries.extend(record(path, path.relative_to(REPO).as_posix(), "repo")
                   for path in selected_repo_files())
    entries.sort(key=lambda x: (x.get("root", "handoff"), x["path"]))
    index = {
        "schema": "non-self-referential-v1",
        "terminal_status": terminal,
        "branch": BRANCH,
        "planned_final_tag": TAG,
        "internal_output_root": "outputs/night6c_handoff",
        "repo_root_marker": "root=repo",
        "files": entries,
    }
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n",
                     encoding="utf-8")
    return index


def verify(index: dict, output_root: Path = OUT, repo_root: Path = REPO) -> None:
    seen: set[tuple[str, str]] = set()
    for item in index["files"]:
        root = item.get("root", "handoff")
        key = (root, item["path"])
        if key in seen:
            raise RuntimeError(f"duplicate delivery key: {key}")
        seen.add(key)
        base = repo_root if root == "repo" else output_root
        path = base / item["path"]
        if not path.is_file():
            raise RuntimeError(f"missing indexed file: {path}")
        if path.stat().st_size != item["size_bytes"]:
            raise RuntimeError(f"size mismatch: {path}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"SHA mismatch: {path}")


def stage(index: dict, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"refusing to reuse delivery stage: {destination}")
    destination.mkdir(parents=True)
    for item in index["files"]:
        root = item.get("root", "handoff")
        source = (REPO if root == "repo" else OUT) / item["path"]
        target = destination / (item["path"] if root == "repo" else Path("handoff") / item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    target_index = destination / "handoff/delivery_index.json"
    target_index.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INDEX, target_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "verify", "stage"])
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    index = build_index() if args.command == "build" else json.loads(INDEX.read_text())
    verify(index)
    if args.command == "stage":
        if args.destination is None:
            parser.error("--destination is required for stage")
        stage(index, args.destination)
    print(json.dumps({
        "status": "PASS",
        "entries": len(index["files"]),
        "delivery_index": str(INDEX),
        "delivery_index_sha256": sha256_file(INDEX),
        "staged_to": str(args.destination) if args.command == "stage" else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
