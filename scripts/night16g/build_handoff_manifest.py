#!/usr/bin/env python3
"""Build a deterministic repository-relative Night-16G handoff manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    patterns = (
        "SpaLORA/night16g*.py",
        "scripts/night16g/*",
        "configs/night16g/*",
        "tests/test_night16g*.py",
        "outputs/night16g_handoff/*",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path.resolve() for path in repo.glob(pattern) if path.is_file())
    paths.discard(output)
    rows = [
        {
            "path": path.relative_to(repo).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(paths)
    ]
    payload = {
        "schema": "night16g-repository-handoff-manifest-v1",
        "root": "repository root",
        "indexed_count": len(rows),
        "files": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "indexed_count": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
