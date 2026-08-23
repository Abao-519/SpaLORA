#!/usr/bin/env python3
"""Create and independently verify a root-relative size/SHA-256 file index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index-name", default="compact_delivery_index.json")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    index_path = root / args.index_name
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.resolve() != index_path.resolve()):
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    payload = {
        "schema": "root-relative-size-sha256-v1",
        "indexed_count": len(files),
        "files": files,
    }
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reread = json.loads(index_path.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in reread["files"]}
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != index_path.resolve()
    }
    missing = sorted(set(expected) - set(actual))
    extras = sorted(set(actual) - set(expected))
    size_mismatch = sorted(key for key in set(expected) & set(actual) if actual[key].stat().st_size != expected[key]["size"])
    sha_mismatch = sorted(key for key in set(expected) & set(actual) if sha256(actual[key]) != expected[key]["sha256"])
    result = {
        "status": "PASS" if not (missing or extras or size_mismatch or sha_mismatch) else "FAILED",
        "indexed_count": len(expected),
        "actual_count": len(actual),
        "missing": missing,
        "extras": extras,
        "size_mismatch": size_mismatch,
        "sha_mismatch": sha_mismatch,
        "index_sha256": sha256(index_path),
    }
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
