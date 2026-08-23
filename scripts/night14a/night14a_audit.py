#!/usr/bin/env python3
"""Small, read-only audit helpers for the Night-14A handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def snapshot_root(declared: str) -> Dict[str, object]:
    root = Path(declared).resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total = 0
    maximum = 0
    for path in files:
        stat = path.stat()
        total += stat.st_size
        maximum = max(maximum, stat.st_mtime_ns)
        digest.update(
            f"{path.relative_to(root).as_posix()}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode()
        )
    return {
        "declared_root": declared,
        "resolved_root": str(root),
        "file_count": len(files),
        "total_bytes": total,
        "max_mtime_ns": maximum,
        "metadata_fingerprint": digest.hexdigest(),
    }


def compare_roots(baseline: Path) -> Dict[str, object]:
    before = json.loads(baseline.read_text(encoding="utf-8"))
    records: List[Dict[str, object]] = []
    fields = (
        "resolved_root", "file_count", "total_bytes", "max_mtime_ns",
        "metadata_fingerprint",
    )
    for old in before["roots"]:
        current = snapshot_root(str(old["declared_root"]))
        current["byte_exact_metadata_match_night13c"] = all(
            current[key] == old[key] for key in fields
        )
        records.append(current)
    return {
        "baseline": "Night13C final metadata inventory (itself matched Night13A)",
        "roots": records,
        "passed": all(row["byte_exact_metadata_match_night13c"] for row in records),
        "changed_root_count": sum(
            not row["byte_exact_metadata_match_night13c"] for row in records
        ),
        "raw_content_files_opened_by_this_audit": 0,
        "audit_semantics": "relative path, size and mtime metadata only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    atomic_json(args.output, compare_roots(args.baseline))


if __name__ == "__main__":
    main()
