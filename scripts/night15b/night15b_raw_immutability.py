#!/usr/bin/env python3
"""Metadata-only immutability snapshot for Night-15B registered input roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOTS = [
    Path("/root/autodl-fs/Human lymph node/A1"),
    Path("/root/autodl-fs/Human lymph node/D1"),
    Path("/root/autodl-fs/P22 mouse brain coronal section"),
    Path("/root/autodl-fs/night8b_raw_runs_20260820/cache/base"),
    Path("/root/autodl-fs/night13a_benchmark_expansion_20260822/data/canonical_tonsil"),
]


def snapshot(root: Path) -> dict:
    resolved = root.resolve(strict=True)
    rows = []
    count = 0
    total = 0
    maximum_mtime_ns = 0
    for current, directories, files in os.walk(resolved):
        directories.sort()
        files.sort()
        for name in files:
            path = Path(current) / name
            stat = path.stat()
            relative = path.relative_to(resolved).as_posix()
            rows.append(f"{relative}\t{stat.st_size}\t{stat.st_mtime_ns}")
            count += 1
            total += stat.st_size
            maximum_mtime_ns = max(maximum_mtime_ns, stat.st_mtime_ns)
    digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
    return {
        "registered_path": str(root),
        "resolved_path": str(resolved),
        "file_count": count,
        "total_bytes": total,
        "max_mtime_ns": maximum_mtime_ns,
        "root_metadata_fingerprint": digest,
        "content_opened_or_rehashed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "method": "path,size,mtime metadata only; file contents not opened",
        "roots": [snapshot(root) for root in ROOTS],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
