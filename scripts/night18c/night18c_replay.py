#!/usr/bin/env python3
"""Compare two independently produced Night-18C artifacts exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--original", required=True); parser.add_argument("--replay", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.original, allow_pickle=False) as left, np.load(args.replay, allow_pickle=False) as right:
        keys_equal = left.files == right.files
        array_equal = keys_equal and all(np.array_equal(left[key], right[key]) for key in left.files)
        partition_equal = np.array_equal(left["partitions"], right["partitions"])
    result = {"status": "PASS" if keys_equal and array_equal else "FAIL", "keys_equal": keys_equal,
              "array_equal": array_equal, "partition_exact": partition_equal,
              "original_sha256": file_sha(Path(args.original)), "replay_sha256": file_sha(Path(args.replay))}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if result["status"] != "PASS": raise SystemExit(2)


if __name__ == "__main__": main()

