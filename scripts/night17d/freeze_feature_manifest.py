#!/usr/bin/env python
"""Seal two independently produced feature CSVs before label-derived evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay1", type=Path, required=True)
    parser.add_argument("--replay2", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lanes = sorted(path.parent.name for path in args.replay1.glob("*/candidate_learned_evidence.csv"))
    if not lanes:
        raise ValueError("no replay feature files")
    features = {}
    for lane in lanes:
        first = args.replay1 / lane / "candidate_learned_evidence.csv"
        second = args.replay2 / lane / "candidate_learned_evidence.csv"
        first_sha, second_sha = sha(first), sha(second)
        if first_sha != second_sha:
            raise RuntimeError(f"fresh-process feature mismatch for {lane}")
        features[lane] = {"replay1_sha256": first_sha, "replay2_sha256": second_sha, "exact": True}
    result = {
        "schema": "night17d-feature-freeze-v1",
        "lanes": lanes,
        "features": features,
        "selector_freeze_sha256": sha(args.freeze),
        "source_sha256": {str(path): sha(path) for path in args.source},
        "candidate_feature_label_reads": 0,
        "fresh_process_exact": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
