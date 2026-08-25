#!/usr/bin/env python
"""Rerun control tables with final candidate-set and partition-SHA assertions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    builder = Path(__file__).with_name("build_control_table.py")
    for lane in ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7"):
        subprocess.run(
            [
                sys.executable, str(builder), "--lane", lane,
                "--features", str(args.feature_root / lane / "candidate_learned_evidence.csv"),
                "--bank", str(args.bank_root / f"{lane}.npz"),
                "--evaluation", str(args.evaluation_root / f"{lane}.csv"),
                "--authority", str(args.authority), "--fixed-config", str(args.freeze),
                "--output", str(args.output_root / f"{lane}.csv"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
