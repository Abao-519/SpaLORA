#!/usr/bin/env python
"""Prepare one-row immutable baseline slices before strict-LOSO fit processes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.authority.open(encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = args.output_dir / f"{row['lane']}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
            writer.writeheader(); writer.writerow(row)


if __name__ == "__main__":
    main()
