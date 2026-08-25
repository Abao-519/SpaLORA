#!/usr/bin/env python
"""Apply a frozen config to locked candidate features without labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpaLORA.night17d_learned_evidence import SelectorWeights, select_candidate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence-prefix", choices=("LEARNED", "ZERO", "PERMUTED"), default="LEARNED")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.features.open(encoding="utf-8")))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    values = config.get("fitted_weights", config.get("fixed_global"))
    weights = SelectorWeights(**{key: float(values[key]) for key in ("molecular", "topology", "learned", "uncertainty")})
    selected = select_candidate(rows, weights, args.evidence_prefix)
    result = {
        "schema": "night17d-locked-selection-v1",
        "lane": args.lane,
        "candidate_id": selected["candidate_id"],
        "candidate_sha256": selected["candidate_sha256"],
        "evidence_prefix": args.evidence_prefix,
        "config_id": weights.config_id,
        "weights": weights.__dict__,
        "features_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "producer_label_reads": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
