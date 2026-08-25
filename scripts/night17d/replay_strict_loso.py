#!/usr/bin/env python
"""Fresh-process replay of the three frozen strict-LOSO selections."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    checks = {}
    for lane in LANES:
        first = args.working / "evaluation/formal/strict_loso" / f"held_{lane}"
        second = args.output_root / f"held_{lane}"
        selection = second / "LEARNED.selection.json"
        evaluation = second / "LEARNED.evaluation.json"
        subprocess.run([
            sys.executable, str(scripts / "apply_selector.py"), "--lane", lane,
            "--features", str(args.working / "formal_features/replay1" / lane / "candidate_learned_evidence.csv"),
            "--config", str(first / "fit/fitted_config.json"), "--output", str(selection),
        ], check=True)
        subprocess.run([
            sys.executable, str(scripts / "evaluate_selection.py"), "--selection", str(selection),
            "--evaluation", str(args.evaluation_root / f"{lane}.csv"), "--output", str(evaluation),
        ], check=True)
        first_selection = first / "LEARNED.selection.json"
        first_evaluation = first / "LEARNED.evaluation.json"
        checks[lane] = {
            "selection_replay1_sha256": sha(first_selection), "selection_replay2_sha256": sha(selection),
            "selection_exact": sha(first_selection) == sha(selection),
            "evaluation_replay1_sha256": sha(first_evaluation), "evaluation_replay2_sha256": sha(evaluation),
            "evaluation_exact": sha(first_evaluation) == sha(evaluation),
        }
        if not checks[lane]["selection_exact"] or not checks[lane]["evaluation_exact"]:
            raise RuntimeError(f"strict LOSO replay mismatch for {lane}")
    result = {"schema": "night17d-strict-loso-replay-v1", "lanes": checks, "all_exact": True}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
