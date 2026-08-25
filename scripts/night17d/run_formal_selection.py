#!/usr/bin/env python
"""Orchestrate separated fixed, public-HPO, and strict-LOSO selector processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")


def run(*items: object) -> None:
    subprocess.run([str(item) for item in items], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-slices", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    script_root = Path(__file__).resolve().parent
    python = sys.executable

    def feature(lane): return args.feature_root / lane / "candidate_learned_evidence.csv"
    def evaluation(lane): return args.evaluation_root / f"{lane}.csv"
    def authority_slice(lane): return args.authority_slices / f"{lane}.csv"

    # Fixed global and preregistered strong controls.
    for lane in LANES:
        control_path = args.output_root / "controls" / f"{lane}.csv"
        run(
            python, script_root / "build_control_table.py",
            "--lane", lane, "--features", feature(lane), "--bank", args.bank_root / f"{lane}.npz",
            "--evaluation", evaluation(lane), "--authority", args.authority,
            "--fixed-config", args.freeze, "--output", control_path,
        )
        for prefix in ("LEARNED", "ZERO", "PERMUTED"):
            selection = args.output_root / "fixed" / lane / f"{prefix}.selection.json"
            result = args.output_root / "fixed" / lane / f"{prefix}.evaluation.json"
            run(python, script_root / "apply_selector.py", "--lane", lane, "--features", feature(lane), "--config", args.freeze, "--evidence-prefix", prefix, "--output", selection)
            run(python, script_root / "evaluate_selection.py", "--selection", selection, "--evaluation", evaluation(lane), "--output", result)

    # Transparent direct HPO uses all three studies and is not a transfer result.
    direct = args.output_root / "public_benchmark_hpo"
    fit = [python, script_root / "fit_selector.py", "--freeze", args.freeze, "--held-out-lane", "NONE", "--output-dir", direct]
    for lane in LANES:
        fit.extend(["--training", f"{lane}={feature(lane)}={evaluation(lane)}"])
        fit.extend(["--training-authority", f"{lane}={authority_slice(lane)}"])
    run(*fit)
    for lane in LANES:
        selection = direct / f"{lane}.selection.json"
        result = direct / f"{lane}.evaluation.json"
        run(python, script_root / "apply_selector.py", "--lane", lane, "--features", feature(lane), "--config", direct / "fitted_config.json", "--output", selection)
        run(python, script_root / "evaluate_selection.py", "--selection", selection, "--evaluation", evaluation(lane), "--output", result)

    # Strict LOSO: fit sees only two training lanes, then a separate producer sees
    # held-out features, and only the final evaluator opens held-out metrics.
    for held in LANES:
        fold = args.output_root / "strict_loso" / f"held_{held}"
        training = [lane for lane in LANES if lane != held]
        fit = [python, script_root / "fit_selector.py", "--freeze", args.freeze, "--held-out-lane", held, "--output-dir", fold / "fit"]
        for lane in training:
            fit.extend(["--training", f"{lane}={feature(lane)}={evaluation(lane)}"])
            fit.extend(["--training-authority", f"{lane}={authority_slice(lane)}"])
        run(*fit)
        for prefix in ("LEARNED", "ZERO", "PERMUTED"):
            selection = fold / f"{prefix}.selection.json"
            result = fold / f"{prefix}.evaluation.json"
            run(python, script_root / "apply_selector.py", "--lane", held, "--features", feature(held), "--config", fold / "fit" / "fitted_config.json", "--evidence-prefix", prefix, "--output", selection)
            run(python, script_root / "evaluate_selection.py", "--selection", selection, "--evaluation", evaluation(held), "--output", result)


if __name__ == "__main__":
    main()
