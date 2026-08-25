#!/usr/bin/env python
"""I/O registry runner; scientific producer remains dataset-name agnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lane", action="append")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    lanes = args.lane or list(registry["lanes"])
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "0"})
    producer = Path(__file__).with_name("build_learned_evidence.py")
    for lane in lanes:
        if lane not in registry["lanes"]:
            raise KeyError(lane)
        command = [
            sys.executable,
            str(producer),
            "--lane", lane,
            "--carrier", registry["carrier_template"].format(lane=lane),
            "--candidate-bank", registry["candidate_bank_template"].format(lane=lane),
            "--feasibility", registry["feasibility_template"].format(lane=lane),
            "--output-dir", str(args.output_root / lane),
            "--device", args.device,
        ]
        for seed, template in sorted(registry["seed_artifact_templates"].items()):
            command.extend(["--seed-artifact", f"{seed}={template.format(lane=lane)}"])
        for seed, template in sorted(registry["seed_checkpoint_templates"].items()):
            command.extend(["--seed-checkpoint", f"{seed}={template.format(lane=lane)}"])
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
