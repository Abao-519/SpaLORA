#!/usr/bin/env python3
"""Isolated descriptive deterministic-algorithm probe; never authorizes P0C."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

os.environ.setdefault("R_HOME", "/opt/R/4.0.3/lib/R")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.model import Encoder_overall
from SpaLORA.preprocess import fix_seed
from SpaLORA.SpaLORA_pyG import Train_SpaLORA
from scripts.night1_benchmark import prepare_legacy
from scripts.night2b_parity_locked_audit import forward_on_reference, frozen_components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["a1", "placenta", "p22"])
    args = parser.parse_args()
    config = json.loads((REPO / "configs/night2c_numerical_equivalence_factorial.json").read_text())
    cfg = config["datasets"][args.dataset]
    report = {"dataset": args.dataset, "ground_truth_accessed": False, "descriptive_only": True}
    try:
        fix_seed(0)
        data, _, _ = prepare_legacy(args.dataset, cfg)
        device = torch.device("cuda:0")
        trainer = Train_SpaLORA(data, datatype=cfg["legacy_datatype"], device=device, random_seed=0, epochs=cfg["epochs"], dim_output=cfg["embedding_dim"])
        base = Encoder_overall(trainer.dim_input1, trainer.dim_output1, trainer.dim_input2, trainer.dim_output2).to(device)
        state = {name: value.detach().clone() for name, value in base.state_dict().items()}
        torch.use_deterministic_algorithms(True)
        models = []
        results = []
        gradients = []
        for _ in range(2):
            model = Encoder_overall(trainer.dim_input1, trainer.dim_output1, trainer.dim_input2, trainer.dim_output2).to(device)
            model.load_state_dict(state)
            result = forward_on_reference(model, trainer)
            loss = frozen_components(trainer, result)["total"]
            model.zero_grad()
            loss.backward()
            torch.cuda.synchronize()
            models.append(model)
            results.append({name: value.detach().clone() for name, value in result.items()})
            gradients.append({name: value.grad.detach().clone() for name, value in model.named_parameters()})
        report["success"] = True
        report["output_max_absolute_difference"] = max(float(torch.abs(results[0][name] - results[1][name]).max().cpu()) for name in results[0])
        report["gradient_max_absolute_difference"] = max(float(torch.abs(gradients[0][name] - gradients[1][name]).max().cpu()) for name in gradients[0])
    except Exception as exc:
        report["success"] = False
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
    finally:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
