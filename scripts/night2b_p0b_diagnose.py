#!/usr/bin/env python3
"""Label-free diagnosis of the already-failed P0B numerical divergence."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


os.environ.setdefault("R_HOME", "/opt/R/4.0.3/lib/R")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.model import Encoder_overall
from SpaLORA.preprocess import fix_seed
from SpaLORA.SpaLORA_pyG import Train_SpaLORA
from scripts.night1_benchmark import prepare_legacy


def forward(model, trainer):
    return model(
        trainer.features_omics1,
        trainer.features_omics2,
        trainer.adj_spatial_omics1,
        trainer.adj_feature_omics1,
        trainer.adj_spatial_omics2,
        trainer.adj_feature_omics2,
    )


def repeated_forward(data, cfg, device, state=None):
    trainer = Train_SpaLORA(
        data,
        datatype=cfg["legacy_datatype"],
        device=device,
        random_seed=0,
        learning_rate=0.0001,
        weight_decay=0.0,
        epochs=cfg["epochs"],
        dim_output=cfg["embedding_dim"],
    )
    model = Encoder_overall(
        trainer.dim_input1, trainer.dim_output1, trainer.dim_input2, trainer.dim_output2
    ).to(device)
    if state is not None:
        model.load_state_dict({name: value.to(device) for name, value in state.items()})
    model.eval()
    with torch.no_grad():
        first = forward(model, trainer)
        if device.type == "cuda":
            torch.cuda.synchronize()
        second = forward(model, trainer)
        if device.type == "cuda":
            torch.cuda.synchronize()
    outputs = {}
    for name in first:
        difference = torch.abs(first[name] - second[name])
        outputs[name] = {
            "exact_equal": bool(torch.equal(first[name], second[name])),
            "max_absolute_difference": float(difference.max().cpu()),
            "mean_absolute_difference": float(difference.mean().cpu()),
            "pass_at_1e_7": bool(torch.allclose(first[name], second[name], atol=1e-7, rtol=1e-7)),
        }
    return {
        "device": str(device),
        "pass": all(item["pass_at_1e_7"] for item in outputs.values()),
        "maximum_output_difference": max(item["max_absolute_difference"] for item in outputs.values()),
        "outputs": outputs,
    }, {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def main():
    night1 = json.loads((REPO / "configs/night1.json").read_text(encoding="utf-8"))
    prior = json.loads((REPO / "reports/night2b_parity_locked.json").read_text(encoding="utf-8"))
    if prior["p0b_pass"] is not False:
        raise RuntimeError("Diagnosis is valid only after the frozen P0B failure")
    report = {
        "schema_version": 1,
        "ground_truth_accessed": False,
        "purpose": "diagnose, not reauthorize or weaken P0B",
        "p0b_remains_failed": True,
        "datasets": {},
    }
    for dataset in ("a1", "placenta", "p22"):
        print("DIAG_START", dataset, flush=True)
        cfg = night1["datasets"][dataset]
        fix_seed(0)
        data, _, _ = prepare_legacy(dataset, cfg)
        gpu, state = repeated_forward(data, cfg, torch.device("cuda:0"))
        cpu, _ = repeated_forward(data, cfg, torch.device("cpu"), state)
        report["datasets"][dataset] = {"gpu_same_model_repeat": gpu, "cpu_same_model_repeat": cpu}
        print("DIAG_DONE", dataset, "gpu", gpu["maximum_output_difference"], "cpu", cpu["maximum_output_difference"], flush=True)
    report["interpretation"] = (
        "If identical-state same-model GPU repeats exceed 1e-7 while CPU repeats do not, the P0B residual is "
        "consistent with nondeterministic/numerically order-sensitive CUDA sparse execution. This does not undo the hard gate."
    )
    (REPO / "reports/night2b_p0b_diagnosis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
