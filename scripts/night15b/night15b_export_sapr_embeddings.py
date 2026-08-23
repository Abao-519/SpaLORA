#!/usr/bin/env python3
"""Export bounded SAPR checkpoint embeddings for the registered Windows CPU replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from SpaLORA.night15b_sapr import SAPRCore  # noqa: E402
from night15b_local_runner import sha256_array  # noqa: E402
from night15b_train_sapr import (  # noqa: E402
    load_lane,
    model_forward_numpy,
    prepare_tensors,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for checkpoint in sorted(args.checkpoints.glob("*.pt")):
        saved = torch.load(checkpoint, map_location="cpu")
        payload = load_lane(args.kit, args.head, str(saved["lane"]))
        tensors = prepare_tensors(payload, device)
        config = saved["config"]
        model = SAPRCore(
            int(saved["base_dim"]), int(saved["view1_dim"]), int(saved["view2_dim"]),
            int(config["latent_dim"]), int(saved["k"]), int(config["hidden_dim"]),
            float(config["dropout"]),
        ).to(device)
        model.load_state_dict(saved["state_dict"], strict=True)
        full = model_forward_numpy(model, tensors, True)
        disabled = model_forward_numpy(model, tensors, False)
        full_sha = sha256_array(full)
        disabled_sha = sha256_array(disabled)
        if full_sha != saved["expected_full_sha256"] or disabled_sha != saved["expected_disabled_sha256"]:
            raise RuntimeError(f"checkpoint replay mismatch: {checkpoint}")
        target = args.output / f"{checkpoint.stem}.npz"
        np.savez_compressed(
            target,
            full=full,
            disabled=disabled,
            lane=np.asarray([saved["lane"]], dtype=str),
            candidate_id=np.asarray([config["candidate_id"]], dtype=str),
            training_seed=np.asarray([saved["seed"]], dtype=np.int64),
            full_sha256=np.asarray([full_sha], dtype=str),
            disabled_sha256=np.asarray([disabled_sha], dtype=str),
        )
        rows.append({
            "checkpoint": str(checkpoint),
            "export": target.name,
            "lane": str(saved["lane"]),
            "candidate_id": str(config["candidate_id"]),
            "training_seed": int(saved["seed"]),
            "full_shape": list(full.shape),
            "disabled_shape": list(disabled.shape),
            "full_sha256": full_sha,
            "disabled_sha256": disabled_sha,
            "strict_load": True,
            "numerical_replay_match": True,
        })
    manifest = {
        "status": "PASS",
        "count": len(rows),
        "device": str(device),
        "raw_or_dense_n_by_n_in_export": 0,
        "rows": rows,
    }
    (args.output / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "count": len(rows)}))


if __name__ == "__main__":
    main()
