#!/usr/bin/env python3
"""Fresh-process checkpoint reload verifier for Night-6C."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night3af_cache import load_cache
from SpaLORA.night3a_ige import model_state_sha256
from SpaLORA.night6c_pipeline import (
    DATASET_CFG, array_sha, atomic_json, forward_model, h00, load_graph_data,
    load_views, make_trainer, observation_sha, sha256_file,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    spec = json.loads((run / "reload_spec.json").read_text(encoding="utf-8"))
    base = load_cache(Path(spec["base_cache_dir"]), spec["base_cache_manifest_sha256"])
    data, graph_manifest = load_graph_data(base, Path(spec["graph_cache_dir"]))
    if graph_manifest["canonical_graph_cache_sha256"] != spec["graph_cache_sha256"]:
        raise RuntimeError("reload graph-cache identity mismatch")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint_path = run / "model_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for key in ("model_state_dict", "canonical_training_config", "dataset_id",
                "graph_candidate_id", "seed", "input_and_cache_sha256",
                "code_commit", "software_versions", "canonical_tensor_state_sha256"):
        if key not in checkpoint:
            raise RuntimeError(f"checkpoint missing {key}")
    if checkpoint["dataset_id"] != spec["dataset"] or checkpoint["graph_candidate_id"] != spec["graph_id"]:
        raise RuntimeError("checkpoint identity mismatch")
    trainer = make_trainer(data, spec["dataset"], int(spec["seed"]), device)
    model = trainer.new_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    observed_state = model_state_sha256(model)
    if observed_state != checkpoint["canonical_tensor_state_sha256"]:
        raise RuntimeError("canonical tensor-state SHA mismatch after reload")
    observed = forward_model(model, data, device)
    expected = load_views(run / "views.npz")
    rows = {}
    global_abs = 0.0; global_rel = 0.0
    for key in expected:
        if observed[key].shape != expected[key].shape:
            raise RuntimeError(f"view shape mismatch: {key}")
        diff = np.abs(observed[key].astype(np.float64) - expected[key].astype(np.float64))
        max_abs = float(diff.max(initial=0.0))
        denom = np.maximum(np.abs(expected[key].astype(np.float64)), 1e-12)
        max_rel = float((diff / denom).max(initial=0.0))
        close = bool(np.allclose(observed[key], expected[key], atol=1e-6, rtol=1e-5))
        rows[key] = {"allclose": close, "max_absolute_error": max_abs,
                     "max_relative_error": max_rel,
                     "expected_array_sha256": array_sha(expected[key]),
                     "observed_array_sha256": array_sha(observed[key])}
        global_abs = max(global_abs, max_abs); global_rel = max(global_rel, max_rel)
        if not close:
            raise RuntimeError(f"view reload mismatch: {key}")
    fresh = h00(observed["SpaLORA_fused"], int(DATASET_CFG[spec["dataset"]]["n_clusters"]))["labels"]
    locked = pd.read_csv(run / "h00_clusters.csv")["cluster"].to_numpy(dtype=np.int64)
    exact = bool(np.array_equal(fresh, locked))
    if not exact:
        raise RuntimeError("H00 cluster labels differ after fresh-process reload")
    audit = {
        "status": "PASS", "fresh_process": True, "strict_state_load": True,
        "dataset": spec["dataset"], "graph_id": spec["graph_id"], "seed": int(spec["seed"]),
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "canonical_tensor_state_sha256": observed_state,
        "views": rows, "max_absolute_error": global_abs,
        "max_relative_error": global_rel, "atol": 1e-6, "rtol": 1e-5,
        "h00_clusters_exact": exact, "h00_cluster_sha256": array_sha(fresh),
        "observation_sha256": observation_sha(base.obs_names.astype(str)),
    }
    atomic_json(run / "checkpoint_reload_audit.json", audit)
    print(json.dumps({"status": "PASS", "run": str(run), "max_abs": global_abs,
                      "max_rel": global_rel, "h00_exact": exact}, sort_keys=True))


if __name__ == "__main__":
    main()
