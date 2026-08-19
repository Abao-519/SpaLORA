#!/usr/bin/env python3
"""Fixed 160-epoch weighted-MNN trainer for Night-7C Stage W."""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha  # noqa: E402
from SpaLORA.night7a_consensus import atomic_json  # noqa: E402
from SpaLORA.night7b_adaptive import (  # noqa: E402
    AdaptiveFusion, VIEWS, deterministic_masks, fixed_mnn_triplets,
    graph_summary, reliability_inputs, row_l2, sparse_relation_edges,
)
from SpaLORA.night7c_conflict import weighted_mnn_weights, weighted_triplet_margin_loss  # noqa: E402
from scripts.night7b_train import (  # noqa: E402
    atomic_npy, configure_seed, load_contract, rng_snapshot, sha_file, state_sha,
)

RAW = Path("/root/autodl-fs/night7c_replay_recovery_20260818")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def train(unit_dir: Path, config_path: Path, feature_path: Path, candidate: str, output: Path) -> None:
    start = time.perf_counter(); require(not output.exists(), "weighted training output exists")
    output.mkdir(parents=True)
    unit, config, ids, arrays, s00, s04 = load_contract(unit_dir, config_path)
    require(config["recipe_id"] == "R02" and config["fusion"] == "equal", "base must be exact R02 equal")
    require(tuple(config["losses"]) == ("RECON", "MNN") and int(config["epochs"]) == 160,
            "base R02 loss/epoch contract mismatch")
    with np.load(feature_path, allow_pickle=False) as f:
        rank_c, quality, support = f["rank_c"], f["quality"], f["support"]
        locked_positive, locked_negative = f["positive"], f["negative"]
    raw_weights, norm_weights = weighted_mnn_weights(candidate, rank_c, quality, support, ids)
    seed = int(config["seed"]); configure_seed(seed)
    require(torch.cuda.is_available(), "GPU required")
    device = torch.device("cuda"); torch.cuda.reset_peak_memory_stats(device)
    z00 = graph_summary({key: arrays[i] for i, key in enumerate(VIEWS)})
    z04 = graph_summary({key: arrays[i + 3] for i, key in enumerate(VIEWS)})
    _, _, rel = reliability_inputs(s00, s04, z00, z04, ids)
    model = AdaptiveFusion([x.shape[1] for x in arrays], int(unit["K"]), False, False).to(device)
    targets = [torch.as_tensor(x, device=device) for x in arrays]
    reliability = torch.as_tensor(rel["scalars"], dtype=torch.float32, device=device)
    row_np, col_np, _ = sparse_relation_edges(arrays, ids, 20)
    # Preserve the complete historical construction order even though relation
    # edges and masks are not active in R02's RECON+MNN objective.
    torch.as_tensor(row_np, dtype=torch.long, device=device)
    torch.as_tensor(col_np, dtype=torch.long, device=device)
    pos_np, neg_np, mnn_audit = fixed_mnn_triplets(z00, z04, ids, "R02", seed)
    require(np.array_equal(pos_np, locked_positive) and np.array_equal(neg_np, locked_negative),
            "weighted trainer fixed MNN indices changed")
    positives = torch.as_tensor(pos_np, dtype=torch.long, device=device)
    negatives = torch.as_tensor(neg_np, dtype=torch.long, device=device)
    masks_np = deterministic_masks(len(ids), "R02", seed)
    for x in masks_np: torch.as_tensor(x, dtype=torch.long, device=device)
    weights = torch.as_tensor(norm_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160)
    initial_rng = rng_snapshot(); curves = []
    for epoch in range(160):
        model.train(); optimizer.zero_grad(set_to_none=True)
        value = model(targets, reliability, None)
        recon = torch.stack([F.mse_loss(x, y) for x, y in zip(value["decoded"], targets)]).mean()
        mnn = weighted_triplet_margin_loss(value["z"], value["projected"][3][positives],
                                           value["projected"][3][negatives], weights, margin=.5)
        loss = recon + .2 * mnn
        require(bool(torch.isfinite(loss)), "non-finite weighted formal loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step(); scheduler.step()
        curves.append({"epoch": epoch + 1, "RECON": float(recon.detach().cpu()),
                       "MNN": float(mnn.detach().cpu()), "total": float(loss.detach().cpu()),
                       "learning_rate": float(optimizer.param_groups[0]["lr"])})
    model.eval()
    with torch.no_grad(): final = model(targets, reliability, None)
    embedding = final["z"].detach().cpu().numpy().astype(np.float32)
    gates = final["gate_weights"].detach().cpu().numpy().astype(np.float32)
    checkpoint = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                  "scheduler_state": scheduler.state_dict(), "base_config": config,
                  "candidate_id": candidate, "unit_contract": unit,
                  "initial_rng": initial_rng, "final_rng": rng_snapshot(),
                  "embedding_sha256": array_sha(embedding), "gate_sha256": array_sha(gates),
                  "feature_file": str(feature_path), "weight_sha256": array_sha(norm_weights)}
    torch.save(checkpoint, output / "model_final.pt")
    atomic_npy(output / "embedding.npy", embedding); atomic_npy(output / "gate_weights.npy", gates)
    np.save(output / "mnn_raw_weights.npy", raw_weights, allow_pickle=False)
    np.save(output / "mnn_normalized_weights.npy", norm_weights, allow_pickle=False)
    with (output / "loss_curve.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["epoch", "RECON", "MNN", "total", "learning_rate"]); w.writeheader(); w.writerows(curves)
    manifest = {
        "schema_version": 1, "status": "success", "candidate_id": candidate,
        "unit_id": unit["unit_id"], "seed": seed, "epochs": 160,
        "scientific_training": True, "label_access": False, "fallback": False, "retry": False,
        "gpu_model": torch.cuda.get_device_name(0), "cuda_tensor_verified": all(x.is_cuda for x in targets) and next(model.parameters()).is_cuda,
        "runtime_seconds": time.perf_counter() - start,
        "peak_gpu_mib": torch.cuda.max_memory_allocated(device) / 1048576.0,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "checkpoint_path": str(output / "model_final.pt"), "checkpoint_sha256": sha_file(output / "model_final.pt"),
        "state_tensor_sha256": state_sha(model.state_dict()),
        "embedding_path": str(output / "embedding.npy"), "embedding_sha256": array_sha(embedding),
        "gate_path": str(output / "gate_weights.npy"), "gate_sha256": array_sha(gates),
        "base_config_sha256": sha_file(config_path), "worker_input_sha256": sha_file(unit_dir / "worker_input.json"),
        "feature_file_sha256": sha_file(feature_path), "raw_weight_sha256": array_sha(raw_weights),
        "normalized_weight_sha256": array_sha(norm_weights),
        "nonzero_weight_count": int(np.count_nonzero(raw_weights)),
        "normalized_nonzero_mean": float(norm_weights[norm_weights > 0].mean()),
        "positive_sha256": array_sha(pos_np), "negative_sha256": array_sha(neg_np),
        "mnn_audit": mnn_audit, "loss_curve_sha256": sha_file(output / "loss_curve.csv"),
    }
    atomic_json(output / "training_manifest.json", manifest)


def reload(unit_dir: Path, config_path: Path, output: Path) -> None:
    unit, config, ids, arrays, s00, s04 = load_contract(unit_dir, config_path)
    checkpoint = torch.load(output / "model_final.pt", map_location="cuda")
    model = AdaptiveFusion([x.shape[1] for x in arrays], int(unit["K"]), False, False).cuda()
    model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    z00 = graph_summary({key: arrays[i] for i, key in enumerate(VIEWS)})
    z04 = graph_summary({key: arrays[i + 3] for i, key in enumerate(VIEWS)})
    _, _, rel = reliability_inputs(s00, s04, z00, z04, ids)
    with torch.no_grad():
        value = model([torch.as_tensor(x, device="cuda") for x in arrays],
                      torch.as_tensor(rel["scalars"], dtype=torch.float32, device="cuda"), None)
    embedding = value["z"].detach().cpu().numpy().astype(np.float32)
    gates = value["gate_weights"].detach().cpu().numpy().astype(np.float32)
    exact_e = np.array_equal(embedding, np.load(output / "embedding.npy", allow_pickle=False))
    exact_g = np.array_equal(gates, np.load(output / "gate_weights.npy", allow_pickle=False))
    expected = json.loads((output / "training_manifest.json").read_text())
    audit = {"schema_version": 1, "status": "PASS" if exact_e and exact_g and state_sha(model.state_dict()) == expected["state_tensor_sha256"] else "FAIL",
             "fresh_process": True, "label_access": False, "embedding_exact": exact_e, "gate_exact": exact_g,
             "state_tensor_sha256": state_sha(model.state_dict()), "expected_state_tensor_sha256": expected["state_tensor_sha256"]}
    atomic_json(output / "reload_forward_audit.json", audit)
    require(audit["status"] == "PASS", "weighted checkpoint round-trip failed")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=("train", "reload")); p.add_argument("--unit-dir", type=Path, required=True); p.add_argument("--config", type=Path, required=True); p.add_argument("--feature", type=Path); p.add_argument("--candidate"); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "train":
        require(a.feature is not None and a.candidate, "weighted train arguments missing")
        train(a.unit_dir, a.config, a.feature, a.candidate, a.output)
    else: reload(a.unit_dir, a.config, a.output)


if __name__ == "__main__": main()
