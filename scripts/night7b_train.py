#!/usr/bin/env python3
"""Opaque, zero-label Night-7B adapter trainer and reload verifier."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha, sparse_sha  # noqa: E402
from SpaLORA.night7a_consensus import atomic_json, canonical_csr  # noqa: E402
from SpaLORA.night7b_adaptive import (  # noqa: E402
    AdaptiveFusion, VIEWS, deterministic_masks, fixed_mnn_triplets,
    graph_summary, loss_components, pseudo_confidence, pseudo_keep,
    reliability_inputs, row_l2, sparse_relation_edges, total_loss,
)


FORBIDDEN_KEYS = {
    "dataset", "tissue", "organism", "ground_truth", "label_path",
    "metric_path", "ari", "nmi", "q", "original_h5ad",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha(state) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, value, allow_pickle=False)
    os.replace(tmp, path)


def load_contract(unit_dir: Path, config_path: Path):
    unit = json.loads((unit_dir / "worker_input.json").read_text())
    config = json.loads(config_path.read_text())
    forbidden = FORBIDDEN_KEYS & (set(unit) | set(config))
    if forbidden:
        raise RuntimeError("worker received forbidden identity keys: %r" % sorted(forbidden))
    if set(unit) != {"unit_id", "K", "observation_count", "ordered_observation_sha256",
                     "g00_views", "g04_views", "observation_ids", "s00", "s04",
                     "pseudo_partition", "pseudo_affinity"}:
        raise RuntimeError("worker input schema is not fail-closed")
    ids = [x.strip() for x in Path(unit["observation_ids"]).read_text().splitlines() if x.strip()]
    arrays = []
    for key in ("g00_views", "g04_views"):
        with np.load(unit[key], allow_pickle=False) as payload:
            arrays.extend([row_l2(np.asarray(payload[name])).astype(np.float32) for name in VIEWS])
    if any(len(x) != int(unit["observation_count"]) for x in arrays) or len(ids) != len(arrays[0]):
        raise RuntimeError("opaque input observation mismatch")
    s00, s04 = sp.load_npz(unit["s00"]), sp.load_npz(unit["s04"])
    if s00.shape != (len(ids), len(ids)) or s04.shape != s00.shape:
        raise RuntimeError("opaque sparse input shape mismatch")
    return unit, config, ids, arrays, canonical_csr(s00), canonical_csr(s04)


def configure_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_snapshot() -> dict:
    return {
        "python_repr": repr(random.getstate()),
        "numpy_repr": repr(np.random.get_state()),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def train(unit_dir: Path, config_path: Path, output: Path, smoke: bool) -> None:
    start = time.perf_counter(); output.mkdir(parents=True, exist_ok=False)
    unit, config, ids, arrays, s00, s04 = load_contract(unit_dir, config_path)
    if config["recipe_id"] not in {"R%02d" % i for i in range(10)}:
        raise RuntimeError("unregistered recipe")
    epochs = 2 if smoke else int(config["epochs"])
    if not smoke and epochs != 160:
        raise RuntimeError("formal endpoint is not exactly 160 epochs")
    seed = int(config["seed"]); configure_seed(seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    torch.cuda.reset_peak_memory_stats(device)
    z00 = graph_summary({key: arrays[i] for i, key in enumerate(VIEWS)})
    z04 = graph_summary({key: arrays[i + 3] for i, key in enumerate(VIEWS)})
    _, _, rel = reliability_inputs(s00, s04, z00, z04, ids)
    recipe_losses = tuple(config["losses"])
    moe = config["fusion"] == "moe"
    semantic = "SEMANTIC" in recipe_losses
    model = AdaptiveFusion([x.shape[1] for x in arrays], int(unit["K"]), moe, semantic).to(device)
    targets = [torch.as_tensor(x, device=device) for x in arrays]
    reliability = torch.as_tensor(rel["scalars"], dtype=torch.float32, device=device)
    row_np, col_np, teacher_neighbors = sparse_relation_edges(arrays, ids, 20)
    rows = torch.as_tensor(row_np, dtype=torch.long, device=device)
    cols = torch.as_tensor(col_np, dtype=torch.long, device=device)
    pos_np, neg_np, mnn_audit = fixed_mnn_triplets(z00, z04, ids, config["recipe_id"], seed)
    positives = torch.as_tensor(pos_np, dtype=torch.long, device=device)
    negatives = torch.as_tensor(neg_np, dtype=torch.long, device=device)
    masks_np = deterministic_masks(len(ids), config["recipe_id"], seed)
    masks = [torch.as_tensor(x, dtype=torch.long, device=device) for x in masks_np]
    mask_flags = [torch.as_tensor(np.isin(np.arange(len(ids)), x), dtype=torch.bool, device=device)
                  for x in masks_np]
    semantic_target = semantic_keep = None
    pseudo_audit = None
    if semantic:
        if not unit["pseudo_partition"] or not unit["pseudo_affinity"]:
            raise RuntimeError("semantic recipe lacks locked pseudo target")
        labels = np.load(unit["pseudo_partition"], allow_pickle=False).astype(np.int64)
        affinity = sp.load_npz(unit["pseudo_affinity"])
        if len(labels) != len(ids) or affinity.shape != s00.shape:
            raise RuntimeError("pseudo target contract mismatch")
        canonical = np.unique(labels, return_inverse=True)[1].astype(np.int64)
        confidence = pseudo_confidence(affinity, canonical)
        keep_np = pseudo_keep(canonical, confidence)
        semantic_target = torch.as_tensor(canonical, dtype=torch.long, device=device)
        semantic_keep = torch.as_tensor(keep_np, dtype=torch.bool, device=device)
        pseudo_audit = {"partition_sha256": array_sha(labels),
                        "affinity_sha256": sparse_sha(affinity),
                        "keep_sha256": array_sha(keep_np.astype(np.uint8)),
                        "keep_count": int(keep_np.sum())}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    initial_rng = rng_snapshot(); curves = []
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output_value = model(targets, reliability, mask_flags if "MASK" in recipe_losses else None)
        components = loss_components(output_value, targets, recipe_losses, rows, cols,
                                     positives, negatives, masks, semantic_target,
                                     semantic_keep)
        loss = total_loss(components)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite formal loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step(); scheduler.step()
        row = {"epoch": epoch + 1, "total": float(loss.detach().cpu()),
               "learning_rate": float(optimizer.param_groups[0]["lr"])}
        for name, value in components.items(): row[name] = float(value.detach().cpu())
        curves.append(row)
    model.eval()
    with torch.no_grad(): final = model(targets, reliability, None)
    embedding = final["z"].detach().cpu().numpy().astype(np.float32)
    gates = final["gate_weights"].detach().cpu().numpy().astype(np.float32)
    checkpoint = {
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "config": config,
        "unit_contract": unit, "initial_rng": initial_rng, "final_rng": rng_snapshot(),
        "embedding_sha256": array_sha(embedding), "gate_sha256": array_sha(gates),
    }
    checkpoint_path = output / "model_final.pt"
    torch.save(checkpoint, checkpoint_path)
    atomic_npy(output / "embedding.npy", embedding); atomic_npy(output / "gate_weights.npy", gates)
    with (output / "loss_curve.csv").open("w", newline="") as handle:
        columns = sorted(set().union(*(row.keys() for row in curves)), key=lambda x: (x != "epoch", x))
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(curves)
    fixed = {
        "relation_rows_sha256": array_sha(row_np), "relation_cols_sha256": array_sha(col_np),
        "teacher_neighbor_sha256": [array_sha(x) for x in teacher_neighbors],
        "mask_sha256": [array_sha(x) for x in masks_np], "mnn": mnn_audit,
        "reliability_scalar_sha256": rel["audit"]["scalar_sha256"], "pseudo": pseudo_audit,
    }
    atomic_json(output / "fixed_indices.json", fixed)
    manifest = {
        "schema_version": 1, "status": "smoke_invalid_for_science" if smoke else "success",
        "unit_id": unit["unit_id"], "recipe_id": config["recipe_id"], "seed": seed,
        "epochs": epochs, "scientific_training": not smoke, "label_access": False,
        "fallback": False, "retry": False, "gpu_model": torch.cuda.get_device_name(0),
        "runtime_seconds": time.perf_counter() - start,
        "peak_gpu_mib": torch.cuda.max_memory_allocated(device) / 1048576.0,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha_file(checkpoint_path),
        "state_tensor_sha256": state_sha(model.state_dict()),
        "embedding_path": str(output / "embedding.npy"), "embedding_sha256": array_sha(embedding),
        "gate_path": str(output / "gate_weights.npy"), "gate_sha256": array_sha(gates),
        "gate_quantiles": {
            "expert_0": [float(x) for x in np.quantile(gates[:, 0], [0, .25, .5, .75, 1])],
            "expert_1": [float(x) for x in np.quantile(gates[:, 1], [0, .25, .5, .75, 1])],
        },
        "config_sha256": sha_file(config_path), "worker_input_sha256": sha_file(unit_dir / "worker_input.json"),
        "fixed_indices_sha256": sha_file(output / "fixed_indices.json"),
        "loss_curve_sha256": sha_file(output / "loss_curve.csv"),
    }
    atomic_json(output / "training_manifest.json", manifest)


def reload_verify(unit_dir: Path, config_path: Path, output: Path) -> None:
    unit, config, ids, arrays, s00, s04 = load_contract(unit_dir, config_path)
    checkpoint_path = output / "model_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    recipe_losses = tuple(config["losses"])
    model = AdaptiveFusion([x.shape[1] for x in arrays], int(unit["K"]),
                           config["fusion"] == "moe", "SEMANTIC" in recipe_losses).cuda()
    model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    z00 = graph_summary({key: arrays[i] for i, key in enumerate(VIEWS)})
    z04 = graph_summary({key: arrays[i + 3] for i, key in enumerate(VIEWS)})
    _, _, rel = reliability_inputs(s00, s04, z00, z04, ids)
    targets = [torch.as_tensor(x, device="cuda") for x in arrays]
    reliability = torch.as_tensor(rel["scalars"], dtype=torch.float32, device="cuda")
    with torch.no_grad(): value = model(targets, reliability, None)
    embedding = value["z"].detach().cpu().numpy().astype(np.float32)
    gates = value["gate_weights"].detach().cpu().numpy().astype(np.float32)
    saved_embedding = np.load(output / "embedding.npy", allow_pickle=False)
    saved_gates = np.load(output / "gate_weights.npy", allow_pickle=False)
    audit = {
        "status": "PASS" if np.array_equal(embedding, saved_embedding) and np.array_equal(gates, saved_gates) else "FAIL",
        "embedding_exact": bool(np.array_equal(embedding, saved_embedding)),
        "gate_exact": bool(np.array_equal(gates, saved_gates)),
        "embedding_max_abs": float(np.max(np.abs(embedding - saved_embedding))),
        "gate_max_abs": float(np.max(np.abs(gates - saved_gates))),
        "state_tensor_sha256": state_sha(model.state_dict()),
        "expected_state_tensor_sha256": json.loads((output / "training_manifest.json").read_text())["state_tensor_sha256"],
        "fresh_process": True, "label_access": False,
    }
    if audit["state_tensor_sha256"] != audit["expected_state_tensor_sha256"]:
        audit["status"] = "FAIL"
    atomic_json(output / "reload_forward_audit.json", audit)
    if audit["status"] != "PASS": raise RuntimeError("checkpoint reload forward mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "reload"))
    parser.add_argument("--unit-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mode == "train": train(args.unit_dir, args.config, args.output, args.smoke)
    else: reload_verify(args.unit_dir, args.config, args.output)


if __name__ == "__main__":
    main()
