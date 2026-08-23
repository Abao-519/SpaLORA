#!/usr/bin/env python3
"""Train and replay one Night-15A raw-feature unified MCDF candidate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from SpaLORA.night14a_tcf import seed_everything, state_sha256  # noqa: E402
from SpaLORA.night14b_atac import spatial_operator  # noqa: E402
from SpaLORA.night15a_mcdf import (  # noqa: E402
    MCDFUnifiedCore,
    array_sha256,
    mcdf_unsupervised_loss,
)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def torch_graph(operator: sp.spmatrix, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    coo = operator.tocoo()
    edge_index = torch.as_tensor(
        np.vstack((coo.row, coo.col)), dtype=torch.long, device=device
    )
    edge_weight = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return edge_index, edge_weight


def resolve_config(grid_path: Path, candidate_id: str) -> dict:
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    candidate = [item for item in grid["candidates"] if item["candidate_id"] == candidate_id]
    if len(candidate) != 1:
        raise ValueError("candidate ID is not unique in grid")
    item = copy.deepcopy(candidate[0])
    model_config = {
        key: item[key]
        for key in (
            "mode",
            "gate_hidden",
            "initial_residual_strength",
            "support_power",
            "joint_power",
        )
    }
    if "modality_dropout_probability" in item:
        model_config["modality_dropout_probability"] = float(
            item["modality_dropout_probability"]
        )
    model_config["base_config"] = copy.deepcopy(grid["base_config"])
    return {
        **copy.deepcopy(grid["shared_training"]),
        "candidate_id": item["candidate_id"],
        "mechanism_family": item["mechanism_family"],
        "graph_k": int(item["graph_k"]),
        "model_config": model_config,
        "loss_weights": copy.deepcopy(item["loss_weights"]),
    }


def load_input(path: Path) -> dict:
    value = np.load(path, allow_pickle=False)
    required = {"x1", "x2", "coordinates", "ids"}
    if not required.issubset(set(value.files)):
        raise ValueError("preprocessed archive schema incomplete")
    result = {
        "x1": np.asarray(value["x1"], dtype=np.float32),
        "x2": np.asarray(value["x2"], dtype=np.float32),
        "coordinates": np.asarray(value["coordinates"], dtype=np.float64),
        "ids": np.asarray(value["ids"], dtype=str),
    }
    n = len(result["ids"])
    if result["x1"].shape[0] != n or result["x2"].shape[0] != n:
        raise ValueError("input observation count mismatch")
    if result["coordinates"].shape != (n, 2):
        raise ValueError("coordinate shape mismatch")
    if not all(np.isfinite(result[key]).all() for key in ("x1", "x2", "coordinates")):
        raise ValueError("input contains non-finite values")
    return result


def forward_numpy(
    model: MCDFUnifiedCore,
    x1: torch.Tensor,
    x2: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> dict:
    model.eval()
    with torch.no_grad():
        output = model(x1, x2, edge_index, edge_weight)
    return {
        "z1": output["z1"].detach().cpu().numpy().astype(np.float32),
        "z2": output["z2"].detach().cpu().numpy().astype(np.float32),
        "base_fused": output["fused"].detach().cpu().numpy().astype(np.float32),
        "mcdf": output["mcdf"].detach().cpu().numpy().astype(np.float32),
        "expert_gate": output["expert_gate"].detach().cpu().numpy().astype(np.float32),
        "scale_weights": output["scale_weights"].detach().cpu().numpy().astype(np.float32),
    }


def instantiate(config: Mapping[str, object], payload: Mapping[str, np.ndarray], device):
    model = MCDFUnifiedCore(
        payload["x1"].shape[1], payload["x2"].shape[1], config["model_config"]
    ).to(device)
    graph = spatial_operator(payload["coordinates"], int(config["graph_k"]))
    edge_index, edge_weight = torch_graph(graph, device)
    x1 = torch.as_tensor(payload["x1"], dtype=torch.float32, device=device)
    x2 = torch.as_tensor(payload["x2"], dtype=torch.float32, device=device)
    return model, x1, x2, edge_index, edge_weight, graph


def train(
    input_path: Path,
    grid_path: Path,
    candidate_id: str,
    seed: int,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    payload = load_input(input_path)
    config = resolve_config(grid_path, candidate_id)
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Night-15A train lane requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    model, x1, x2, edge_index, edge_weight, graph = instantiate(config, payload, device)
    initial_sha = state_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    trace = []
    first_gradient = None
    dropout_rng = np.random.default_rng(int(seed) + 150_822)
    dropout_probability = float(
        config["model_config"].get("modality_dropout_probability", 0.0)
    )
    dropout_counts = {"none": 0, "rna": 0, "atac": 0}
    started = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for step in range(int(config["steps"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        draw = float(dropout_rng.random())
        if draw < dropout_probability:
            # Mask exactly one assay while preserving the other.  Targets below
            # remain the unmasked tensors, so this is a cross-modal masked
            # reconstruction objective rather than input corruption at inference.
            if int(dropout_rng.integers(0, 2)) == 0:
                model_x1, model_x2 = torch.zeros_like(x1), x2
                dropout_counts["rna"] += 1
            else:
                model_x1, model_x2 = x1, torch.zeros_like(x2)
                dropout_counts["atac"] += 1
        else:
            model_x1, model_x2 = x1, x2
            dropout_counts["none"] += 1
        model_output = model(model_x1, model_x2, edge_index, edge_weight)
        loss, audit = mcdf_unsupervised_loss(
            model, model_output, x1, x2, edge_index, config["loss_weights"]
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Night-15A loss")
        loss.backward()
        norm = torch.sqrt(
            sum(
                (
                    parameter.grad.detach().square().sum()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ),
                torch.zeros((), device=device),
            )
        )
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise RuntimeError("zero or non-finite Night-15A gradient")
        if first_gradient is None:
            first_gradient = float(norm.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        optimizer.step()
        if step == 0 or step == int(config["steps"]) - 1 or (step + 1) % 50 == 0:
            trace.append({"step": step + 1, "gradient_norm": float(norm.detach().cpu()), **audit})
    end_event.record()
    torch.cuda.synchronize(device)
    gpu_seconds = float(start_event.elapsed_time(end_event) / 1000.0)
    final_sha = state_sha256(model.state_dict())
    if final_sha == initial_sha:
        raise RuntimeError("Night-15A parameters did not change")
    views = forward_numpy(model, x1, x2, edge_index, edge_weight)
    archive = output / "views.npz"
    np.savez_compressed(archive, ids=payload["ids"], coordinates=payload["coordinates"], **views)
    checkpoint = output / "checkpoint.pt"
    temporary = output / "checkpoint.pt.tmp"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "config_sha256": canonical_sha256(config),
            "seed": int(seed),
            "input_sha256": file_sha256(input_path),
        },
        temporary,
    )
    os.replace(str(temporary), str(checkpoint))
    audit = {
        "candidate_id": candidate_id,
        "mechanism_family": config["mechanism_family"],
        "seed": int(seed),
        "optimizer_steps": int(config["steps"]),
        "training_labels_read": 0,
        "labels_in_loss_gradient_or_checkpoint_selection": False,
        "checkpoint_selection": "FINAL_REGISTERED_STEP",
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "input_shapes": {"rna": list(payload["x1"].shape), "atac": list(payload["x2"].shape)},
        "coordinate_shape": list(payload["coordinates"].shape),
        "ordered_id_count": len(payload["ids"]),
        "graph_shape": list(graph.shape),
        "graph_nnz": int(graph.nnz),
        "dense_n_by_n_count": 0,
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "first_gradient_norm": first_gradient,
        "initial_state_sha256": initial_sha,
        "final_state_sha256": final_sha,
        "parameters_changed": True,
        "config": config,
        "config_sha256": canonical_sha256(config),
        "checkpoint_sha256": file_sha256(checkpoint),
        "views_sha256": file_sha256(archive),
        "view_hashes": {key: array_sha256(value) for key, value in views.items()},
        "gate_mean": views["expert_gate"].mean(axis=0).astype(float).tolist(),
        "scale_weights": views["scale_weights"].astype(float).tolist(),
        "trace": trace,
        "modality_dropout_probability": dropout_probability,
        "modality_dropout_step_counts": dropout_counts,
        "wall_seconds": time.perf_counter() - started,
        "gpu_seconds": gpu_seconds,
        "peak_gpu_mib": torch.cuda.max_memory_allocated(device) / 1048576.0,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    atomic_json(output / "training_audit.json", audit)
    del model, x1, x2, edge_index, edge_weight
    torch.cuda.empty_cache()
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "replay",
            "--input",
            str(input_path),
            "--output",
            str(output),
        ],
        cwd=str(REPO),
        check=True,
    )


def replay(input_path: Path, output: Path) -> None:
    payload = load_input(input_path)
    checkpoint_path = output / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, x1, x2, edge_index, edge_weight, _ = instantiate(config, payload, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    actual = forward_numpy(model, x1, x2, edge_index, edge_weight)
    expected = np.load(output / "views.npz", allow_pickle=False)
    rows = []
    for key, value in actual.items():
        rows.append(
            {
                "key": key,
                "shape": list(value.shape),
                "actual_sha256": array_sha256(value),
                "expected_sha256": array_sha256(expected[key]),
                "byte_exact": bool(np.array_equal(value, expected[key])),
                "max_abs_error": float(np.max(np.abs(value - expected[key]))),
            }
        )
    atomic_json(
        output / "fresh_process_reload.json",
        {
            "fresh_process": True,
            "strict_state_dict": True,
            "row_count": len(rows),
            "exact_count": sum(item["byte_exact"] for item in rows),
            "all_exact": all(item["byte_exact"] for item in rows),
            "numerical_tolerance": 5e-6,
            "numerically_close_count": sum(
                item["max_abs_error"] <= 5e-6 for item in rows
            ),
            "all_numerically_close": all(
                item["max_abs_error"] <= 5e-6 for item in rows
            ),
            "rows": rows,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--input", required=True)
    train_parser.add_argument("--grid", required=True)
    train_parser.add_argument("--candidate", required=True)
    train_parser.add_argument("--seed", required=True, type=int)
    train_parser.add_argument("--output", required=True)
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--input", required=True)
    replay_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "train":
        train(Path(args.input), Path(args.grid), args.candidate, args.seed, Path(args.output))
    else:
        replay(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
