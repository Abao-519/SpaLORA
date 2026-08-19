#!/usr/bin/env python3
"""Night-7C replay portability recovery: historical-order initial-forward audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha  # noqa: E402
from SpaLORA.night7a_consensus import atomic_json, sha256_file  # noqa: E402
from SpaLORA.night7b_adaptive import (  # noqa: E402
    AdaptiveFusion, VIEWS, deterministic_masks, fixed_mnn_triplets,
    graph_summary, loss_components, reliability_inputs, sparse_relation_edges,
)
from scripts.night7b_train import configure_seed, load_contract, rng_snapshot, state_sha  # noqa: E402

RAW7B = Path("/root/autodl-fs/night7b_score_rnd_20260818")
RAW = Path("/root/autodl-fs/night7c_replay_recovery_20260818")
HANDOFF7B = RAW7B / "official_compact/handoff"
OUT = REPO / "outputs/night7c_replay_recovery_handoff"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def tensor_sha(value: torch.Tensor) -> str:
    a = value.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes(order="C")); return h.hexdigest()


def rng_hash(snapshot: dict) -> dict:
    return {
        "python_sha256": hashlib.sha256(snapshot["python_repr"].encode()).hexdigest(),
        "numpy_sha256": hashlib.sha256(snapshot["numpy_repr"].encode()).hexdigest(),
        "torch_cpu_sha256": tensor_sha(snapshot["torch_cpu"]),
        "torch_cuda_sha256": [tensor_sha(x) for x in snapshot["torch_cuda"]],
    }


def training_map() -> dict:
    result = {}
    for stage in ("R1", "R2"):
        path = HANDOFF7B / f"locked_{stage}_manifest.json"
        locked = json.loads(path.read_text())
        for cell in locked["training_cells"]:
            if cell["recipe_id"] == "R02":
                result[cell["unit_id"]] = {"stage": stage, "cell": cell, "locked_path": path}
    require(len(result) == 30, "R02 training map is not exactly 30")
    return result


def source_rows() -> list[dict]:
    rows = list(csv.DictReader((HANDOFF7B / "source_unit_index.csv").open(newline="")))
    require(len(rows) == 30 and [int(x["ordinal"]) for x in rows] == list(range(1, 31)),
            "source order is not the locked 1..30 order")
    return rows


def alpha_values(m: float) -> tuple[float, float, bool]:
    return (float(np.clip((m - .20) / .15, 0, 1)),
            float(np.clip((m - .25) / .10, 0, 1)), bool(m >= .30))


def environment() -> dict:
    return {
        "gpu_model": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
    }


def historical_context(unit_id: str):
    entry = training_map()[unit_id]; cell = entry["cell"]
    manifest = cell["training_manifest"]
    worker = Path(manifest["checkpoint_path"]).parent
    config_path = Path(cell["config_path"])
    unit_dir = RAW7B / "adapter_inputs" / unit_id
    require(sha256_file(worker / "loss_curve.csv") == manifest["loss_curve_sha256"], "loss curve SHA mismatch")
    require(sha256_file(worker / "training_manifest.json") == sha256_file(worker / "training_manifest.json"), "manifest unreadable")
    with (worker / "loss_curve.csv").open(newline="") as handle:
        first = next(csv.DictReader(handle))
    historical = float(first["MNN"])
    return entry, manifest, worker, config_path, unit_dir, historical


def construct_first_forward(unit_id: str, old_order: bool) -> dict:
    start = time.perf_counter()
    entry, manifest, worker, config_path, unit_dir, historical = historical_context(unit_id)

    # CPU-only read occurs before any CUDA context or tensor construction.  The
    # checkpoint is used solely for historical RNG hashes and, in old-order
    # diagnostics, the explicitly registered prior final-forward disturbance.
    checkpoint = torch.load(worker / "model_final.pt", map_location="cpu")
    unit, config, ids, arrays, s00, s04 = load_contract(unit_dir, config_path)
    require(config["recipe_id"] == "R02" and config["fusion"] == "equal", "R02 contract mismatch")
    require(tuple(config["losses"]) == ("RECON", "MNN"), "R02 loss contract mismatch")
    seed = int(config["seed"])

    if old_order:
        # Reproduce the invalid Night-7C P1 ordering only as a four-shape
        # diagnostic.  It is never an authority source.
        final_model = AdaptiveFusion([x.shape[1] for x in arrays], int(unit["K"]), False, False).cuda()
        final_model.load_state_dict(checkpoint["model_state"], strict=True); final_model.eval()
        z00_old = graph_summary({key: arrays[i] for i, key in enumerate(VIEWS)})
        z04_old = graph_summary({key: arrays[i + 3] for i, key in enumerate(VIEWS)})
        _, _, rel_old = reliability_inputs(s00, s04, z00_old, z04_old, ids)
        with torch.no_grad():
            final_model([torch.as_tensor(x, device="cuda") for x in arrays],
                        torch.as_tensor(rel_old["scalars"], dtype=torch.float32, device="cuda"), None)
        del final_model

    # Historical Night-7B order starts here.
    configure_seed(seed)
    z00 = graph_summary({key: arrays[i] for i, key in enumerate(VIEWS)})
    z04 = graph_summary({key: arrays[i + 3] for i, key in enumerate(VIEWS)})
    _, _, rel = reliability_inputs(s00, s04, z00, z04, ids)
    model = AdaptiveFusion([x.shape[1] for x in arrays], int(unit["K"]), False, False).cuda()
    initial_state = state_sha(model.state_dict())
    targets = [torch.as_tensor(x, device="cuda") for x in arrays]
    reliability = torch.as_tensor(rel["scalars"], dtype=torch.float32, device="cuda")
    row_np, col_np, _ = sparse_relation_edges(arrays, ids, 20)
    rows = torch.as_tensor(row_np, dtype=torch.long, device="cuda")
    cols = torch.as_tensor(col_np, dtype=torch.long, device="cuda")
    pos_np, neg_np, mnn_audit = fixed_mnn_triplets(z00, z04, ids, "R02", seed)
    positives = torch.as_tensor(pos_np, dtype=torch.long, device="cuda")
    negatives = torch.as_tensor(neg_np, dtype=torch.long, device="cuda")
    masks_np = deterministic_masks(len(ids), "R02", seed)
    masks = [torch.as_tensor(x, dtype=torch.long, device="cuda") for x in masks_np]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160)
    pre_rng = rng_snapshot(); pre_rng_sha = rng_hash(pre_rng)
    model.train(); optimizer.zero_grad(set_to_none=True)
    value = model(targets, reliability, None)
    components = loss_components(value, targets, ("RECON", "MNN"), rows, cols,
                                 positives, negatives, masks, None, None)
    current = float(components["MNN"].detach().cpu())
    require(np.isfinite(current), "non-finite first MNN")
    fixed = json.loads((worker / "fixed_indices.json").read_text())
    require(array_sha(pos_np) == fixed["mnn"]["positive_sha256"], "positive index mismatch")
    require(array_sha(neg_np) == fixed["mnn"]["negative_sha256"], "negative index mismatch")
    a2h, a3h, d4h = alpha_values(historical); a2c, a3c, d4c = alpha_values(current)
    return {
        "schema_version": 1, "status": "PASS", "unit_id": unit_id,
        "order": "night7c_old_order" if old_order else "night7b_historical_order",
        "label_access": False, "formal_training": 0, "formal_transform": 0,
        "historical_m_initial": historical, "recomputed_m_initial": current,
        "absolute_error": abs(current - historical),
        "relative_error": abs(current - historical) / max(abs(historical), 1e-12),
        "t02_alpha_historical": a2h, "t02_alpha_current": a2c,
        "t02_alpha_abs_delta": abs(a2c - a2h),
        "t03_alpha_historical": a3h, "t03_alpha_current": a3c,
        "t03_alpha_abs_delta": abs(a3c - a3h),
        "t04_historical": d4h, "t04_current": d4c,
        "initial_state_tensor_sha256": initial_state,
        "pre_forward_rng_sha256": pre_rng_sha,
        "historical_pre_forward_rng_sha256": rng_hash(checkpoint["initial_rng"]),
        "fixed_positive_sha256": array_sha(pos_np),
        "fixed_negative_sha256": array_sha(neg_np),
        "fixed_indices_file_sha256": sha256_file(worker / "fixed_indices.json"),
        "loss_curve_path": str(worker / "loss_curve.csv"),
        "loss_curve_sha256": sha256_file(worker / "loss_curve.csv"),
        "training_manifest_path": str(worker / "training_manifest.json"),
        "training_manifest_sha256": sha256_file(worker / "training_manifest.json"),
        "worker_input_sha256": sha256_file(unit_dir / "worker_input.json"),
        "config_sha256": sha256_file(config_path),
        "mnn_audit": mnn_audit, "environment": environment(),
        "runtime_seconds": time.perf_counter() - start,
    }


def cell(unit_id: str, output: Path, old_order: bool) -> None:
    require(not output.exists(), "replay output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    value = construct_first_forward(unit_id, old_order)
    atomic_json(output, value)


def authority_manifest() -> list[dict]:
    mapping = training_map(); result = []
    for src in source_rows():
        unit_id = src["unit_id"]
        entry, manifest, worker, config, unit_dir, historical = historical_context(unit_id)
        base = {
            "ordinal": int(src["ordinal"]), "unit_id": unit_id,
            "historical_loss_path": str(worker / "loss_curve.csv"),
            "historical_loss_sha256": sha256_file(worker / "loss_curve.csv"),
            "training_manifest_path": str(worker / "training_manifest.json"),
            "training_manifest_sha256": sha256_file(worker / "training_manifest.json"),
            "fixed_indices_path": str(worker / "fixed_indices.json"),
            "fixed_indices_sha256": sha256_file(worker / "fixed_indices.json"),
            "worker_input_sha256": sha256_file(unit_dir / "worker_input.json"),
            "config_sha256": sha256_file(config), "m_initial_authority": historical,
            "authority": "night7b_sha_locked_loss_curve_first_row_MNN",
        }
        base["authority_row_sha256"] = hashlib.sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result.append(base)
    require(len(result) == 30, "historical authority manifest not 30 rows")
    return result


def driver() -> None:
    root = RAW / "p1a_historical_order_replay"
    require(not root.exists(), "P1A root already exists")
    root.mkdir(parents=True)
    auth = authority_manifest()
    with (root / "historical_m_initial_authority.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(auth[0])); writer.writeheader(); writer.writerows(auth)

    rows = []
    for src in source_rows():
        unit_id = src["unit_id"]
        repeat_values = []
        for repeat in (1, 2):
            target = root / unit_id / f"historical_order_repeat{repeat}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "cell", "--unit-id", unit_id,
                   "--output", str(target)]
            done = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (root / unit_id).mkdir(parents=True, exist_ok=True)
            (root / unit_id / f"repeat{repeat}.log").write_text(done.stdout + done.stderr)
            require(done.returncode == 0, f"historical-order replay failed: {unit_id} repeat {repeat}")
            repeat_values.append(json.loads(target.read_text()))
        a, b = repeat_values
        rows.append({
            "ordinal": int(src["ordinal"]), "unit_id": unit_id,
            "historical_m_initial": a["historical_m_initial"],
            "repeat1_m_initial": a["recomputed_m_initial"], "repeat2_m_initial": b["recomputed_m_initial"],
            "within_current_abs_error": abs(a["recomputed_m_initial"] - b["recomputed_m_initial"]),
            "historical_abs_error_max": max(a["absolute_error"], b["absolute_error"]),
            "historical_relative_error_max": max(a["relative_error"], b["relative_error"]),
            "t02_alpha_abs_delta_max": max(a["t02_alpha_abs_delta"], b["t02_alpha_abs_delta"]),
            "t03_alpha_abs_delta_max": max(a["t03_alpha_abs_delta"], b["t03_alpha_abs_delta"]),
            "t04_mismatch": int(a["t04_current"] != a["t04_historical"] or b["t04_current"] != b["t04_historical"]),
            "initial_state_repeat_match": a["initial_state_tensor_sha256"] == b["initial_state_tensor_sha256"],
            "rng_repeat_match": a["pre_forward_rng_sha256"] == b["pre_forward_rng_sha256"],
            "fixed_indices_match": (a["fixed_positive_sha256"] == b["fixed_positive_sha256"] and
                                      a["fixed_negative_sha256"] == b["fixed_negative_sha256"]),
        })

    first_by_shape = []
    seen = set()
    for src in source_rows():
        family = src["dataset"]
        if family in seen:
            continue
        seen.add(family); unit_id = src["unit_id"]
        target = root / "old_order_diagnostic" / f"{unit_id}.json"
        cmd = [sys.executable, str(Path(__file__).resolve()), "cell", "--unit-id", unit_id,
               "--output", str(target), "--old-order"]
        done = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        target.parent.mkdir(parents=True, exist_ok=True)
        (target.parent / f"{unit_id}.log").write_text(done.stdout + done.stderr)
        require(done.returncode == 0, f"old-order shape diagnostic failed: {unit_id}")
        first_by_shape.append(json.loads(target.read_text()))
    require(len(first_by_shape) == 4, "old-order diagnostic is not four shapes")

    with (root / "replay_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    gates = {
        "all_30_historical_order_repeats_finite": len(rows) == 30 and all(np.isfinite(x["repeat1_m_initial"]) and np.isfinite(x["repeat2_m_initial"]) for x in rows),
        "within_current_hardware_repeat_absolute_error_max": max(x["within_current_abs_error"] for x in rows),
        "historical_vs_current_absolute_error_max": max(x["historical_abs_error_max"] for x in rows),
        "historical_vs_current_relative_error_max": max(x["historical_relative_error_max"] for x in rows),
        "T04_hard_decision_mismatch_count": sum(x["t04_mismatch"] for x in rows),
        "T02_T03_alpha_absolute_delta_max": max(max(x["t02_alpha_abs_delta_max"], x["t03_alpha_abs_delta_max"]) for x in rows),
        "reconstructed_initial_state_sha_repeat_mismatch_count": sum(not x["initial_state_repeat_match"] for x in rows),
        "pre_forward_rng_repeat_mismatch_count": sum(not x["rng_repeat_match"] for x in rows),
        "fixed_index_or_formula_mismatch_count": sum(not x["fixed_indices_match"] for x in rows),
    }
    passed = (gates["all_30_historical_order_repeats_finite"] and
              gates["within_current_hardware_repeat_absolute_error_max"] <= 1e-7 and
              gates["historical_vs_current_absolute_error_max"] <= .001 and
              gates["historical_vs_current_relative_error_max"] <= .01 and
              gates["T04_hard_decision_mismatch_count"] == 0 and
              gates["T02_T03_alpha_absolute_delta_max"] <= .01 and
              gates["reconstructed_initial_state_sha_repeat_mismatch_count"] == 0 and
              gates["pre_forward_rng_repeat_mismatch_count"] == 0 and
              gates["fixed_index_or_formula_mismatch_count"] == 0)
    result = {
        "schema_version": 1,
        "status": "PASS_BY_LOCKED_HISTORICAL_FEATURE_AUTHORITY" if passed else "BLOCKED_REPLAY_PORTABILITY_RECOVERY",
        "formal_training": 0, "formal_transforms": 0, "label_access": False,
        "units": 30, "independent_replays": 60, "old_order_shape_diagnostics": 4,
        "routing_m_initial_source": "night7b_sha_locked_loss_curve_first_row_MNN",
        "current_hardware_replay_used_for_routing": False,
        "historical_authority_manifest_sha256": sha256_file(root / "historical_m_initial_authority.csv"),
        "replay_summary_sha256": sha256_file(root / "replay_summary.csv"),
        "gates": gates,
    }
    atomic_json(OUT / "p1a_replay_portability_contract.json", result)
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit(42)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("driver", "cell"))
    parser.add_argument("--unit-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--old-order", action="store_true")
    args = parser.parse_args()
    if args.mode == "driver":
        driver()
    else:
        require(bool(args.unit_id) and args.output is not None, "cell arguments missing")
        cell(args.unit_id, args.output, args.old_order)


if __name__ == "__main__":
    main()
