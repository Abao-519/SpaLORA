#!/usr/bin/env python3
"""Fresh-process Night-7C semantic parity and label-free feature freeze."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha, sparse_sha  # noqa: E402
from SpaLORA.night7a_consensus import atomic_json, canonical_partition, sha256_file  # noqa: E402
from SpaLORA.night7b_adaptive import (  # noqa: E402
    AdaptiveFusion, VIEWS, fixed_mnn_triplets, graph_summary, reliability_inputs,
    row_l2, run_partition,
)
from SpaLORA.night7c_conflict import (  # noqa: E402
    conflict_rank, matching_quality, shared_neighbor_support,
)
from scripts.night7b_adapter_stage import endpoint_affinity  # noqa: E402
from scripts.night7b_train import configure_seed, load_contract, state_sha  # noqa: E402


RAW7B = Path("/root/autodl-fs/night7b_score_rnd_20260818")
RAW = Path("/root/autodl-fs/night7c_conflict_rnd_20260818")
HANDOFF7B = RAW7B / "official_compact/handoff"
OUT = REPO / "outputs/night7c_handoff"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def training_map() -> dict:
    result = {}
    for stage in ("R1", "R2"):
        locked = json.loads((HANDOFF7B / ("locked_%s_manifest.json" % stage)).read_text())
        for cell in locked["training_cells"]:
            if cell["recipe_id"] == "R02":
                result[cell["unit_id"]] = (stage, cell)
    require(len(result) == 30, "R02 training map is not 30 units")
    return result


def transform_map() -> dict:
    result = {}
    for stage in ("R1", "R2"):
        locked = json.loads((HANDOFF7B / ("locked_%s_manifest.json" % stage)).read_text())
        for cell in locked["transforms"]:
            if (cell["recipe_id"] == "R02" and cell["endpoint"] == "E1_ADAPTER_C06_MEAN"
                    and cell["head_id"] == "H01"):
                result[cell["unit_id"]] = cell
    require(len(result) == 30, "R02 E1/H01 transform map is not 30 units")
    return result


def cell(unit_id: str) -> None:
    start = time.perf_counter()
    train_stage, train = training_map()[unit_id]
    expected_transform = transform_map()[unit_id]
    config_path = Path(train["config_path"])
    unit_dir = RAW7B / "adapter_inputs" / unit_id
    unit, config, ids, arrays, s00, s04 = load_contract(unit_dir, config_path)
    require(config["recipe_id"] == "R02" and config["fusion"] == "equal",
            "P1 must use the exact R02 equal-fusion contract")
    require(tuple(config["losses"]) == ("RECON", "MNN"), "R02 loss contract mismatch")
    worker = Path(train["training_manifest"]["checkpoint_path"]).parent
    checkpoint = torch.load(worker / "model_final.pt", map_location="cuda")
    model = AdaptiveFusion([x.shape[1] for x in arrays], int(unit["K"]), False, False).cuda()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    require(state_sha(model.state_dict()) == train["training_manifest"]["state_tensor_sha256"],
            "fresh checkpoint state SHA mismatch")
    z00 = graph_summary({key: arrays[i] for i, key in enumerate(VIEWS)})
    z04 = graph_summary({key: arrays[i + 3] for i, key in enumerate(VIEWS)})
    _, _, rel = reliability_inputs(s00, s04, z00, z04, ids)
    targets = [torch.as_tensor(x, device="cuda") for x in arrays]
    reliability = torch.as_tensor(rel["scalars"], dtype=torch.float32, device="cuda")
    model.eval()
    with torch.no_grad():
        final = model(targets, reliability, None)
    embedding = final["z"].detach().cpu().numpy().astype(np.float32)
    gates = final["gate_weights"].detach().cpu().numpy().astype(np.float32)
    require(np.array_equal(embedding, np.load(worker / "embedding.npy", allow_pickle=False)),
            "fresh R02 embedding parity failure")
    require(np.array_equal(gates, np.load(worker / "gate_weights.npy", allow_pickle=False)),
            "fresh R02 gate parity failure")

    positives, negatives, mnn_audit = fixed_mnn_triplets(
        z00, z04, ids, "R02", int(config["seed"]))
    fixed = json.loads((worker / "fixed_indices.json").read_text())
    require(array_sha(positives) == fixed["mnn"]["positive_sha256"],
            "fixed MNN positive SHA mismatch")
    require(array_sha(negatives) == fixed["mnn"]["negative_sha256"],
            "fixed MNN negative SHA mismatch")

    configure_seed(int(config["seed"]))
    initial_model = AdaptiveFusion([x.shape[1] for x in arrays], int(unit["K"]), False, False).cuda()
    initial_model.train()
    # The checkpoint stores the exact pre-forward RNG state.  Restoring it
    # avoids device/kernel-dependent CUDA RNG advancement while preserving the
    # registered deterministic initialization and first-forward semantics.
    torch.set_rng_state(checkpoint["initial_rng"]["torch_cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint["initial_rng"]["torch_cuda"]])
    initial = initial_model(targets, reliability, None)
    positive_t = torch.as_tensor(positives, dtype=torch.long, device="cuda")
    negative_t = torch.as_tensor(negatives, dtype=torch.long, device="cuda")
    m_initial = float(F.triplet_margin_loss(
        initial["z"], initial["projected"][3][positive_t],
        initial["projected"][3][negative_t], margin=.5).detach().cpu())
    curve_first = float(pd.read_csv(worker / "loss_curve.csv").iloc[0]["MNN"])
    m_error = abs(m_initial - curve_first)
    require(m_error <= 1e-7,
            "initial MNN parity exceeds 1e-7: recomputed=%.17g saved=%.17g error=%.17g"
            % (m_initial, curve_first, m_error))

    u00 = final["u00"].detach().cpu().numpy().astype(np.float64)
    u04 = final["u04"].detach().cpu().numpy().astype(np.float64)
    conflict, rank_c = conflict_rank(u00, u04)
    quality = matching_quality(u00, u04, positives, ids)
    support = shared_neighbor_support(u00, u04, ids, k=10)

    c06 = sp.load_npz(RAW7B / "source" / unit_id / "c06_affinity.npz")
    affinity = endpoint_affinity("E1_ADAPTER_C06_MEAN", embedding, c06, ids)
    labels, partition = run_partition("H01", affinity, int(unit["K"]), [])
    require(sparse_sha(affinity) == expected_transform["canonical_affinity_sha256"],
            "fresh R02 endpoint affinity parity failure")
    require(array_sha(canonical_partition(labels)) == expected_transform["canonical_partition_sha256"],
            "fresh R02 endpoint partition parity failure")

    target = RAW / "p1_features" / unit_id
    target.mkdir(parents=True, exist_ok=False)
    np.savez(target / "features.npz", m_initial=np.asarray(m_initial, dtype=np.float64),
             conflict=conflict, rank_c=rank_c, quality=quality, support=support,
             positive=positives, negative=negatives)
    manifest = {
        "schema_version": 1, "status": "PASS", "unit_id": unit_id,
        "label_access": False, "fresh_process": True,
        "checkpoint_sha256": train["training_manifest"]["checkpoint_sha256"],
        "checkpoint_state_sha256": train["training_manifest"]["state_tensor_sha256"],
        "embedding_exact": True, "gate_exact": True,
        "endpoint_affinity_exact": True, "endpoint_partition_exact": True,
        "m_initial": m_initial, "loss_curve_first_mnn": curve_first,
        "m_initial_abs_error": m_error,
        "fixed_positive_sha256": array_sha(positives),
        "fixed_negative_sha256": array_sha(negatives),
        "conflict_sha256": array_sha(conflict), "rank_c_sha256": array_sha(rank_c),
        "quality_sha256": array_sha(quality), "support_sha256": array_sha(support),
        "feature_file_sha256": sha256_file(target / "features.npz"),
        "feature_summary": {
            "conflict": [float(conflict.min()), float(conflict.mean()), float(conflict.max())],
            "rank_c": [float(rank_c.min()), float(rank_c.mean()), float(rank_c.max())],
            "quality": [float(quality.min()), float(quality.mean()), float(quality.max())],
            "support": [float(support.min()), float(support.mean()), float(support.max())],
        },
        "canonical_affinity_sha256": sparse_sha(affinity),
        "canonical_partition_sha256": array_sha(canonical_partition(labels)),
        "gpu_model": torch.cuda.get_device_name(0),
        "runtime_seconds": time.perf_counter() - start,
    }
    atomic_json(target / "manifest.json", manifest)


def driver() -> None:
    feature_root = RAW / "p1_features"
    require(not feature_root.exists(), "P1 feature root already exists")
    feature_root.mkdir(parents=True)
    units = list(csv.DictReader((HANDOFF7B / "source_unit_index.csv").open(newline="")))
    rows = []
    for source in units:
        command = [sys.executable, str(Path(__file__).resolve()), "cell", "--unit-id", source["unit_id"]]
        completed = subprocess.run(command, cwd=REPO, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log = feature_root / source["unit_id"] / "cell.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout + completed.stderr)
        require(completed.returncode == 0, "P1 cell failed: %s" % source["unit_id"])
        manifest_path = feature_root / source["unit_id"] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        rows.append({
            "ordinal": int(source["ordinal"]), "unit_id": source["unit_id"],
            "dataset": source["dataset"], "seed": int(source["seed"]),
            "status": manifest["status"], "label_access": False,
            "m_initial": manifest["m_initial"],
            "m_initial_abs_error": manifest["m_initial_abs_error"],
            "feature_file": str(feature_root / source["unit_id"] / "features.npz"),
            "feature_file_sha256": manifest["feature_file_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
            "runtime_seconds": manifest["runtime_seconds"],
        })
    require(len(rows) == 30 and all(x["status"] == "PASS" for x in rows),
            "P1 did not pass 30/30 units")
    with (OUT / "routing_feature_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {
        "schema_version": 1, "status": "PASS", "label_access": False,
        "fresh_reload_endpoint_parity": "30/30",
        "m_initial_parity_within_1e-7": "30/30",
        "feature_files_locked": 30,
        "routing_feature_manifest_sha256": sha256_file(OUT / "routing_feature_manifest.csv"),
        "maximum_m_initial_abs_error": max(x["m_initial_abs_error"] for x in rows),
    }
    atomic_json(OUT / "p1_semantic_contract.json", result)
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("driver", "cell"))
    parser.add_argument("--unit-id")
    args = parser.parse_args()
    if args.mode == "driver":
        driver()
    else:
        require(bool(args.unit_id), "cell requires unit id")
        cell(args.unit_id)


if __name__ == "__main__":
    main()
