#!/usr/bin/env python3
"""Fresh-process final checkpoint parity and label-free feature freeze."""
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
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha, sparse_sha  # noqa: E402
from SpaLORA.night7a_consensus import atomic_json, canonical_partition, sha256_file  # noqa: E402
from SpaLORA.night7b_adaptive import (  # noqa: E402
    AdaptiveFusion, VIEWS, fixed_mnn_triplets, graph_summary, reliability_inputs,
    row_l2, run_partition,
)
from SpaLORA.night7c_conflict import conflict_rank, matching_quality, shared_neighbor_support  # noqa: E402
from scripts.night7b_adapter_stage import endpoint_affinity  # noqa: E402
from scripts.night7b_train import load_contract, state_sha  # noqa: E402
from scripts.night7c_p1 import training_map, transform_map  # noqa: E402

RAW7B = Path("/root/autodl-fs/night7b_score_rnd_20260818")
RAW = Path("/root/autodl-fs/night7c_replay_recovery_20260818")
HANDOFF7B = RAW7B / "official_compact/handoff"
OUT = REPO / "outputs/night7c_replay_recovery_handoff"
AUTHORITY = RAW / "p1a_historical_order_replay/historical_m_initial_authority.csv"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def authority_map() -> dict[str, dict]:
    rows = list(csv.DictReader(AUTHORITY.open(newline="")))
    require(len(rows) == 30 and len({x["unit_id"] for x in rows}) == 30,
            "historical authority map is not 30 unique rows")
    return {x["unit_id"]: x for x in rows}


def cell(unit_id: str) -> None:
    start = time.perf_counter()
    train_stage, train = training_map()[unit_id]
    expected_transform = transform_map()[unit_id]
    config_path = Path(train["config_path"])
    unit_dir = RAW7B / "adapter_inputs" / unit_id
    unit, config, ids, arrays, s00, s04 = load_contract(unit_dir, config_path)
    require(config["recipe_id"] == "R02" and config["fusion"] == "equal", "R02 contract mismatch")
    require(tuple(config["losses"]) == ("RECON", "MNN"), "R02 loss contract mismatch")
    worker = Path(train["training_manifest"]["checkpoint_path"]).parent

    # This fresh process performs final-state parity only.  It does not
    # reconstruct or forward the initial model, which is P1A's separate role.
    checkpoint = torch.load(worker / "model_final.pt", map_location="cuda")
    model = AdaptiveFusion([x.shape[1] for x in arrays], int(unit["K"]), False, False).cuda()
    model.load_state_dict(checkpoint["model_state"], strict=True); model.eval()
    require(state_sha(model.state_dict()) == train["training_manifest"]["state_tensor_sha256"],
            "final state tensor SHA mismatch")
    z00 = graph_summary({key: arrays[i] for i, key in enumerate(VIEWS)})
    z04 = graph_summary({key: arrays[i + 3] for i, key in enumerate(VIEWS)})
    _, _, rel = reliability_inputs(s00, s04, z00, z04, ids)
    targets = [torch.as_tensor(x, device="cuda") for x in arrays]
    reliability = torch.as_tensor(rel["scalars"], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        final = model(targets, reliability, None)
    embedding = final["z"].detach().cpu().numpy().astype(np.float32)
    gates = final["gate_weights"].detach().cpu().numpy().astype(np.float32)
    require(np.array_equal(embedding, np.load(worker / "embedding.npy", allow_pickle=False)),
            "R02 embedding exact parity failure")
    require(np.array_equal(gates, np.load(worker / "gate_weights.npy", allow_pickle=False)),
            "R02 gate exact parity failure")

    positives, negatives, _ = fixed_mnn_triplets(z00, z04, ids, "R02", int(config["seed"]))
    fixed = json.loads((worker / "fixed_indices.json").read_text())
    require(array_sha(positives) == fixed["mnn"]["positive_sha256"], "positive index SHA mismatch")
    require(array_sha(negatives) == fixed["mnn"]["negative_sha256"], "negative index SHA mismatch")

    u00 = final["u00"].detach().cpu().numpy().astype(np.float64)
    u04 = final["u04"].detach().cpu().numpy().astype(np.float64)
    conflict, rank_c = conflict_rank(u00, u04)
    quality = matching_quality(u00, u04, positives, ids)
    support = shared_neighbor_support(u00, u04, ids, k=10)
    c06 = sp.load_npz(RAW7B / "source" / unit_id / "c06_affinity.npz")
    affinity = endpoint_affinity("E1_ADAPTER_C06_MEAN", embedding, c06, ids)
    labels, _ = run_partition("H01", affinity, int(unit["K"]), [])
    require(sparse_sha(affinity) == expected_transform["canonical_affinity_sha256"],
            "R02 endpoint affinity parity failure")
    require(array_sha(canonical_partition(labels)) == expected_transform["canonical_partition_sha256"],
            "R02 endpoint partition parity failure")

    authority = authority_map()[unit_id]
    m_initial = float(authority["m_initial_authority"])
    target = RAW / "p1b_features" / unit_id
    require(not target.exists(), "P1B feature cell already exists")
    target.mkdir(parents=True)
    np.savez(target / "features.npz", m_initial=np.asarray(m_initial, dtype=np.float64),
             conflict=conflict, rank_c=rank_c, quality=quality, support=support,
             positive=positives, negative=negatives)
    manifest = {
        "schema_version": 1, "status": "PASS", "unit_id": unit_id,
        "label_access": False, "fresh_final_process": True, "initial_forward_performed": False,
        "m_initial": m_initial, "m_initial_authority_row_sha256": authority["authority_row_sha256"],
        "m_initial_source": "night7b_sha_locked_loss_curve_first_row_MNN",
        "checkpoint_sha256": train["training_manifest"]["checkpoint_sha256"],
        "checkpoint_state_sha256": train["training_manifest"]["state_tensor_sha256"],
        "embedding_exact": True, "gate_exact": True,
        "endpoint_affinity_exact": True, "endpoint_partition_exact": True,
        "fixed_positive_sha256": array_sha(positives), "fixed_negative_sha256": array_sha(negatives),
        "conflict_sha256": array_sha(conflict), "rank_c_sha256": array_sha(rank_c),
        "quality_sha256": array_sha(quality), "support_sha256": array_sha(support),
        "feature_file_sha256": sha256_file(target / "features.npz"),
        "canonical_affinity_sha256": sparse_sha(affinity),
        "canonical_partition_sha256": array_sha(canonical_partition(labels)),
        "gpu_model": torch.cuda.get_device_name(0), "runtime_seconds": time.perf_counter() - start,
    }
    atomic_json(target / "manifest.json", manifest)


def driver() -> None:
    feature_root = RAW / "p1b_features"
    require(not feature_root.exists(), "P1B feature root already exists")
    feature_root.mkdir(parents=True)
    units = list(csv.DictReader((HANDOFF7B / "source_unit_index.csv").open(newline="")))
    rows = []
    for source in units:
        unit_id = source["unit_id"]
        cmd = [sys.executable, str(Path(__file__).resolve()), "cell", "--unit-id", unit_id]
        done = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log = feature_root / unit_id / "cell.log"; log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(done.stdout + done.stderr)
        require(done.returncode == 0, f"P1B cell failed: {unit_id}")
        mp = feature_root / unit_id / "manifest.json"; manifest = json.loads(mp.read_text())
        rows.append({
            "ordinal": int(source["ordinal"]), "unit_id": unit_id,
            "status": manifest["status"], "label_access": False,
            "m_initial": manifest["m_initial"], "m_initial_source": manifest["m_initial_source"],
            "feature_file": str(feature_root / unit_id / "features.npz"),
            "feature_file_sha256": manifest["feature_file_sha256"],
            "manifest_sha256": sha256_file(mp), "runtime_seconds": manifest["runtime_seconds"],
        })
    require(len(rows) == 30 and all(x["status"] == "PASS" for x in rows), "P1B is not 30/30")
    target = OUT / "routing_feature_manifest.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {
        "schema_version": 1, "status": "PASS", "label_access": False,
        "fresh_final_checkpoint_endpoint_parity": "30/30", "feature_files_locked": 30,
        "initial_replay_and_final_parity_separate_processes": True,
        "routing_m_initial_source": "night7b_sha_locked_loss_curve_first_row_MNN",
        "routing_feature_manifest_sha256": sha256_file(target),
    }
    atomic_json(OUT / "p1b_feature_freeze_contract.json", result)
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("driver", "cell")); parser.add_argument("--unit-id")
    args = parser.parse_args()
    if args.mode == "driver": driver()
    else:
        require(bool(args.unit_id), "cell requires unit id"); cell(args.unit_id)


if __name__ == "__main__":
    main()
