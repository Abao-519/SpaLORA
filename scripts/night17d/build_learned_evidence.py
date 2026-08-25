#!/usr/bin/env python
"""Build locked Night-17D candidate evidence without opening annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpaLORA.night17b_sfrd import (  # noqa: E402
    canonical_pair_bank,
    csr_from_carrier,
    deterministic_relation_smooth,
    relation_posterior,
)
from SpaLORA.night17c_zero_start import (  # noqa: E402
    node_trust_gate,
    reload_zero_start,
    stratified_permute_relation,
    train_zero_start,
)
from SpaLORA.night17d_learned_evidence import (  # noqa: E402
    compress_seed_features,
    percentile,
    permutation_order,
    raw_candidate_features,
    sha256_array,
)


Z01 = {
    "config_id": "Z01_CONSERVATIVE",
    "hidden_dim": 32,
    "residual_scale": 0.05,
    "learning_rate": 0.0007,
    "steps": 40,
    "relation_weight": 1.0,
    "anchor_weight": 4.0,
    "self_return_weight": 4.0,
    "consistency_weight": 0.5,
    "variance_weight": 0.1,
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def expected_hash(artifact: dict[str, np.ndarray], run_id: str) -> str:
    matches = np.flatnonzero(artifact["run_ids"] == run_id)
    if matches.size != 1:
        raise ValueError(f"expected exactly one locked artifact {run_id}")
    return str(artifact["representation_sha256"][int(matches[0])])


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--feasibility", type=Path, required=True)
    parser.add_argument("--seed-artifact", action="append", required=True, help="SEED=PATH")
    parser.add_argument("--seed-checkpoint", action="append", required=True, help="SEED=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {int(item.split("=", 1)[0]): Path(item.split("=", 1)[1]) for item in args.seed_artifact}
    checkpoint_paths = {int(item.split("=", 1)[0]): Path(item.split("=", 1)[1]) for item in args.seed_checkpoint}
    if sorted(artifact_paths) != [0, 1, 2] or sorted(checkpoint_paths) != [0, 1, 2]:
        raise ValueError("exactly seeds 0,1,2 are required")

    carrier = load_npz(args.carrier)
    bank = load_npz(args.candidate_bank)
    records = list(csv.DictReader(args.feasibility.open(encoding="utf-8")))
    if not np.array_equal(carrier["ids"], bank["ids"]):
        raise ValueError("carrier/candidate ordered IDs differ")
    if bank["partitions"].shape[0] != 89:
        raise ValueError("Night-17D requires the locked 89-candidate bank")
    if [row["candidate_id"] for row in records] != bank["candidate_ids"].tolist():
        raise ValueError("candidate order differs from feasibility authority")

    graph = csr_from_carrier(carrier, "graph0")
    pair_i, pair_j, is_spatial = canonical_pair_bank(graph, carrier["retained"], feature_neighbors=6)
    posterior = relation_posterior(bank["partitions"], records, pair_i, pair_j, "UNBIASED_BANK", True)
    permuted = stratified_permute_relation(posterior, is_spatial)
    smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, posterior, alpha=0.2)
    permuted_smooth = deterministic_relation_smooth(carrier["retained"], pair_i, pair_j, permuted, alpha=0.2)
    gate, gate_diag = node_trust_gate(carrier["ids"].size, pair_i, pair_j, is_spatial, posterior)
    permuted_gate, permuted_gate_diag = node_trust_gate(
        carrier["ids"].size, pair_i, pair_j, is_spatial, permuted
    )
    perm_order = permutation_order(is_spatial)
    if not np.array_equal(permuted.probability_same, posterior.probability_same[perm_order]):
        raise RuntimeError("permutation-order reconstruction differs from Night-17C")

    representations: dict[str, list[np.ndarray]] = {"LEARNED": [], "ZERO": [], "PERMUTED": []}
    representation_hashes: dict[str, dict[str, str]] = {key: {} for key in representations}
    checkpoint_hashes: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    with threadpool_limits(limits=1):
        for seed in (0, 1, 2):
            artifact = load_npz(artifact_paths[seed])
            if not np.array_equal(artifact["ids"], carrier["ids"]):
                raise ValueError(f"seed {seed} artifact IDs differ")
            checkpoint = load_checkpoint(checkpoint_paths[seed])
            checkpoint_configs = {
                str(item["config_id"]): dict(item) for item in checkpoint.get("configs", [])
            }
            if checkpoint_configs.get("Z01_CONSERVATIVE") != Z01:
                raise RuntimeError(f"seed {seed} checkpoint Z01 config differs from frozen builder config")
            state = checkpoint["unbiased_full_state_dict"]["Z01_CONSERVATIVE"]
            learned = reload_zero_start(
                state,
                carrier["view1"],
                carrier["view2"],
                carrier["retained"],
                smooth,
                gate,
                Z01,
                device=device,
            )
            locked_learned_hash = expected_hash(artifact, f"Z01_CONSERVATIVE__UNBIASED_FULL__S{seed}")
            if sha256_array(learned) != locked_learned_hash:
                raise RuntimeError(f"seed {seed} learned representation differs from Night-17C authority")
            locked_zero_hash = expected_hash(artifact, "BASELINE__ZERO_RESIDUAL")
            if sha256_array(smooth.astype(np.float32)) != locked_zero_hash:
                raise RuntimeError(f"seed {seed} zero-residual smooth differs from Night-17C authority")
            perm_result = train_zero_start(
                carrier["view1"],
                carrier["view2"],
                carrier["retained"],
                permuted_smooth,
                permuted_gate,
                pair_i,
                pair_j,
                is_spatial,
                permuted,
                Z01,
                seed,
                device=device,
            )
            locked_permuted_hash = expected_hash(artifact, f"Z01_CONSERVATIVE__PERMUTED_RELATION__S{seed}")
            if sha256_array(perm_result.representation) != locked_permuted_hash:
                raise RuntimeError(f"seed {seed} permuted representation differs from Night-17C authority")
            representations["LEARNED"].append(learned)
            representations["ZERO"].append(smooth.astype(np.float32))
            representations["PERMUTED"].append(perm_result.representation)
            for arm, value in (("LEARNED", learned), ("ZERO", smooth), ("PERMUTED", perm_result.representation)):
                representation_hashes[arm][str(seed)] = sha256_array(np.asarray(value, dtype=np.float32))
            checkpoint_hashes[str(seed)] = file_sha(checkpoint_paths[seed])
            artifact_hashes[str(seed)] = file_sha(artifact_paths[seed])

    feasible = np.asarray(
        [str(row["feasible_SMALLEST_SCALE_INTERNAL_EDGE"]).lower() == "true" for row in records], dtype=bool
    )
    feasible_indices = np.flatnonzero(feasible)
    output_rows = [dict(row) for row in records]
    raw_by_arm: dict[str, list[list[dict[str, float | bool]]]] = {}
    for arm in ("LEARNED", "ZERO", "PERMUTED"):
        arm_posterior = permuted if arm == "PERMUTED" else posterior
        raw_by_arm[arm] = []
        for seed, representation in enumerate(representations[arm]):
            seed_rows = []
            for index in feasible_indices:
                partition = bank["partitions"][index]
                same = partition[pair_i] == partition[pair_j]
                contribution = same[perm_order] if arm == "PERMUTED" else same
                feature = raw_candidate_features(
                    representation,
                    partition,
                    pair_i,
                    pair_j,
                    is_spatial,
                    arm_posterior,
                    int(index),
                    contribution,
                )
                seed_rows.append(feature)
                for key, value in feature.items():
                    output_rows[index][f"{arm}_S{seed}_{key}"] = value
            raw_by_arm[arm].append(seed_rows)
            seed_evidence = compress_seed_features(seed_rows)
            for position, index in enumerate(feasible_indices):
                output_rows[index][f"{arm}_S{seed}_evidence"] = float(seed_evidence[position])
        matrix = np.column_stack(
            [[float(output_rows[index][f"{arm}_S{seed}_evidence"]) for index in feasible_indices] for seed in (0, 1, 2)]
        )
        evidence = matrix.mean(axis=1)
        uncertainty = matrix.std(axis=1)
        uncertainty_rank = percentile(uncertainty)
        for position, index in enumerate(feasible_indices):
            output_rows[index][f"{arm}_evidence"] = float(evidence[position])
            output_rows[index][f"{arm}_uncertainty"] = float(uncertainty[position])
            output_rows[index][f"{arm}_uncertainty_rank"] = float(uncertainty_rank[position])

    # Infeasible rows remain visible but cannot be selected.
    for index in np.flatnonzero(~feasible):
        for arm in ("LEARNED", "ZERO", "PERMUTED"):
            output_rows[index][f"{arm}_evidence"] = ""
            output_rows[index][f"{arm}_uncertainty"] = ""
            output_rows[index][f"{arm}_uncertainty_rank"] = ""

    fieldnames = list(output_rows[0])
    for row in output_rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path = args.output_dir / "candidate_learned_evidence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    manifest = {
        "schema": "night17d-learned-evidence-v1",
        "lane": args.lane,
        "n": int(carrier["ids"].size),
        "candidate_count": int(bank["partitions"].shape[0]),
        "feasible_candidate_count": int(feasible.sum()),
        "pair_count": int(pair_i.size),
        "spatial_pair_count": int(is_spatial.sum()),
        "candidate_feature_label_reads": 0,
        "relation_alignment": "CANDIDATE_SPECIFIC_LEAVE_ONE_OUT_WITH_FULL_POSTERIOR_SENSITIVITY",
        "unbiased_candidate_count": int(posterior.selected_candidate_count),
        "representation_hashes": representation_hashes,
        "checkpoint_hashes": checkpoint_hashes,
        "artifact_hashes": artifact_hashes,
        "carrier_sha256": file_sha(args.carrier),
        "candidate_bank_sha256": file_sha(args.candidate_bank),
        "feasibility_sha256": file_sha(args.feasibility),
        "feature_csv_sha256": file_sha(csv_path),
        "gate_diagnostics": gate_diag,
        "permuted_gate_diagnostics": permuted_gate_diag,
        "wall_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "device": device,
        "thread_limits": {"OMP": 1, "MKL": 1, "OPENBLAS": 1, "threadpoolctl": 1},
    }
    (args.output_dir / "candidate_learned_evidence.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("lane", "n", "feasible_candidate_count", "feature_csv_sha256", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
