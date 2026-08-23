#!/usr/bin/env python3
"""Post-freeze offline ablations and corruption controls for Night-14A.

This script never trains.  It reads the byte-recorded latent arrays produced by
one frozen checkpoint, applies registered TCF variants, and evaluates every
variant with the same common endpoint protocol used by the formal runner.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.night14a import night14a_run as run


ROOT = Path("/root/autodl-fs/night14a_topology_conflict_sprint_20260823")
CANDIDATE = "C15_BAL_XREC_600_WEAK_ALIGN"
CORRUPTION_SEED = 20260823


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_directories(formal_root: Path) -> List[Path]:
    return sorted((formal_root / CANDIDATE).glob("*/seed_*"))


def recorded_objects(run_dir: Path) -> Tuple[dict, dict]:
    expected = np.load(run_dir / "roundtrip_expected.npz", allow_pickle=False)
    values = np.load(run_dir / "roundtrip_input.npz", allow_pickle=False)
    objects = {
        "embeddings": {
            key: np.asarray(expected[key], dtype=np.float32)
            for key in ("z1", "z2", "fused")
        },
        "edge_index": np.asarray(values["edge_index"], dtype=np.int64),
    }
    audit = read_json(run_dir / "training_audit.json")
    return objects, audit


def deterministic_derangement(n: int, seed: int = CORRUPTION_SEED) -> np.ndarray:
    """Seeded random permutation with fixed points removed mechanically."""
    if n < 2:
        raise ValueError("derangement requires at least two observations")
    permutation = np.random.default_rng(int(seed)).permutation(n)
    fixed = np.flatnonzero(permutation == np.arange(n))
    if len(fixed) == 1:
        left = int(fixed[0])
        right = (left + 1) % n
        permutation[left], permutation[right] = permutation[right], permutation[left]
    elif len(fixed) > 1:
        permutation[fixed] = np.roll(permutation[fixed], 1)
    if np.any(permutation == np.arange(n)):
        raise RuntimeError("failed to construct derangement")
    return permutation


def offline_ablation(formal_root: Path, filter_path: Path, output: Path,
                     endpoint_seed_count: int, n_init: int) -> None:
    filters = read_json(filter_path)
    for value in filters:
        run.validate_filter(value)
    rows: List[dict] = []
    summaries: List[dict] = []
    artifacts: List[dict] = []
    endpoint_seeds = tuple(range(endpoint_seed_count))
    for run_dir in run_directories(formal_root):
        objects, audit = recorded_objects(run_dir)
        dataset = str(audit["dataset"])
        model_seed = int(audit["model_seed"])
        payload = run.n13b.base_payload(dataset)
        filtered, filter_audits = run.filter_embeddings(
            objects, filters, torch.device("cpu")
        )
        cache: Dict[str, Tuple[List[dict], dict, str]] = {}
        for filter_id, embedding in filtered.items():
            embedding_sha = run.array_sha256(embedding)
            if embedding_sha in cache:
                old_rows, old_summary, source_filter = cache[embedding_sha]
                endpoint, summary = run.retag_reused_endpoint(
                    old_rows, old_summary, filter_id, source_filter
                )
            else:
                endpoint, summary, _ = run.endpoint_rows(
                    dataset, "NIGHT14A_UNIFIED_TCF", model_seed, embedding,
                    payload, endpoint_seeds, n_init, "OFFLINE_KEY_ABLATION",
                    CANDIDATE, filter_id,
                )
                summary["endpoint_reused"] = False
                summary["endpoint_reuse_of_filter_id"] = None
                for value in endpoint:
                    value["endpoint_reused"] = False
                    value["endpoint_reuse_of_filter_id"] = None
                    value["source_endpoint_wall_seconds"] = value["endpoint_wall_seconds"]
                cache[embedding_sha] = (
                    [dict(value) for value in endpoint], dict(summary), filter_id
                )
            rows.extend(endpoint)
            summaries.append({**summary, **filter_audits[filter_id]})
        artifacts.append({
            "dataset": dataset,
            "model_seed": model_seed,
            "checkpoint_sha256": str(audit["checkpoint_sha256"]),
            "source_embedding_sha256": audit["embedding_sha256"],
            "optimizer_steps_during_ablation": 0,
            "single_frozen_checkpoint_for_filter_grid": True,
            "filter_count": len(filters),
            "unique_endpoint_embedding_count": len(cache),
            "status": "PASS",
        })
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "offline_ablation_endpoint_rows.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "offline_ablation_endpoint_summary.csv", index=False)
    run.atomic_json(output / "offline_ablation_manifest.json", {
        "candidate": CANDIDATE,
        "source_formal_root": str(formal_root),
        "filter_path": str(filter_path),
        "filter_sha256": run.file_sha256(filter_path),
        "endpoint_seeds": list(endpoint_seeds),
        "endpoint_n_init": int(n_init),
        "run_count": len(artifacts),
        "all_status_pass": all(row["status"] == "PASS" for row in artifacts),
        "optimizer_steps_during_ablation": 0,
        "rows": artifacts,
    })


def corruption_audit(formal_root: Path, filter_path: Path, output: Path) -> None:
    filters = read_json(filter_path)
    selected = [value for value in filters if value["filter_id"] == "W02_TCF_FINAL"]
    if len(selected) != 1:
        raise RuntimeError("expected exactly one W02_TCF_FINAL filter")
    config = selected[0]
    rows = []
    for run_dir in run_directories(formal_root):
        objects, audit = recorded_objects(run_dir)
        permutation = deterministic_derangement(len(objects["embeddings"]["z2"]))
        corrupted = {
            "embeddings": {
                "z1": objects["embeddings"]["z1"],
                "z2": objects["embeddings"]["z2"][permutation],
                "fused": objects["embeddings"]["fused"],
            },
            "edge_index": objects["edge_index"],
        }
        filtered, diagnostics = run.filter_embeddings(
            corrupted, [config], torch.device("cpu")
        )
        observed = filtered["W02_TCF_FINAL"]
        identity = corrupted["embeddings"]["fused"]
        diag = diagnostics["W02_TCF_FINAL"]
        rows.append({
            "dataset": str(audit["dataset"]),
            "model_seed": int(audit["model_seed"]),
            "corruption": "SEEDED_RANDOM_Z2_SPOT_DERANGEMENT",
            "corruption_seed": CORRUPTION_SEED,
            "fixed_point_count": int(np.sum(permutation == np.arange(len(permutation)))),
            "checkpoint_sha256": str(audit["checkpoint_sha256"]),
            "exact_identity": bool(np.array_equal(observed, identity)),
            "max_abs_difference_from_identity": float(np.max(np.abs(observed - identity))),
            "global_gate": float(diag["global_gate"]),
            "integrity_gate": float(diag["integrity_gate"]),
            "dense_n_by_n_count": 0,
        })
    result = {
        "filter_id": "W02_TCF_FINAL",
        "source_formal_root": str(formal_root),
        "fixed_permutation": True,
        "permutation_kind": "SEEDED_RANDOM_DERANGEMENT",
        "permutation_seed": CORRUPTION_SEED,
        "run_count": len(rows),
        "exact_identity_count": sum(int(row["exact_identity"]) for row in rows),
        "all_exact_identity": all(row["exact_identity"] for row in rows),
        "rows": rows,
    }
    run.atomic_json(output, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    ablation = sub.add_parser("offline-ablation")
    ablation.add_argument("--formal-root", required=True)
    ablation.add_argument("--filters", required=True)
    ablation.add_argument("--output", required=True)
    ablation.add_argument("--endpoint-seed-count", type=int, default=20)
    ablation.add_argument("--n-init", type=int, default=10)
    corruption = sub.add_parser("corruption")
    corruption.add_argument("--formal-root", required=True)
    corruption.add_argument("--filters", required=True)
    corruption.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "offline-ablation":
        offline_ablation(
            Path(args.formal_root), Path(args.filters), Path(args.output),
            int(args.endpoint_seed_count), int(args.n_init),
        )
    else:
        corruption_audit(
            Path(args.formal_root), Path(args.filters), Path(args.output)
        )


if __name__ == "__main__":
    main()
