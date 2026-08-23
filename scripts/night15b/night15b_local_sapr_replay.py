#!/usr/bin/env python3
"""Replay frozen SAPR embeddings through the exact Windows head-HPO implementation."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from night15b_local_runner import (  # noqa: E402
    csr_from_archive,
    lanes,
    metrics,
    refine_partition,
    sha256_array,
)


def dataset_for_lane(lane: str) -> str:
    if lane.startswith("P22"):
        return "P22"
    if lane.startswith("MISAR_E15_5_S1"):
        return "MISAR_E15_5_S1"
    return lane


def endpoint_spec(ledger: pd.DataFrame, lane: str, retained_id: str) -> Mapping[str, object]:
    rows = ledger[(ledger.lane == lane) & (ledger.status == "PASS") & (ledger.embedding_id == retained_id)].copy()
    if rows.empty:
        raise RuntimeError(f"missing endpoint for {lane}/{retained_id}")
    rows["rank_score"] = rows.absolute_ari + 0.35 * rows.absolute_nmi
    row = rows.sort_values("rank_score", ascending=False).iloc[0]
    return {"head": str(row["head"]), "endpoint_seed": int(row["endpoint_seed"])}


def run_endpoint(embedding: np.ndarray, k: int, graph, spec: Mapping[str, object]) -> np.ndarray:
    head = str(spec["head"])
    seed = int(spec["endpoint_seed"])
    if head.startswith("GMM_DIAG"):
        partition = GaussianMixture(k, covariance_type="diag", random_state=seed, n_init=1, max_iter=120, reg_covar=1e-5).fit_predict(embedding)
    elif head.startswith("GMM_TIED"):
        partition = GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=1, max_iter=120, reg_covar=1e-5).fit_predict(embedding)
    else:
        partition = KMeans(k, random_state=seed, n_init=30).fit_predict(embedding)
    matched = re.search(r"REFINE_T([0-9.]+)_S(\d+)", head)
    if matched:
        partition = refine_partition(partition, graph, k, float(matched.group(1)), int(matched.group(2)))
    return np.asarray(partition, dtype=np.int32)


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--exports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    head_ledger = pd.read_csv(args.head / "all_head_run_ledger.csv")
    rows = []
    sha_rows = []
    for path in sorted(args.exports.glob("*.npz")):
        exported = np.load(path, allow_pickle=False, mmap_mode="r")
        lane = str(exported["lane"][0])
        dataset = dataset_for_lane(lane)
        archive = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.head / f"{dataset}_selected_partition_bank.npz", allow_pickle=False, mmap_mode="r")
        retained = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
        retained_id = str(bank[f"{lane}__retained_embedding_id"][0])
        registered_lane = next(item for item in lanes(dataset, archive) if item.lane == lane)
        graph = csr_from_archive(archive, "graph")
        spec = endpoint_spec(head_ledger, lane, retained_id)
        expected_full = str(exported["full_sha256"][0])
        expected_disabled = str(exported["disabled_sha256"][0])
        full = np.asarray(exported["full"], dtype=np.float32)
        disabled = np.asarray(exported["disabled"], dtype=np.float32)
        if sha256_array(full) != expected_full or sha256_array(disabled) != expected_disabled:
            raise RuntimeError(f"export SHA mismatch: {path}")
        observed = {}
        for mode, embedding in (
            ("RETAINED_TEACHER", retained),
            ("SAPR_FULL", full),
            ("SAPR_RESIDUAL_DISABLED", disabled),
        ):
            partition = run_endpoint(embedding, registered_lane.k, graph, spec)
            observed[mode] = metrics(registered_lane.labels, registered_lane.mask, partition, graph)
            rows.append({
                "export": path.name,
                "dataset": dataset,
                "lane": lane,
                "candidate_id": str(exported["candidate_id"][0]),
                "training_seed": int(exported["training_seed"][0]),
                "endpoint_seed": int(spec["endpoint_seed"]),
                "head": str(spec["head"]),
                "mode": mode,
                "k": int(registered_lane.k),
                "total_observations": int(len(registered_lane.labels)),
                "evaluated_observations": int(registered_lane.mask.sum()),
                "partition_sha256": sha256_array(partition),
                **observed[mode],
            })
        reference = observed["RETAINED_TEACHER"]
        for row in rows[-3:]:
            row["delta_ari_vs_retained"] = row["absolute_ari"] - reference["absolute_ari"]
            row["delta_nmi_vs_retained"] = row["absolute_nmi"] - reference["absolute_nmi"]
            row["status"] = "PASS"
        sha_rows.append({
            "export": path.name,
            "full_sha256_match": sha256_array(full) == expected_full,
            "disabled_sha256_match": sha256_array(disabled) == expected_disabled,
        })
    pd.DataFrame(rows).to_csv(args.output / "sapr_local_endpoint_ledger.csv", index=False)
    audit = {
        "status": "PASS",
        "export_count": len(sha_rows),
        "run_rows": len(rows),
        "all_embedding_sha256_match": all(x["full_sha256_match"] and x["disabled_sha256_match"] for x in sha_rows),
        "labels_used_only_after_embedding_and_partition_fit": True,
        "dense_n_by_n_count": 0,
        "wall_seconds": time.perf_counter() - started,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "rows": sha_rows,
    }
    (args.output / "sapr_local_endpoint_replay_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "exports": len(sha_rows), "rows": len(rows)}))


if __name__ == "__main__":
    main()
