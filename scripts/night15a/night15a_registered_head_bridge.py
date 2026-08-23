#!/usr/bin/env python3
"""Bridge one frozen Night-15A representation through registered project heads.

The public benchmark label is loaded only after a partition has been generated.
All registered heads receive the same ordered observations, K, coordinates and
filtered representation triplet.  This isolates representation/filter effects
from clustering-head effects without entering labels into model or cluster fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/night13b"), str(REPO / "scripts/night14b")]

import night13b_run as n13b  # noqa: E402
from SpaLORA.night6c_pipeline import run_head  # noqa: E402
from SpaLORA.night14b_atac import diffuse, spatial_operator  # noqa: E402
import night14b_head_search as n14heads  # noqa: E402


HEADS = (
    {"id": "H00_FUSED_PCA20_MCLUST_EEE"},
    {"id": "H01_FUSED_DIRECT_MCLUST_EEE"},
    {"id": "H03_CONCAT_PRIVATE_PCA20_MCLUST_EEE"},
    {"id": "H05_EQUAL3_AFFINITY_SPECTRAL"},
)
FILTERS = (
    {"id": "NONE", "kind": "NONE"},
    {"id": "LOW_K12_B80_S2", "kind": "LOW", "graph_k": 12, "beta": 0.80, "steps": 2},
    {"id": "LOW_K18_B90_S3", "kind": "LOW", "graph_k": 18, "beta": 0.90, "steps": 3},
    {"id": "BILAT_K12_Q50_B85_S2", "kind": "BILATERAL", "graph_k": 12, "quantile": 0.50, "beta": 0.85, "steps": 2},
    {"id": "BILAT_K18_Q50_B85_S2", "kind": "BILATERAL", "graph_k": 18, "quantile": 0.50, "beta": 0.85, "steps": 2},
)


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def load_run(run_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    audit = json.loads((run_dir / "training_audit.json").read_text(encoding="utf-8"))
    reload_audit = json.loads((run_dir / "fresh_process_reload.json").read_text(encoding="utf-8"))
    if not reload_audit["all_numerically_close"]:
        raise RuntimeError("fresh-process checkpoint round-trip failed")
    archive = np.load(run_dir / "views.npz", allow_pickle=False)
    return audit, {
        "emb_latent_omics1": np.asarray(archive["z1"], dtype=np.float32),
        "emb_latent_omics2": np.asarray(archive["z2"], dtype=np.float32),
        "SpaLORA_fused": np.asarray(archive["mcdf"], dtype=np.float32),
        "ids": np.asarray(archive["ids"], dtype=str),
        "coordinates": np.asarray(archive["coordinates"], dtype=np.float64),
    }


def protocol_payload(dataset: str, protocol: str, ids: np.ndarray) -> tuple[dict, tuple[int, ...], str]:
    payload = n13b.base_payload(dataset)
    if not np.array_equal(np.asarray(payload["ids"], dtype=str), ids):
        raise RuntimeError("registered observation order mismatch")
    if protocol != "P22_3DOT_K18":
        if dataset == "P22":
            ks = (9,)
        elif dataset == "MISAR_E15_5_S1":
            ks = (7, 12)
        else:
            ks = (int(payload["k"]),)
        return payload, ks, "PROJECT_PUBLIC_GROUND_TRUTH"
    if dataset != "P22":
        raise ValueError("P22_3DOT_K18 requires dataset P22")
    path = Path(
        "/root/autodl-fs/night15a_multimodal_contribution_and_score_stability_20260823/"
        "protocol_inputs/3dot_zenodo_15089427/3d-OT.h5ad"
    )
    carrier = ad.read_h5ad(str(path), backed="r")
    carrier_ids = carrier.obs_names.astype(str).to_numpy()
    if not np.array_equal(carrier_ids, ids):
        raise RuntimeError("official K18 observation order mismatch")
    labels = carrier.obs["3d-OT"].astype(str).to_numpy()
    if int(pd.Series(labels).nunique()) != 18:
        raise RuntimeError("official 3d-OT annotation is not K18")
    return {
        **payload,
        "labels": labels,
        "label_mask": np.ones(len(labels), dtype=bool),
    }, (18,), "OFFICIAL_3DOT_SUPPLIED_18_STATE_ANNOTATION_NOT_K9_SPLIT"


def filtered_views(views: dict[str, np.ndarray], spec: dict) -> dict[str, np.ndarray]:
    content = {key: views[key] for key in ("emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused")}
    if spec["kind"] == "NONE":
        return content
    if spec["kind"] == "LOW":
        operator = spatial_operator(views["coordinates"], int(spec["graph_k"]))
    else:
        operator, _ = n14heads.anisotropic_operator(
            content, views["coordinates"], int(spec["graph_k"]), float(spec["quantile"])
        )
    return {
        key: diffuse(value, operator, float(spec["beta"]), int(spec["steps"]))
        for key, value in content.items()
    }


def run(dataset: str, protocol: str, run_dir: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started_all = time.perf_counter()
    audit, views = load_run(run_dir)
    payload, ks, label_semantics = protocol_payload(dataset, protocol, views["ids"])
    if not np.array_equal(np.asarray(payload["coordinates"]), views["coordinates"]):
        raise RuntimeError("registered coordinate order mismatch")
    rows: list[dict] = []
    partitions: dict[str, np.ndarray] = {}
    for spec in FILTERS:
        filtered = filtered_views(views, spec)
        for cluster_k in ks:
            for head in HEADS:
                started = time.perf_counter()
                base = {
                    "dataset": "P22_3DOT_K18" if protocol == "P22_3DOT_K18" else dataset,
                    "candidate_id": audit["candidate_id"],
                    "model_seed": int(audit["seed"]),
                    "cluster_k": int(cluster_k),
                    "filter_id": spec["id"],
                    "head_id": head["id"],
                    "known_k_unsupervised": True,
                    "labels_in_model_or_cluster_fit": False,
                    "public_labels_used_after_partition_for_hpo_and_evaluation": True,
                }
                try:
                    partition, head_audit = run_head(
                        head, filtered, int(cluster_k), views["coordinates"], views["ids"]
                    )
                    partition = np.asarray(partition, dtype=np.int64)
                    key = f"{spec['id']}|{cluster_k}|{head['id']}"
                    partitions[key] = partition
                    rows.append({
                        **base,
                        "status": "PASS",
                        "partition_sha256": sha256_array(partition),
                        "head_audit": json.dumps(head_audit, sort_keys=True, default=str),
                        "wall_seconds": time.perf_counter() - started,
                        **n13b.partition_metrics(
                            payload["labels"], payload["label_mask"], partition, payload["metric_graph"]
                        ),
                    })
                except Exception as error:
                    rows.append({
                        **base,
                        "status": "FAIL",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "wall_seconds": time.perf_counter() - started,
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "registered_head_ledger.csv", index=False)
    np.savez_compressed(output / "partitions.npz", **partitions)
    atomic_json(output / "registered_head_manifest.json", {
        "dataset": dataset,
        "result_dataset": "P22_3DOT_K18" if protocol == "P22_3DOT_K18" else dataset,
        "protocol": protocol,
        "label_semantics": label_semantics,
        "candidate_id": audit["candidate_id"],
        "model_seed": int(audit["seed"]),
        "rows": len(frame),
        "pass_rows": int((frame.status == "PASS").sum()),
        "fail_rows": int((frame.status == "FAIL").sum()),
        "deterministic_partitions_registered_once": True,
        "labels_in_model_or_cluster_fit": False,
        "dense_n_by_n_count": 0,
        "wall_seconds": time.perf_counter() - started_all,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3", "P22", "MISAR_E15_5_S1"),
        required=True,
    )
    parser.add_argument("--protocol", choices=("PROJECT", "P22_3DOT_K18"), default="PROJECT")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.dataset, args.protocol, Path(args.run_dir), Path(args.output))


if __name__ == "__main__":
    main()
