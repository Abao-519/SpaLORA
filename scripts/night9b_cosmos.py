#!/usr/bin/env python3
"""One fixed-endpoint, label-free COSMOS P22 unit with two shared outputs."""
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

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
COSMOS_PARENT = Path("/root/autodl-fs/night8a_external_sources_20260820/COSMOS")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(COSMOS_PARENT))
from COSMOS.cosmos import Cosmos  # noqa: E402
from SpaLORA.night6c_pipeline import h00, run_head  # noqa: E402
from SpaLORA.night6d_pipeline import HEADS  # noqa: E402
from SpaLORA.night7a_consensus import canonical_partition  # noqa: E402
from SpaLORA.night9b_racf import array_sha, canonical_json_sha  # noqa: E402

BASE = Path("/root/autodl-fs/night3af_p0d_builds_20260810/process_a/p22")
SOURCE_COMMIT = "56ea355be51e64d9253e2871b8bd447fdfd0d230"
CONFIG = {
    "source_commit": SOURCE_COMMIT, "preprocessing": {"do_norm": False, "do_log": False,
    "n_top_genes": None, "do_pca": False, "n_neighbors": 10},
    "training": {"spatial_regularization_strength": .05, "z_dim": 50, "lr": 1e-3,
    "wnn_epoch": 100, "total_epoch": 1000, "max_patience_bef": 10,
    "max_patience_aft": 30, "min_stop": 100, "regularization_acceleration": True,
    "edge_subset_sz": 1000000},
    "K": 9,
    "native_endpoint": "row_L2_COSMOS_embedding_then_deterministic_PCA20_mclust_EEE_random_state_2020_fixed_K",
    "common_endpoint": "repeat_same_COSMOS_embedding_as_three_H05_views_fixed_K",
    "label_metric_selection": False, "fallback": False, "scientific_retry": 0,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); os.replace(tmp, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle: np.save(handle, value)
    os.replace(tmp, path)


def clusters(path: Path, ids: list[str], labels: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["observation_id", "cluster"])
        writer.writerows(zip(ids, map(int, labels)))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    if args.seed not in range(5): raise RuntimeError("unregistered COSMOS seed")
    if args.output.exists() and any(args.output.iterdir()): raise RuntimeError("refusing overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("COSMOS formal unit requires CUDA")
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    x1 = np.load(BASE / "features_omics1.npy", allow_pickle=False)
    x2 = np.load(BASE / "features_omics2.npy", allow_pickle=False)
    coords = np.load(BASE / "coordinates.npy", allow_pickle=False)
    ids_path = BASE / "observation_ids.tsv"
    with ids_path.open(newline="", encoding="utf-8") as handle:
        ids = [str(r["observation_id"]) for r in csv.DictReader(handle, delimiter="\t")]
    if not (len(ids) == len(x1) == len(x2) == len(coords) == 9196):
        raise RuntimeError("COSMOS P22 modeling input contract failed")
    model = Cosmos(count_matrix1=x1, count_matrix2=x2, spatial_locs=coords,
                   sample_names=ids)
    prep_started = time.perf_counter()
    model.preprocessing_data(**CONFIG["preprocessing"])
    preprocessing_seconds = time.perf_counter() - prep_started
    training_started = time.perf_counter()
    embedding = model.train(random_seed=int(args.seed), gpu=0, **CONFIG["training"])
    training_seconds = time.perf_counter() - training_started
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.shape != (9196, 50) or not np.isfinite(embedding).all():
        raise RuntimeError("COSMOS embedding contract failed")
    embedding_path = args.output / "embedding.npy"; atomic_npy(embedding_path, embedding)
    weights = np.asarray(model.weights, dtype=np.float32)
    weights_path = args.output / "weights.npy"; atomic_npy(weights_path, weights)
    endpoint_started = time.perf_counter()
    native = h00(embedding, 9)["labels"]
    native_seconds = time.perf_counter() - endpoint_started
    common_started = time.perf_counter()
    repeated = {"emb_latent_omics1": embedding, "emb_latent_omics2": embedding,
                "SpaLORA_fused": embedding}
    common, common_audit = run_head(HEADS["H05_EQUAL3_AFFINITY_SPECTRAL"], repeated,
                                    9, coords, ids, args.output / "common_head")
    common_seconds = time.perf_counter() - common_started
    native_path = args.output / "COSMOS_NATIVE_FIXED_clusters.csv"
    common_path = args.output / "COSMOS_COMMON_HEAD_clusters.csv"
    clusters(native_path, ids, native); clusters(common_path, ids, common)
    config_path = args.output / "resolved_config.json"; atomic_json(config_path, {**CONFIG, "seed": args.seed})
    artifacts = {p.name: {"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
                 for p in (embedding_path, weights_path, native_path, common_path, config_path,
                           args.output / "common_head/affinity.npz")}
    manifest = {
        "schema_version": 1, "status": "SUCCESS_PRE_LABEL", "seed": args.seed,
        "source_commit": SOURCE_COMMIT, "config_sha256": canonical_json_sha(CONFIG),
        "label_access": False, "scientific_retry": 0, "fallback": False,
        "cuda_used": True, "cuda_device": torch.cuda.get_device_name(0),
        "input": {"cache_manifest_sha256": sha256_file(BASE / "manifest.json"),
                  "features_omics1_sha256": sha256_file(BASE / "features_omics1.npy"),
                  "features_omics2_sha256": sha256_file(BASE / "features_omics2.npy"),
                  "coordinates_sha256": sha256_file(BASE / "coordinates.npy"),
                  "observation_ids_sha256": sha256_file(ids_path), "spots": len(ids),
                  "feature_dimensions": [int(x1.shape[1]), int(x2.shape[1])]},
        "embedding_canonical_sha256": array_sha(embedding),
        "weights_canonical_sha256": array_sha(weights),
        "partitions": {"COSMOS_NATIVE_FIXED": array_sha(canonical_partition(native)),
                       "COSMOS_COMMON_HEAD": array_sha(canonical_partition(common))},
        "common_head": common_audit,
        "runtime": {"preprocessing_seconds": preprocessing_seconds,
                    "training_seconds": training_seconds, "checkpoint_reload_seconds": 0.0,
                    "native_endpoint_seconds": native_seconds,
                    "common_head_seconds": common_seconds,
                    "total_seconds": time.perf_counter() - started},
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "artifacts": artifacts,
    }
    atomic_json(args.output / "cosmos_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "seed": args.seed,
                      "runtime_seconds": manifest["runtime"]["total_seconds"]}, sort_keys=True))


if __name__ == "__main__": main()
