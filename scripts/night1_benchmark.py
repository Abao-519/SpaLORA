#!/usr/bin/env python3
"""Run preregistered SpaLORA Night 1 experiments without label leakage."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

os.environ.setdefault("R_HOME", "/opt/R/4.0.3/lib/R")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SpaLORA.night1_evaluation import evaluate, load_evaluation_labels
from SpaLORA.preprocess import clr_normalize_each_cell, construct_neighbor_graph, fix_seed, pca
from SpaLORA.SpaLORA_pyG import Train_SpaLORA
from SpaLORA.utils import clustering


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _drop_observation_metadata(obj: ad.AnnData) -> None:
    """Make label leakage structurally impossible during training."""
    obj.obs = pd.DataFrame(index=obj.obs_names.copy())


def load_training_data(cfg: dict) -> Tuple[ad.AnnData, ad.AnnData]:
    rna = sc.read_h5ad(cfg["rna"])
    mod2 = sc.read_h5ad(cfg["modality2"])
    rna.var_names_make_unique()
    mod2.var_names_make_unique()
    if not rna.obs_names.equals(mod2.obs_names):
        raise AssertionError("Paired modalities must have identical ordered observation IDs")
    if "spatial" not in rna.obsm or "spatial" not in mod2.obsm:
        raise AssertionError("Both modalities require spatial coordinates")
    if not np.array_equal(rna.obsm["spatial"], mod2.obsm["spatial"]):
        raise AssertionError("Paired modalities require identical spatial coordinates")
    _drop_observation_metadata(rna)
    _drop_observation_metadata(mod2)
    return rna, mod2


def prepare_legacy(dataset: str, cfg: dict) -> Tuple[Dict[str, ad.AnnData], pd.Index, np.ndarray]:
    """Mirror the public tutorials and original package code exactly."""
    rna, mod2 = load_training_data(cfg)
    sc.pp.filter_genes(rna, min_cells=10)
    if dataset == "p22":
        sc.pp.filter_cells(rna, min_genes=200)
        mod2 = mod2[rna.obs_names].copy()
    sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=cfg["hvg"])
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.scale(rna)

    rna_high = rna[:, rna.var["highly_variable"]]
    rna.obsm["raw_feat"] = rna_high.X.copy()
    if dataset == "p22":
        rna.obsm["feat"] = pca(rna_high, n_comps=50)
        if "X_lsi" not in mod2.obsm:
            raise AssertionError("P22 modality 2 requires its deposited X_lsi representation")
        mod2.obsm["feat"] = mod2.obsm["X_lsi"].copy()
    else:
        rna.obsm["feat"] = pca(rna_high, n_comps=mod2.n_vars - 1)
        mod2 = clr_normalize_each_cell(mod2)
        sc.pp.scale(mod2)
        mod2.obsm["feat"] = pca(mod2, n_comps=mod2.n_vars - 1)

    data = construct_neighbor_graph(rna, mod2, datatype=cfg["legacy_datatype"])
    return data, rna.obs_names.copy(), np.asarray(rna.obsm["spatial"])


def cluster_exact(embedding: np.ndarray, n_clusters: int, random_seed: int) -> np.ndarray:
    clustered = ad.AnnData(np.zeros((embedding.shape[0], 1), dtype=np.float32))
    clustered.obsm["SpaLORA"] = embedding
    clustering(
        clustered,
        key="SpaLORA",
        add_key="SpaLORA",
        n_clusters=n_clusters,
        use_pca=True,
        n_comps=20,
    )
    # The original helper fixes Mclust to seed 2020; enforce the preregistered value.
    if random_seed != 2020:
        raise AssertionError("Legacy clustering seed must remain 2020")
    return clustered.obs["SpaLORA"].astype(int).to_numpy()


def _validate_attention(output: dict, n_obs: int) -> Dict[str, float]:
    deviations = {}
    for key in ("alpha", "alpha_omics1", "alpha_omics2"):
        value = np.asarray(output[key])
        if value.shape != (n_obs, 2):
            raise AssertionError("%s has shape %r instead of (%d, 2)" % (key, value.shape, n_obs))
        deviations[key] = float(np.max(np.abs(value.sum(axis=1) - 1.0)))
    return deviations


def run_one(repo: Path, config_path: Path, config: dict, dataset: str, variant: str, seed: int, force: bool) -> Path:
    cfg = config["datasets"][dataset]
    run_dir = repo / "results" / "night1" / "raw" / dataset / variant / ("seed_%d" % seed)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        print("SKIP", dataset, variant, seed, metrics_path)
        return metrics_path
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    failure_path = run_dir / "failure.json"
    if failure_path.exists():
        failure_path.unlink()

    try:
        fix_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        pre_start = time.perf_counter()
        if variant != "legacy_exact":
            from SpaLORA.night1_pipeline import prepare_corrected, train_corrected

            prepared = prepare_corrected(dataset, cfg, config, variant)
            data, obs_names, coordinates = prepared.data, prepared.obs_names, prepared.coordinates
            gene_table = prepared.gene_table
        else:
            data, obs_names, coordinates = prepare_legacy(dataset, cfg)
            gene_table = None
        preprocessing_seconds = time.perf_counter() - pre_start

        training_start = time.perf_counter()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if variant == "legacy_exact":
            trainer = Train_SpaLORA(data, datatype=cfg["legacy_datatype"], device=device, random_seed=seed)
            output = trainer.train()
        else:
            output = train_corrected(data, cfg, variant, seed, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - training_start
        embedding = np.asarray(output["SpaLORA"])
        attention_deviation = _validate_attention(output, len(obs_names))

        cluster_start = time.perf_counter()
        predicted = cluster_exact(
            embedding,
            n_clusters=cfg["n_clusters"],
            random_seed=config["clustering"]["random_seed"],
        )
        clustering_seconds = time.perf_counter() - cluster_start

        # The first ground-truth access in the run occurs here, after training and clustering.
        evaluation_positions, true_labels = load_evaluation_labels(dataset, cfg, obs_names)
        evaluation_start = time.perf_counter()
        result_metrics = evaluate(
            true_labels=true_labels,
            predicted_labeled=predicted[evaluation_positions],
            predicted_all=predicted,
            embedding=embedding,
            coordinates=coordinates,
            spatial_neighbors=cfg["spatial_neighbors"],
        )
        evaluation_seconds = time.perf_counter() - evaluation_start

        np.savez_compressed(
            run_dir / "attention.npz",
            alpha=np.asarray(output["alpha"]),
            alpha_omics1=np.asarray(output["alpha_omics1"]),
            alpha_omics2=np.asarray(output["alpha_omics2"]),
        )
        pd.DataFrame({"observation_id": obs_names.astype(str), "cluster": predicted}).to_csv(
            run_dir / "clusters.csv", index=False
        )
        if gene_table is not None:
            interpretation = repo / "results" / "night1" / "interpretability"
            interpretation.mkdir(parents=True, exist_ok=True)
            gene_table.to_csv(interpretation / ("%s_%s_gene_scores.csv" % (dataset, variant)), index=False)

        timings = {
            "preprocessing_seconds": preprocessing_seconds,
            "training_seconds": training_seconds,
            "clustering_seconds": clustering_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        payload = {
            "schema_version": 1,
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "n_observations_trained": int(len(obs_names)),
            "n_observations_evaluated": int(evaluation_positions.size),
            "n_clusters_requested": int(cfg["n_clusters"]),
            "n_clusters_observed": int(np.unique(predicted).size),
            "config_sha256": sha256(config_path),
            "git_commit_at_run": git_value(repo, "rev-parse", "HEAD"),
            "metrics": result_metrics,
            "timings": timings,
            "memory": {
                "gpu_peak_allocated_mib": (
                    float(torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0.0
                ),
                "process_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            },
            "attention_max_row_sum_deviation": attention_deviation,
        }
        metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(
            "DONE",
            dataset,
            variant,
            seed,
            "ARI=%.6f" % result_metrics["ari"],
            "seconds=%.1f" % timings["total_seconds"],
            flush=True,
        )
        return metrics_path
    except Exception as exc:
        failure_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/night1.json")
    parser.add_argument("--dataset", choices=["a1", "placenta", "p22", "all"], default="all")
    parser.add_argument(
        "--variant",
        choices=["legacy_exact", "corrected_unweighted", "abundance_only", "asr_hvg", "asr_rescue", "all"],
        default="all",
    )
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = (repo / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    variants = list(config["variants"]) if args.variant == "all" else [args.variant]
    seeds = config["seeds"] if args.seeds is None or len(args.seeds) == 0 else args.seeds
    if any(seed not in config["seeds"] for seed in seeds):
        raise ValueError("Seeds must be a subset of the preregistered %r" % config["seeds"])

    for dataset in datasets:
        for variant in variants:
            for seed in seeds:
                run_one(repo, config_path, config, dataset, variant, seed, args.force)


if __name__ == "__main__":
    main()
