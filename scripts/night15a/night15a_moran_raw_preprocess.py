#!/usr/bin/env python3
"""Spatially screen raw RNA/ATAC features without using annotations.

This is a clean-room, sparse implementation of a standard Moran feature bank.
The use of Moran-ranked features is attributed to the SEPAR tutorial, while the
implementation and downstream MCDF model remain project code.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from night15a_raw_preprocess import (
    array_sha256,
    atomic_json,
    file_sha256,
    load_misar,
    load_p22,
    normalize_atac,
    normalize_rna,
    ordered_id_sha256,
    select_features,
    sparse_stats,
    text_sequence_sha256,
)
from SpaLORA.night14b_atac import spatial_operator


def moran_scores(value: sp.csr_matrix, operator: sp.csr_matrix, block: int = 256) -> np.ndarray:
    """Return column-wise Moran ratios using only N x block dense work arrays."""
    if value.shape[0] != operator.shape[0] or operator.shape[0] != operator.shape[1]:
        raise ValueError("Moran graph and feature matrix dimensions differ")
    scores = np.full(value.shape[1], -np.inf, dtype=np.float64)
    scale = float(value.shape[0] / operator.sum())
    for start in range(0, value.shape[1], int(block)):
        stop = min(value.shape[1], start + int(block))
        dense = value[:, start:stop].toarray().astype(np.float64, copy=False)
        dense -= dense.mean(axis=0, keepdims=True)
        denominator = np.einsum("ij,ij->j", dense, dense)
        lag = operator @ dense
        numerator = np.einsum("ij,ij->j", dense, lag)
        valid = denominator > 1e-12
        block_scores = np.full(stop - start, -np.inf, dtype=np.float64)
        block_scores[valid] = scale * numerator[valid] / denominator[valid]
        scores[start:stop] = block_scores
    return scores


def select_moran(
    value: sp.csr_matrix,
    features: np.ndarray,
    operator: sp.csr_matrix,
    variance_cap: int,
    moran_cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    variance_screened, variance_index = select_features(value, int(variance_cap))
    scores = moran_scores(variance_screened, operator)
    finite = np.isfinite(scores)
    if int(finite.sum()) < int(moran_cap):
        raise RuntimeError("too few finite Moran features")
    ranking = np.argsort(scores, kind="mergesort")[::-1]
    ranking = ranking[np.isfinite(scores[ranking])][: int(moran_cap)]
    selected_index = variance_index[ranking]
    selected = value[:, selected_index].toarray().astype(np.float32)
    mean = selected.mean(axis=0, keepdims=True)
    standard = selected.std(axis=0, keepdims=True)
    standard[standard < 1e-6] = 1.0
    selected = np.clip((selected - mean) / standard, -8.0, 8.0).astype(np.float32)
    return selected, selected_index, scores[ranking], {
        "variance_screen_shape": list(variance_screened.shape),
        "variance_cap": int(variance_cap),
        "moran_cap": int(moran_cap),
        "moran_best": float(scores[ranking[0]]),
        "moran_median_selected": float(np.median(scores[ranking])),
        "moran_min_selected": float(scores[ranking[-1]]),
        "selected_feature_ids_sha256": text_sequence_sha256(
            np.asarray(features)[selected_index]
        ),
    }


def run(
    dataset: str,
    output: Path,
    graph_k: int,
    rna_variance_cap: int,
    atac_variance_cap: int,
    rna_moran_cap: int,
    atac_moran_cap: int,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    raw = load_p22() if dataset == "P22" else load_misar()
    rna_stats = sparse_stats(raw["rna"])
    atac_stats = sparse_stats(raw["atac"])
    rna, rna_transform = normalize_rna(raw["rna"])
    atac, atac_transform = normalize_atac(raw["atac"])
    operator = spatial_operator(raw["coordinates"], int(graph_k)).tocsr()
    x1, rna_index, rna_scores, rna_audit = select_moran(
        rna,
        raw["rna_features"],
        operator,
        rna_variance_cap,
        rna_moran_cap,
    )
    x2, atac_index, atac_scores, atac_audit = select_moran(
        atac,
        raw["atac_features"],
        operator,
        atac_variance_cap,
        atac_moran_cap,
    )
    archive = output / "preprocessed_feature_level.npz"
    np.savez_compressed(
        archive,
        x1=x1,
        x2=x2,
        coordinates=np.asarray(raw["coordinates"], dtype=np.float64),
        ids=np.asarray(raw["ids"], dtype=str),
        rna_feature_index=rna_index,
        atac_feature_index=atac_index,
        rna_moran_scores=rna_scores,
        atac_moran_scores=atac_scores,
    )
    audit = {
        "dataset": raw["dataset"],
        "preprocessor_id": "MORAN_RAW_BANK_V1",
        "upstream_mechanism_attribution": {
            "mechanism": "Moran-ranked spatial feature selection",
            "source": "SEPAR Tutorial4 and SEPAR_model.py",
            "commit": "6d3475fa0bd749d3b1b5592b68323439d473f9fc",
            "license": "MIT",
            "implementation": "clean-room sparse block implementation",
        },
        "training_labels_read": 0,
        "evaluation_labels_read": 0,
        "labels_in_preprocessing": False,
        "ordered_id_sha256": ordered_id_sha256(raw["ids"]),
        "coordinate_sha256": array_sha256(np.asarray(raw["coordinates"])),
        "graph_k": int(graph_k),
        "graph_nnz": int(operator.nnz),
        "dense_n_by_n_count": 0,
        "raw_feature_shapes": {"rna": rna_stats, "atac": atac_stats},
        "transforms": {"rna": rna_transform, "atac": atac_transform},
        "selection": {"rna": rna_audit, "atac": atac_audit},
        "processed": {
            "rna": {"shape": list(x1.shape), "sha256": array_sha256(x1)},
            "atac": {"shape": list(x2.shape), "sha256": array_sha256(x2)},
        },
        "source_files": [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in raw["source_paths"]
        ],
        "archive_path": str(archive),
        "archive_size": archive.stat().st_size,
        "archive_sha256": file_sha256(archive),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "raw_feature_preflight.json", audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("P22", "MISAR_E15_5_S1"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--rna-variance-cap", type=int, default=6000)
    parser.add_argument("--atac-variance-cap", type=int, default=15000)
    parser.add_argument("--rna-moran-cap", type=int, default=1000)
    parser.add_argument("--atac-moran-cap", type=int, default=1000)
    args = parser.parse_args()
    run(
        args.dataset,
        Path(args.output),
        args.graph_k,
        args.rna_variance_cap,
        args.atac_variance_cap,
        args.rna_moran_cap,
        args.atac_moran_cap,
    )


if __name__ == "__main__":
    main()
