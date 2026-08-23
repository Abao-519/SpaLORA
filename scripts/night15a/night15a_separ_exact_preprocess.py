#!/usr/bin/env python3
"""Build the MISAR lane directly from the official SEPAR/MISAR 10x carrier.

The output keeps the registered 1,949 project observations in byte-exact order,
but obtains both molecular views from Zenodo record 7480069.  No ``Y`` or other
annotation dataset is opened by this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from night15a_raw_preprocess import (
    array_sha256,
    atomic_json,
    file_sha256,
    normalize_atac,
    normalize_rna,
    ordered_id_sha256,
    reduce,
    select_features,
    sparse_stats,
    text_sequence_sha256,
)


PROJECT_ROOT = Path("/root/autodl-fs/night8b_raw_runs_20260820")
OFFICIAL_ROOT = Path(
    "/root/autodl-fs/night15a_multimodal_contribution_and_score_stability_20260823/"
    "protocol_inputs/misar_zenodo_7480069"
)
MATRIX = OFFICIAL_ROOT / "E15_5-S1_raw_feature_bc_matrix.h5"
BARCODE_MAP = OFFICIAL_ROOT / "MISAR-seq_barcode_filter.csv"
TISSUE_ARRAYS = OFFICIAL_ROOT / "position_E15_5-S1.txt"


def _decode(values) -> np.ndarray:
    return np.asarray(
        [item.decode() if isinstance(item, bytes) else str(item) for item in values],
        dtype=str,
    )


def official_tissue_order() -> tuple[np.ndarray, np.ndarray]:
    with TISSUE_ARRAYS.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.reader(handle))
    arrays = np.asarray([item for item in row if item], dtype=str)
    mapping = pd.read_csv(BARCODE_MAP, dtype=str)
    if list(mapping.columns) != ["array", "barcode"]:
        raise RuntimeError("unexpected official barcode-map schema")
    if not mapping["array"].is_unique or not mapping["barcode"].is_unique:
        raise RuntimeError("official barcode map is not one-to-one")
    lookup = dict(zip(mapping["array"], mapping["barcode"]))
    if not set(arrays).issubset(lookup):
        raise RuntimeError("official tissue-array entry is absent from barcode map")
    return arrays, np.asarray([lookup[item] for item in arrays], dtype=str)


def load_exact_matrix() -> dict:
    project_ids = (
        pd.read_csv(PROJECT_ROOT / "cache/base/observation_ids.tsv", sep="\t")
        .iloc[:, 0]
        .astype(str)
        .to_numpy()
    )
    arrays, tissue_barcodes = official_tissue_order()
    if not np.array_equal(tissue_barcodes, project_ids):
        raise RuntimeError("official MISAR tissue order differs from project authority")

    with h5py.File(str(MATRIX), "r") as handle:
        matrix = handle["matrix"]
        raw_barcodes = _decode(matrix["barcodes"][:])
        if len(set(raw_barcodes.tolist())) != len(raw_barcodes):
            raise RuntimeError("official raw barcodes are not unique")
        column_index = pd.Index(raw_barcodes).get_indexer(project_ids)
        if np.any(column_index < 0):
            raise RuntimeError("registered observation is missing from official raw matrix")
        shape = tuple(int(item) for item in matrix["shape"][:])
        full = sp.csc_matrix(
            (
                np.asarray(matrix["data"][:], dtype=np.float32),
                np.asarray(matrix["indices"][:], dtype=np.int64),
                np.asarray(matrix["indptr"][:], dtype=np.int64),
            ),
            shape=shape,
        )
        selected = full[:, column_index].T.tocsr()
        feature_type = _decode(matrix["features/feature_type"][:])
        feature_id = _decode(matrix["features/id"][:])
        feature_name = _decode(matrix["features/name"][:])
        genome = _decode(matrix["features/genome"][:])

    rna_mask = feature_type == "Gene Expression"
    atac_mask = feature_type == "Peaks"
    if int(rna_mask.sum()) != 32285 or int(atac_mask.sum()) != 141420:
        raise RuntimeError("official MISAR modality dimensions changed")
    if set(genome.tolist()) != {"mm10"}:
        raise RuntimeError("official MISAR genome is not uniquely mm10")
    coordinates = np.load(
        PROJECT_ROOT / "cache/base/coordinates.npy", allow_pickle=False
    ).astype(np.float64)
    if coordinates.shape != (len(project_ids), 2):
        raise RuntimeError("registered coordinate shape mismatch")
    return {
        "ids": project_ids,
        "arrays": arrays,
        "coordinates": coordinates,
        "rna": selected[:, rna_mask].tocsr(),
        "atac": selected[:, atac_mask].tocsr(),
        "rna_features": feature_name[rna_mask],
        "rna_feature_ids": feature_id[rna_mask],
        "atac_features": feature_name[atac_mask],
        "atac_feature_ids": feature_id[atac_mask],
        "raw_matrix_shape": list(shape),
    }


def run(output: Path, seed: int, dimension: int, rna_cap: int, atac_cap: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    raw = load_exact_matrix()
    rna_stats = sparse_stats(raw["rna"])
    atac_stats = sparse_stats(raw["atac"])
    rna, rna_transform = normalize_rna(raw["rna"])
    atac, atac_transform = normalize_atac(raw["atac"])
    rna, rna_index = select_features(rna, int(rna_cap))
    atac, atac_index = select_features(atac, int(atac_cap))
    rna_reduced, rna_audit = reduce(rna, dimension, seed)
    atac_reduced, atac_audit = reduce(atac, dimension, seed)

    archive = output / "preprocessed_feature_level.npz"
    np.savez_compressed(
        archive,
        x1=rna_reduced,
        x2=atac_reduced,
        coordinates=raw["coordinates"],
        ids=np.asarray(raw["ids"], dtype=str),
        rna_feature_index=rna_index,
        atac_feature_index=atac_index,
    )
    audit = {
        "dataset": "MISAR_E15_5_S1",
        "source_protocol": "SEPAR_MISAR_OFFICIAL_ZENODO_7480069_EXACT_RAW",
        "seed": int(seed),
        "training_labels_read": 0,
        "evaluation_labels_read": 0,
        "labels_in_preprocessing": False,
        "official_tissue_order_matches_project_byte_exact": True,
        "ordered_id_sha256": ordered_id_sha256(raw["ids"]),
        "array_order_sha256": text_sequence_sha256(raw["arrays"]),
        "coordinate_sha256": array_sha256(raw["coordinates"]),
        "official_raw_matrix_shape": raw["raw_matrix_shape"],
        "raw_feature_shapes": {"rna": rna_stats, "atac": atac_stats},
        "registered_transforms": {
            "rna": rna_transform,
            "atac": atac_transform,
            "rna_feature_cap": int(rna_cap),
            "atac_feature_cap": int(atac_cap),
            "rna_selected_feature_names_sha256": text_sequence_sha256(
                raw["rna_features"][rna_index]
            ),
            "rna_selected_feature_ids_sha256": text_sequence_sha256(
                raw["rna_feature_ids"][rna_index]
            ),
            "atac_selected_feature_names_sha256": text_sequence_sha256(
                raw["atac_features"][atac_index]
            ),
            "atac_selected_feature_ids_sha256": text_sequence_sha256(
                raw["atac_feature_ids"][atac_index]
            ),
        },
        "processed": {
            "rna": {
                **rna_audit,
                "shape": list(rna_reduced.shape),
                "sha256": array_sha256(rna_reduced),
            },
            "atac": {
                **atac_audit,
                "shape": list(atac_reduced.shape),
                "sha256": array_sha256(atac_reduced),
            },
        },
        "source_files": [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (MATRIX, BARCODE_MAP, TISSUE_ARRAYS)
        ],
        "archive_path": str(archive),
        "archive_size": archive.stat().st_size,
        "archive_sha256": file_sha256(archive),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "raw_feature_preflight.json", audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--rna-cap", type=int, default=6000)
    parser.add_argument("--atac-cap", type=int, default=30000)
    args = parser.parse_args()
    run(Path(args.output), args.seed, args.dimension, args.rna_cap, args.atac_cap)


if __name__ == "__main__":
    main()
