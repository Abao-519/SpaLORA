#!/usr/bin/env python3
"""Feature-level RNA/ATAC preprocessing for the two Night-15A lanes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence, Tuple

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/night13b")]

import night13b_run as n13b  # noqa: E402
from SpaLORA.night13b_unified import ordered_id_sha256  # noqa: E402
from SpaLORA.night15a_mcdf import array_sha256  # noqa: E402


P22_RNA = Path("/root/autodl-fs/P22 mouse brain coronal section/mousebrain_rna.h5ad")
P22_ATAC = Path("/root/autodl-fs/P22 mouse brain coronal section/mousebrain_atac.h5ad")
MISAR_ROOT = Path("/root/autodl-fs/night8b_raw_runs_20260820")
MISAR_RNA = MISAR_ROOT / "annotation_carrier/MISAR_seq_mouse_E15_brain_mRNA_data.h5"
MISAR_ATAC = MISAR_ROOT / "annotation_carrier/MISAR_seq_mouse_E15_brain_ATAC_data.h5"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        raw = str(value).encode("utf-8")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _as_csr(value) -> sp.csr_matrix:
    result = value.tocsr() if sp.issparse(value) else sp.csr_matrix(value)
    result = result.astype(np.float32)
    result.sum_duplicates()
    result.eliminate_zeros()
    return result


def sparse_stats(value: sp.csr_matrix) -> dict:
    data = value.data
    sample = data[: min(len(data), 1000000)]
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nnz": int(value.nnz),
        "density": float(value.nnz / (value.shape[0] * value.shape[1])),
        "minimum_nonzero": float(data.min()) if len(data) else 0.0,
        "maximum": float(data.max()) if len(data) else 0.0,
        "integer_fraction_sample": float(
            np.mean(np.isclose(sample, np.round(sample), atol=1e-6))
        )
        if len(sample)
        else 1.0,
    }


def feature_variance(value: sp.csr_matrix) -> np.ndarray:
    mean = np.asarray(value.mean(axis=0)).ravel()
    square = value.copy()
    square.data **= 2
    second = np.asarray(square.mean(axis=0)).ravel()
    return np.maximum(second - mean * mean, 0.0)


def select_features(value: sp.csr_matrix, cap: int) -> Tuple[sp.csr_matrix, np.ndarray]:
    if value.shape[1] <= int(cap):
        index = np.arange(value.shape[1], dtype=np.int64)
        return value, index
    variance = feature_variance(value)
    index = np.argpartition(variance, -int(cap))[-int(cap) :]
    index = index[np.argsort(variance[index], kind="mergesort")][::-1]
    return value[:, index].tocsr(), index.astype(np.int64)


def normalize_rna(value: sp.csr_matrix) -> Tuple[sp.csr_matrix, str]:
    stats = sparse_stats(value)
    if stats["integer_fraction_sample"] >= 0.99 and stats["minimum_nonzero"] >= 0:
        row_sum = np.asarray(value.sum(axis=1)).ravel()
        scale = 10000.0 / np.maximum(row_sum, 1.0)
        result = sp.diags(scale.astype(np.float32)) @ value
        result = result.tocsr()
        result.data = np.log1p(result.data)
        return result, "LIBRARY_1E4_LOG1P"
    return value, "REGISTERED_FEATURE_LEVEL_VALUES"


def normalize_atac(value: sp.csr_matrix) -> Tuple[sp.csr_matrix, str]:
    stats = sparse_stats(value)
    if stats["integer_fraction_sample"] >= 0.99 and stats["minimum_nonzero"] >= 0:
        row_sum = np.asarray(value.sum(axis=1)).ravel()
        tf = sp.diags(1.0 / np.maximum(row_sum, 1.0)) @ value
        document_frequency = np.diff(value.tocsc().indptr).astype(np.float64)
        idf = np.log1p(value.shape[0] / np.maximum(document_frequency, 1.0))
        result = (tf @ sp.diags(idf.astype(np.float32))).tocsr()
        result = normalize(result, norm="l2", axis=1, copy=False)
        return result.astype(np.float32), "TFIDF_L2"
    result = normalize(value, norm="l2", axis=1, copy=True)
    return result.astype(np.float32), "REGISTERED_VALUES_L2"


def reduce(value: sp.csr_matrix, dimension: int, seed: int) -> Tuple[np.ndarray, dict]:
    dimension = min(int(dimension), value.shape[0] - 1, value.shape[1] - 1)
    model = TruncatedSVD(n_components=dimension, random_state=int(seed), n_iter=7)
    reduced = model.fit_transform(value)
    reduced = StandardScaler().fit_transform(reduced).astype(np.float32)
    return reduced, {
        "dimension": int(dimension),
        "explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum()),
        "singular_values_sha256": array_sha256(model.singular_values_.astype(np.float64)),
    }


def load_p22():
    rna = ad.read_h5ad(str(P22_RNA))
    atac = ad.read_h5ad(str(P22_ATAC))
    registry = pd.read_csv(
        "/root/autodl-fs/P22 mouse brain coronal section/MouseBrain_groundtruth.csv",
        usecols=["Barcode"],
    )
    ids = registry["Barcode"].astype(str).to_numpy()
    if not set(ids).issubset(set(rna.obs_names.astype(str))):
        raise RuntimeError("P22 RNA observation IDs do not contain registered order")
    if not set(ids).issubset(set(atac.obs_names.astype(str))):
        raise RuntimeError("P22 ATAC observation IDs do not contain registered order")
    rna_x = _as_csr(rna[ids].X)
    atac_x = _as_csr(atac[ids].X)
    coordinates = np.asarray(rna[ids].obsm["spatial"], dtype=np.float64)
    if not np.array_equal(coordinates, np.asarray(atac[ids].obsm["spatial"], dtype=np.float64)):
        raise RuntimeError("P22 RNA/ATAC coordinates differ after ID alignment")
    return {
        "dataset": "P22",
        "ids": ids,
        "coordinates": coordinates,
        "rna": rna_x,
        "atac": atac_x,
        "rna_features": rna.var_names.astype(str).to_numpy(),
        "atac_features": atac.var_names.astype(str).to_numpy(),
        "source_paths": [P22_RNA, P22_ATAC],
    }


def _decode(values) -> np.ndarray:
    return np.asarray(
        [item.decode() if isinstance(item, bytes) else str(item) for item in values],
        dtype=str,
    )


def load_misar():
    ids = (
        pd.read_csv(MISAR_ROOT / "cache/base/observation_ids.tsv", sep="\t")
        .iloc[:, 0]
        .astype(str)
        .to_numpy()
    )
    with h5py.File(str(MISAR_RNA), "r") as handle:
        source_ids = _decode(handle["cell"][:])
        index = pd.Index(source_ids).get_indexer(ids)
        if np.any(index < 0):
            raise RuntimeError("MISAR RNA registered observation missing")
        rna_x = _as_csr(np.asarray(handle["X"][:], dtype=np.float32)[index])
        rna_features = _decode(handle["gene"][:])
        coordinates = np.asarray(handle["pos"][:], dtype=np.float64)[index]
    with h5py.File(str(MISAR_ATAC), "r") as handle:
        source_ids = _decode(handle["cell"][:])
        index = pd.Index(source_ids).get_indexer(ids)
        if np.any(index < 0):
            raise RuntimeError("MISAR ATAC registered observation missing")
        # Y is intentionally not read here; labels remain evaluator-only.
        atac_x = _as_csr(np.asarray(handle["X"][:], dtype=np.float32)[index])
        atac_features = _decode(handle["peak"][:])
        atac_coordinates = np.asarray(handle["pos"][:], dtype=np.float64)[index]
    registered_coordinates = np.load(
        MISAR_ROOT / "cache/base/coordinates.npy", allow_pickle=False
    ).astype(np.float64)
    if not np.array_equal(coordinates, atac_coordinates):
        raise RuntimeError("MISAR RNA/ATAC coordinates differ after ID alignment")
    # Registered coordinates can use a different affine scale; preserve the
    # project authority order and values for graphs and endpoint comparison.
    return {
        "dataset": "MISAR_E15_5_S1",
        "ids": ids,
        "coordinates": registered_coordinates,
        "rna": rna_x,
        "atac": atac_x,
        "rna_features": rna_features,
        "atac_features": atac_features,
        "source_paths": [MISAR_RNA, MISAR_ATAC],
        "carrier_coordinate_sha256": array_sha256(coordinates),
    }


def run(dataset: str, output: Path, seed: int, dimension: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    raw = load_p22() if dataset == "P22" else load_misar()
    rna_stats = sparse_stats(raw["rna"])
    atac_stats = sparse_stats(raw["atac"])
    rna, rna_transform = normalize_rna(raw["rna"])
    atac, atac_transform = normalize_atac(raw["atac"])
    rna, rna_index = select_features(rna, 3000)
    atac, atac_index = select_features(atac, 12000)
    rna_reduced, rna_audit = reduce(rna, dimension, seed)
    atac_reduced, atac_audit = reduce(atac, dimension, seed)
    archive = output / "preprocessed_feature_level.npz"
    np.savez_compressed(
        archive,
        x1=rna_reduced,
        x2=atac_reduced,
        coordinates=np.asarray(raw["coordinates"], dtype=np.float64),
        ids=np.asarray(raw["ids"], dtype=str),
        rna_feature_index=rna_index,
        atac_feature_index=atac_index,
    )
    audit = {
        "dataset": raw["dataset"],
        "seed": int(seed),
        "training_labels_read": 0,
        "labels_in_preprocessing": False,
        "ordered_id_sha256": ordered_id_sha256(raw["ids"]),
        "coordinate_sha256": array_sha256(np.asarray(raw["coordinates"])),
        "raw_feature_shapes": {"rna": rna_stats, "atac": atac_stats},
        "registered_transforms": {
            "rna": rna_transform,
            "atac": atac_transform,
            "rna_feature_cap": 3000,
            "atac_feature_cap": 12000,
            "rna_selected_feature_ids_sha256": text_sequence_sha256(
                np.asarray(raw["rna_features"])[rna_index]
            ),
            "atac_selected_feature_ids_sha256": text_sequence_sha256(
                np.asarray(raw["atac_features"])[atac_index]
            ),
        },
        "processed": {
            "rna": {**rna_audit, "shape": list(rna_reduced.shape), "sha256": array_sha256(rna_reduced)},
            "atac": {**atac_audit, "shape": list(atac_reduced.shape), "sha256": array_sha256(atac_reduced)},
        },
        "source_files": [
            {"path": str(path), "size": path.stat().st_size, "sha256": file_sha256(path)}
            for path in raw["source_paths"]
        ],
        "archive_path": str(archive),
        "archive_size": archive.stat().st_size,
        "archive_sha256": file_sha256(archive),
        "wall_seconds": time.perf_counter() - started,
    }
    if "carrier_coordinate_sha256" in raw:
        audit["carrier_coordinate_sha256"] = raw["carrier_coordinate_sha256"]
    atomic_json(output / "raw_feature_preflight.json", audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("P22", "MISAR_E15_5_S1"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--dimension", type=int, default=64)
    args = parser.parse_args()
    run(args.dataset, Path(args.output), args.seed, args.dimension)


if __name__ == "__main__":
    main()
