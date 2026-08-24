#!/usr/bin/env python3
"""Close the SCP2176 melanoma K=2 protocol and make sanitized numeric H5ADs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_max_abs_difference(left: object, right: object) -> float:
    left = sp.csr_matrix(left)
    right = sp.csr_matrix(right)
    if left.shape != right.shape:
        return float("inf")
    delta = left - right
    return float(np.max(np.abs(delta.data))) if delta.nnz else 0.0


def read_annotations(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or set(rows[0]) != {"NAME", "cluster"}:
        raise ValueError("unexpected SCP2176 annotation schema")
    ids = [row["NAME"] for row in rows]
    mapping = {row["NAME"]: row["cluster"] for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("duplicate SCP2176 annotation IDs")
    return ids, mapping


def read_spatial(path: Path) -> tuple[list[str], dict[str, tuple[float, float]], list[str]]:
    payload = json.loads(path.read_text())
    data = payload["data"]
    ids = [str(value) for value in data["cells"]]
    coordinates = {
        identifier: (float(x), float(y))
        for identifier, x, y in zip(ids, data["x"], data["y"])
    }
    annotations = [str(value) for value in data["annotations"]]
    if not (len(ids) == len(coordinates) == len(annotations)):
        raise ValueError("SCP2176 spatial carrier length mismatch")
    return ids, coordinates, annotations


def sanitized_copy(source: ad.AnnData, ids: np.ndarray, coordinates: np.ndarray) -> ad.AnnData:
    result = ad.AnnData(
        X=source.X.copy(),
        obs=None,
        var=source.var.copy(),
        obsm={"spatial": np.asarray(coordinates, dtype=np.float64)},
    )
    result.obs_names = ids.astype(str)
    result.var_names = source.var_names.astype(str)
    result.uns["night16f_numeric_carrier"] = {
        "annotation_columns_removed": True,
        "formal_producer_label_reads": 0,
        "coordinate_authority": "SCP2176 official spatial endpoint",
    }
    return result


def run(args: argparse.Namespace) -> None:
    paths = {name: Path(getattr(args, name)) for name in (
        "figshare_rna", "figshare_atac", "official_rna", "official_atac",
        "annotations", "spatial",
    )}
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    fig_rna = ad.read_h5ad(paths["figshare_rna"])
    fig_atac = ad.read_h5ad(paths["figshare_atac"])
    official_rna = ad.read_h5ad(paths["official_rna"])
    official_atac = ad.read_h5ad(paths["official_atac"])
    if not (fig_rna.n_obs == fig_atac.n_obs == official_rna.n_obs == official_atac.n_obs == 833):
        raise ValueError("official tumor-only processed carrier is not 833 cells")
    if fig_rna.shape != official_rna.shape or fig_atac.shape != official_atac.shape:
        raise ValueError("Figshare/official-code carrier shapes differ")
    # Figshare also replaces feature names by exact positional placeholders.
    # Require those placeholders and use the row/column-identical official
    # reproduce carriers for the biological feature names and metadata.
    expected_rna_feature_ids = np.asarray([str(index) for index in range(fig_rna.n_vars)])
    expected_atac_feature_ids = np.asarray([str(index) for index in range(fig_atac.n_vars)])
    if not np.array_equal(np.asarray(fig_rna.var_names.astype(str)), expected_rna_feature_ids):
        raise ValueError("Figshare RNA placeholder feature order is unexpected")
    if not np.array_equal(np.asarray(fig_atac.var_names.astype(str)), expected_atac_feature_ids):
        raise ValueError("Figshare ATAC placeholder feature order is unexpected")
    if len(np.unique(np.asarray(official_rna.var_names.astype(str)))) != official_rna.n_vars:
        raise ValueError("official-code RNA feature IDs are non-unique")
    if len(np.unique(np.asarray(official_atac.var_names.astype(str)))) != official_atac.n_vars:
        raise ValueError("official-code ATAC feature IDs are non-unique")
    # The Figshare export deliberately uses positional obs_names 0..832,
    # whereas the official reproduce carrier restores biological barcodes.
    # Require the placeholder order explicitly before transferring the exact
    # row-wise barcode carrier; never infer this mapping from a filename.
    expected_positional_ids = np.asarray([str(index) for index in range(833)])
    if not np.array_equal(np.asarray(fig_rna.obs_names.astype(str)), expected_positional_ids):
        raise ValueError("Figshare RNA placeholder observation order is unexpected")
    if not np.array_equal(np.asarray(fig_atac.obs_names.astype(str)), expected_positional_ids):
        raise ValueError("Figshare ATAC placeholder observation order is unexpected")
    if matrix_max_abs_difference(fig_rna.X, official_rna.X) != 0:
        raise ValueError("Figshare RNA numeric matrix differs from official-code carrier")
    if matrix_max_abs_difference(fig_atac.X, official_atac.X) != 0:
        raise ValueError("Figshare ATAC numeric matrix differs from official-code carrier")
    ids = np.asarray(official_rna.obs_names.astype(str))
    atac_ids = np.asarray(official_atac.obs_names.astype(str))
    if not np.array_equal(ids, atac_ids) or len(np.unique(ids)) != len(ids):
        raise ValueError("official RNA/ATAC IDs are not byte-exact aligned and unique")

    annotation_ids, annotation_map = read_annotations(paths["annotations"])
    spatial_ids, spatial_map, spatial_annotations = read_spatial(paths["spatial"])
    if annotation_ids != spatial_ids:
        raise ValueError("official annotation/spatial endpoint orders differ")
    if any(annotation_map[identifier] != label for identifier, label in zip(spatial_ids, spatial_annotations)):
        raise ValueError("official annotation/spatial endpoint labels differ")
    if not set(ids).issubset(annotation_map) or not set(ids).issubset(spatial_map):
        raise ValueError("processed tumor carrier IDs are absent from SCP2176 authority")
    labels = np.asarray([annotation_map[identifier] for identifier in ids])
    if set(labels) != {"tumour_1", "tumour_2"}:
        raise ValueError("processed carrier is not the registered tumor-only K=2 mask")
    coordinates = np.asarray([spatial_map[identifier] for identifier in ids], dtype=np.float64)

    rna_output = output / "HumanMelanoma_RNA_sanitized.h5ad"
    atac_output = output / "HumanMelanoma_ATAC_lsi_sanitized.h5ad"
    sanitized_copy(official_rna, ids, coordinates).write_h5ad(rna_output, compression="gzip")
    sanitized_copy(official_atac, ids, coordinates).write_h5ad(atac_output, compression="gzip")
    with (output / "tumor_k2_reference.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["cell_id", "public_author_cluster"])
        writer.writerows(zip(ids, labels))

    all_counts: dict[str, int] = {}
    for label in annotation_map.values():
        all_counts[label] = all_counts.get(label, 0) + 1
    tumor_counts = {label: int(np.sum(labels == label)) for label in sorted(set(labels))}
    omitted = [identifier for identifier in annotation_ids if identifier not in set(ids)]
    reference_hash = hashlib.sha256(
        b"\0".join(f"{identifier}\t{annotation_map[identifier]}".encode() for identifier in ids)
    ).hexdigest()
    registry = {
        "schema": "night16f-scp2176-protocol-registry-v1",
        "study": "SCP2176 Slide-tags human metastatic melanoma multiome",
        "authority_cell_count": len(annotation_ids),
        "authority_category_counts": all_counts,
        "authority_first_id": annotation_ids[0],
        "authority_last_id": annotation_ids[-1],
        "processed_cell_count": len(ids),
        "figshare_observation_ids": (
            "exact positional placeholders 0..832; biological barcode order transferred only after "
            "exact matrix-shape and row/column numeric identity to the official reproduce carrier"
        ),
        "figshare_feature_ids": (
            "exact positional placeholders; biological feature IDs transferred from the official "
            "reproduce carrier after exact matrix-shape and row/column numeric identity"
        ),
        "processed_id_intersection_with_authority": int(sum(x in annotation_map for x in ids)),
        "omitted_authority_cells": len(omitted),
        "omitted_id_sha256": hashlib.sha256(b"\0".join(x.encode() for x in omitted)).hexdigest(),
        "prompt_expected_2529_disposition": (
            "DISPROVEN_BY_EXACT_OFFICIAL_FIGSHARE_FILES; both processed H5ADs contain 833 tumor cells"
        ),
        "protocols": {
            "ALL_CELL_K10": {
                "status": "DATA_INPUT_BLOCKED_AUTH_REQUIRED",
                "n_labels_and_coordinates": len(annotation_ids),
                "k": 10,
                "label_type": "public computational cell-type/tumor-state annotation",
                "reason": (
                    "anonymous SCP endpoints expose labels/coordinates, but exact full RNA/ATAC matrices "
                    "require authenticated SCP asset access; Figshare processed H5ADs are tumor-only"
                ),
            },
            "TUMOR_ONLY_K2": {
                "status": "READY",
                "n": len(ids),
                "k": 2,
                "label_counts": tumor_counts,
                "label_type": "public author-derived computational tumor-state/compartment annotation",
                "ordered_reference_sha256": reference_hash,
                "mask": "exact ID intersection of official 833-cell processed carrier with SCP2176 annotations",
            },
        },
        "input_files": {
            key: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": file_hash(path),
                "md5": file_hash(path, "md5"),
            }
            for key, path in paths.items()
        },
        "sanitized_files": {
            path.name: {"size": path.stat().st_size, "sha256": file_hash(path)}
            for path in (rna_output, atac_output)
        },
        "explicit_id_alignment": "PASS",
        "coordinate_alignment": "PASS",
        "formal_numeric_carrier_obs_columns": [],
    }
    (output / "scp2176_protocol_registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figshare-rna", required=True)
    parser.add_argument("--figshare-atac", required=True)
    parser.add_argument("--official-rna", required=True)
    parser.add_argument("--official-atac", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--spatial", required=True)
    parser.add_argument("--output-root", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
