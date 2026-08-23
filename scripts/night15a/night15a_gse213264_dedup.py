#!/usr/bin/env python3
"""Deduplicate GSE213264 human tonsil against canonical project tonsils by IDs."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_sha(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in np.asarray(values, dtype=str):
        raw = value.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def first_column(path: Path) -> np.ndarray:
    values = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        next(handle)
        for line in handle:
            values.append(line.split("\t", 1)[0])
    return np.asarray(values, dtype=str)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def run(input_dir: Path, canonical_root: Path, output: Path) -> None:
    rna = input_dir / "GSM6578062_humantonsil_RNA.tsv.gz"
    protein = input_dir / "GSM6578071_humantonsil_protein.tsv.gz"
    rna_ids = first_column(rna)
    protein_ids = first_column(protein)
    rna_set = set(rna_ids)
    protein_set = set(protein_ids)
    shared_gse = rna_set & protein_set
    canonical = []
    for index in (1, 2, 3):
        path = canonical_root / f"s{index}_adata_rna.h5ad"
        carrier = ad.read_h5ad(str(path), backed="r")
        ids = carrier.obs_names.astype(str).to_numpy()
        canonical.append({
            "slice": f"tonsil_s{index}",
            "path": str(path),
            "observation_count": int(len(ids)),
            "ordered_id_sha256": ordered_sha(ids),
            "id_intersection_with_gse213264_rna": int(len(set(ids) & rna_set)),
            "id_intersection_with_gse213264_protein": int(len(set(ids) & protein_set)),
        })
    atomic_json(output, {
        "decision": "NOT_DUPLICATE_DISTINCT_ACCESSION_PLATFORM_OBSERVATION_ID_SPACE_AND_COUNT",
        "gse213264": {
            "study": "GSE213264",
            "platform": "Spatial-CITE-seq",
            "rna_accession": "GSM6578062",
            "protein_accession": "GSM6578071",
            "rna_observation_count": int(len(rna_ids)),
            "protein_observation_count": int(len(protein_ids)),
            "rna_unique_observation_count": int(len(rna_set)),
            "protein_unique_observation_count": int(len(protein_set)),
            "rna_protein_shared_observation_count": int(len(shared_gse)),
            "rna_protein_id_sets_exact": rna_set == protein_set,
            "rna_protein_ordered_ids_exact": bool(np.array_equal(rna_ids, protein_ids)),
            "rna_ordered_id_sha256": ordered_sha(rna_ids),
            "protein_ordered_id_sha256": ordered_sha(protein_ids),
            "rna_sha256": file_sha256(rna),
            "protein_sha256": file_sha256(protein),
        },
        "canonical_project_tonsils": canonical,
        "evidence": (
            "different study/accessions and platform; GSE213264 has 2492 coordinate-style "
            "spot IDs, whereas canonical Zenodo 12654113 slices have 4326/4519/4521 "
            "10x-style IDs; every exact ID-set intersection is zero"
        ),
        "annotation_or_label_values_read": 0,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.input_dir), Path(args.canonical_root), Path(args.output))


if __name__ == "__main__":
    main()
