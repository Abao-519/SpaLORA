#!/usr/bin/env python3
"""Label-sealed external dataset provenance and integrity preflight for Night-8A."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import h5py


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/night8a_handoff"
DATA = Path("/root/autodl-fs/night8a_external_data_20260820")


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def zenodo(record_id: str) -> dict:
    path = DATA / f"zenodo{record_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload["id"]) != record_id:
        raise RuntimeError(f"Zenodo record mismatch for {record_id}")
    return {
        "record_id": record_id,
        "record_url": f"https://zenodo.org/records/{record_id}",
        "title": payload["metadata"]["title"],
        "metadata_snapshot": str(path),
        "metadata_snapshot_sha256": digest(path),
        "files": [
            {"name": item["key"], "bytes": int(item["size"]), "checksum": item["checksum"]}
            for item in payload["files"]
        ],
    }


def geo(accession: str) -> dict:
    path = DATA / f"{accession}_brief.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    title = next(x.split(" = ", 1)[1] for x in lines if x.startswith("!Series_title = "))
    files = [x.split(" = ", 1)[1] for x in lines if x.startswith("!Series_supplementary_file = ")]
    return {
        "accession": accession,
        "record_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
        "title": title,
        "metadata_snapshot": str(path),
        "metadata_snapshot_sha256": digest(path),
        "supplementary_files": files,
        "downloaded_in_night8a": False,
    }


def misar_lock() -> tuple[dict, dict]:
    matrix = DATA / "E15_5-S1_raw_feature_bc_matrix.h5"
    positions = DATA / "position_E15_5-S1.txt"
    barcode_map = DATA / "MISAR-seq_barcode_filter.csv"
    barcode_grid = DATA / "MISAR-seq_barcode.csv"
    expected = {
        matrix.name: (48966670, "md5:85068d59bfecc7ff19394594bb94b071"),
        positions.name: (11083, "md5:ab7cc4c996b29ab7a4a60838933141f8"),
        barcode_map.name: (64115, "md5:73dc8a0fad504cc883ae394669fb8fe0"),
        barcode_grid.name: (64129, "md5:9404a58a3252a181b0de6ab613ce990b"),
    }
    files = []
    for path in (matrix, positions, barcode_map, barcode_grid):
        size, checksum = expected[path.name]
        actual_md5 = digest(path, "md5")
        if path.stat().st_size != size or f"md5:{actual_md5}" != checksum:
            raise RuntimeError(f"official MISAR file mismatch: {path.name}")
        files.append({
            "name": path.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "official_checksum": checksum,
        })

    with h5py.File(matrix, "r") as handle:
        group = handle["matrix"]
        shape = [int(x) for x in group["shape"][:]]
        barcodes = [x.decode() if isinstance(x, bytes) else str(x) for x in group["barcodes"][:]]
        feature_types = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in group["features"]["feature_type"][:]
        ]
        nnz = int(group["data"].shape[0])

    grid_rows = list(csv.DictReader(barcode_grid.open(encoding="utf-8", newline="")))
    filter_rows = list(csv.DictReader(barcode_map.open(encoding="utf-8", newline="")))
    array_to_barcode = {row["array"]: row["barcode"] for row in filter_rows}
    selected_arrays = [x for x in positions.read_text(encoding="utf-8").strip().split(",") if x]
    if len(set(selected_arrays)) != len(selected_arrays):
        raise RuntimeError("duplicate position identifier")
    missing_arrays = sorted(set(selected_arrays) - set(array_to_barcode))
    selected_barcodes = [array_to_barcode[x] for x in selected_arrays if x in array_to_barcode]
    missing_barcodes = sorted(set(selected_barcodes) - set(barcodes))
    if missing_arrays or missing_barcodes:
        raise RuntimeError("MISAR spot/coordinate alignment mismatch")
    grid_pairs = {(int(row["array_col"]), int(row["array_row"])) for row in grid_rows}
    coordinate_pairs = {tuple(map(int, value.split("x"))) for value in selected_arrays}
    if not coordinate_pairs.issubset(grid_pairs):
        raise RuntimeError("MISAR coordinate grid mismatch")

    counts = dict(Counter(feature_types))
    if shape != [173705, 2500] or counts != {"Gene Expression": 32285, "Peaks": 141420}:
        raise RuntimeError("MISAR matrix modality contract mismatch")
    if len(selected_arrays) != 1949:
        raise RuntimeError(f"MISAR expected 1949 tissue spots, got {len(selected_arrays)}")

    integrity = {
        "status": "PASS_LABEL_SEALED",
        "matrix_shape_features_by_all_barcodes": shape,
        "matrix_nnz": nnz,
        "all_barcodes": len(barcodes),
        "selected_tissue_spots": len(selected_arrays),
        "selected_barcodes_unique": len(set(selected_barcodes)),
        "coordinates_present_and_unique": True,
        "rna_features": counts["Gene Expression"],
        "atac_peak_features": counts["Peaks"],
        "rna_atac_same_spot_alignment": True,
        "annotation_values_deserialized": False,
        "files": files,
    }
    lock = {
        "status": "LOCKED_FOR_FUTURE_NIGHT8B_CONFIRMATION",
        "dataset_id": "MISAR_E15_5_S1",
        "family": "RNA_EPIGENOME",
        "primary_accession": "OEP003285",
        "official_processed_release": "Zenodo 7480069",
        "cross_reference": "GSE213264 / SRP491963 raw reads",
        "stage": "E15.5",
        "section": "S1",
        "spots": len(selected_arrays),
        "features": {"RNA": counts["Gene Expression"], "ATAC_peaks": counts["Peaks"]},
        "coordinates": {"available": True, "grid_axis_fields": ["array_col", "array_row"]},
        "modalities_same_spot_aligned": True,
        "annotation_provenance": {
            "classification": "manual anatomical regions with atlas-guided interpretation; not treated as a pristine expert gold standard",
            "night8a_use": "sealed; no values or distribution read",
            "night8b_requirement": "re-audit the exact annotation file and provenance before the one-time evaluation window",
        },
        "labels_sealed": True,
        "formal_benchmark_run_in_night8a": False,
        "files": files,
    }
    return integrity, lock


def main() -> None:
    primary_metadata = zenodo("7480069")
    smart_metadata = zenodo("17093158")
    tonsil_metadata = zenodo("12654113")
    starmap_metadata = zenodo("8041114")
    integrity, lock = misar_lock()
    preflight = {
        "status": "PASS_PRIMARY_EXTERNAL_DATASET_LOCKED_LABEL_SEALED",
        "authorized_role": "night8a_external_data_steward",
        "formal_external_training_or_benchmark": False,
        "label_files_opened": False,
        "label_values_or_distribution_read": False,
        "windows_new_download_bytes": sum(x["bytes"] for x in integrity["files"]),
        "windows_download_limit_bytes": 104857600,
        "primary": {
            "dataset": "MISAR-seq mouse brain E15.5 S1",
            "integrity": integrity,
            "official_release_metadata": primary_metadata,
            "SMART_processed_bundle_metadata_only": smart_metadata,
            "SMART_bundle_not_downloaded_reason": "10.74 GB monolithic archive; outside Night-8A selective preflight scope",
        },
        "secondary": {
            "GSE205055_mouse_embryo_E13": geo("GSE205055"),
            "Zenodo_12654113_tonsil_sections_2_3": {
                **tonsil_metadata,
                "downloaded_in_night8a": False,
                "role": "same-tissue section replication; metadata-only preflight",
            },
            "GSE198353_SPOTS_spleen": {
                **geo("GSE198353"),
                "role": "label-free replication candidate; metadata-only preflight",
            },
        },
        "large_scale_deferred": {
            **starmap_metadata,
            "downloaded_in_night8a": False,
            "reason": "approximately 58k aligned spots and very large matrix files; registered only by protocol",
        },
    }
    if preflight["windows_new_download_bytes"] > preflight["windows_download_limit_bytes"]:
        raise RuntimeError("Windows external download budget exceeded")
    atomic_json(OUT / "external_dataset_preflight.json", preflight)
    atomic_json(OUT / "external_dataset_lock.json", lock)
    print(json.dumps({"status": preflight["status"], "locked": lock["dataset_id"], "spots": lock["spots"]}, sort_keys=True))


if __name__ == "__main__":
    main()
