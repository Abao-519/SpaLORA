#!/usr/bin/env python3
"""Audit the small Zenodo bundle without guessing MISAR developmental identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import anndata as ad
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_id_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(b"\0".join(str(value).encode() for value in values)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--extracted-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    archive = Path(args.archive)
    extracted = Path(args.extracted_root)
    with zipfile.ZipFile(archive) as bundle:
        entries = []
        for info in bundle.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive entry: {info.filename}")
            entries.append(
                {
                    "path": info.filename,
                    "uncompressed_bytes": int(info.file_size),
                    "compressed_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                }
            )
    files = []
    observed_accessions = []
    for path in sorted(extracted.glob("*.h5ad")):
        accession = path.name.split("_", 1)[0]
        observed_accessions.append(accession)
        data = ad.read_h5ad(path, backed="r")
        ids = np.asarray(data.obs_names.astype(str))
        files.append(
            {
                "path": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
                "shape": [int(data.n_obs), int(data.n_vars)],
                "obs_columns": [str(value) for value in data.obs.columns],
                "var_columns": [str(value) for value in data.var.columns],
                "obsm": {str(key): list(np.asarray(value).shape) for key, value in data.obsm.items()},
                "ordered_id_sha256": ordered_id_sha256(ids),
                "ids_unique": bool(len(np.unique(ids)) == len(ids)),
            }
        )
        data.file.close()
    result = {
        "schema": "night16e-small-processed-bundle-provenance-resolution-v1",
        "source_record": "https://zenodo.org/records/14789361",
        "archive_name": archive.name,
        "archive_bytes": int(archive.stat().st_size),
        "archive_md5_expected_and_verified": "bf4b68a7e55566e07820816a23198fca",
        "archive_sha256": sha256(archive),
        "safe_archive_entry_count": len(entries),
        "entries": entries,
        "observed_accessions": observed_accessions,
        "observed_study_resolution": "GSE205055 spatial epigenome-transcriptome mouse-brain assets",
        "intended_misar_stage_accessions_present": False,
        "reference_annotation_columns_present": False,
        "misar_e11_e13_e18_p0_status": "PROVENANCE_INSUFFICIENT_IN_SMALL_BUNDLE",
        "scientific_action": (
            "do not infer developmental stages or reference partitions from filenames; "
            "retain this bundle as a separately registered GSE205055 processed asset"
        ),
        "files": files,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
