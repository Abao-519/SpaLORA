#!/usr/bin/env python3
"""Finalize a fail-closed Night-9C P0-DATA handoff without reading labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


ANNOTATION = "Annotation_for_Combined"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def command(*parts: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(parts, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def h5_shape_and_keys(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        if isinstance(x, h5py.Dataset):
            shape = list(map(int, x.shape))
        else:
            shape = list(map(int, x.attrs["shape"]))
        return {
            "shape": shape,
            "root_keys_names_only": sorted(map(str, handle.keys())),
            "obs_keys_names_only": sorted(map(str, handle["obs"].keys())),
            "obsm_keys_names_only": sorted(map(str, handle.get("obsm", {}).keys())),
            "annotation_key_present": f"obs/{ANNOTATION}" in handle,
            "annotation_values_read": False,
            "opened_with_anndata": False,
        }


def file_row(path: Path, *, include_h5: bool = False) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if include_h5:
        row["schema"] = h5_shape_and_keys(path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    attempt_dir = args.output_dir / "attempts/p0_attempt1_spaddm_public_blob"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    required = [
        "e18_5_data_provenance_and_firewall.json",
        "e18_5_source_file_manifest.json",
        "e18_5_label_free_copy_manifest.json",
        "night9c_p0_contract.json",
    ]
    for name in required:
        source = args.output_dir / name
        target = attempt_dir / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)

    paths = {
        "spaddm_rna": args.data_root / "source/E18_5_expr.h5ad",
        "spaddm_atac": args.data_root / "source/E18_5_atac.h5ad",
        "spamosaic_rna": args.data_root / "source_spamosaic_zenodo/E18_5_expr.h5ad",
        "spamosaic_atac": args.data_root / "source_spamosaic_zenodo/E18_5_atac.h5ad",
        "smart_rna": args.data_root / "source_smart_zenodo/E18_5_expr.h5ad",
        "smart_atac": args.data_root / "source_smart_zenodo/E18_5_atac.h5ad",
    }
    candidates = {name: file_row(path, include_h5=True) for name, path in paths.items()}

    supporting = {}
    for name in (
        "pre_download_filename_audit.txt",
        "pre_download_text_presence_paths.txt",
        "misar_official_baidu_inventory.json",
        "zenodo_15681100_record.json",
        "zenodo_15681100_Data_rar_index.json",
        "zenodo_16925549_record.json",
        "zenodo_16925549_SpaMosaic_data_zip_index.json",
        "zenodo_17093158_record.json",
        "zenodo_17093158_SMART_data_zip_index.json",
        "spamosaic_raw_meta_data.csv",
        "smart_e18_anno.csv",
    ):
        path = args.data_root / "p0_audit" / name
        if path.exists():
            supporting[name] = file_row(path)

    official_expected = {
        "spots": 2129,
        "rna_features": 32285,
        "atac_features": 294734,
        "annotation_key": ANNOTATION,
        "k": 10,
    }
    comparisons = {
        name: {
            "spots_match": row["schema"]["shape"][0] == official_expected["spots"],
            "feature_count": row["schema"]["shape"][1],
            "feature_count_matches_expected": row["schema"]["shape"][1]
            == (official_expected["rna_features"] if "rna" in name else official_expected["atac_features"]),
            "annotation_key_present": row["schema"]["annotation_key_present"],
        }
        for name, row in candidates.items()
    }
    exact_pair_available = any(
        comparisons[f"{prefix}_rna"]["feature_count_matches_expected"]
        and comparisons[f"{prefix}_atac"]["feature_count_matches_expected"]
        and comparisons[f"{prefix}_rna"]["annotation_key_present"]
        and comparisons[f"{prefix}_atac"]["annotation_key_present"]
        for prefix in ("spaddm", "spamosaic", "smart")
    )

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal_status": "BLOCKED_DATA_PROVENANCE",
        "scope": "Night-9C P0-DATA only",
        "reason": (
            "No public candidate simultaneously satisfies the preregistered 2129 x 32285 RNA, "
            "2129 x 294734 ATAC, and in-file obs/Annotation_for_Combined contract. "
            "Substitution, aliasing, reconstruction, or sidecar-label injection was not authorized."
        ),
        "official_expected_contract": official_expected,
        "candidate_files": candidates,
        "candidate_comparisons": comparisons,
        "supporting_evidence": supporting,
        "source_findings": [
            {
                "source": "MISAR-seq official Baidu share",
                "finding": "E18_5-S1.arrow is present, but the preregistered paired H5AD files are not present.",
                "label_values_read": False,
            },
            {
                "source": "SpaDDM public Git blobs and Zenodo 15681100",
                "finding": "RNA has 32285 features; ATAC has 117473 features; the required annotation key is absent.",
                "label_values_read": False,
            },
            {
                "source": "SpaMosaic reproduce branch and Zenodo 16925549",
                "finding": "E18 processed pair is 25740 RNA features and 191034 ATAC features; required annotation key is absent.",
                "label_values_read": False,
            },
            {
                "source": "SMART reproduce branch and Zenodo 17093158",
                "finding": "RNA has 32285 features; ATAC has 117473 features; required annotation key is absent and labels are a sidecar.",
                "label_values_read": False,
            },
            {
                "source": "PRESENT official tutorial and repository issues",
                "finding": (
                    "Tutorial documents the expected in-file schema, while the maintainer points requesters "
                    "to OEP003285 rather than publishing the tutorial H5AD pair."
                ),
                "label_values_read": False,
            },
        ],
        "annotation_provenance_tier": "TIER_B_PUBLISHED_REFERENCE_CLUSTER",
        "annotation_claim": "Published reference clusters with anatomical interpretation; not manual ground truth",
        "history_role": "PRISTINE_EXTERNAL_CONFIRMATION",
        "exact_pair_available": exact_pair_available,
        "prohibited_actions_not_taken": [
            "No alias from Combined_Clusters_annotation or a sidecar cluster column",
            "No reconstruction of a 294734-feature ATAC H5AD",
            "No annotation value access",
            "No training, embedding, partition, or evaluation",
            "No AutoDL API call",
        ],
        "label_firewall": {
            "annotation_value_reads": 0,
            "annotation_value_hashes": 0,
            "anndata_deserializations_of_label_bearing_source": 0,
            "authorized_evaluator_processes_started": 0,
        },
        "scientific_counts": {"training": 0, "embedding": 0, "partition": 0, "evaluation": 0},
    }
    atomic_json(args.output_dir / "p0_source_recovery_audit.json", audit)

    source_audit = {
        "status": "NOT_STARTED_P0_DATA_HARD_STOP",
        "smart": "repository and data locations identified only for P0 provenance; no model execution",
        "present": "repository and tutorial schema audited only for P0 provenance; no model execution",
        "candies": "SOURCE_UNAVAILABLE_NOT_EXECUTED",
        "sofusion": "SOURCE_UNAVAILABLE_NOT_EXECUTED",
        "label_values_read": False,
    }
    atomic_json(args.output_dir / "night9c_modern_source_code_audit.json", source_audit)
    atomic_text(
        args.output_dir / "night9c_modern_source_code_audit.md",
        "# Night-9C modern source-code audit\n\n"
        "Status: `NOT_STARTED_P0_DATA_HARD_STOP`.\n\n"
        "SMART and PRESENT repositories were located only as part of source-data provenance. "
        "P1 semantic adapters and model execution were not started because P0-DATA did not pass. "
        "CANDIES and soFusion were not reimplemented or executed.\n",
    )

    counts = {
        "terminal_status": "BLOCKED_DATA_PROVENANCE",
        "training_units": 0,
        "embeddings": 0,
        "partitions": 0,
        "label_value_reads": 0,
        "label_value_hashes": 0,
        "evaluation_processes": 0,
        "cuda_training_started": False,
        "gpu_training_seconds": 0,
    }
    atomic_json(args.output_dir / "scientific_budget_and_label_access.json", counts)

    report = f"""# SpaLORA Night-9C report

## Terminal status

`BLOCKED_DATA_PROVENANCE`

Night-9C stopped at P0-DATA. No scientific training, embedding, clustering, label evaluation, SMART run, or PRESENT run was started.

## Why the external confirmation did not run

The preregistered input contract requires a paired E18.5 S1 source with 2,129 spots, 32,285 RNA features, 294,734 ATAC features, and `obs/Annotation_for_Combined` in both source H5AD files. All accessible public candidates failed at least one of these immutable checks:

| Public source | RNA shape | ATAC shape | Required annotation key | Decision |
|---|---:|---:|---|---|
| SpaDDM Git/Zenodo | 2129 x 32285 | 2129 x 117473 | absent | reject |
| SpaMosaic Zenodo | 2129 x 25740 | 2129 x 191034 | absent | reject |
| SMART Zenodo | 2129 x 32285 | 2129 x 117473 | absent; sidecar only | reject |
| MISAR official share | E18_5-S1.arrow only | E18_5-S1.arrow only | no paired tutorial H5AD | reject |

The PRESENT tutorial documents the expected full schema, but the repository does not distribute those H5AD files; its maintainer directs data requests to OEP003285. Reconstructing the full ATAC matrix, renaming `Combined_Clusters_annotation`, or injecting a sidecar label would change the source contract and was not authorized. The run therefore failed closed rather than producing a scientifically incomparable result.

## Provenance and firewall

- Raw lineage: MISAR-seq, OEP003285 / SRP491963.
- Reference tier: `TIER_B_PUBLISHED_REFERENCE_CLUSTER`; it must not be described as manual ground truth.
- Historical role remains `PRISTINE_EXTERNAL_CONFIRMATION`; no earlier E18.5 model/selection evidence was found before download.
- Label value reads: 0.
- Annotation value hashes: 0.
- Training / embedding / partition / evaluation: 0 / 0 / 0 / 0.
- K remained preregistered at 10 from the published tutorial, never inferred from local labels.

## What would be required to resume

A planning erratum must explicitly authorize one of the following: (a) provide the exact two PRESENT-style source H5AD files; or (b) define and hash a reproducible OEP/Arrow-to-H5AD conversion, including the exact 294,734-peak feature universe and an explicit, provenance-backed mapping to `Annotation_for_Combined`. Without that authority, Night-9C must remain blocked.

## Technical authority

- Parent commit: `{command('git', 'rev-parse', 'HEAD', cwd=args.repo)}`
  - Parent tag peel: `{command('git', 'rev-parse', 'night9b-final-20260820^{commit}', cwd=args.repo)}`
- Branch: `{command('git', 'branch', '--show-current', cwd=args.repo)}`
- N02 implementation SHA-256: `{sha256(args.repo / 'SpaLORA/night9b_racf.py')}`
- Host: `{platform.node()}`
"""
    atomic_text(args.output_dir / "night9c_report.md", report)

    print(json.dumps({
        "status": audit["terminal_status"],
        "exact_pair_available": exact_pair_available,
        "label_reads": 0,
        "training": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
