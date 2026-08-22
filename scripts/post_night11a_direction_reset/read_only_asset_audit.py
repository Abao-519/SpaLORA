#!/usr/bin/env python3
"""Read-only metadata/schema audit for the Post-Night-11A direction reset.

The script deliberately reads only filesystem metadata, HDF5 schema, ordered
observation identifiers, feature identifiers, spatial coordinates, and declared
genome metadata. It never deserializes observation annotations or label files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np


LABEL_TOKENS = (
    "groundtruth", "ground_truth", "manual-anno", "manual_anno",
    "annotation", "annotations", "labels", "label_",
)
GENOME_TOKENS = ("genome", "assembly", "build", "reference")


ROOTS = [
    "/root/autodl-fs/Human lymph node/A1",
    "/root/autodl-fs/Human lymph node/D1",
    "/root/autodl-fs/P22 mouse brain coronal section",
    "/root/autodl-fs/night4a_external_data",
    "/root/autodl-fs/night6c_cache_20260817",
    "/root/autodl-fs/night6c_raw_runs_20260817",
    "/root/autodl-fs/night6d_cache_20260817",
    "/root/autodl-fs/night6d_raw_runs_20260817",
    "/root/autodl-fs/night7b_score_rnd_20260818",
    "/root/autodl-fs/night10a_rev2_qcrd_20260821",
    "/root/autodl-fs/night10b_family_policy_integration_20260821",
    "/root/autodl-fs/night11a_selective_transfer_identifiability_20260822",
]


PAIRS = {
    "A1_LYMPH_NODE": {
        "family": "RNA+PROTEIN",
        "rna": "/root/autodl-fs/Human lymph node/A1/humanlymphnode_rna.h5ad",
        "mod2": "/root/autodl-fs/Human lymph node/A1/humanlymphnode_adt.h5ad",
    },
    "D1_LYMPH_NODE_LABEL_FREE": {
        "family": "RNA+PROTEIN",
        "rna": "/root/autodl-fs/night6d_data_20260817/d1_label_free/d1_rna_label_free.h5ad",
        "mod2": "/root/autodl-fs/night6d_data_20260817/d1_label_free/d1_adt_label_free.h5ad",
    },
    "A1_TONSIL": {
        "family": "RNA+PROTEIN",
        "rna": "/root/autodl-fs/night4a_external_data/extracted/GSE263617/GSM8195495_A1_TNSL.h5ad",
        "mod2": "/root/autodl-fs/night4a_external_data/extracted/GSE263617/GSM8195499_A1_TNSL_Protein.h5ad",
    },
    "P22_MOUSE_BRAIN": {
        "family": "RNA+ATAC",
        "rna": "/root/autodl-fs/P22 mouse brain coronal section/mousebrain_rna.h5ad",
        "mod2": "/root/autodl-fs/P22 mouse brain coronal section/mousebrain_atac.h5ad",
    },
}


TENX = {
    "GSE198353_SPLEEN_REP1": "/root/autodl-fs/night4a_external_data/raw/GSE198353/GSE198353_spleen_rep_1_filtered_feature_bc_matrix.h5",
    "GSE198353_SPLEEN_REP2": "/root/autodl-fs/night4a_external_data/raw/GSE198353/GSE198353_spleen_rep_2_filtered_feature_bc_matrix.h5",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha_text(parts: list[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def decode_strings(dataset: h5py.Dataset) -> list[str]:
    values = dataset[()]
    values = np.atleast_1d(values)
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values.tolist()]


def node_shape(node: h5py.Dataset | h5py.Group) -> dict:
    if isinstance(node, h5py.Dataset):
        return {"storage": "dense", "shape": list(node.shape), "dtype": str(node.dtype)}
    shape = node.attrs.get("shape")
    if shape is not None:
        shape = [int(x) for x in np.asarray(shape).tolist()]
    dtype = str(node["data"].dtype) if "data" in node and isinstance(node["data"], h5py.Dataset) else None
    return {
        "storage": str(node.attrs.get("encoding-type", "group")),
        "shape": shape,
        "dtype": dtype,
        "keys": sorted(node.keys()),
    }


def index_dataset(group: h5py.Group) -> h5py.Dataset:
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    if key not in group:
        key = "_index"
    return group[str(key)]


def safe_scalar(node: h5py.Dataset) -> str | None:
    if node.shape not in ((), (1,)) or node.size != 1:
        return None
    value = np.atleast_1d(node[()]).tolist()[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return None


def h5ad_schema(path: str) -> dict:
    result = {"path": str(Path(path).resolve()), "exists": Path(path).is_file()}
    if not result["exists"]:
        return result
    with h5py.File(path, "r") as handle:
        obs = decode_strings(index_dataset(handle["obs"]))
        var = decode_strings(index_dataset(handle["var"]))
        result.update({
            "shape": [len(obs), len(var)],
            "ordered_observation_sha256": sha_text(obs),
            "observation_ids_unique": len(obs) == len(set(obs)),
            "feature_identifier_python_type": "str",
            "feature_identifiers_unique": len(var) == len(set(var)),
            "feature_identifier_count": len(var),
            "feature_identifier_sample": var[:10],
            "X": node_shape(handle["X"]),
            "raw": node_shape(handle["raw"]) if "raw" in handle else None,
            "layers": {key: node_shape(handle["layers"][key]) for key in sorted(handle.get("layers", {}).keys())},
            "obsm": {key: node_shape(handle["obsm"][key]) for key in sorted(handle.get("obsm", {}).keys())},
            "varm": {key: node_shape(handle["varm"][key]) for key in sorted(handle.get("varm", {}).keys())},
            "uns_keys": sorted(handle.get("uns", {}).keys()),
            "obs_storage_keys_not_read": sorted(k for k in handle["obs"].keys() if k != str(handle["obs"].attrs.get("_index", "_index"))),
            "var_storage_keys": sorted(handle["var"].keys()),
        })
        declared = {}
        if "uns" in handle:
            def visit(name: str, node: h5py.Dataset | h5py.Group) -> None:
                if isinstance(node, h5py.Dataset) and any(t in name.lower() for t in GENOME_TOKENS):
                    value = safe_scalar(node)
                    if value is not None:
                        declared[name] = value
            handle["uns"].visititems(visit)
        result["declared_genome_metadata"] = declared
        spatial = handle.get("obsm/spatial")
        if isinstance(spatial, h5py.Dataset):
            result["spatial"] = node_shape(spatial)
            result["spatial_sha256"] = hashlib.sha256(np.asarray(spatial[()]).tobytes()).hexdigest()
        result["_observation_ids"] = obs
        result["_feature_ids"] = var
    return result


PEAK_PATTERNS = (
    re.compile(r"^(chr[^: _-]+)[:_](\d+)[-_](\d+)$", re.I),
    re.compile(r"^(chr[^: _-]+)-(\d+)-(\d+)$", re.I),
)


def peak_summary(values: list[str]) -> dict:
    parsed = []
    for value in values:
        match = next((p.match(value) for p in PEAK_PATTERNS if p.match(value)), None)
        if match:
            parsed.append((match.group(1), int(match.group(2)), int(match.group(3))))
    return {
        "coordinate_parseable": len(parsed),
        "coordinate_unparseable": len(values) - len(parsed),
        "coordinate_fraction": len(parsed) / max(1, len(values)),
        "coordinate_examples": values[:10],
        "coordinates_valid_order": bool(parsed) and all(a < b for _, a, b in parsed),
    }


def pair_schema(name: str, spec: dict) -> dict:
    rna = h5ad_schema(spec["rna"])
    mod2 = h5ad_schema(spec["mod2"])
    ids1 = rna.pop("_observation_ids", [])
    ids2 = mod2.pop("_observation_ids", [])
    f1 = rna.pop("_feature_ids", [])
    f2 = mod2.pop("_feature_ids", [])
    out = {
        "unit": name,
        "family": spec["family"],
        "rna": rna,
        "modality2": mod2,
        "paired_observation_ids_byte_exact": ids1 == ids2,
        "spot_count": len(ids1),
        "rna_feature_count": len(f1),
        "modality2_feature_count": len(f2),
        "rna_feature_identifiers_unique": len(f1) == len(set(f1)),
        "modality2_feature_identifiers_unique": len(f2) == len(set(f2)),
        "coordinates_byte_exact": rna.get("spatial_sha256") == mod2.get("spatial_sha256") and rna.get("spatial_sha256") is not None,
    }
    if spec["family"] == "RNA+PROTEIN":
        out["protein_target_identifiers"] = f2
        rna_names = set(f1)
        matched = [target for target in f2 if target in rna_names]
        out["deposited_exact_name_correspondence"] = {
            "matched": len(matched),
            "total_protein_targets": len(f2),
            "one_to_one_exact": len(matched),
            "one_to_many": 0,
            "unmapped": [target for target in f2 if target not in rna_names],
            "normalization_or_similarity_guessing_used": False,
        }
    else:
        out["atac_peak_identifiers"] = peak_summary(f2)
    return out


def tenx_schema(path: str) -> dict:
    result = {"path": str(Path(path).resolve()), "exists": Path(path).is_file()}
    if not result["exists"]:
        return result
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        barcodes = decode_strings(matrix["barcodes"])
        features = matrix["features"]
        names = decode_strings(features["name"])
        kinds = decode_strings(features["feature_type"])
        counts = {kind: kinds.count(kind) for kind in sorted(set(kinds))}
        genes = {name for name, kind in zip(names, kinds) if kind == "Gene Expression"}
        protein = [name for name, kind in zip(names, kinds) if kind == "Antibody Capture"]
        result.update({
            "shape_features_by_spots": [int(x) for x in matrix["shape"][()].tolist()],
            "matrix_storage": "csc_sparse_10x",
            "matrix_data_dtype": str(matrix["data"].dtype),
            "matrix_nnz": int(matrix["data"].shape[0]),
            "ordered_observation_sha256": sha_text(barcodes),
            "observation_ids_unique": len(barcodes) == len(set(barcodes)),
            "feature_identifier_count": len(names),
            "feature_identifiers_unique": len(names) == len(set(names)),
            "feature_type_counts": counts,
            "feature_identifier_samples": {
                kind: [name for name, observed in zip(names, kinds) if observed == kind][:10]
                for kind in sorted(set(kinds))
            },
            "deposited_exact_name_correspondence": {
                "matched": sum(name in genes for name in protein),
                "total_protein_targets": len(protein),
                "unmapped": [name for name in protein if name not in genes],
                "normalization_or_similarity_guessing_used": False,
            },
        })
    return result


def root_snapshot(path: str) -> dict:
    root = Path(path)
    row = {"root": str(root.resolve()), "exists": root.exists()}
    if not root.exists():
        return row
    records = []
    label_candidates = []
    file_count = 0
    total_bytes = 0
    for item in sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root))):
        stat = item.lstat()
        relative = str(item.relative_to(root))
        kind = "dir" if item.is_dir() else "file"
        size = int(stat.st_size) if item.is_file() else 0
        records.append(f"{kind}\t{relative}\t{size}\t{stat.st_mtime_ns}")
        if item.is_file():
            file_count += 1
            total_bytes += size
            lower = relative.lower()
            if any(token in lower for token in LABEL_TOKENS):
                label_candidates.append({"relative_path": relative, "size_bytes": size, "mtime_ns": stat.st_mtime_ns})
    root_stat = root.stat()
    row.update({
        "file_count": file_count,
        "total_file_bytes": total_bytes,
        "root_mtime_ns": root_stat.st_mtime_ns,
        "metadata_fingerprint_sha256": sha_text(records),
        "label_or_ground_truth_path_metadata_only": label_candidates,
        "content_hashes_recomputed": 0,
    })
    return row


def compare_snapshots(before: dict, after: dict) -> dict:
    old = {x["root"]: x for x in before["roots"]}
    rows = []
    for current in after["roots"]:
        previous = old.get(current["root"])
        exact = previous == current
        rows.append({
            "root": current["root"],
            "byte_exact_metadata_unchanged": exact,
            "before": previous,
            "after": current,
        })
    return {
        "schema": "spalora.post_night11a.raw_root_invariance.v1",
        "status": "PASS" if all(x["byte_exact_metadata_unchanged"] for x in rows) else "FAIL",
        "roots": rows,
        "raw_content_opened_for_hashing": False,
        "label_file_contents_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before", "schema", "after"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.phase in ("before", "after"):
        value = {
            "schema": "spalora.post_night11a.raw_root_metadata_snapshot.v1",
            "phase": args.phase,
            "roots": [root_snapshot(path) for path in ROOTS],
            "label_file_contents_opened": False,
            "gpu_used_mib": 0,
        }
        target = args.output_root / f"raw_root_metadata_{args.phase}.json"
        atomic_json(target, value)
        if args.phase == "after":
            before = json.loads((args.output_root / "raw_root_metadata_before.json").read_text(encoding="utf-8"))
            atomic_json(args.output_root / "raw_root_invariance_audit.json", compare_snapshots(before, value))
    else:
        value = {
            "schema": "spalora.post_night11a.asset_schema.v1",
            "h5ad_pairs": {name: pair_schema(name, spec) for name, spec in PAIRS.items()},
            "independent_tenx": {name: tenx_schema(path) for name, path in TENX.items()},
            "read_boundary": {
                "observation_annotation_values_read": False,
                "label_files_opened": False,
                "feature_identifiers_read": True,
                "ordered_observation_identifiers_read": True,
                "spatial_coordinates_read": True,
                "genome_metadata_values_read_only_when_key_name_declares_genome_semantics": True,
            },
        }
        atomic_json(args.output_root / "asset_schema_audit.json", value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
