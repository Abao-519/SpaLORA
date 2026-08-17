#!/usr/bin/env python3
"""Authorized low-level HDF5 ontology audit and label-free copy builder."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path("/root/autodl-fs/SpaLORA-night6b")
sys.path.insert(0, str(REPO))
from SpaLORA.night6b_firewall import guard_path

SOURCE = Path("/root/autodl-fs/datasets/human_tonsil_official/section1")
DEST = Path("/root/autodl-fs/night6b_data_20260817/tonsil_label_free")
OUT = REPO / "outputs/night6b_handoff"
LOG = OUT / "firewall/data_steward_access.jsonl"
EXPECTED = {
    "rna": "e1d99b34685805c93a7314f8ef4e07b2f92d244a8c29ac9fced9ee19ff3b96a9",
    "adt": "f7e7c2b723faf59fbf1d949c131b2f345c268787cd2b72177e0c3a139afa38be",
}
COLUMNS = ["final_annot", "lab", "lab_lynn", "src"]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode_array(values):
    return [decode_scalar(x) for x in np.asarray(values)]


def read_index(obs: h5py.Group):
    key = decode_scalar(obs.attrs.get("_index", "_index"))
    return [str(x) for x in decode_array(obs[key][...])]


def read_column(obs: h5py.Group, name: str):
    if name not in obs:
        return None, {"present": False}
    node = obs[name]
    encoding = decode_scalar(node.attrs.get("encoding-type", "dataset"))
    if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
        categories = [str(x) for x in decode_array(node["categories"][...])]
        codes = np.asarray(node["codes"][...], dtype=int)
        values = [None if int(c) < 0 else categories[int(c)] for c in codes]
        meta = {"present": True, "encoding": encoding, "categories_storage_order": categories}
        return values, meta
    raw = decode_array(node[...])
    values = [None if x is None or str(x) in {"", "nan", "None"} else str(x) for x in raw]
    return values, {"present": True, "encoding": encoding}


def read_x(handle: h5py.File):
    node = handle["X"]
    if isinstance(node, h5py.Dataset):
        return np.asarray(node)
    shape = tuple(map(int, node.attrs["shape"]))
    encoding = decode_scalar(node.attrs.get("encoding-type", "csr_matrix"))
    cls = sp.csc_matrix if encoding == "csc_matrix" else sp.csr_matrix
    return cls((node["data"][...], node["indices"][...], node["indptr"][...]), shape=shape)


def inspect(path: Path):
    guard_path(path, role="data_steward", operation="low_level_hdf5_ontology_read", audit_log=LOG)
    with h5py.File(path, "r") as h:
        barcodes = read_index(h["obs"])
        columns = {}
        vectors = {}
        for name in COLUMNS:
            values, meta = read_column(h["obs"], name)
            if values is not None:
                nonmissing = [x for x in values if x is not None]
                counts = Counter(nonmissing)
                meta.update({
                    "missing_count": len(values) - len(nonmissing),
                    "unique_nonmissing_count": len(counts),
                    "categories_lexical": sorted(counts),
                    "category_counts": {k: counts[k] for k in sorted(counts)},
                    "per_spot_sequence_sha256": digest_json(values),
                })
                vectors[name] = values
            columns[name] = meta
        spatial = np.asarray(h["obsm/spatial"])
        var_key = decode_scalar(h["var"].attrs.get("_index", "_index"))
        var_names = [str(x) for x in decode_array(h["var"][var_key][...])]
    return {"barcodes": barcodes, "barcodes_sha256": digest_json(barcodes), "columns": columns, "vectors": vectors, "spatial": spatial, "var_names": var_names}


def create_label_free(source: Path, target: Path, audit):
    guard_path(source, role="data_steward", operation="low_level_hdf5_label_free_copy", audit_log=LOG)
    with h5py.File(source, "r") as h:
        x = read_x(h)
    obj = ad.AnnData(X=x, obs=pd.DataFrame(index=pd.Index(audit["barcodes"])), var=pd.DataFrame(index=pd.Index(audit["var_names"])))
    obj.obsm["spatial"] = audit["spatial"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError(f"refuse to overwrite label-free output: {target}")
    obj.write_h5ad(target, compression="gzip")
    check = ad.read_h5ad(target, backed="r")
    proof = {
        "path": str(target),
        "sha256": sha_file(target),
        "size_bytes": target.stat().st_size,
        "shape": list(map(int, check.shape)),
        "obs_columns": list(check.obs.columns),
        "barcodes_sha256": digest_json(list(map(str, check.obs_names))),
        "spatial_shape": list(map(int, check.obsm["spatial"].shape)),
    }
    check.file.close()
    if proof["obs_columns"]:
        raise RuntimeError("label-free output contains obs columns")
    return proof


paths = {"rna": SOURCE / "s1_adata_rna.h5ad", "adt": SOURCE / "s1_adata_adt.h5ad"}
source_hashes = {k: sha_file(guard_path(v, role="data_steward", operation="source_sha256", audit_log=LOG)) for k, v in paths.items()}
if source_hashes != EXPECTED:
    raise SystemExit(f"source hash mismatch: {source_hashes}")
audits = {k: inspect(v) for k, v in paths.items()}
target = "final_annot"
if not audits["rna"]["columns"][target].get("present"):
    raise SystemExit("target final_annot missing from RNA")
k = audits["rna"]["columns"][target]["unique_nonmissing_count"]
if k not in (4, 6, 7):
    raise SystemExit(f"actual final_annot K={k} outside preregistered provenance values")
adt_has_target = audits["adt"]["columns"][target].get("present", False)
barcodes_equal = audits["rna"]["barcodes"] == audits["adt"]["barcodes"]
target_equal = None
if adt_has_target and barcodes_equal:
    target_equal = audits["rna"]["vectors"][target] == audits["adt"]["vectors"][target]
    if not target_equal:
        raise SystemExit("RNA/ADT final_annot conflict")

contract = {
    "status": "P0_ONTOLOGY_PASS",
    "authorized_role": "data_steward",
    "target_column_locked_before_read": target,
    "known_k": k,
    "source_sha256": source_hashes,
    "barcodes_sha256": audits["rna"]["barcodes_sha256"],
    "paired_barcodes_exact": barcodes_equal,
    "rna_adt_target_column_presence": {"rna": True, "adt": adt_has_target},
    "rna_adt_final_annot_exact": target_equal,
    "columns": {mod: audits[mod]["columns"] for mod in ("rna", "adt")},
    "target_per_spot_label_vector_sha256": audits["rna"]["columns"][target]["per_spot_sequence_sha256"],
    "provenance": {
        "four": "coarse manually described regions in SpaMICS paper; not selected",
        "six": "SpaMosaic official tonsil section-1 tutorial n_cluster; not selected",
        "seven": "finer published/domain scheme and SpaMICS Human_tonsil branch; not selected",
        "resolution_rule": "actual nonmissing unique count in preregistered final_annot",
        "sources": [
            "https://spamosaic.readthedocs.io/en/latest/tutorials/integration/vertical/Tonsil_vertical.html",
            "https://www.sciencedirect.com/science/article/pii/S1566253525005019",
            "https://github.com/SZU-CGC/SpaMICS",
        ],
    },
    "access_semantics": {
        "deserialized_into_memory": True,
        "explicitly_indexed_or_observed": True,
        "used_for_training_or_selection": False,
        "authorized_role": "data_steward",
    },
}
OUT.joinpath("ontology").mkdir(parents=True, exist_ok=True)
OUT.joinpath("firewall").mkdir(parents=True, exist_ok=True)
(OUT / "ontology/tonsil_ontology_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

created = {
    mod: create_label_free(paths[mod], DEST / f"tonsil_s1_{mod}_label_free.h5ad", audits[mod])
    for mod in ("rna", "adt")
}
manifest = {
    "status": "LABEL_FREE_COPY_PASS",
    "authorized_role": "data_steward",
    "source_sha256": source_hashes,
    "created": created,
    "paired_barcodes_exact": barcodes_equal,
    "obs_zero_columns": all(not x["obs_columns"] for x in created.values()),
    "original_h5ad_opened_by_anndata": False,
}
(OUT / "firewall/source_to_label_free_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
known = {"dataset": "tonsil_s1", "known_k": k, "source_contract_sha256": sha_file(OUT / "ontology/tonsil_ontology_contract.json"), "contains_per_spot_labels": False}
Path("/root/autodl-fs/night6b_data_20260817/known_k.json").write_text(json.dumps(known, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "P0_ONTOLOGY_PASS", "known_k": k, "created": created}, sort_keys=True))
