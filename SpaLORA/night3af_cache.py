"""Deterministic, content-addressed preprocessing cache for Night-3A-F."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import torch

from .night1_pipeline import PreparedData
from .night3a_ige import input_sha256


ARRAYS = (
    "features_omics1", "features_omics2", "weight_vector_omics1",
    "rna_pca_scores", "rna_pca_explained_variance",
    "rna_pca_explained_variance_ratio", "coordinates",
)
GRAPHS = (
    "adj_spatial_omics1", "adj_spatial_omics2",
    "adj_feature_omics1", "adj_feature_omics2",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _write_tsv(path: Path, header: str, values) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(header + "\n")
        for value in values:
            handle.write(str(value) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def _file_rows(directory: Path) -> dict:
    return {
        path.name: {"sha256": sha256_file(path), "size_bytes": int(path.stat().st_size)}
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }


def save_cache(directory: Path, dataset: str, prepared: PreparedData,
               protocol: Mapping[str, object]) -> dict:
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("Refusing to overwrite non-empty cache: %s" % directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = prepared.data
    arrays = {
        "features_omics1": np.asarray(data["features_omics1"], dtype=np.float32),
        "features_omics2": np.asarray(data["features_omics2"], dtype=np.float32),
        "weight_vector_omics1": np.asarray(data["weight_vector_omics1"], dtype=np.float32),
        "rna_pca_scores": np.asarray(data["rna_pca_scores"], dtype=np.float32),
        "rna_pca_explained_variance": np.asarray(data["rna_pca_explained_variance"], dtype=np.float64),
        "rna_pca_explained_variance_ratio": np.asarray(data["rna_pca_explained_variance_ratio"], dtype=np.float64),
        "coordinates": np.asarray(prepared.coordinates),
    }
    for name, value in arrays.items():
        np.save(directory / (name + ".npy"), value, allow_pickle=False)
    _write_tsv(directory / "observation_ids.tsv", "observation_id", prepared.obs_names.astype(str))
    _write_tsv(directory / "selected_genes.tsv", "gene", data["selected_gene_names"])
    graph_metadata = {}
    for name in GRAPHS:
        graph = data[name]
        if not isinstance(graph, torch.Tensor) or not graph.is_sparse:
            raise AssertionError("Cache graph must remain sparse: %s" % name)
        graph = graph.coalesce().cpu()
        indices = graph.indices().numpy().astype(np.int64, copy=False)
        values = graph.values().numpy().astype(np.float32, copy=False)
        np.save(directory / (name + "_indices.npy"), indices, allow_pickle=False)
        np.save(directory / (name + "_values.npy"), values, allow_pickle=False)
        graph_metadata[name] = {
            "shape": list(map(int, graph.shape)), "nnz": int(graph._nnz()),
            "coalesced": True, "indices_shape": list(map(int, indices.shape)),
            "values_shape": list(map(int, values.shape)),
        }
    model_hash = input_sha256(
        data, prepared.obs_names.astype(str), data["selected_gene_names"]
    )
    metadata = {
        "schema_version": 1, "dataset": dataset,
        "pca_svd_solver": protocol["pca_svd_solver"],
        "pca_random_state": int(protocol["pca_random_state"]),
        "rna_pca_metadata": data["rna_pca_metadata"],
        "arrays": {name: {"shape": list(map(int, value.shape)), "dtype": str(value.dtype)}
                   for name, value in arrays.items()},
        "graphs": graph_metadata,
        "n_observations": int(len(prepared.obs_names)),
        "n_selected_genes": int(len(data["selected_gene_names"])),
        "canonical_model_input_sha256": model_hash,
        "semantic_label_values_read": False,
    }
    atomic_json(directory / "metadata.json", metadata)
    files = _file_rows(directory)
    digest = hashlib.sha256()
    for name, row in sorted(files.items()):
        digest.update(name.encode("utf-8")); digest.update(row["sha256"].encode("ascii"))
    manifest = {
        "schema_version": 1, "dataset": dataset,
        "canonical_model_input_sha256": model_hash,
        "canonical_cache_content_sha256": digest.hexdigest(),
        "files": files, "manifest_excludes_itself": True,
    }
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def verify_cache(directory: Path, expected_manifest_sha256: Optional[str] = None) -> dict:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Missing cache manifest: %s" % directory)
    if expected_manifest_sha256 and sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("Cache manifest SHA mismatch: %s" % directory)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_names = set(manifest["files"]) | {"manifest.json"}
    actual_names = {path.name for path in directory.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError("Partial/extra cache files: %s" % sorted(actual_names ^ expected_names))
    for name, row in manifest["files"].items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
            raise RuntimeError("Damaged cache file: %s" % path)
    return manifest


def load_cache(directory: Path, expected_manifest_sha256: Optional[str] = None) -> PreparedData:
    manifest = verify_cache(directory, expected_manifest_sha256)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    arrays = {name: np.load(directory / (name + ".npy"), allow_pickle=False) for name in ARRAYS}
    obs = pd.Index(pd.read_csv(directory / "observation_ids.tsv", sep="\t")["observation_id"].astype(str))
    genes = pd.read_csv(directory / "selected_genes.tsv", sep="\t")["gene"].astype(str).to_numpy()
    data = {
        "features_omics1": arrays["features_omics1"],
        "features_omics2": arrays["features_omics2"],
        "weight_vector_omics1": arrays["weight_vector_omics1"],
        "selected_gene_names": genes,
        "rna_pca_scores": arrays["rna_pca_scores"],
        "rna_pca_explained_variance": arrays["rna_pca_explained_variance"],
        "rna_pca_explained_variance_ratio": arrays["rna_pca_explained_variance_ratio"],
        "rna_pca_metadata": metadata["rna_pca_metadata"],
    }
    for name in GRAPHS:
        indices = torch.as_tensor(np.load(directory / (name + "_indices.npy"), allow_pickle=False), dtype=torch.long)
        values = torch.as_tensor(np.load(directory / (name + "_values.npy"), allow_pickle=False), dtype=torch.float32)
        graph = torch.sparse_coo_tensor(indices, values, size=metadata["graphs"][name]["shape"]).coalesce()
        if not graph.is_sparse or not graph.is_coalesced():
            raise AssertionError("Loaded graph is not sparse/coalesced")
        data[name] = graph
    observed_hash = input_sha256(data, obs, genes)
    if observed_hash != manifest["canonical_model_input_sha256"]:
        raise RuntimeError("Loaded canonical model-input SHA mismatch")
    return PreparedData(
        data=data, obs_names=obs, coordinates=arrays["coordinates"],
        gene_table=pd.DataFrame(index=np.arange(len(genes))),
    )
