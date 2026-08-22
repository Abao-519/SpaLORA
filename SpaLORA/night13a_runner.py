"""Night-13A identity-blind unified baseline primitives.

The model and endpoint accept tensors, coordinates, identifiers, adapter type,
and explicit numerical configuration only.  Dataset/tissue/path strings never
enter model semantics.  All neighborhood objects remain sparse.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler

from .night12a_schema_p0 import (
    UnifiedZeroStepAutoencoder,
    array_sha,
    assert_model_identity_blind,
    canonical_partition,
    file_sha256,
    fixed_linked_features,
    gene_score_intervals,
    h05_endpoint,
    inspect_csv_matrix,
    load_selected_counts,
    normalize_counts,
    normalize_gene_scores,
    parse_ensembl79_genes,
    read_coordinates,
    reconstruction_loss,
    scan_fragments,
    seurat_clr_counts,
    sparse_spatial_graph,
    text_sha256,
)


SEED = 0
LATENT_DIM = 64
P0_K = 2
RNA_COMPONENTS = 30
OTHER_COMPONENTS = 30


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _finite_float32(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError("adapter produced non-finite or non-matrix output")
    return result


def _log_normalize_sparse(matrix: sp.spmatrix) -> sp.csr_matrix:
    value = matrix.tocsr().astype(np.float64)
    if value.nnz and (np.min(value.data) < 0 or not np.all(np.isfinite(value.data))):
        raise ValueError("count adapter requires finite non-negative values")
    library = np.asarray(value.sum(axis=1)).ravel()
    if np.any(library <= 0):
        raise ValueError("count adapter encountered zero library")
    value = value.multiply((10000.0 / library)[:, None]).tocsr()
    value.data = np.log1p(value.data)
    return value


def rna_adapter(matrix: object, components: int = RNA_COMPONENTS) -> np.ndarray:
    """Registered RNA preprocessing: total-normalize, log1p, sparse SVD."""
    if not sp.issparse(matrix):
        dense = np.asarray(matrix)
        matrix = sp.csr_matrix(dense)
    value = _log_normalize_sparse(matrix)
    n_components = max(1, min(int(components), value.shape[0] - 1,
                              value.shape[1] - 1))
    return _finite_float32(
        TruncatedSVD(n_components=n_components, random_state=SEED).fit_transform(value)
    )


def protein_adapter(matrix: object, components: int = OTHER_COMPONENTS) -> np.ndarray:
    value = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    value = seurat_clr_counts(np.asarray(value, dtype=np.float64))
    value = StandardScaler().fit_transform(value)
    n_components = max(1, min(int(components), value.shape[0] - 1,
                              value.shape[1]))
    if n_components < value.shape[1]:
        value = PCA(n_components=n_components, random_state=SEED).fit_transform(value)
    return _finite_float32(value)


def atac_adapter(adata: ad.AnnData, components: int = 50) -> np.ndarray:
    """Use deposited LSI when traceable; otherwise sparse TF-IDF-like SVD."""
    if "X_lsi" in adata.obsm:
        value = np.asarray(adata.obsm["X_lsi"])
        if value.shape[0] != adata.n_obs:
            raise ValueError("deposited X_lsi observation contract mismatch")
        return _finite_float32(value[:, :min(int(components), value.shape[1])])
    matrix = adata.X if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
    matrix = matrix.tocsr().astype(np.float64)
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(row_sum <= 0):
        raise ValueError("ATAC adapter encountered zero library")
    matrix = matrix.multiply((1.0 / row_sum)[:, None]).tocsr()
    document = np.asarray((matrix > 0).sum(axis=0)).ravel()
    idf = np.log1p(matrix.shape[0] / np.maximum(document, 1))
    matrix = matrix.multiply(idf).tocsr()
    n_components = max(1, min(int(components), matrix.shape[0] - 1,
                              matrix.shape[1] - 1))
    return _finite_float32(
        TruncatedSVD(n_components=n_components, random_state=SEED).fit_transform(matrix)
    )


def _exact_pair(left: ad.AnnData, right: ad.AnnData) -> ad.AnnData:
    left_ids = list(map(str, left.obs_names))
    right_ids = list(map(str, right.obs_names))
    if len(set(left_ids)) != len(left_ids) or len(set(right_ids)) != len(right_ids):
        raise ValueError("paired observations must be unique")
    if set(left_ids) != set(right_ids):
        raise ValueError("paired observation identifier sets differ")
    return right[left_ids].copy()


def load_h5ad_pair(rna_path: Path, other_path: Path,
                    adapter: str,
                    observation_ids: Sequence[str] | None = None) -> Dict[str, object]:
    rna = ad.read_h5ad(rna_path)
    other_all = _exact_pair(rna, ad.read_h5ad(other_path))
    if observation_ids is not None:
        requested = list(map(str, observation_ids))
        if len(set(requested)) != len(requested):
            raise ValueError("registered observation filter contains duplicates")
        if not set(requested) <= set(map(str, rna.obs_names)):
            raise ValueError("registered observation filter is not a source subset")
        rna = rna[requested].copy()
        other = other_all[requested].copy()
    else:
        other = other_all
    ids = list(map(str, rna.obs_names))
    if "spatial" not in rna.obsm:
        raise ValueError("RNA input lacks registered spatial coordinates")
    coordinates = np.asarray(rna.obsm["spatial"], dtype=np.float64)
    if coordinates.shape != (len(ids), 2) or not np.all(np.isfinite(coordinates)):
        raise ValueError("registered spatial coordinate contract invalid")
    view1 = rna_adapter(rna.X)
    if adapter == "protein":
        view2 = protein_adapter(other.X)
    elif adapter == "atac":
        view2 = atac_adapter(other)
    else:
        raise ValueError("adapter must be explicit protein or atac")
    return {
        "ids": ids, "coordinates": coordinates,
        "view1": view1, "view2": view2,
        "raw_shapes": [list(rna.shape), list(other.shape)],
        "source_sha256": {
            "rna": file_sha256(rna_path), "other": file_sha256(other_path),
        },
    }


def load_spots(h5_path: Path, positions_path: Path) -> Dict[str, object]:
    adata = sc.read_10x_h5(h5_path, gex_only=False)
    if "feature_types" not in adata.var:
        raise ValueError("10x file lacks feature_type registry")
    feature_types = adata.var["feature_types"].astype(str)
    counts = feature_types.value_counts().to_dict()
    if counts != {"Gene Expression": 32285, "Antibody Capture": 21}:
        raise ValueError(f"unexpected SPOTS feature_type split: {counts}")
    positions = pd.read_csv(positions_path, header=None)
    if positions.shape != (4992, 6):
        raise ValueError("SPOTS coordinate archive shape mismatch")
    positions[0] = positions[0].astype(str)
    tissue = positions.loc[positions[1].astype(int) == 1].copy()
    ids = list(map(str, adata.obs_names))
    if len(ids) != 2653 or tissue[0].nunique() != 2653 or set(ids) != set(tissue[0]):
        raise ValueError("SPOTS barcode+tissue exact alignment did not close at 2653")
    tissue = tissue.set_index(0).loc[ids]
    coordinates = tissue[[5, 4]].to_numpy(dtype=np.float64)
    rna = adata[:, feature_types == "Gene Expression"]
    protein = adata[:, feature_types == "Antibody Capture"]
    return {
        "ids": ids, "coordinates": coordinates,
        "view1": rna_adapter(rna.X), "view2": protein_adapter(protein.X),
        "raw_shapes": [list(rna.shape), list(protein.shape)],
        "position_shape": list(positions.shape),
        "feature_type_counts": counts,
        "source_sha256": {
            "matrix": file_sha256(h5_path), "coordinates": file_sha256(positions_path),
        },
    }


def load_p5(rna_path: Path, positions_path: Path, fragments_path: Path,
            gtf_path: Path) -> Dict[str, object]:
    coordinate = read_coordinates(positions_path)
    rna_audit = inspect_csv_matrix(rna_path, coordinate["ordered_ids"])
    registry = parse_ensembl79_genes(gtf_path)
    linked = fixed_linked_features(rna_audit["feature_ids"],
                                   registry["unique"].keys(), 256)
    if len(linked) != 256:
        raise ValueError("P5 engineering lexical-first panel did not close at 256")
    loaded = load_selected_counts(rna_path, rna_audit, linked)
    view1 = normalize_counts(loaded["counts"], loaded["library_size"])
    fragments = scan_fragments(
        fragments_path, rna_audit["ordered_observation_ids"],
        gene_score_intervals(registry, linked, upstream_bp=5000),
    )
    if fragments["fragment_rows"] != 83593412:
        raise ValueError("P5S1 fragment row count mismatch")
    if fragments["registered_missing_count"] != 0:
        raise ValueError("P5S1 registered RNA spots missing from fragments")
    view2 = normalize_gene_scores(
        fragments["counts"], fragments["registered_fragment_depth"]
    )
    ids = list(rna_audit["ordered_observation_ids"])
    coordinates = np.asarray([
        [coordinate["records"][value]["pixel_col"],
         coordinate["records"][value]["pixel_row"]] for value in ids
    ], dtype=np.float64)
    return {
        "ids": ids, "coordinates": coordinates,
        "view1": _finite_float32(view1), "view2": _finite_float32(view2),
        "raw_shapes": [list(rna_audit["observation_by_feature_shape"]),
                       [int(fragments["fragment_rows"]), 5]],
        "fragment_rows": int(fragments["fragment_rows"]),
        "linked_features": linked,
        "linked_feature_sha256": text_sha256(linked),
        "source_sha256": {
            "rna": file_sha256(rna_path), "coordinates": file_sha256(positions_path),
            "fragments": file_sha256(fragments_path), "gtf": file_sha256(gtf_path),
        },
        "gene_score_boundary": (
            "clean-room Ensembl79 GRCm38 gene body plus strand-aware 5kb upstream; "
            "not ArchR GeneScoreMatrix equivalence"
        ),
    }


def run_zero_step(payload: Mapping[str, object], artifact_dir: Path,
                  endpoint_k: int = P0_K) -> dict:
    started = time.monotonic()
    artifact_dir.mkdir(parents=True, exist_ok=False)
    ids = list(payload["ids"])
    coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
    view1 = _finite_float32(payload["view1"])
    view2 = _finite_float32(payload["view2"])
    if len(ids) != view1.shape[0] or view1.shape[0] != view2.shape[0]:
        raise ValueError("zero-step observation contract mismatch")
    graph = sparse_spatial_graph(coordinates, ids, k=6)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedZeroStepAutoencoder([view1.shape[1], view2.shape[1]], LATENT_DIM)
    assert_model_identity_blind(model)
    model.to(device).eval()
    left = torch.as_tensor(view1, dtype=torch.float32, device=device)
    right = torch.as_tensor(view2, dtype=torch.float32, device=device)
    with torch.no_grad():
        result = model(left, right)
        loss = reconstruction_loss(result, left, right)
    arrays = {name: result[name].detach().cpu().numpy()
              for name in ["private1", "private2", "fused"]}
    partition, _ = h05_endpoint(
        arrays["private1"], arrays["private2"], arrays["fused"], coordinates,
        ids, int(endpoint_k), artifact_dir / "endpoint_initial",
    )
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dims": [view1.shape[1], view2.shape[1]],
        "latent_dim": LATENT_DIM, "seed": SEED, "optimizer_steps": 0,
        "ordered_id_sha256": text_sha256(ids),
    }
    atomic_torch(artifact_dir / "checkpoint.pt", checkpoint)
    atomic_npz(
        artifact_dir / "roundtrip.npz", view1=view1, view2=view2,
        coordinates=coordinates, ordered_ids=np.asarray(ids, dtype=str),
        private1=arrays["private1"], private2=arrays["private2"],
        fused=arrays["fused"], partition=partition.astype(np.int64),
    )
    audit = {
        "status": "FORWARD_COMPLETE_RELOAD_PENDING",
        "seed": SEED, "optimizer_steps": 0,
        "real_observations": len(ids),
        "raw_shapes": payload.get("raw_shapes"),
        "processed_shapes": [list(view1.shape), list(view2.shape)],
        "ordered_id_sha256": text_sha256(ids),
        "finite_reconstruction_loss": bool(torch.isfinite(loss).item()),
        "reconstruction_loss": float(loss.detach().cpu()),
        "sparse_graph_nnz": int(graph.nnz), "dense_n_by_n_count": 0,
        "embedding_sha256": array_sha(arrays["fused"]),
        "partition_sha256": array_sha(partition),
        "checkpoint_sha256": file_sha256(artifact_dir / "checkpoint.pt"),
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_mib": (torch.cuda.max_memory_allocated() / 1048576.0
                         if torch.cuda.is_available() else 0.0),
        "peak_rss_mib": psutil.Process().memory_info().rss / 1048576.0,
        "label_reads": 0,
        "source_sha256": payload.get("source_sha256", {}),
    }
    for key in ["fragment_rows", "position_shape", "feature_type_counts",
                "linked_feature_sha256", "gene_score_boundary"]:
        if key in payload:
            audit[key] = payload[key]
    atomic_json(artifact_dir / "forward.json", audit)
    return audit


def reload_zero_step(artifact_dir: Path, endpoint_k: int = P0_K) -> dict:
    started = time.monotonic()
    bundle = np.load(artifact_dir / "roundtrip.npz")
    checkpoint = torch.load(artifact_dir / "checkpoint.pt", map_location="cpu")
    ids = list(map(str, bundle["ordered_ids"].tolist()))
    if text_sha256(ids) != checkpoint["ordered_id_sha256"]:
        raise ValueError("fresh-process ordered-ID SHA mismatch")
    torch.manual_seed(int(checkpoint["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedZeroStepAutoencoder(checkpoint["input_dims"],
                                       checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    with torch.no_grad():
        result = model(
            torch.as_tensor(bundle["view1"], device=device),
            torch.as_tensor(bundle["view2"], device=device),
        )
    exact = {}
    actual = {}
    for name in ["private1", "private2", "fused"]:
        actual[name] = result[name].detach().cpu().numpy()
        exact[name] = bool(np.allclose(actual[name], bundle[name], rtol=1e-5,
                                       atol=1e-6))
    partition, _ = h05_endpoint(
        actual["private1"], actual["private2"], actual["fused"],
        bundle["coordinates"], ids, int(endpoint_k),
        artifact_dir / "endpoint_reload",
    )
    partition_exact = bool(np.array_equal(partition, bundle["partition"]))
    audit = {
        "status": "PASS" if all(exact.values()) and partition_exact else "FAIL",
        "fresh_process": True, "checkpoint_strict_load": True,
        "numerical_roundtrip": exact,
        "canonical_partition_exact": partition_exact,
        "ordered_id_sha256": text_sha256(ids),
        "partition_sha256": array_sha(partition),
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(artifact_dir / "fresh_process_reload.json", audit)
    return audit


def simple_embedding(payload: Mapping[str, object]) -> np.ndarray:
    views = []
    for value in [payload["view1"], payload["view2"]]:
        value = StandardScaler().fit_transform(np.asarray(value, dtype=np.float64))
        views.append(value)
    combined = np.concatenate(views, axis=1)
    n_components = max(1, min(LATENT_DIM, combined.shape[0] - 1,
                              combined.shape[1]))
    return _finite_float32(
        PCA(n_components=n_components, random_state=SEED).fit_transform(combined)
    )


def mean_binary_moran(labels: Sequence[int], graph: sp.csr_matrix) -> float:
    labels = np.asarray(labels)
    graph = graph.tocsr().astype(np.float64)
    if graph.shape != (len(labels), len(labels)):
        raise ValueError("Moran graph shape mismatch")
    weight_sum = float(graph.sum())
    values = []
    for group in np.unique(labels):
        x = (labels == group).astype(np.float64)
        centered = x - x.mean()
        denominator = float(centered @ centered)
        if denominator == 0:
            continue
        numerator = float(centered @ (graph @ centered))
        values.append(len(x) * numerator / (weight_sum * denominator))
    return float(np.mean(values)) if values else 0.0


def evaluate_embedding(embedding: np.ndarray, labels: Sequence[object],
                       coordinates: np.ndarray, ids: Sequence[str], k: int) -> dict:
    encoded, _ = pd.factorize(pd.Series(labels, dtype="object"), sort=True)
    if np.any(encoded < 0):
        raise ValueError("canonical label contains missing values")
    partition = canonical_partition(
        KMeans(n_clusters=int(k), random_state=SEED, n_init=20).fit_predict(embedding)
    )
    graph = sparse_spatial_graph(coordinates, ids, k=6)
    return {
        "partition": partition,
        "absolute_ari": float(adjusted_rand_score(encoded, partition)),
        "absolute_nmi": float(normalized_mutual_info_score(encoded, partition)),
        "ami": float(adjusted_mutual_info_score(encoded, partition)),
        "fmi": float(fowlkes_mallows_score(encoded, partition)),
        "homogeneity": float(homogeneity_score(encoded, partition)),
        "v_measure": float(v_measure_score(encoded, partition)),
        "morans_i": mean_binary_moran(partition, graph),
        "sparse_graph_nnz": int(graph.nnz),
        "dense_n_by_n_count": 0,
    }
