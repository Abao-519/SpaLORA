#!/usr/bin/env python3
"""Night-2 hard-gate parity audit. This script never loads ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.model import Encoder_overall
from SpaLORA.model_corrected import EncoderOverallCorrected
import SpaLORA.night1_pipeline as corrected_pipeline
from SpaLORA.night1_pipeline import normalize_graph_sparse
from SpaLORA.preprocess import fix_seed
from scripts.night1_benchmark import prepare_legacy


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def compare_array(first: np.ndarray, second: np.ndarray, atol: float, rtol: float) -> dict:
    first = np.asarray(first)
    second = np.asarray(second)
    shape_equal = first.shape == second.shape
    if not shape_equal:
        return {"shape_equal": False, "first_shape": list(first.shape), "second_shape": list(second.shape), "pass": False}
    difference = np.abs(first.astype(np.float64) - second.astype(np.float64))
    exact = np.array_equal(first, second)
    return {
        "shape_equal": True,
        "shape": list(first.shape),
        "exact_equal": bool(exact),
        "allclose": bool(np.allclose(first, second, atol=atol, rtol=rtol)),
        "max_absolute_difference": float(difference.max(initial=0.0)),
        "mean_absolute_difference": float(difference.mean()) if difference.size else 0.0,
        "first_sha256": array_sha256(first) if exact else None,
        "second_sha256": array_sha256(second) if exact else None,
        "pass": bool(np.allclose(first, second, atol=atol, rtol=rtol)),
    }


def compare_pca_up_to_sign(first: np.ndarray, second: np.ndarray, atol: float, rtol: float) -> dict:
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape:
        return {"shape_equal": False, "first_shape": list(first.shape), "second_shape": list(second.shape), "pass": False}
    dots = np.sum(first.astype(np.float64) * second.astype(np.float64), axis=0)
    signs = np.where(dots < 0, -1.0, 1.0)
    aligned = second * signs
    result = compare_array(first, aligned, atol=atol, rtol=rtol)
    result["column_sign_flips"] = int(np.sum(signs < 0))
    return result


def symmetrize_legacy_graph(graph: sp.spmatrix) -> sp.csr_matrix:
    graph = graph.tocsr().astype(np.float32)
    result = graph.maximum(graph.T).tocsr()
    result = result - sp.diags(result.diagonal())
    result.eliminate_zeros()
    result.data[:] = 1.0
    return result


def legacy_spatial_graph(frame, n_obs: int) -> sp.csr_matrix:
    directed = sp.coo_matrix(
        (frame["value"].to_numpy(dtype=np.float32), (frame["x"].to_numpy(), frame["y"].to_numpy())),
        shape=(n_obs, n_obs),
    )
    return symmetrize_legacy_graph(directed)


def edge_set(graph: sp.spmatrix) -> set:
    rows, cols = graph.tocsr().nonzero()
    return set(zip(rows.tolist(), cols.tolist()))


def compare_edges(first: sp.spmatrix, second: sp.spmatrix) -> dict:
    first_edges = edge_set(first)
    second_edges = edge_set(second)
    union = len(first_edges | second_edges)
    intersection = len(first_edges & second_edges)
    return {
        "first_nnz": int(first.nnz),
        "second_nnz": int(second.nnz),
        "exact_edge_set_equal": first_edges == second_edges,
        "only_first": len(first_edges - second_edges),
        "only_second": len(second_edges - first_edges),
        "intersection": intersection,
        "union": union,
        "jaccard": float(intersection / union) if union else 1.0,
        "pass": first_edges == second_edges,
    }


def torch_sparse_to_scipy(value: torch.Tensor) -> sp.csr_matrix:
    value = value.coalesce().cpu()
    indices = value.indices().numpy()
    data = value.values().numpy()
    return sp.coo_matrix((data, (indices[0], indices[1])), shape=tuple(value.shape)).tocsr()


def compare_sparse(first: sp.spmatrix, second: sp.spmatrix, atol: float, rtol: float) -> dict:
    first = first.tocsr()
    second = second.tocsr()
    if first.shape != second.shape:
        return {"shape_equal": False, "first_shape": list(first.shape), "second_shape": list(second.shape), "pass": False}
    difference = (first - second).tocsr()
    max_difference = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    mean_difference = float(np.sum(np.abs(difference.data)) / (first.shape[0] * first.shape[1]))
    scale = max(float(np.max(np.abs(first.data))) if first.nnz else 0.0, float(np.max(np.abs(second.data))) if second.nnz else 0.0)
    passed = max_difference <= atol + rtol * scale
    return {
        "shape_equal": True,
        "shape": list(first.shape),
        "first_nnz": int(first.nnz),
        "second_nnz": int(second.nnz),
        "difference_nnz": int(difference.nnz),
        "max_absolute_difference": max_difference,
        "mean_absolute_difference_over_all_entries": mean_difference,
        "pass": bool(passed and first.nnz == second.nnz),
    }


def dataset_parity(dataset: str, dataset_cfg: dict, night1: dict, parity: dict) -> dict:
    fix_seed(0)
    legacy, legacy_ids, coordinates = prepare_legacy(dataset, dataset_cfg)
    fix_seed(0)
    captured_pca = []
    captured_graphs = []
    original_pca = corrected_pipeline.pca
    original_graph = corrected_pipeline.symmetric_knn_graph

    def capture_pca(*args, **kwargs):
        result = original_pca(*args, **kwargs)
        captured_pca.append(np.asarray(result).copy())
        return result

    def capture_graph(*args, **kwargs):
        result = original_graph(*args, **kwargs)
        captured_graphs.append(result.copy())
        return result

    corrected_pipeline.pca = capture_pca
    corrected_pipeline.symmetric_knn_graph = capture_graph
    try:
        corrected = corrected_pipeline.prepare_corrected(dataset, dataset_cfg, night1, "corrected_unweighted")
    finally:
        corrected_pipeline.pca = original_pca
        corrected_pipeline.symmetric_knn_graph = original_graph
    if len(captured_pca) != (1 if dataset == "p22" else 2) or len(captured_graphs) != 4:
        raise AssertionError("Unexpected corrected preprocessing call structure")
    atol = parity["absolute_tolerance"]
    rtol = parity["relative_tolerance"]

    legacy_rna = legacy["adata_omics1"]
    legacy_mod2 = legacy["adata_omics2"]
    legacy_hvg = legacy_rna.var_names[legacy_rna.var["highly_variable"]].astype(str).to_numpy()
    corrected_hvg = np.asarray(corrected.data["selected_gene_names"], dtype=str)
    observations_equal = np.array_equal(legacy_ids.astype(str), corrected.obs_names.astype(str))
    hvg_equal = np.array_equal(legacy_hvg, corrected_hvg)

    corrected_rna_pca = captured_pca[0]

    legacy_spatial1 = legacy_spatial_graph(legacy_rna.uns["adj_spatial"], legacy_rna.n_obs)
    legacy_spatial2 = legacy_spatial_graph(legacy_mod2.uns["adj_spatial"], legacy_mod2.n_obs)
    corrected_spatial1 = captured_graphs[0]
    corrected_spatial2 = captured_graphs[1]
    legacy_feature1 = symmetrize_legacy_graph(legacy_rna.obsm["adj_feature"])
    legacy_feature2 = symmetrize_legacy_graph(legacy_mod2.obsm["adj_feature"])
    corrected_feature1 = captured_graphs[2]
    corrected_feature2 = captured_graphs[3]

    graph_pairs = {
        "spatial_omics1": (legacy_spatial1, corrected_spatial1, corrected.data["adj_spatial_omics1"]),
        "spatial_omics2": (legacy_spatial2, corrected_spatial2, corrected.data["adj_spatial_omics2"]),
        "feature_omics1": (legacy_feature1, corrected_feature1, corrected.data["adj_feature_omics1"]),
        "feature_omics2": (legacy_feature2, corrected_feature2, corrected.data["adj_feature_omics2"]),
    }
    graphs = {}
    for name, (legacy_raw, corrected_raw, corrected_normalized_torch) in graph_pairs.items():
        graphs[name] = {
            "edge_set": compare_edges(legacy_raw, corrected_raw),
            "normalized_adjacency": compare_sparse(
                torch_sparse_to_scipy(normalize_graph_sparse(legacy_raw)),
                torch_sparse_to_scipy(corrected_normalized_torch),
                atol,
                rtol,
            ),
        }

    checks = {
        "observation_ids": {
            "legacy_count": int(len(legacy_ids)),
            "corrected_count": int(len(corrected.obs_names)),
            "exact_names_and_order": bool(observations_equal),
            "pass": bool(observations_equal),
        },
        "hvg_gene_names": {
            "legacy_count": int(len(legacy_hvg)),
            "corrected_count": int(len(corrected_hvg)),
            "exact_names_and_order": bool(hvg_equal),
            "legacy_sha256": array_sha256(legacy_hvg) if hvg_equal else None,
            "corrected_sha256": array_sha256(corrected_hvg) if hvg_equal else None,
            "pass": bool(hvg_equal),
        },
        "rna_scaled_hvg_matrix": compare_array(
            np.asarray(legacy_rna.obsm["raw_feat"]), corrected.data["features_omics1"], atol, rtol
        ),
        "modality2_features": compare_array(
            np.asarray(legacy_mod2.obsm["feat"]), corrected.data["features_omics2"], atol, rtol
        ),
        "graphs": graphs,
    }

    checks["rna_pca_features"] = compare_pca_up_to_sign(
        np.asarray(legacy_rna.obsm["feat"]),
        corrected_rna_pca,
        parity["pca_absolute_tolerance_after_sign_alignment"],
        rtol,
    )

    flat_passes = [
        checks["observation_ids"]["pass"],
        checks["hvg_gene_names"]["pass"],
        checks["rna_scaled_hvg_matrix"]["pass"],
        checks["rna_pca_features"]["pass"],
        checks["modality2_features"]["pass"],
    ]
    for graph in graphs.values():
        flat_passes.extend([graph["edge_set"]["pass"], graph["normalized_adjacency"]["pass"]])
    checks["pass"] = bool(all(flat_passes))
    return checks


def copy_parameters(legacy: Encoder_overall, corrected: EncoderOverallCorrected) -> dict:
    pairs = {
        "encoder_omics1.weight->encoder1.weight": (legacy.encoder_omics1.weight, corrected.encoder1.weight),
        "decoder_omics1.weight->decoder1.weight": (legacy.decoder_omics1.weight, corrected.decoder1.weight),
        "encoder_omics2.weight->encoder2.weight": (legacy.encoder_omics2.weight, corrected.encoder2.weight),
        "decoder_omics2.weight->decoder2.weight": (legacy.decoder_omics2.weight, corrected.decoder2.weight),
        "atten_omics1.w_omega->attention1.w_omega": (legacy.atten_omics1.w_omega, corrected.attention1.w_omega),
        "atten_omics1.u_omega->attention1.u_omega": (legacy.atten_omics1.u_omega, corrected.attention1.u_omega),
        "atten_omics2.w_omega->attention2.w_omega": (legacy.atten_omics2.w_omega, corrected.attention2.w_omega),
        "atten_omics2.u_omega->attention2.u_omega": (legacy.atten_omics2.u_omega, corrected.attention2.u_omega),
        "atten_cross.w_omega->cross_attention.w_omega": (legacy.atten_cross.w_omega, corrected.cross_attention.w_omega),
        "atten_cross.u_omega->cross_attention.u_omega": (legacy.atten_cross.u_omega, corrected.cross_attention.u_omega),
    }
    with torch.no_grad():
        for source, target in pairs.values():
            target.copy_(source)
    return {name: list(source.shape) for name, (source, _) in pairs.items()}


def model_forward_parity(night1: dict, parity: dict) -> dict:
    cfg = night1["datasets"]["placenta"]
    fix_seed(17)
    prepared = corrected_pipeline.prepare_corrected("placenta", cfg, night1, "corrected_unweighted")
    features1 = torch.as_tensor(prepared.data["features_omics1"], dtype=torch.float32)
    features2 = torch.as_tensor(prepared.data["features_omics2"], dtype=torch.float32)
    adj = [
        prepared.data["adj_spatial_omics1"],
        prepared.data["adj_feature_omics1"],
        prepared.data["adj_spatial_omics2"],
        prepared.data["adj_feature_omics2"],
    ]
    fix_seed(31415)
    legacy = Encoder_overall(features1.shape[1], cfg["embedding_dim"], features2.shape[1], cfg["embedding_dim"])
    corrected = EncoderOverallCorrected(
        features1.shape[1], cfg["embedding_dim"], features2.shape[1], cfg["embedding_dim"]
    )
    mapping = copy_parameters(legacy, corrected)
    legacy.eval()
    corrected.eval()
    with torch.no_grad():
        first = legacy(features1, features2, *adj)
        second = corrected(features1, features2, *adj)
    keys = [
        "emb_latent_omics1",
        "emb_latent_omics2",
        "emb_latent_combined",
        "emb_recon_omics1",
        "emb_recon_omics2",
        "emb_latent_omics1_across_recon",
        "emb_latent_omics2_across_recon",
        "alpha_omics1",
        "alpha_omics2",
        "alpha",
    ]
    outputs = {
        key: compare_array(
            first[key].detach().cpu().numpy(),
            second[key].detach().cpu().numpy(),
            parity["model_forward_absolute_tolerance"],
            parity["relative_tolerance"],
        )
        for key in keys
    }
    return {
        "dataset": "placenta",
        "environment": "same-process controlled comparison in legacy PyTorch environment",
        "parameter_mapping": mapping,
        "outputs": outputs,
        "pass": bool(all(item["pass"] for item in outputs.values())),
    }


def softmax_parity(parity: dict) -> dict:
    fixed = torch.tensor([[1.0, -1.0], [0.25, 0.75], [-3.0, 2.0], [0.0, 0.0]], dtype=torch.float32)
    with torch.no_grad():
        legacy = F.softmax(fixed)
        explicit = torch.softmax(fixed, dim=1)
    result = compare_array(
        legacy.numpy(), explicit.numpy(), parity["model_forward_absolute_tolerance"], parity["relative_tolerance"]
    )
    result["input_shape"] = list(fixed.shape)
    result["legacy_call"] = "torch.nn.functional.softmax(x)"
    result["explicit_call"] = "torch.softmax(x, dim=1)"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/night2_loss_audit.json")
    parser.add_argument("--output", default="reports/night2_parity.json")
    args = parser.parse_args()
    config = json.loads((REPO / args.config).read_text(encoding="utf-8"))
    night1 = json.loads((REPO / config["night1_config"]).read_text(encoding="utf-8"))
    parity = config["parity"]
    report = {
        "schema_version": 1,
        "gate": "P0",
        "ground_truth_accessed": False,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "comparison_note": "Both legacy and corrected semantics are evaluated in one process to isolate code semantics; Night-1 package versions other than PyTorch were identical.",
        },
        "declared_tolerances": parity,
        "legacy_softmax_semantics": softmax_parity(parity),
        "datasets": {},
    }
    for dataset in ("a1", "placenta", "p22"):
        print("PARITY", dataset, flush=True)
        report["datasets"][dataset] = dataset_parity(dataset, night1["datasets"][dataset], night1, parity)
    print("MODEL_FORWARD placenta", flush=True)
    report["model_forward"] = model_forward_parity(night1, parity)
    report["p0_pass"] = bool(
        report["legacy_softmax_semantics"]["pass"]
        and all(value["pass"] for value in report["datasets"].values())
        and report["model_forward"]["pass"]
    )
    report["factorial_authorized"] = report["p0_pass"]
    output = REPO / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("P0_PASS=%s" % report["p0_pass"], flush=True)
    if not report["p0_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
