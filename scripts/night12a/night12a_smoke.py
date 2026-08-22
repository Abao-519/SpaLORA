#!/usr/bin/env python3
"""Night-12A registered feature-level zero-step real-path smoke runner."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night12a_schema_p0 import (
    UnifiedZeroStepAutoencoder,
    array_sha,
    assert_model_identity_blind,
    atomic_json,
    atomic_npz,
    atomic_torch,
    file_sha256,
    fixed_linked_features,
    gene_score_intervals,
    h05_endpoint,
    inspect_csv_matrix,
    load_selected_counts,
    map_adt_targets,
    normalize_counts,
    normalize_gene_scores,
    parse_ensembl79_genes,
    parse_ncbi_gene_info,
    read_coordinates,
    reconstruction_loss,
    scale_features,
    scan_fragments,
    seurat_clr_counts,
    sparse_sha,
    sparse_spatial_graph,
    text_sha256,
)

SEED = 20260822
LATENT = 64
ENGINEERING_K = 2
FEATURE_CAP = 256


def compact_audit(value: dict) -> dict:
    return {k: v for k, v in value.items()
            if k not in {"ordered_observation_ids", "feature_ids", "records", "counts",
                         "registered_fragment_depth"}}


def coordinates_for(records: dict, ordered_ids: list[str]) -> np.ndarray:
    return np.asarray([[records[x]["pixel_col"], records[x]["pixel_row"]]
                       for x in ordered_ids], dtype=np.float64)


def prepare_rna(path: Path, coordinate: dict, selected: list[str]):
    audit = inspect_csv_matrix(path, coordinate["ordered_ids"])
    loaded = load_selected_counts(path, audit, selected)
    value = normalize_counts(loaded["counts"], loaded["library_size"])
    return audit, value


def prepare_atac(args):
    coord = read_coordinates(args.coordinates)
    rna_audit = inspect_csv_matrix(args.rna, coord["ordered_ids"])
    registry = parse_ensembl79_genes(args.gtf)
    linked = fixed_linked_features(rna_audit["feature_ids"],
                                   registry["unique"].keys(), FEATURE_CAP)
    if len(linked) != FEATURE_CAP:
        raise ValueError("P5 linked feature cap did not close at 256")
    rna_loaded = load_selected_counts(args.rna, rna_audit, linked)
    view1 = normalize_counts(rna_loaded["counts"], rna_loaded["library_size"])
    intervals = gene_score_intervals(registry, linked, upstream_bp=5000)
    fragment = scan_fragments(args.other, rna_audit["ordered_observation_ids"],
                              intervals)
    if fragment["registered_missing_count"] != 0:
        raise ValueError("registered RNA spots missing from ATAC fragments")
    view2 = normalize_gene_scores(fragment["counts"],
                                  fragment["registered_fragment_depth"])
    metadata = {
        "kind": "RNA+ATAC",
        "rna_audit": compact_audit(rna_audit),
        "coordinate_audit": compact_audit(coord),
        "fragment_audit": compact_audit(fragment),
        "linked_features": linked,
        "linked_feature_sha256": text_sha256(linked),
        "mapping": {
            "authority": "Ensembl release 79 GRCm38 unique gene symbols",
            "interval_formula": "zero-based gene body plus strand-aware 5000 bp upstream",
            "fragment_assignment": "fragment midpoint inside linked interval",
            "normalization": "log1p(10000 * linked count / full registered fragment multiplicity)",
        },
    }
    return rna_audit["ordered_observation_ids"], coord, view1, view2, metadata


def prepare_protein(args):
    coord = read_coordinates(args.coordinates)
    rna_audit = inspect_csv_matrix(args.rna, coord["ordered_ids"])
    adt_audit = inspect_csv_matrix(args.other, coord["ordered_ids"])
    if rna_audit["ordered_observation_ids"] != adt_audit["ordered_observation_ids"]:
        raise ValueError("RNA and ADT ordered spot identifiers differ")
    mapping = map_adt_targets(adt_audit["feature_ids"],
                              parse_ncbi_gene_info(args.gene_info))
    rna_features = set(rna_audit["feature_ids"])
    eligible = sorted(
        [(row["canonical_identifier"], row["raw_target"])
         for row in mapping
         if row["status"] == "unique" and row["canonical_identifier"] in rna_features],
        key=lambda pair: pair[0],
    )[:FEATURE_CAP]
    if not eligible:
        raise ValueError("no authoritative RNA-ADT linked features")
    canonical = [x[0] for x in eligible]
    raw_targets = [x[1] for x in eligible]
    if len(set(canonical)) != len(canonical) or len(set(raw_targets)) != len(raw_targets):
        raise ValueError("RNA-ADT mapping is not one-to-one")
    rna_loaded = load_selected_counts(args.rna, rna_audit, canonical)
    view1 = normalize_counts(rna_loaded["counts"], rna_loaded["library_size"])
    adt_loaded = load_selected_counts(args.other, adt_audit, adt_audit["feature_ids"])
    adt_all = scale_features(seurat_clr_counts(adt_loaded["counts"]))
    adt_index = {name: i for i, name in enumerate(adt_audit["feature_ids"])}
    view2 = adt_all[:, [adt_index[name] for name in raw_targets]].astype(np.float32)
    metadata = {
        "kind": "RNA+protein",
        "rna_audit": compact_audit(rna_audit),
        "adt_audit": compact_audit(adt_audit),
        "coordinate_audit": compact_audit(coord),
        "mapping_status_counts": {
            status: sum(row["status"] == status for row in mapping)
            for status in ["unique", "ambiguous", "control", "unmapped"]
        },
        "linked_features": canonical,
        "linked_raw_targets": raw_targets,
        "linked_feature_sha256": text_sha256(canonical),
        "mapping_rows": mapping,
        "mapping": {
            "authority": "NCBI Gene exact case-folded Symbol or Synonym token",
            "selection": "canonical identifier lexical order, all eligible up to 256",
            "normalization": "RNA full-library log normalization; ADT CLR over all deposited targets then ddof=1 feature scaling",
        },
    }
    return rna_audit["ordered_observation_ids"], coord, view1, view2, metadata


def smoke(args):
    start = time.monotonic()
    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    if args.mode == "smoke-atac":
        ids, coordinate, view1, view2, metadata = prepare_atac(args)
    else:
        ids, coordinate, view1, view2, metadata = prepare_protein(args)
    coords = coordinates_for(coordinate["records"], ids)
    graph = sparse_spatial_graph(coords, ids, k=6)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedZeroStepAutoencoder([view1.shape[1], view2.shape[1]], LATENT).to(device)
    assert_model_identity_blind(model)
    model.eval()
    t1 = torch.as_tensor(view1, dtype=torch.float32, device=device)
    t2 = torch.as_tensor(view2, dtype=torch.float32, device=device)
    with torch.no_grad():
        result = model(t1, t2)
        loss = reconstruction_loss(result, t1, t2)
    arrays = {name: result[name].detach().cpu().numpy()
              for name in ["private1", "private2", "fused"]}
    partition, endpoint_aux = h05_endpoint(
        arrays["private1"], arrays["private2"], arrays["fused"],
        coords, ids, ENGINEERING_K, args.artifact_dir / "endpoint_initial",
    )
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dims": [int(view1.shape[1]), int(view2.shape[1])],
        "latent_dim": LATENT,
        "seed": SEED,
        "training_steps": 0,
        "ordered_id_sha256": text_sha256(ids),
        "source_sha256": {
            "rna": file_sha256(args.rna),
            "coordinates": file_sha256(args.coordinates),
            "other": file_sha256(args.other),
        },
    }
    atomic_torch(args.artifact_dir / "checkpoint.pt", checkpoint)
    atomic_npz(
        args.artifact_dir / "roundtrip_inputs_outputs.npz",
        view1=view1.astype(np.float32), view2=view2.astype(np.float32),
        coordinates=coords, ordered_ids=np.asarray(ids, dtype=str),
        private1=arrays["private1"], private2=arrays["private2"],
        fused=arrays["fused"], partition=partition.astype(np.int64),
    )
    gpu_peak = (int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0)
    audit = {
        "status": "SMOKE_FORWARD_COMPLETE_RELOAD_PENDING",
        "kind": metadata["kind"],
        "real_observations": len(ids),
        "input_shapes": [list(view1.shape), list(view2.shape)],
        "input_dtypes": [str(view1.dtype), str(view2.dtype)],
        "ordered_id_sha256": text_sha256(ids),
        "coordinates_shape": list(coords.shape),
        "coordinates_sha256": array_sha(coords),
        "sparse_graph_shape": list(graph.shape),
        "sparse_graph_nnz": int(graph.nnz),
        "sparse_graph_sha256": sparse_sha(graph),
        "dense_n_by_n_count": 0,
        "training_steps": 0,
        "reconstruction_loss": float(loss.detach().cpu()),
        "finite_reconstruction_loss": bool(torch.isfinite(loss).item()),
        "partition_shape": list(partition.shape),
        "partition_sha256": array_sha(partition),
        "partition_k": int(len(np.unique(partition))),
        "endpoint_aux_keys": sorted(endpoint_aux),
        "checkpoint_sha256": file_sha256(args.artifact_dir / "checkpoint.pt"),
        "roundtrip_payload_sha256": file_sha256(args.artifact_dir / "roundtrip_inputs_outputs.npz"),
        "gpu_peak_bytes": gpu_peak,
        "wall_seconds": time.monotonic() - start,
        "label_reads": 0,
        "metric_calls": 0,
        "metadata": metadata,
    }
    atomic_json(args.artifact_dir / "smoke_forward.json", audit)
    print(json.dumps({k: audit[k] for k in ["kind", "real_observations", "input_shapes", "reconstruction_loss", "partition_k", "wall_seconds"]}, sort_keys=True))


def reload(args):
    start = time.monotonic()
    bundle = np.load(args.artifact_dir / "roundtrip_inputs_outputs.npz")
    checkpoint = torch.load(args.artifact_dir / "checkpoint.pt", map_location="cpu")
    ids = [str(x) for x in bundle["ordered_ids"].tolist()]
    if text_sha256(ids) != checkpoint["ordered_id_sha256"]:
        raise ValueError("fresh-process ordered ID SHA mismatch")
    torch.manual_seed(int(checkpoint["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedZeroStepAutoencoder(checkpoint["input_dims"], checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    with torch.no_grad():
        result = model(
            torch.as_tensor(bundle["view1"], dtype=torch.float32, device=device),
            torch.as_tensor(bundle["view2"], dtype=torch.float32, device=device),
        )
    numerical = {}
    actual = {}
    for name in ["private1", "private2", "fused"]:
        actual[name] = result[name].detach().cpu().numpy()
        numerical[name] = bool(np.allclose(actual[name], bundle[name], rtol=1e-5, atol=1e-6))
    partition, _ = h05_endpoint(
        actual["private1"], actual["private2"], actual["fused"],
        bundle["coordinates"], ids, ENGINEERING_K,
        args.artifact_dir / "endpoint_reload",
    )
    partition_exact = bool(np.array_equal(partition, bundle["partition"]))
    status = "PASS" if all(numerical.values()) and partition_exact else "FAIL"
    audit = {
        "status": status,
        "fresh_process": True,
        "checkpoint_strict_load": True,
        "numerical_roundtrip": numerical,
        "canonical_partition_exact": partition_exact,
        "partition_sha256": array_sha(partition),
        "ordered_id_sha256": text_sha256(ids),
        "wall_seconds": time.monotonic() - start,
        "label_reads": 0,
        "metric_calls": 0,
    }
    atomic_json(args.artifact_dir / "fresh_process_reload.json", audit)
    print(json.dumps(audit, sort_keys=True))
    if status != "PASS":
        raise SystemExit(2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke-atac", "smoke-protein", "reload"])
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--rna", type=Path)
    parser.add_argument("--coordinates", type=Path)
    parser.add_argument("--other", type=Path)
    parser.add_argument("--gtf", type=Path)
    parser.add_argument("--gene-info", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "reload":
        reload(arguments)
    else:
        required = [arguments.rna, arguments.coordinates, arguments.other]
        if any(value is None for value in required):
            raise SystemExit("registered input paths are required")
        if arguments.mode == "smoke-atac" and arguments.gtf is None:
            raise SystemExit("registered Ensembl79 GTF is required")
        if arguments.mode == "smoke-protein" and arguments.gene_info is None:
            raise SystemExit("registered NCBI Gene mapping is required")
        smoke(arguments)
