"""Night-12B replicate-calibrated linked-feature identifiability primitives.

The implementation is deliberately label-free and identity-blind after input
registration.  Family-specific code is limited to the registered preprocessing
adapters; folds, OLS, decoys, bootstrap, leave-one-replicate-out scores, and
gates are shared.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats

from SpaLORA.night12a_schema_p0 import (
    atomic_json,
    atomic_npz,
    exact_reindex,
    file_sha256,
    gene_score_intervals,
    inspect_csv_matrix,
    load_selected_counts,
    normalize_counts,
    normalize_gene_scores,
    parse_ensembl79_genes,
    read_coordinates,
    scan_fragments,
    seurat_clr_counts,
    text_sha256,
)


P10_EXPECTED_COUNT = 83
P10_EXPECTED_SHA = "ac1b801fceba359bbd7e9454c6811400443440575e4022376c9f64940e612a43"
UNITS = ("P5S1", "P5S2", "P5S3", "P10S1", "P10S2", "P10S3")
FAMILIES = {
    "RNA_PLUS_ATAC": ("P5S1", "P5S2", "P5S3"),
    "RNA_PLUS_PROTEIN": ("P10S1", "P10S2", "P10S3"),
}
FOLD_COUNT = 5
BLOCK_BOOTSTRAPS = 64
BLOCK_SEED_BASE = 120064
DECISION_BOOTSTRAPS = 2000
DECISION_SEED = 120777
AXES = ("link_only", "bootstrap_only", "replicate_only", "combined")
ZERO_COUNTS = {
    "new_downloads": 0,
    "new_external_datasets": 0,
    "training_label_reads": 0,
    "evaluation_label_reads": 0,
    "total_label_reads": 0,
    "ARI": 0,
    "NMI": 0,
    "AMI": 0,
    "FMI": 0,
    "Q": 0,
    "annotation_based_spatial_metrics": 0,
    "clustering_endpoint_calls": 0,
    "neural_candidate_training_steps": 0,
    "third_party_benchmarks": 0,
    "MISAR_Y": 0,
    "E18_5": 0,
    "GSE263333_or_GSE213264_downloads": 0,
    "QCRD_calls": 0,
    "dataset_name_routing": 0,
    "family_specific_scientific_formula_or_threshold": 0,
    "dense_N_by_N_allocations": 0,
    "scientific_retries_or_fallbacks": 0,
    "spotwise_cross_replicate_alignment": 0,
    "Night12A_historical_raw_mutations": 0,
}


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def array_hash(value: np.ndarray) -> str:
    x = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(str(tuple(x.shape)).encode())
    h.update(x.tobytes(order="C"))
    return h.hexdigest()


def line_text_hash(values: Sequence[str]) -> str:
    """SHA-256 of one UTF-8 identifier per line, including the final newline."""
    return hashlib.sha256("".join(f"{x}\n" for x in values).encode("utf-8")).hexdigest()


def read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def raw_root_snapshot(root: Path) -> dict:
    root = Path(root).resolve()
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            st = path.stat()
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
                "mode": oct(st.st_mode & 0o777),
            })
    return {
        "resolved_root": str(root),
        "file_count": len(rows),
        "total_bytes": sum(x["size"] for x in rows),
        "metadata_sha256": canonical_hash(rows),
    }


def _load_schema(raw_root: Path, unit: str) -> dict:
    return read_json(Path(raw_root) / "derived" / "schema" / f"{unit}.json")


def _mapping_rows(repo: Path) -> List[dict]:
    path = Path(repo) / "outputs" / "night12a_handoff" / "adt_target_mapping.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_bridge_panel(repo: Path, raw_root: Path, out_dir: Path) -> dict:
    """Identifier-only panel construction; this function never opens matrices."""
    started = time.monotonic()
    repo, raw_root, out_dir = Path(repo), Path(raw_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = _mapping_rows(repo)
    by_unit: Dict[str, Dict[str, str]] = {}
    for unit in FAMILIES["RNA_PLUS_PROTEIN"]:
        rows = [r for r in mapping if r["unit_id"] == unit and r["status"] == "unique"]
        canonical_to_raw = {r["canonical_identifier"]: r["raw_target"] for r in rows}
        if len(canonical_to_raw) != len(rows):
            raise ValueError(f"{unit} unique ADT mapping is not one-to-one")
        by_unit[unit] = canonical_to_raw
    p10 = sorted(set.intersection(*(set(by_unit[u]) for u in FAMILIES["RNA_PLUS_PROTEIN"])))
    p10_sha = line_text_hash(p10)
    if len(p10) != P10_EXPECTED_COUNT or p10_sha != P10_EXPECTED_SHA:
        raise RuntimeError("P10-only registered 83-gene authority mismatch")

    schemas = {u: _load_schema(raw_root, u) for u in UNITS}
    rna_sets = {u: set(map(str, schemas[u]["rna_feature_ids"])) for u in UNITS}
    gtf_candidates = sorted((Path(raw_root) / "downloads").glob("Mus_musculus.GRCm38.79.gtf*"))
    if len(gtf_candidates) != 1:
        raise RuntimeError("unique Ensembl79 GRCm38 GTF not found")
    registry = parse_ensembl79_genes(gtf_candidates[0])
    final = sorted(set(p10) & set.intersection(*(rna_sets[u] for u in UNITS)) & set(registry["unique"]))
    if not 3 <= len(final) <= 83:
        raise RuntimeError(f"final structural panel count {len(final)} outside [3,83]")
    deleted = []
    for gene in p10:
        reasons = []
        for unit in UNITS:
            if gene not in rna_sets[unit]:
                reasons.append(f"missing_RNA_identifier:{unit}")
        if gene not in registry["unique"]:
            reasons.append("not_exactly_one_Ensembl79_interval")
        if reasons:
            deleted.append({"canonical_identifier": gene, "structural_reasons": reasons})

    panel_csv = out_dir / "bridge_panel.csv"
    with panel_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = ["panel_index", "canonical_identifier", "P10S1_raw_target",
                  "P10S2_raw_target", "P10S3_raw_target", "ensembl_gene_id",
                  "chrom", "start_1based", "end_1based", "strand"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, gene in enumerate(final):
            g = registry["unique"][gene]
            writer.writerow({
                "panel_index": i, "canonical_identifier": gene,
                **{f"{u}_raw_target": by_unit[u][gene] for u in FAMILIES["RNA_PLUS_PROTEIN"]},
                "ensembl_gene_id": g["gene_id"], "chrom": g["chrom"],
                "start_1based": g["start_1based"], "end_1based": g["end_1based"],
                "strand": g["strand"],
            })
    panel_txt = out_dir / "bridge_panel_identifiers.txt"
    with panel_txt.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(f"{x}\n" for x in final))
    audit = {
        "schema": "spalora.night12b.bridge_panel.v1",
        "numeric_matrix_reads": 0,
        "label_reads": 0,
        "p5_lexical_first_256_used": False,
        "p10_only_count": len(p10),
        "p10_only_ordered_sha256": p10_sha,
        "final_m": len(final),
        "final_ordered_identifier_sha256": line_text_hash(final),
        "panel_text_sha256": file_sha256(panel_txt),
        "panel_csv_sha256": file_sha256(panel_csv),
        "structural_deletion_count": len(deleted),
        "structural_deletions": deleted,
        "all_six_rna_feature_hashes": {u: text_sha256(schemas[u]["rna_feature_ids"]) for u in UNITS},
        "gtf_path": str(gtf_candidates[0].resolve()),
        "gtf_sha256": file_sha256(gtf_candidates[0]),
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(out_dir / "bridge_panel_construction_audit.json", audit)
    return audit


def load_panel(path: Path) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    genes = [r["canonical_identifier"] for r in rows]
    targets = {u: {r["canonical_identifier"]: r[f"{u}_raw_target"] for r in rows}
               for u in FAMILIES["RNA_PLUS_PROTEIN"]}
    return genes, targets


def normalized_coordinates(coordinate: Mapping[str, object], ids: Sequence[str]) -> Tuple[np.ndarray, dict]:
    records = coordinate["records"]
    coords = np.asarray([[records[x]["pixel_col"], records[x]["pixel_row"]] for x in ids], dtype=np.float64)
    low, high = coords.min(axis=0), coords.max(axis=0)
    span = high - low
    degenerate = span == 0
    safe = span.copy()
    safe[degenerate] = 1.0
    result = (coords - low) / safe
    result[:, degenerate] = 0.0
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite normalized coordinates")
    return result, {"min": low.tolist(), "max": high.tolist(), "degenerate_axes": degenerate.tolist()}


def spatial_blocks(coords: np.ndarray, ids: Sequence[str], count: int = FOLD_COUNT) -> Tuple[np.ndarray, dict]:
    value = np.asarray(coords, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or len(ids) != len(value):
        raise ValueError("coordinate/identifier shape mismatch")
    centered = value - value.mean(axis=0)
    covariance = centered.T @ centered / float(len(value))
    eigvals, eigvecs = np.linalg.eigh(covariance)
    fallback = bool(eigvals[-1] == eigvals[-2])
    if fallback:
        vector = np.asarray([1.0, 0.0])
    else:
        vector = eigvecs[:, -1]
        largest = 0 if abs(vector[0]) >= abs(vector[1]) else 1
        if vector[largest] < 0:
            vector = -vector
    projection = centered @ vector
    order = sorted(range(len(value)), key=lambda i: (projection[i], value[i, 0], value[i, 1], str(ids[i])))
    blocks = np.empty(len(value), dtype=np.int8)
    pieces = np.array_split(np.asarray(order, dtype=np.int64), count)
    if any(len(x) == 0 for x in pieces):
        raise ValueError("empty spatial block")
    for block, indices in enumerate(pieces):
        blocks[indices] = block
    return blocks, {
        "count": count,
        "sizes": [len(x) for x in pieces],
        "leading_eigenvalues": eigvals.tolist(),
        "leading_vector": vector.tolist(),
        "exact_tie_fallback_to_x": fallback,
    }


def _download_paths(repo: Path) -> Dict[Tuple[str, str], Path]:
    manifest = read_json(Path(repo) / "outputs" / "night12a_handoff" / "download_manifest.json")
    result = {}
    for r in manifest["records"]:
        result[(r["replicate"], r["modality"])] = Path(r["absolute_path"])
    return result


def complete_header_contract(audit: Mapping[str, object], path: Path) -> dict:
    """Restore the parser flag omitted from some compact Night-12A audits."""
    result = dict(audit)
    if "header_index_field_present" in result:
        return result
    opener = gzip.open if str(path).endswith(".gz") else open
    delimiter = "\t" if ".tsv" in Path(path).name else ","
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader)
        first = next(reader)
    if len(first) == len(header):
        result["header_index_field_present"] = True
    elif len(first) == len(header) + 1:
        result["header_index_field_present"] = False
    else:
        raise ValueError("matrix header/data width contract invalid on reload")
    return result


def prepare_unit(repo: Path, raw_root: Path, panel_path: Path, unit: str, out_root: Path) -> dict:
    if unit not in UNITS:
        raise ValueError("unit is not registered")
    started = time.monotonic()
    genes, raw_targets = load_panel(panel_path)
    schema = _load_schema(raw_root, unit)
    download = _download_paths(repo)
    rna_path = Path(schema["rna"]["source_path"])
    rna_audit = complete_header_contract(schema["rna"], rna_path)
    rna_audit["ordered_observation_ids"] = schema["rna_ordered_observation_ids"]
    rna_audit["feature_ids"] = schema["rna_feature_ids"]
    rna_loaded = load_selected_counts(rna_path, rna_audit, genes)
    rna = normalize_counts(rna_loaded["counts"], rna_loaded["library_size"])
    ids = list(rna_loaded["observations"])
    coord_path = download[(unit, "spatial_coordinates")]
    coordinate = read_coordinates(coord_path)
    coords, coordinate_audit = normalized_coordinates(coordinate, ids)
    blocks, fold_audit = spatial_blocks(coords, ids)

    join_audit: Dict[str, object]
    if unit.startswith("P5"):
        gtf = next((Path(raw_root) / "downloads").glob("Mus_musculus.GRCm38.79.gtf*"))
        registry = parse_ensembl79_genes(gtf)
        intervals = gene_score_intervals(registry, genes, upstream_bp=5000)
        fragment = scan_fragments(download[(unit, "ATAC_fragments")], ids, intervals)
        if fragment["registered_missing_count"] != 0:
            raise ValueError("registered RNA spots missing from ATAC fragments")
        target = normalize_gene_scores(fragment["counts"], fragment["registered_fragment_depth"])
        join_audit = {
            "adapter": "registered_P5_clean_room_gene_score",
            "rna_rows": len(ids),
            "all_atac_barcodes": fragment["all_barcode_count"],
            "registered_atac_barcodes": fragment["registered_barcode_count"],
            "joined_rows": len(ids),
            "rna_unmatched": fragment["registered_missing_count"],
            "atac_unmatched_from_full_barcode_set": fragment["all_barcode_count"] - fragment["registered_barcode_count"],
            "fragment_rows": fragment["fragment_rows"],
            "clean_room_not_ArchR_equivalent": True,
        }
    else:
        adt_path = Path(schema["adt"]["source_path"])
        adt_audit = inspect_csv_matrix(adt_path, ids)
        adt_loaded = load_selected_counts(adt_path, adt_audit, adt_audit["feature_ids"])
        clr_all = seurat_clr_counts(adt_loaded["counts"])
        reindex = exact_reindex(adt_loaded["observations"], ids)
        adt_index = {x: i for i, x in enumerate(adt_audit["feature_ids"])}
        selected = [adt_index[raw_targets[unit][g]] for g in genes]
        target = clr_all[reindex][:, selected].astype(np.float32)
        join_audit = {
            "adapter": "registered_P10_all_target_CLR_then_structural_select",
            "rna_rows": len(ids), "adt_rows": len(adt_loaded["observations"]),
            "joined_rows": len(ids), "rna_unmatched": 0, "adt_unmatched": 0,
            "global_feature_scaling": False,
        }
    if rna.shape != target.shape or rna.shape != (len(ids), len(genes)):
        raise ValueError("bridge matrices do not close to n by m")
    if not (np.all(np.isfinite(rna)) and np.all(np.isfinite(target)) and np.all(np.isfinite(coords))):
        raise ValueError("non-finite registered unit")

    unit_dir = Path(out_root) / "preflight" / unit
    unit_dir.mkdir(parents=True, exist_ok=True)
    artifact = unit_dir / "registered_matrices.npz"
    atomic_npz(artifact, rna=rna.astype(np.float32), target=target.astype(np.float32),
               coordinates=coords.astype(np.float64), blocks=blocks,
               observation_ids=np.asarray(ids), genes=np.asarray(genes))
    fold_path = unit_dir / "spatial_fold_assignments.csv.gz"
    with gzip.open(fold_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["ordered_index", "canonical_spot_id", "block"])
        writer.writerows((i, ids[i], int(blocks[i])) for i in range(len(ids)))

    smoke = reference_crossfit(rna[:, [0]], target[:, [0]], coords, blocks)
    if not all(np.all(np.isfinite(x)) for x in smoke):
        raise ValueError("preflight baseline/linked path non-finite")
    result = {
        "schema": "spalora.night12b.real_unit_preflight.v1",
        "unit_id": unit, "m": len(genes), "observations": len(ids),
        "rna_shape": list(rna.shape), "target_shape": list(target.shape),
        "coordinate_shape": list(coords.shape), "dtype": "float32",
        "ordered_id_sha256": text_sha256(ids), "gene_sha256": text_sha256(genes),
        "rna_array_sha256": array_hash(rna), "target_array_sha256": array_hash(target),
        "coordinate_array_sha256": array_hash(coords), "block_array_sha256": array_hash(blocks),
        "artifact_path": str(artifact.resolve()), "artifact_sha256": file_sha256(artifact),
        "fold_assignment_path": str(fold_path.resolve()), "fold_assignment_sha256": file_sha256(fold_path),
        "finite": True, "fresh_process_reload": False,
        "coordinate_audit": coordinate_audit, "fold_audit": fold_audit,
        "join_audit": join_audit, "scientific_utility_reported": False,
        "label_reads": 0, "forbidden_actions": 0,
        "wall_seconds": time.monotonic() - started,
        "peak_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    atomic_json(unit_dir / "preflight.json", result)
    return result


def reload_unit(artifact: Path, preflight_json: Path, output: Path) -> dict:
    expected = read_json(preflight_json)
    with np.load(artifact, allow_pickle=False) as data:
        actual = {
            "rna_array_sha256": array_hash(data["rna"]),
            "target_array_sha256": array_hash(data["target"]),
            "coordinate_array_sha256": array_hash(data["coordinates"]),
            "block_array_sha256": array_hash(data["blocks"]),
            "ordered_id_sha256": text_sha256(data["observation_ids"].tolist()),
            "gene_sha256": text_sha256(data["genes"].tolist()),
            "rna_shape": list(data["rna"].shape),
            "target_shape": list(data["target"].shape),
            "finite": bool(np.all(np.isfinite(data["rna"])) and np.all(np.isfinite(data["target"]))),
        }
    keys = ["rna_array_sha256", "target_array_sha256", "coordinate_array_sha256",
            "block_array_sha256", "ordered_id_sha256", "gene_sha256", "rna_shape", "target_shape"]
    passed = actual["finite"] and all(actual[k] == expected[k] for k in keys)
    result = {"unit_id": expected["unit_id"], "fresh_process": True,
              "round_trip_passed": passed, "actual": actual}
    atomic_json(output, result)
    if not passed:
        raise RuntimeError("fresh-process registered artifact reload mismatch")
    return result


def spatial_basis(coords: np.ndarray) -> np.ndarray:
    x, y = np.asarray(coords, dtype=np.float64).T
    return np.column_stack([np.ones(len(x)), x, y, x * x, x * y, y * y])


def _standardize_train_apply(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64)
    std = train.std(axis=0, ddof=0, dtype=np.float64)
    zero = std == 0
    safe = std.copy()
    safe[zero] = 1.0
    a, b = (train - mean) / safe, (test - mean) / safe
    a[:, zero] = 0.0
    b[:, zero] = 0.0
    return a, b, zero


def reference_crossfit(rna: np.ndarray, target: np.ndarray, coords: np.ndarray,
                       blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Frozen per-candidate OLS; returns block baseline/linked SSE and zero flags."""
    rna, target = np.asarray(rna, dtype=np.float64), np.asarray(target, dtype=np.float64)
    m = rna.shape[1]
    baseline = np.zeros((FOLD_COUNT, m), dtype=np.float64)
    linked = np.zeros((FOLD_COUNT, m, m), dtype=np.float64)
    zero_rna = np.zeros((FOLD_COUNT, m), dtype=bool)
    zero_target = np.zeros((FOLD_COUNT, m), dtype=bool)
    basis = spatial_basis(coords)
    for fold in range(FOLD_COUNT):
        test_mask = blocks == fold
        train_mask = ~test_mask
        ytr, yte, zy = _standardize_train_apply(target[train_mask], target[test_mask])
        xtr, xte, zx = _standardize_train_apply(rna[train_mask], rna[test_mask])
        zero_rna[fold], zero_target[fold] = zx, zy
        btr, bte = basis[train_mask], basis[test_mask]
        beta = np.linalg.lstsq(btr, ytr, rcond=1e-12)[0]
        residual = yte - bte @ beta
        baseline[fold] = np.sum(residual * residual, axis=0, dtype=np.float64)
        for candidate in range(m):
            dtr = np.column_stack([btr, xtr[:, candidate]])
            dte = np.column_stack([bte, xte[:, candidate]])
            beta = np.linalg.lstsq(dtr, ytr, rcond=1e-12)[0]
            residual = yte - dte @ beta
            linked[fold, :, candidate] = np.sum(residual * residual, axis=0, dtype=np.float64)
    return baseline, linked, zero_rna, zero_target


def optimized_crossfit(rna: np.ndarray, target: np.ndarray, coords: np.ndarray,
                       blocks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Frisch-Waugh equivalent of the frozen per-candidate OLS.

    It keeps the exact training-only standardization and held-out predictions,
    but evaluates every candidate/target SSE through matrix cross-products.
    Numerically rank-deficient candidate columns fall back to the literal
    reference fit.
    """
    rna, target = np.asarray(rna, dtype=np.float64), np.asarray(target, dtype=np.float64)
    m = rna.shape[1]
    baseline = np.zeros((FOLD_COUNT, m), dtype=np.float64)
    linked = np.zeros((FOLD_COUNT, m, m), dtype=np.float64)
    zero_rna = np.zeros((FOLD_COUNT, m), dtype=bool)
    zero_target = np.zeros((FOLD_COUNT, m), dtype=bool)
    basis = spatial_basis(coords)
    for fold in range(FOLD_COUNT):
        test_mask = blocks == fold
        train_mask = ~test_mask
        ytr, yte, zy = _standardize_train_apply(target[train_mask], target[test_mask])
        xtr, xte, zx = _standardize_train_apply(rna[train_mask], rna[test_mask])
        zero_rna[fold], zero_target[fold] = zx, zy
        btr, bte = basis[train_mask], basis[test_mask]
        beta_y = np.linalg.lstsq(btr, ytr, rcond=1e-12)[0]
        gamma_x = np.linalg.lstsq(btr, xtr, rcond=1e-12)[0]
        residual_y_train = ytr - btr @ beta_y
        residual_x_train = xtr - btr @ gamma_x
        residual_y_test = yte - bte @ beta_y
        residual_x_test = xte - bte @ gamma_x
        base = np.sum(residual_y_test * residual_y_test, axis=0, dtype=np.float64)
        baseline[fold] = base
        denominator = np.sum(residual_x_train * residual_x_train, axis=0, dtype=np.float64)
        numerator = residual_x_train.T @ residual_y_train
        test_cross = residual_x_test.T @ residual_y_test
        test_norm = np.sum(residual_x_test * residual_x_test, axis=0, dtype=np.float64)
        for candidate in range(m):
            tolerance = np.finfo(np.float64).eps * max(1.0, float(np.sum(xtr[:, candidate] ** 2))) * 100.0
            if denominator[candidate] <= tolerance:
                dtr = np.column_stack([btr, xtr[:, candidate]])
                dte = np.column_stack([bte, xte[:, candidate]])
                beta = np.linalg.lstsq(dtr, ytr, rcond=1e-12)[0]
                residual = yte - dte @ beta
                linked[fold, :, candidate] = np.sum(residual * residual, axis=0, dtype=np.float64)
            else:
                coefficient = numerator[candidate] / denominator[candidate]
                linked[fold, :, candidate] = (
                    base - 2.0 * coefficient * test_cross[candidate]
                    + coefficient * coefficient * test_norm[candidate]
                )
    return baseline, linked, zero_rna, zero_target


def utility_and_tail(baseline: np.ndarray, linked: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    base = np.sum(baseline, axis=0)
    link = np.sum(linked, axis=0)
    utility = (base[:, None] - link) / np.maximum(base[:, None], 1e-12)
    m = utility.shape[0]
    p = np.empty(m, dtype=np.float64)
    for j in range(m):
        true = utility[j, j]
        p[j] = (1.0 + np.count_nonzero(np.delete(utility[j], j) >= true)) / float(m)
    return utility, p


def unit_bootstrap(unit: str, baseline: np.ndarray, linked: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    seed = BLOCK_SEED_BASE + (int(hashlib.sha256(unit.encode()).hexdigest()[:8], 16) % 1000000)
    rng = np.random.default_rng(seed)
    draws = np.empty((BLOCK_BOOTSTRAPS, linked.shape[1]), dtype=np.float64)
    for b in range(BLOCK_BOOTSTRAPS):
        indices = rng.integers(0, FOLD_COUNT, size=FOLD_COUNT)
        _, draws[b] = utility_and_tail(baseline[indices], linked[indices])
    unstable = (1.0 + np.count_nonzero(draws > 0.5, axis=0)) / float(BLOCK_BOOTSTRAPS + 1)
    return draws, unstable, seed


def run_unit_formal(unit: str, artifact: Path, output_dir: Path) -> dict:
    started = time.monotonic()
    with np.load(artifact, allow_pickle=False) as data:
        rna = data["rna"].astype(np.float64)
        target = data["target"].astype(np.float64)
        coords = data["coordinates"].astype(np.float64)
        blocks = data["blocks"]
        genes = data["genes"].tolist()
    baseline, linked, zero_rna, zero_target = optimized_crossfit(rna, target, coords, blocks)
    utility, p_link = utility_and_tail(baseline, linked)
    p_boot_draws, p_unstable, seed = unit_bootstrap(unit, baseline, linked)
    nonident = np.any(zero_rna, axis=0) | np.any(zero_target, axis=0)
    out = Path(output_dir) / unit
    out.mkdir(parents=True, exist_ok=True)
    atomic_npz(out / "formal_arrays.npz", baseline_sse=baseline, linked_sse=linked,
               utility=utility, p_link=p_link, p_link_bootstrap=p_boot_draws,
               p_unstable=p_unstable, non_identifiable=nonident.astype(np.uint8),
               genes=np.asarray(genes))
    summary = {
        "unit_id": unit, "m": len(genes), "finite": bool(all(np.all(np.isfinite(x)) for x in
            [baseline, linked, utility, p_link, p_boot_draws, p_unstable])),
        "block_bootstrap_count": BLOCK_BOOTSTRAPS, "block_bootstrap_seed": seed,
        "non_identifiable_count": int(nonident.sum()),
        "zero_predictor_by_fold_count": int(zero_rna.sum()),
        "zero_target_by_fold_count": int(zero_target.sum()),
        "formal_array_sha256": file_sha256(out / "formal_arrays.npz"),
        "wall_seconds": time.monotonic() - started,
        "peak_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    atomic_json(out / "formal_summary.json", summary)
    return summary


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, "undefined_or_constant_set_to_zero"
    value = float(stats.spearmanr(x, y).statistic)
    if not np.isfinite(value):
        return 0.0, "nonfinite_set_to_zero"
    return value, "finite"


def fisher_pool(values: Sequence[float]) -> float:
    x = np.clip(np.asarray(values, dtype=np.float64), -1 + 1e-12, 1 - 1e-12)
    return float(np.tanh(np.mean(np.arctanh(x))))


def _scores(p_a: np.ndarray, u_a: np.ndarray, p_b: np.ndarray, u_b: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "link_only": -np.log10(np.sqrt(p_a * p_b)),
        "bootstrap_only": -np.log10(np.maximum(u_a, u_b)),
        "replicate_only": -np.log10(np.maximum(p_a, p_b)),
        "combined": -np.log10(np.maximum.reduce([p_a, p_b, u_a, u_b])),
    }


def summarize_formal(formal_root: Path, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for unit in UNITS:
        with np.load(Path(formal_root) / unit / "formal_arrays.npz", allow_pickle=False) as z:
            data[unit] = {k: z[k] for k in ("utility", "p_link", "p_unstable", "non_identifiable", "genes")}
    genes = data[UNITS[0]]["genes"].tolist()
    if any(data[u]["genes"].tolist() != genes for u in UNITS):
        raise ValueError("formal gene identity mismatch")

    per_candidate = out_dir / "per_candidate_crossfit_utility.csv.gz"
    with gzip.open(per_candidate, "wt", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["unit_id", "target_gene", "candidate_gene", "is_true_link", "crossfit_utility"])
        for unit in UNITS:
            for j, target_gene in enumerate(genes):
                for g, candidate_gene in enumerate(genes):
                    w.writerow([unit, target_gene, candidate_gene, int(j == g), format(float(data[unit]["utility"][j, g]), ".17g")])

    per_feature = out_dir / "per_feature_replicate_evidence.csv"
    with per_feature.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["unit_id", "canonical_gene", "p_link", "p_unstable", "non_identifiable"])
        for unit in UNITS:
            for j, gene in enumerate(genes):
                w.writerow([unit, gene, format(float(data[unit]["p_link"][j]), ".17g"),
                            format(float(data[unit]["p_unstable"][j]), ".17g"), int(data[unit]["non_identifiable"][j])])

    fold_rows, score_rows, family_results, draw_rows = [], [], {}, []
    for family, units in FAMILIES.items():
        correlations = {axis: [] for axis in AXES}
        fold_cache = []
        for h_index, heldout in enumerate(units):
            discovery = [u for u in units if u != heldout]
            s = _scores(data[discovery[0]]["p_link"], data[discovery[0]]["p_unstable"],
                        data[discovery[1]]["p_link"], data[discovery[1]]["p_unstable"])
            outcome = -np.log10(data[heldout]["p_link"])
            row = {"family": family, "heldout_unit": heldout,
                   "median_heldout_p_link": float(np.median(data[heldout]["p_link"]))}
            reasons = {}
            for axis in AXES:
                rho, reason = safe_spearman(s[axis], outcome)
                correlations[axis].append(rho)
                row[f"spearman_{axis}"] = rho
                reasons[axis] = reason
            row["correlation_reasons"] = reasons
            fold_rows.append(row)
            fold_cache.append((s, outcome))
            for j, gene in enumerate(genes):
                score_rows.append([family, heldout, gene, outcome[j], *(s[a][j] for a in AXES)])

        pooled = {axis: fisher_pool(correlations[axis]) for axis in AXES}
        rng = np.random.default_rng(DECISION_SEED)
        draws = {axis: np.empty(DECISION_BOOTSTRAPS) for axis in AXES}
        difference = np.empty(DECISION_BOOTSTRAPS)
        m = len(genes)
        for b in range(DECISION_BOOTSTRAPS):
            idx = rng.integers(0, m, size=m)
            pooled_b = {}
            for axis in AXES:
                rs = [safe_spearman(s[axis][idx], y[idx])[0] for s, y in fold_cache]
                pooled_b[axis] = fisher_pool(rs)
                draws[axis][b] = pooled_b[axis]
            difference[b] = pooled_b["combined"] - max(pooled_b[a] for a in AXES[:3])
            draw_rows.append([family, b, *(draws[a][b] for a in AXES), difference[b]])
        ci = {axis: np.quantile(draws[axis], [0.025, 0.975], method="linear").tolist() for axis in AXES}
        diff_ci = np.quantile(difference, [0.025, 0.975], method="linear").tolist()
        family_fold_rows = [x for x in fold_rows if x["family"] == family]
        gates = {
            "combined_pooled_gt_zero": pooled["combined"] > 0,
            "all_three_combined_folds_gt_zero": all(x["spearman_combined"] > 0 for x in family_fold_rows),
            "combined_pooled_ci_lower_gt_zero": ci["combined"][0] > 0,
            "combined_minus_best_single_ci_lower_gt_zero": diff_ci[0] > 0,
            "all_three_median_p_link_lt_half": all(x["median_heldout_p_link"] < 0.5 for x in family_fold_rows),
        }
        gates["family_pass"] = all(gates.values())
        family_results[family] = {
            "folds": family_fold_rows, "pooled_spearman": pooled,
            "pooled_ci95": ci, "combined_minus_best_single_ci95": diff_ci,
            "gates": gates,
        }

    scores_path = out_dir / "leave_one_replicate_out_scores.csv"
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["family", "heldout_unit", "canonical_gene", "heldout_outcome",
                    "score_link_only", "score_bootstrap_only", "score_replicate_only", "score_combined"])
        w.writerows(score_rows)
    fold_path = out_dir / "heldout_fold_summary.csv"
    with fold_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["family", "heldout_unit", "median_heldout_p_link"] + [f"spearman_{a}" for a in AXES]
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(fold_rows)
    draws_path = out_dir / "decision_bootstrap_draws.csv.gz"
    with gzip.open(draws_path, "wt", encoding="utf-8", newline="") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["family", "draw", *(f"pooled_{a}" for a in AXES), "combined_minus_best_single"])
        w.writerows(draw_rows)
    result = {
        "schema": "spalora.night12b.decision_bootstrap.v1", "m": len(genes),
        "decision_bootstrap_count": DECISION_BOOTSTRAPS, "decision_seed": DECISION_SEED,
        "families": family_results,
        "all_six_finite": all(read_json(Path(formal_root) / u / "formal_summary.json")["finite"] for u in UNITS),
    }
    for family in family_results:
        family_results[family]["gates"]["six_units_and_firewall_pass"] = result["all_six_finite"]
        family_results[family]["gates"]["family_pass"] = all(family_results[family]["gates"].values())
    result["overall_pass"] = all(x["gates"]["family_pass"] for x in family_results.values())
    atomic_json(out_dir / "decision_bootstrap_summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("panel")
    a.add_argument("--repo", type=Path, required=True); a.add_argument("--raw-root", type=Path, required=True); a.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--repo", type=Path, required=True); a.add_argument("--raw-root", type=Path, required=True); a.add_argument("--panel", type=Path, required=True); a.add_argument("--unit", choices=UNITS, required=True); a.add_argument("--out-root", type=Path, required=True)
    a = sub.add_parser("reload")
    a.add_argument("--artifact", type=Path, required=True); a.add_argument("--preflight", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("formal-unit")
    a.add_argument("--unit", choices=UNITS, required=True); a.add_argument("--artifact", type=Path, required=True); a.add_argument("--output-dir", type=Path, required=True)
    a = sub.add_parser("summarize")
    a.add_argument("--formal-root", type=Path, required=True); a.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("snapshot")
    a.add_argument("--raw-root", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "panel":
        build_bridge_panel(args.repo, args.raw_root, args.out)
    elif args.command == "prepare":
        prepare_unit(args.repo, args.raw_root, args.panel, args.unit, args.out_root)
    elif args.command == "reload":
        reload_unit(args.artifact, args.preflight, args.output)
    elif args.command == "formal-unit":
        run_unit_formal(args.unit, args.artifact, args.output_dir)
    elif args.command == "summarize":
        summarize_formal(args.formal_root, args.out)
    elif args.command == "snapshot":
        atomic_json(args.output, raw_root_snapshot(args.raw_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
