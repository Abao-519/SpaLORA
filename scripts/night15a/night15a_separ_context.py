#!/usr/bin/env python3
"""Run the single official SEPAR MISAR tutorial lane as protocol context.

This wrapper imports the fixed MIT-licensed upstream source from the external
source root.  It performs byte-exact ID alignment before concatenation and does
not open ``Y`` until SEPAR has emitted a partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


UPSTREAM = Path("/root/autodl-fs/night15a_method_sources_20260823/SEPAR")
PROJECT = Path("/root/autodl-fs/night8b_raw_runs_20260820")
RNA = PROJECT / "annotation_carrier/MISAR_seq_mouse_E15_brain_mRNA_data.h5"
ATAC = PROJECT / "annotation_carrier/MISAR_seq_mouse_E15_brain_ATAC_data.h5"
sys.path.insert(0, str(UPSTREAM))

from SEPAR_model import SEPAR  # noqa: E402


def _decode(values) -> np.ndarray:
    return np.asarray(
        [item.decode() if isinstance(item, bytes) else str(item) for item in values],
        dtype=str,
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    ids = (
        pd.read_csv(PROJECT / "cache/base/observation_ids.tsv", sep="\t")
        .iloc[:, 0]
        .astype(str)
        .to_numpy()
    )
    with h5py.File(str(ATAC), "r") as handle:
        atac_ids = _decode(handle["cell"][:])
        atac_index = pd.Index(atac_ids).get_indexer(ids)
        if np.any(atac_index < 0):
            raise RuntimeError("registered ID missing in ATAC carrier")
        x_atac = np.asarray(handle["X"][:], dtype=np.float32)[atac_index]
        positions = np.asarray(handle["pos"][:], dtype=np.float64)[atac_index]
        peaks = _decode(handle["peak"][:])
        # Y deliberately remains unopened until after clustering.
    with h5py.File(str(RNA), "r") as handle:
        rna_ids = _decode(handle["cell"][:])
        rna_index = pd.Index(rna_ids).get_indexer(ids)
        if np.any(rna_index < 0):
            raise RuntimeError("registered ID missing in RNA carrier")
        x_rna = np.asarray(handle["X"][:], dtype=np.float32)[rna_index]
        rna_positions = np.asarray(handle["pos"][:], dtype=np.float64)[rna_index]
        genes = _decode(handle["gene"][:])
    if not np.array_equal(positions, rna_positions):
        raise RuntimeError("modality coordinate mismatch after exact ID alignment")

    atac = ad.AnnData(
        X=x_atac,
        obs=pd.DataFrame(index=ids),
        var=pd.DataFrame(index=pd.Index(peaks).astype(str)),
    )
    rna = ad.AnnData(
        X=x_rna,
        obs=pd.DataFrame(index=ids),
        var=pd.DataFrame(index=pd.Index(genes).astype(str)),
    )
    sc.pp.normalize_total(atac, target_sum=1e4)
    sc.pp.log1p(atac)
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    var = pd.concat(
        [atac.var.assign(batch="ATAC"), rna.var.assign(batch="RNA")], axis=0
    )
    combined = ad.AnnData(
        X=np.concatenate((np.asarray(atac.X), np.asarray(rna.X)), axis=1),
        obs=pd.DataFrame(index=ids),
        var=var,
    )
    combined.obsm["spatial"] = positions

    model = SEPAR(combined, n_cluster=7, use_gpu=False, dtype=np.float32)
    model.preprocess(min_cells=50, normalize=False)
    model.compute_graph(radius_rate=1.3)
    model.select_morani(nslt=5000)
    selected_batches = model.adata.var["batch"].value_counts().to_dict()
    model.compute_weight(n_cluster=7)
    model.separ_algorithm(r=30, alpha=0.5, beta=0.01, gamma=0.5, mean=False)
    # Upstream 6d3475f can leave NumPy ArrayView subclasses attached when used
    # with newer AnnData releases.  In-place boolean assignment in clustering()
    # then routes through AnnData view semantics and fails with a row/column
    # mask mismatch.  Materializing the same numerical arrays is an API
    # compatibility correction only; no formula, pattern or threshold changes.
    model.Wpn = np.asarray(model.Wpn, dtype=np.float32).copy()
    model.Hpn = np.asarray(model.Hpn, dtype=np.float32).copy()
    model.exp_mat = np.asarray(model.exp_mat, dtype=np.float32).copy()
    np.savez_compressed(
        output / "preclustering_state.npz",
        Wpn=model.Wpn,
        Hpn=model.Hpn,
        exp_mat=model.exp_mat,
        ids=ids,
    )
    partition = np.asarray(
        model.clustering(n_cluster=12, N1=15, N2=1), dtype=np.int64
    )

    # Evaluation boundary: the first and only Y access occurs after partition.
    with h5py.File(str(ATAC), "r") as handle:
        raw = np.asarray(handle["Y"][:]).reshape(-1)[atac_index]
    labels = _decode(raw)
    result = {
        "lane": "EXTERNAL_SEPAR_OFFICIAL_TUTORIAL4_CONTEXT",
        "upstream_commit": "6d3475fa0bd749d3b1b5592b68323439d473f9fc",
        "license": "MIT",
        "observations": int(len(ids)),
        "input_shapes": {
            "rna": list(x_rna.shape),
            "atac": list(x_atac.shape),
            "combined_before_selection": list(combined.shape),
            "combined_after_moran_selection": list(model.adata.shape),
        },
        "selected_batches": {str(k): int(v) for k, v in selected_batches.items()},
        "cluster_k": 12,
        "ground_truth_unique": int(len(np.unique(labels))),
        "absolute_ari": float(adjusted_rand_score(labels, partition)),
        "absolute_nmi": float(normalized_mutual_info_score(labels, partition)),
        "partition_sha256": _array_sha(partition),
        "labels_in_model_input_or_loss": False,
        "Y_reads_before_partition": 0,
        "Y_reads_after_partition": 1,
        "compatibility_correction": (
            "materialize upstream AnnData ArrayView subclasses as byte-equivalent "
            "NumPy arrays before unchanged clustering"
        ),
        "source_sha256": {"rna": _sha(RNA), "atac": _sha(ATAC)},
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    temporary = output / "separ_context_result.json.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(output / "separ_context_result.json"))
    np.savez_compressed(output / "partition.npz", ids=ids, partition=partition)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.output))


if __name__ == "__main__":
    main()
