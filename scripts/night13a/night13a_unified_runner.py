#!/usr/bin/env python3
"""Night-13A common CLI for real P0, reload, and the simple baseline lane."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import torch

from SpaLORA.night13a_runner import (
    SEED,
    atomic_json,
    atomic_npz,
    evaluate_embedding,
    file_sha256,
    load_h5ad_pair,
    load_p5,
    load_spots,
    reload_zero_step,
    run_zero_step,
    simple_embedding,
    text_sha256,
)


def _labels(args, ids):
    if args.label_kind == "obs":
        source = ad.read_h5ad(args.rna)
        values = source.obs.loc[list(ids), args.label_column]
    elif args.label_kind == "csv":
        frame = pd.read_csv(args.label_source, index_col=0)
        if args.label_column not in frame.columns:
            if frame.shape[1] != 1:
                raise ValueError("label column not present and CSV is not single-column")
            values = frame.iloc[:, 0]
        else:
            values = frame[args.label_column]
        values.index = [args.label_id_prefix + str(value) for value in values.index]
        missing = [value for value in ids if value not in values.index]
        if missing:
            raise ValueError(f"canonical label missing {len(missing)} ordered IDs")
        values = values.loc[list(ids)]
    else:
        raise ValueError("explicit obs or csv label source required")
    if values.isna().any():
        raise ValueError("canonical public annotation contains missing values")
    return values.astype(str).tolist()


def p0(args):
    if args.kind == "spots":
        payload = load_spots(args.matrix, args.coordinates)
    elif args.kind == "p5":
        payload = load_p5(args.rna, args.coordinates, args.fragments, args.gtf)
    else:
        raise ValueError("unknown real P0 kind")
    audit = run_zero_step(payload, args.artifact_dir)
    print(json.dumps(audit, sort_keys=True))


def reload(args):
    audit = reload_zero_step(args.artifact_dir)
    print(json.dumps(audit, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(2)


def simple(args):
    started = time.monotonic()
    args.artifact_dir.mkdir(parents=True, exist_ok=False)
    registered_ids = None
    if args.label_kind == "csv":
        registered_ids = [args.label_id_prefix + str(value) for value in
                          pd.read_csv(args.label_source, index_col=0).index]
    elif args.label_kind == "obs":
        label_frame = ad.read_h5ad(args.rna, backed="r")
        registered_ids = list(map(str, label_frame.obs_names[
            label_frame.obs[args.label_column].notna()
        ]))
        label_frame.file.close()
    payload = load_h5ad_pair(args.rna, args.other, args.adapter, registered_ids)
    embedding = simple_embedding(payload)
    labels = _labels(args, payload["ids"])
    k = len(set(labels))
    metrics = evaluate_embedding(
        embedding, labels, payload["coordinates"], payload["ids"], k
    )
    atomic_npz(
        args.artifact_dir / "embedding_partition.npz",
        embedding=embedding,
        partition=metrics.pop("partition"),
        ordered_ids=np.asarray(payload["ids"], dtype=str),
    )
    result = {
        "status": "PASS", "method": "simple_standardized_concatenation",
        "endpoint": "COMMON_KMEANS_SEED0", "seed": SEED, "k": k,
        "evaluated_observations": len(payload["ids"]),
        "raw_shapes": payload["raw_shapes"],
        "processed_shapes": [list(payload["view1"].shape),
                             list(payload["view2"].shape)],
        "ordered_id_sha256": text_sha256(payload["ids"]),
        "source_sha256": payload["source_sha256"],
        "artifact_sha256": file_sha256(args.artifact_dir / "embedding_partition.npz"),
        "wall_seconds": time.monotonic() - started,
        "gpu_seconds": 0.0,
        "peak_gpu_mib": (torch.cuda.max_memory_allocated() / 1048576.0
                         if torch.cuda.is_available() else 0.0),
        "peak_rss_mib": psutil.Process().memory_info().rss / 1048576.0,
        **metrics,
    }
    atomic_json(args.artifact_dir / "result.json", result)
    print(json.dumps(result, sort_keys=True))


def parser():
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    q = commands.add_parser("p0")
    q.add_argument("--kind", choices=["spots", "p5"], required=True)
    q.add_argument("--artifact-dir", type=Path, required=True)
    q.add_argument("--matrix", type=Path)
    q.add_argument("--rna", type=Path)
    q.add_argument("--coordinates", type=Path, required=True)
    q.add_argument("--fragments", type=Path)
    q.add_argument("--gtf", type=Path)
    q.set_defaults(function=p0)
    q = commands.add_parser("reload")
    q.add_argument("--artifact-dir", type=Path, required=True)
    q.set_defaults(function=reload)
    q = commands.add_parser("simple")
    q.add_argument("--rna", type=Path, required=True)
    q.add_argument("--other", type=Path, required=True)
    q.add_argument("--adapter", choices=["protein", "atac"], required=True)
    q.add_argument("--label-kind", choices=["obs", "csv"], required=True)
    q.add_argument("--label-source", type=Path)
    q.add_argument("--label-column", required=True)
    q.add_argument("--label-id-prefix", default="")
    q.add_argument("--artifact-dir", type=Path, required=True)
    q.set_defaults(function=simple)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
