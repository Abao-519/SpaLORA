#!/usr/bin/env python3
"""Label-closed producer for one Night-21A AMCF arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
import torch
from threadpoolctl import threadpool_limits

import SpaLORA.night21a_amcf as amcf_module
from SpaLORA.night21a_amcf import common_kmeans_endpoint, make_config, reload_amcf, row_normalize, sha256_array, standardize, train_amcf


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph(archive: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix((archive[f"{prefix}__data"], archive[f"{prefix}__indices"], archive[f"{prefix}__indptr"]),
                         shape=tuple(int(x) for x in archive[f"{prefix}__shape"]))


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    carrier_path = Path(args.carrier)
    with np.load(carrier_path, allow_pickle=False) as archive:
        discovered_keys = list(archive.files)
        ids = np.asarray(archive["ids"]).astype("U")
        view1 = np.asarray(archive["view1"], dtype=np.float32)
        view2 = np.asarray(archive["view2"], dtype=np.float32)
        retained = np.asarray(archive["retained"], dtype=np.float32)
        graphs = [graph(archive, f"graph{index}") for index in range(3)]
    accessed = ["ids", "view1", "view2", "retained"] + [f"graph{i}__{s}" for i in range(3) for s in ("data", "indices", "indptr", "shape")]
    if len(set(ids.tolist())) != len(ids) or not (len(ids) == len(view1) == len(view2) == len(retained)):
        raise RuntimeError("carrier ordered ID/shape contract failed")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with threadpool_limits(1):
        if args.arm in {"CARRIER_ONLY", "LOWPASS_CARRIER_CONTROL"}:
            representation = row_normalize(standardize(retained))
            if args.arm == "LOWPASS_CARRIER_CONTROL":
                operator = graphs[0].astype(np.float64).maximum(graphs[0].astype(np.float64).T).tocsr()
                operator.setdiag(0); operator.eliminate_zeros()
                degree = np.asarray(operator.sum(1)).ravel(); inverse = np.zeros_like(degree); inverse[degree > 0] = 1.0 / degree[degree > 0]
                operator = sp.diags(inverse) @ operator
                representation = (0.80 * representation + 0.20 * (operator @ representation)).astype(np.float32)
            checkpoint = None
            config_dict = {"config_id": args.arm, "arm": args.arm, "steps": 0}
            diagnostics = {"parameter_changed": False, "optimizer_steps": 0, "identity_initial_max_abs": 0.0,
                           "representation_sha256": sha256_array(representation), "view1_shape": list(view1.shape),
                           "view2_shape": list(view2.shape), "retained_shape": list(retained.shape),
                           "graph_nnz": [int(value.nnz) for value in graphs]}
        else:
            config = make_config(args.arm, args.profile, args.steps)
            trained, state, diagnostics = train_amcf(view1, view2, retained, graphs, args.k, config, args.training_seed, device)
            representation = reload_amcf(view1, view2, retained, graphs, args.k, config, args.training_seed, state, "cpu")
            diagnostics["gpu_to_cpu_max_abs"] = float(np.max(np.abs(trained - representation)))
            diagnostics["representation_sha256"] = sha256_array(representation)
            checkpoint = output.with_suffix(".pt")
            torch.save({"config": config.to_dict(), "training_seed": args.training_seed, "state_dict": state}, checkpoint)
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
            replay = reload_amcf(view1, view2, retained, graphs, args.k, config, args.training_seed, payload["state_dict"], "cpu")
            if not np.array_equal(representation, replay):
                raise RuntimeError("same-process strict checkpoint replay mismatch")
            diagnostics["strict_reload_exact"] = True
            diagnostics["checkpoint_sha256"] = file_sha(checkpoint)
            config_dict = config.to_dict()
        partition = common_kmeans_endpoint(representation, args.k, args.endpoint_seed)
    np.savez_compressed(output, ids=ids, representation=representation, partition=partition)
    with np.load(output, allow_pickle=False) as replay:
        if not np.array_equal(replay["ids"], ids) or not np.array_equal(replay["representation"], representation) or not np.array_equal(replay["partition"], partition):
            raise RuntimeError("artifact reload mismatch")
    sizes = np.bincount(partition, minlength=args.k)
    manifest = {
        "schema": "night21a-amcf-producer-v1", "status": "PARTITION_LOCKED_BEFORE_EVALUATION",
        "lane": args.lane, "family": args.family, "k": args.k, "n": len(ids), "arm": args.arm,
        "profile": args.profile, "training_seed": args.training_seed, "endpoint_seed": args.endpoint_seed,
        "config": config_dict, "carrier_sha256": file_sha(carrier_path), "ordered_ids_sha256": sha256_array(ids),
        "core_source_sha256": file_sha(Path(amcf_module.__file__)), "producer_source_sha256": file_sha(Path(__file__)),
        "representation_sha256": sha256_array(representation), "partition_sha256": sha256_array(partition),
        "cluster_sizes": [int(x) for x in sizes], "min_cluster_size": int(sizes.min()),
        "discovered_carrier_keys": discovered_keys, "accessed_numeric_keys": accessed,
        "annotation_keys_accessed": 0, "dense_n_by_n_allocations": 0, "diagnostics": diagnostics,
        "artifact_sha256": file_sha(output), "artifact_reload": "PASS",
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True); parser.add_argument("--lane", required=True); parser.add_argument("--family", required=True)
    parser.add_argument("--k", required=True, type=int); parser.add_argument("--arm", required=True, choices=["CARRIER_ONLY", "LOWPASS_CARRIER_CONTROL", *sorted(__import__('SpaLORA.night21a_amcf', fromlist=['ARM_CONFIG']).ARM_CONFIG)])
    parser.add_argument("--profile", default="FAMILY_BASE_V1"); parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--training-seed", type=int, default=0); parser.add_argument("--endpoint-seed", type=int, default=0)
    parser.add_argument("--output", required=True); run(parser.parse_args())


if __name__ == "__main__": main()
