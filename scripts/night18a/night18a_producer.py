#!/usr/bin/env python3
"""Train one locked Night-18A backbone artifact and decode the same embedding."""

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

from SpaLORA.night18a_backbone import (
    BackboneConfig, decode_embedding, reload_representation, row_normalize,
    sha256_array, standardize, train_backbone,
)


CONFIGS = {
    "C01_FULL": BackboneConfig("C01_FULL", hidden_dim=96, residual_scale=0.05, steps=80,
                                anchor_weight=0.10, use_graph=True),
    "C02_NO_GRAPH": BackboneConfig("C02_NO_GRAPH", hidden_dim=96, residual_scale=0.05, steps=80,
                                    anchor_weight=0.10, use_graph=False, spatial_smooth_weight=0.0),
    "C03_NO_ANCHOR": BackboneConfig("C03_NO_ANCHOR", hidden_dim=96, residual_scale=0.05, steps=80,
                                     anchor_weight=0.0, use_graph=True),
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csr(archive: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix((archive[f"{prefix}__data"], archive[f"{prefix}__indices"], archive[f"{prefix}__indptr"]),
                         shape=tuple(int(x) for x in archive[f"{prefix}__shape"]))


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    carrier_path = Path(args.carrier)
    with np.load(carrier_path, allow_pickle=False) as archive:
        allowed = {"ids", "view1", "view2", "retained"} | {
            f"graph{index}__{suffix}" for index in range(3)
            for suffix in ("data", "indices", "indptr", "shape")
        }
        ids = np.asarray(archive["ids"]).astype("U")
        view1 = np.asarray(archive["view1"], dtype=np.float32)
        view2 = np.asarray(archive["view2"], dtype=np.float32)
        retained = np.asarray(archive["retained"], dtype=np.float32)
        graphs = [csr(archive, f"graph{i}") for i in range(3)]
    if len(set(ids.tolist())) != len(ids) or not (len(ids) == len(view1) == len(view2) == len(retained)):
        raise ValueError("carrier ordered-ID or shape contract failed")
    config = CONFIGS.get(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        if args.config == "FROZEN_RETAINED":
            representation = row_normalize(standardize(retained))
            diagnostics = {"optimizer_steps": 0, "parameter_changed": False,
                           "representation_sha256": sha256_array(representation),
                           "view1_shape": list(view1.shape), "view2_shape": list(view2.shape),
                           "retained_shape": list(retained.shape), "spatial_graph_nnz": int(graphs[0].nnz)}
            checkpoint = {}
            config_dict = {"config_id": "FROZEN_RETAINED", "residual_scale": 0.0}
        else:
            assert config is not None
            training_representation, checkpoint, diagnostics = train_backbone(view1, view2, retained, graphs[0], args.k,
                                                                               config, args.seed, device)
            # Canonical artifacts are produced by CPU inference from the frozen
            # checkpoint.  This separates GPU training from the exact portable
            # representation consumer and makes fresh-process replay byte exact.
            representation = reload_representation(view1, view2, retained, graphs[0], args.k, config,
                                                   checkpoint, "cpu")
            diagnostics["gpu_to_canonical_cpu_max_abs"] = float(np.max(np.abs(training_representation - representation)))
            diagnostics["canonical_consumer_device"] = "cpu"
            diagnostics["representation_sha256"] = sha256_array(representation)
            checkpoint_path = output.with_suffix(".pt")
            torch.save({"config": config.to_dict(), "state_dict": checkpoint, "seed": args.seed}, checkpoint_path)
            try:
                payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            except TypeError:
                payload = torch.load(checkpoint_path, map_location=device)
            replay = reload_representation(view1, view2, retained, graphs[0], args.k, config,
                                           payload["state_dict"], "cpu")
            if sha256_array(replay) != sha256_array(representation) or not np.array_equal(replay, representation):
                raise RuntimeError("strict checkpoint representation replay mismatch")
            diagnostics["checkpoint_sha256"] = file_sha(checkpoint_path)
            diagnostics["strict_reload_exact"] = True
            config_dict = config.to_dict()
        partitions, records, selections = decode_embedding(representation, view1, view2, graphs, args.k)
    np.savez_compressed(output, ids=ids, representation=representation, partitions=partitions,
                        candidate_ids=np.asarray([row["candidate_id"] for row in records]),
                        selection_names=np.asarray(list(selections)),
                        selection_indices=np.asarray(list(selections.values()), dtype=np.int32))
    with np.load(output, allow_pickle=False) as replay:
        if not np.array_equal(replay["representation"], representation) or not np.array_equal(replay["partitions"], partitions):
            raise RuntimeError("artifact reload mismatch")
    manifest = {
        "schema": "night18a-backbone-producer-v1", "lane": args.lane, "k": args.k, "n": len(ids),
        "config": config_dict, "training_seed": args.seed, "device": device,
        "carrier_sha256": file_sha(carrier_path), "ordered_ids_sha256": sha256_array(ids),
        "representation_sha256": sha256_array(representation), "candidate_count": len(records),
        "candidate_records": records, "selections": {
            name: {"index": int(index), "candidate_id": records[index]["candidate_id"],
                   "partition_sha256": records[index]["candidate_sha256"]}
            for name, index in selections.items()
        },
        "diagnostics": diagnostics, "producer_annotation_reads": 0,
        "producer_allowed_numeric_keys": sorted(allowed), "dense_observation_by_observation_count": 0,
        "artifact_reload": "PASS", "artifact_sha256": file_sha(output),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_peak_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--config", choices=[*CONFIGS, "FROZEN_RETAINED"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
