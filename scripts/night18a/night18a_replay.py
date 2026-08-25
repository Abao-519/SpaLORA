#!/usr/bin/env python3
"""Fresh-process exact replay of locked Night-18A representation and partitions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
import torch
from threadpoolctl import threadpool_limits

from SpaLORA.night18a_backbone import BackboneConfig, decode_embedding, reload_representation, row_normalize, sha256_array, standardize


def csr(archive: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix((archive[f"{prefix}__data"], archive[f"{prefix}__indices"], archive[f"{prefix}__indptr"]),
                         shape=tuple(int(x) for x in archive[f"{prefix}__shape"]))


def run(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    with np.load(args.carrier, allow_pickle=False) as carrier:
        ids = carrier["ids"].astype("U"); view1 = carrier["view1"].astype(np.float32); view2 = carrier["view2"].astype(np.float32)
        retained = carrier["retained"].astype(np.float32); graphs = [csr(carrier, f"graph{i}") for i in range(3)]
    with np.load(args.artifact, allow_pickle=False) as artifact:
        expected_rep = artifact["representation"].astype(np.float32); expected_partitions = artifact["partitions"].astype(np.int32)
        expected_ids = artifact["ids"].astype("U")
    if not np.array_equal(ids, expected_ids): raise ValueError("replay ordered-ID mismatch")
    config_id = manifest["config"]["config_id"]
    with threadpool_limits(1):
        if config_id == "FROZEN_RETAINED":
            representation = row_normalize(standardize(retained))
        else:
            config = BackboneConfig(**manifest["config"])
            try: payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            except TypeError: payload = torch.load(args.checkpoint, map_location="cpu")
            representation = reload_representation(view1, view2, retained, graphs[0], args.k, config, payload["state_dict"], "cpu")
        partitions, _, _ = decode_embedding(representation, view1, view2, graphs, args.k)
    result = {"schema": "night18a-fresh-process-replay-v1", "lane": manifest["lane"], "config_id": config_id,
              "representation_exact": bool(np.array_equal(representation, expected_rep)),
              "representation_sha256": sha256_array(representation),
              "partition_exact": bool(np.array_equal(partitions, expected_partitions)),
              "partitions_sha256": sha256_array(partitions)}
    if not result["representation_exact"] or not result["partition_exact"]:
        raise RuntimeError("fresh-process replay mismatch")
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--carrier", required=True); parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest", required=True); parser.add_argument("--checkpoint"); parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True); run(parser.parse_args())


if __name__ == "__main__": main()
