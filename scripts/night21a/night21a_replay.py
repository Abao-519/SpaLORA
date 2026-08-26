#!/usr/bin/env python3
"""Fresh-process replay for a locked Night-21A artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night21a_amcf import common_kmeans_endpoint, make_config, reload_amcf, row_normalize, sha256_array, standardize


def graph(archive, prefix):
    return sp.csr_matrix((archive[f"{prefix}__data"], archive[f"{prefix}__indices"], archive[f"{prefix}__indptr"]), shape=tuple(archive[f"{prefix}__shape"]))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--carrier", required=True); parser.add_argument("--artifact", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint"); parser.add_argument("--output", required=True); args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    with np.load(args.carrier, allow_pickle=False) as archive:
        ids = archive["ids"].astype("U"); view1 = archive["view1"].astype(np.float32); view2 = archive["view2"].astype(np.float32); retained = archive["retained"].astype(np.float32)
        graphs = [graph(archive, f"graph{i}") for i in range(3)]
    if manifest["arm"] in {"CARRIER_ONLY", "LOWPASS_CARRIER_CONTROL"}:
        representation = row_normalize(standardize(retained))
        if manifest["arm"] == "LOWPASS_CARRIER_CONTROL":
            operator = graphs[0].astype(np.float64).maximum(graphs[0].astype(np.float64).T).tocsr()
            operator.setdiag(0); operator.eliminate_zeros(); degree = np.asarray(operator.sum(1)).ravel(); inverse = np.zeros_like(degree); inverse[degree > 0] = 1.0 / degree[degree > 0]
            representation = (0.80 * representation + 0.20 * ((sp.diags(inverse) @ operator) @ representation)).astype(np.float32)
    else:
        config = make_config(manifest["arm"], manifest["profile"], int(manifest["config"]["steps"]))
        try: payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError: payload = torch.load(args.checkpoint, map_location="cpu")
        representation = reload_amcf(view1, view2, retained, graphs, manifest["k"], config, manifest["training_seed"], payload["state_dict"], "cpu")
    partition = common_kmeans_endpoint(representation, manifest["k"], manifest["endpoint_seed"])
    with np.load(args.artifact, allow_pickle=False) as artifact:
        exact = np.array_equal(artifact["ids"], ids) and np.array_equal(artifact["representation"], representation) and np.array_equal(artifact["partition"], partition)
    if not exact or sha256_array(representation) != manifest["representation_sha256"] or sha256_array(partition) != manifest["partition_sha256"]:
        raise RuntimeError("fresh-process replay mismatch")
    Path(args.output).write_text(json.dumps({"status": "PASS", "byte_exact": True, "representation_sha256": sha256_array(representation), "partition_sha256": sha256_array(partition)}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__": main()
