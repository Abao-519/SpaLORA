"""Fresh-process exact replay for a locked Night-21C official embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night21c_official_math import OfficialMathConfig, common_kmeans, reload_official_embedding, sha256_array


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.artifact, allow_pickle=False) as z:
        ids = np.asarray(z["ids"])
        expected_representation = np.asarray(z["representation"])
        expected_partition = np.asarray(z["partition"])
    with np.load(args.carrier, allow_pickle=False) as z:
        if any(token in name.lower() for name in z.files for token in ("label", "truth", "annot")):
            raise RuntimeError("annotation-like carrier key present")
        carrier_ids = np.asarray(z["ids"])
        view1 = np.asarray(z["view1"], dtype=np.float32)
        view2 = np.asarray(z["view2"], dtype=np.float32)
        graph = sp.csr_matrix(
            (z["graph0__data"], z["graph0__indices"], z["graph0__indptr"]),
            shape=tuple(int(x) for x in z["graph0__shape"]),
        )
    if not np.array_equal(ids, carrier_ids):
        raise RuntimeError("ordered ID mismatch")
    payload = safe_load(Path(args.checkpoint))
    config = OfficialMathConfig(**payload["config"])
    representation = reload_official_embedding(
        view1, view2, graph, len(np.unique(expected_partition)), config, args.snapshot_root,
        int(payload["training_seed"]), payload["state_dict"], args.device,
    )
    partition = common_kmeans(representation, len(np.unique(expected_partition)), 0, n_init=20)
    max_abs = float(np.max(np.abs(representation - expected_representation)))
    result = {
        "schema": "night21c-official-fresh-replay-v1",
        "ids_exact": True,
        "representation_max_abs": max_abs,
        "representation_exact": bool(np.array_equal(representation, expected_representation)),
        "representation_close": bool(np.allclose(representation, expected_representation, rtol=2e-5, atol=2e-5)),
        "partition_exact": bool(np.array_equal(partition, expected_partition)),
        "representation_sha256": sha256_array(representation),
        "partition_sha256": sha256_array(partition),
    }
    if not (result["representation_close"] and result["partition_exact"]):
        raise RuntimeError(result)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
