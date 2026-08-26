"""Label-closed official-math producer for one Night-21C embedding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night21c_official_math import (
    OfficialMathConfig,
    UPSTREAM_COMMIT,
    common_kmeans,
    reload_official_embedding,
    sha256_array,
    train_official_math,
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_graph(archive: np.lib.npyio.NpzFile) -> sp.csr_matrix:
    return sp.csr_matrix(
        (archive["graph0__data"], archive["graph0__indices"], archive["graph0__indptr"]),
        shape=tuple(int(x) for x in archive["graph0__shape"]),
    )


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--arm", choices=["OFFICIAL_FULL", "OFFICIAL_AE_ONLY", "OFFICIAL_GRAPH_ONLY"], required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--endpoint-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.time()
    carrier_path, profile_path, output = Path(args.carrier), Path(args.profile), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    config_payload = dict(profile_payload["config"])
    if args.arm == "OFFICIAL_AE_ONLY":
        config_payload["sigma"] = 0.0
    elif args.arm == "OFFICIAL_GRAPH_ONLY":
        config_payload["sigma"] = 1.0
    config = OfficialMathConfig(**config_payload)

    with np.load(carrier_path, allow_pickle=False) as archive:
        discovered = sorted(archive.files)
        accessed = [
            "ids", "view1", "view2", "graph0__data", "graph0__indices", "graph0__indptr", "graph0__shape"
        ]
        if any(token in name.lower() for name in discovered for token in ("label", "truth", "annot")):
            raise RuntimeError("annotation-like carrier key present")
        ids = np.asarray(archive["ids"])
        view1 = np.asarray(archive["view1"], dtype=np.float32)
        view2 = np.asarray(archive["view2"], dtype=np.float32)
        graph = load_graph(archive)
    if not (len(ids) == len(view1) == len(view2) == graph.shape[0] == graph.shape[1]):
        raise RuntimeError("ordered carrier contract failed")

    representation, state, diagnostics = train_official_math(
        view1, view2, graph, args.k, config, args.snapshot_root, args.training_seed, args.device
    )
    checkpoint = output.with_suffix(".pt")
    torch.save(
        {
            "schema": "night21c-official-math-checkpoint-v1",
            "upstream_commit": UPSTREAM_COMMIT,
            "config": config.to_dict(),
            "arm": args.arm,
            "training_seed": args.training_seed,
            "state_dict": state,
        },
        checkpoint,
    )
    payload = safe_load(checkpoint)
    reloaded = reload_official_embedding(
        view1, view2, graph, args.k, config, args.snapshot_root, args.training_seed,
        payload["state_dict"], device=args.device,
    )
    max_abs = float(np.max(np.abs(representation - reloaded)))
    if max_abs > 2e-5:
        raise RuntimeError(f"same-process strict reload mismatch: {max_abs}")
    partition = common_kmeans(representation, args.k, args.endpoint_seed, n_init=20)
    np.savez_compressed(output, ids=ids, representation=representation, partition=partition)
    with np.load(output, allow_pickle=False) as saved:
        if not np.array_equal(saved["ids"], ids):
            raise RuntimeError("artifact ID reload failed")
        if not np.array_equal(saved["representation"], representation):
            raise RuntimeError("artifact representation reload failed")
        if not np.array_equal(saved["partition"], partition):
            raise RuntimeError("artifact partition reload failed")

    source_root = Path(args.snapshot_root)
    source_files = [
        source_root / "LICENSE",
        source_root / "MGCN-main/model/AE.py",
        source_root / "MGCN-main/model/IGAE.py",
        source_root / "MGCN-main/model/spaMGCN.py",
        source_root / "MGCN-main/train/train3.py",
    ]
    manifest = {
        "schema": "night21c-official-math-producer-v1",
        "lane": args.lane,
        "k": args.k,
        "arm": args.arm,
        "profile_authority": profile_payload,
        "config": config.to_dict(),
        "training_seed": args.training_seed,
        "endpoint_seed": args.endpoint_seed,
        "carrier_path": str(carrier_path),
        "carrier_sha256": file_sha(carrier_path),
        "discovered_carrier_keys": discovered,
        "accessed_numeric_keys": accessed,
        "annotation_arrays_accessed": [],
        "ground_truth_label_values_read": 0,
        "ordered_ids_sha256": sha256_array(ids),
        "representation_sha256": sha256_array(representation),
        "partition_sha256": sha256_array(partition),
        "artifact_sha256": file_sha(output),
        "checkpoint_sha256": file_sha(checkpoint),
        "same_process_reload_max_abs": max_abs,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_sha256": {str(path.relative_to(source_root)): file_sha(path) for path in source_files},
        "compatibility_scope": [
            "sanitized reduced feature inputs instead of repository-specific raw files",
            "registered graph support converted with upstream binary symmetrize plus D^-1/2 normalization",
            "fixed final epoch with no label monitoring",
            "encoder-only exact subpath used for final embedding and replay",
        ],
        "dense_objective_changed": False,
        "diagnostics": diagnostics,
        "wall_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
