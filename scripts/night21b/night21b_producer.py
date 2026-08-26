"""Label-closed producer for one Night-21B arm."""
from __future__ import annotations

import argparse, hashlib, json, os, resource, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night21b_msrd import (
    ARM_NAMES, MSRDConfig, common_kmeans_endpoint, reload_msrd,
    robust_standardize, sha256_array, train_msrd,
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_graph(archive: np.lib.npyio.NpzFile, prefix: str = "graph0") -> sp.csr_matrix:
    return sp.csr_matrix((archive[f"{prefix}__data"], archive[f"{prefix}__indices"], archive[f"{prefix}__indptr"]),
                         shape=tuple(int(x) for x in archive[f"{prefix}__shape"]))


def run(args: argparse.Namespace) -> None:
    started = time.time(); carrier_path = Path(args.carrier); output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(carrier_path, allow_pickle=False) as archive:
        discovered = sorted(archive.files)
        accessed = ["ids", "view1", "view2", "retained", "graph0__data", "graph0__indices", "graph0__indptr", "graph0__shape"]
        ids = np.asarray(archive["ids"]); view1 = np.asarray(archive["view1"], dtype=np.float32)
        view2 = np.asarray(archive["view2"], dtype=np.float32); retained = np.asarray(archive["retained"], dtype=np.float32)
        graph = load_graph(archive)
    if any(token in key.lower() for key in discovered for token in ("label", "truth", "annot")):
        raise RuntimeError("annotation-like carrier key present; sanitized carrier required")
    if not (len(ids) == len(view1) == len(view2) == len(retained) == graph.shape[0]):
        raise RuntimeError("ordered carrier contract failed")
    config = MSRDConfig(config_id=args.config_id, family=args.family, steps=args.steps,
                        hidden_dim=args.hidden_dim, learning_rate=args.learning_rate,
                        point_anchor_weight=args.point_anchor_weight,
                        positive_relation_weight=args.positive_relation_weight,
                        boundary_relation_weight=args.boundary_relation_weight)
    checkpoint = None
    if args.arm == "T0_STRONG_CARRIER":
        representation = robust_standardize(retained)
        state = None
        diagnostics = {"optimizer_steps": 0, "parameter_changed": False, "parameter_count": 0,
                       "representation_sha256": sha256_array(representation),
                       "input_shapes": {"view1": list(view1.shape), "view2": list(view2.shape), "retained": list(retained.shape)},
                       "graph_shape": list(graph.shape), "graph_nnz": int(graph.nnz), "dense_nxn_allocated": False}
    else:
        representation, state, diagnostics = train_msrd(view1, view2, retained, graph, config, args.arm,
                                                        args.training_seed, args.device)
        checkpoint = output.with_suffix(".pt")
        torch.save({"config": config.to_dict(), "arm": args.arm, "training_seed": args.training_seed,
                    "state_dict": state}, checkpoint)
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        reloaded = reload_msrd(view1, view2, retained, graph, config, args.training_seed,
                               payload["state_dict"], device="cpu")
        max_abs = float(np.max(np.abs(representation - reloaded)))
        diagnostics["same_process_cpu_reload_max_abs"] = max_abs
        if max_abs > 2e-5:
            raise RuntimeError(f"checkpoint reload mismatch: {max_abs}")
        diagnostics["checkpoint_sha256"] = file_sha(checkpoint)
    partition = common_kmeans_endpoint(representation, args.k, args.endpoint_seed)
    np.savez_compressed(output, ids=ids, representation=representation, partition=partition)
    with np.load(output, allow_pickle=False) as saved:
        if not (np.array_equal(saved["ids"], ids) and np.array_equal(saved["partition"], partition)
                and np.array_equal(saved["representation"], representation)):
            raise RuntimeError("artifact reload failed")
    manifest = {
        "schema": "night21b-producer-v1", "lane": args.lane, "family": args.family,
        "k": args.k, "arm": args.arm, "config": config.to_dict(),
        "training_seed": args.training_seed, "endpoint_seed": args.endpoint_seed,
        "carrier_path": str(carrier_path), "carrier_sha256": file_sha(carrier_path),
        "ordered_ids_sha256": sha256_array(ids), "partition_sha256": sha256_array(partition),
        "representation_sha256": sha256_array(representation), "artifact_sha256": file_sha(output),
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "discovered_carrier_keys": discovered, "accessed_numeric_keys": accessed,
        "annotation_arrays_accessed": [], "ground_truth_label_values_read": 0,
        "artifact_reload": "PASS", "diagnostics": diagnostics,
        "wall_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True); parser.add_argument("--lane", required=True)
    parser.add_argument("--family", required=True, choices=["RNA_PROTEIN", "RNA_CHROMATIN"])
    parser.add_argument("--k", type=int, required=True); parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument("--config-id", default="P0_COMMON_V1"); parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=96); parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--point-anchor-weight", type=float, default=0.20)
    parser.add_argument("--positive-relation-weight", type=float, default=0.25)
    parser.add_argument("--boundary-relation-weight", type=float, default=0.25)
    parser.add_argument("--training-seed", type=int, default=0); parser.add_argument("--endpoint-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()

