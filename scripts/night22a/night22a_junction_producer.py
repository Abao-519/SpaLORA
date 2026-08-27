"""Train and lock a Night-22A direct-partition junction bank without labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night22a_geometry import array_sha
from SpaLORA.night22a_junction import (
    GeometryJunction,
    JunctionConfig,
    graph_bank,
    hard_partition,
    upper_edges,
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_graph(archive, prefix: str = "graph0") -> sp.csr_matrix:
    return sp.csr_matrix(
        (archive[prefix + "__data"], archive[prefix + "__indices"], archive[prefix + "__indptr"]),
        shape=tuple(int(x) for x in archive[prefix + "__shape"]),
    )


def train_one(
    x: np.ndarray,
    start: np.ndarray,
    k: int,
    edges,
    config: JunctionConfig,
    arm: str,
    seed: int,
    device: torch.device,
):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = GeometryJunction(x, start, k, edges, config, arm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    ledger = []
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = model.loss(step)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite junction loss")
        loss.backward()
        finite_grad = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not finite_grad:
            raise RuntimeError("non-finite junction gradient")
        optimizer.step()
        if step in {0, 1, 5, 20, 60, 120, config.steps - 1}:
            ledger.append({"step": step, **{key: float(value) for key, value in diagnostics.items()}})
    partition, probabilities, repair_count = hard_partition(model)
    changed = int(np.sum(partition != start))
    parameter_delta = float(
        sum(
            torch.sum(torch.abs(value.detach().cpu() - start_state[name])).item()
            for name, value in model.state_dict().items()
            if torch.is_floating_point(value)
        )
    )
    if parameter_delta <= 0:
        raise RuntimeError("junction parameters did not update")
    final_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    diagnostics = {
        "loss_ledger": ledger,
        "changed_spots_vs_start": changed,
        "repair_count": int(repair_count),
        "min_cluster_size": int(np.bincount(partition, minlength=k).min()),
        "cluster_sizes": np.sort(np.bincount(partition, minlength=k)).astype(int).tolist(),
        "parameter_l1_change": parameter_delta,
        "probability_sha256": array_sha(probabilities.astype(np.float32)),
        "partition_sha256": array_sha(partition),
    }
    return partition, final_state, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--embedding-key", default="representation")
    parser.add_argument("--representation-source", required=True)
    parser.add_argument("--parent-bank", required=True)
    parser.add_argument("--start-bank", required=True)
    parser.add_argument("--start-candidate", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    torch.set_num_threads(1)
    started = time.time()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(".pt")
    manifest_path = output.with_suffix(".json")
    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not registry.get("frozen_before_junction_label_evaluation", False):
        raise RuntimeError("registry is not a pre-evaluation freeze")
    repository = Path(__file__).resolve().parents[2]
    core_source = repository / "SpaLORA" / "night22a_junction.py"
    expected_source = registry.get("source_sha256", {})
    if file_sha(core_source) != expected_source.get("core"):
        raise RuntimeError("active junction core differs from frozen source SHA")
    if file_sha(Path(__file__).resolve()) != expected_source.get("producer"):
        raise RuntimeError("active junction producer differs from frozen source SHA")

    carrier_path = Path(args.carrier)
    embedding_path = Path(args.embedding)
    parent_bank_path = Path(args.parent_bank)
    parent_manifest_path = parent_bank_path.with_suffix(".json")
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest["lane"] != args.lane or parent_manifest["representation_source"] != args.representation_source:
        raise RuntimeError("Night-21C parent bank mismatch")
    if file_sha(carrier_path) != parent_manifest["carrier_sha256"]:
        raise RuntimeError("carrier SHA mismatch")
    if file_sha(embedding_path) != parent_manifest["embedding_file_sha256"]:
        raise RuntimeError("embedding SHA mismatch")

    with np.load(carrier_path, allow_pickle=False) as carrier:
        discovered_carrier_keys = list(carrier.files)
        ids = np.asarray(carrier["ids"])
        view1 = np.asarray(carrier["view1"], dtype=np.float32)
        view2 = np.asarray(carrier["view2"], dtype=np.float32)
        spatial = load_graph(carrier, "graph0")
    with np.load(embedding_path, allow_pickle=False) as embedding:
        discovered_embedding_keys = list(embedding.files)
        embedding_ids = np.asarray(embedding["ids"])
        x = np.asarray(embedding[args.embedding_key], dtype=np.float32)
    if not np.array_equal(ids, embedding_ids) or array_sha(ids) != parent_manifest["ordered_ids_sha256"]:
        raise RuntimeError("ordered ID authority mismatch")
    if array_sha(x) != parent_manifest["embedding_array_sha256"]:
        raise RuntimeError("embedding array authority mismatch")

    start_bank_path = Path(args.start_bank)
    with np.load(start_bank_path, allow_pickle=False) as bank:
        bank_ids = np.asarray(bank["ids"])
        candidate_ids = np.asarray(bank["candidate_ids"]).astype(str)
        partitions = np.asarray(bank["partitions"], dtype=np.int32)
    if not np.array_equal(bank_ids, ids):
        raise RuntimeError("start bank ID mismatch")
    matches = np.flatnonzero(candidate_ids == args.start_candidate)
    if len(matches) != 1:
        raise RuntimeError("start candidate is not unique")
    start = partitions[int(matches[0])]
    if np.unique(start).size != args.k:
        raise RuntimeError("start does not have exact K")

    graph_neighbors = [int(value) for value in registry["graph_neighbors"]]
    if len(graph_neighbors) != 2 or not graph_neighbors[0] < graph_neighbors[1]:
        raise RuntimeError("frozen multiscale graph-neighbor contract is invalid")
    named_graphs = graph_bank(
        x,
        view1,
        view2,
        spatial,
        neighbors=graph_neighbors[0],
        secondary_neighbors=graph_neighbors[1],
    )
    graph_names = list(named_graphs)
    edge_bank = [upper_edges(named_graphs[name]) for name in graph_names]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_ids = ["INPUT_GEOMETRY_START"]
    output_partitions = [start.astype(np.int32)]
    checkpoint_payload = {}
    diagnostics = {
        "INPUT_GEOMETRY_START": {
            "partition_sha256": array_sha(start.astype(np.int32)),
            "changed_spots_vs_start": 0,
            "min_cluster_size": int(np.bincount(start, minlength=args.k).min()),
            "cluster_sizes": np.sort(np.bincount(start, minlength=args.k)).astype(int).tolist(),
        }
    }
    for profile in registry["profiles"]:
        config = JunctionConfig(**profile["config"])
        for arm in registry["arms"]:
            candidate_id = f"{profile['profile_id']}__{arm}__S{args.seed}"
            partition, state, row = train_one(x, start, args.k, edge_bank, config, arm, args.seed, device)
            output_ids.append(candidate_id)
            output_partitions.append(partition)
            checkpoint_payload[candidate_id] = state
            diagnostics[candidate_id] = row

    output_partitions_array = np.stack(output_partitions).astype(np.int32)
    np.savez_compressed(
        output,
        ids=ids,
        candidate_ids=np.asarray(output_ids, dtype="U96"),
        partitions=output_partitions_array,
    )
    torch.save(
        {
            "schema": "night22a-junction-checkpoint-bank-v1",
            "lane": args.lane,
            "k": args.k,
            "seed": args.seed,
            "representation_source": args.representation_source,
            "start_candidate": args.start_candidate,
            "graph_names": graph_names,
            "states": checkpoint_payload,
            "registry": registry,
        },
        checkpoint,
    )
    with np.load(output, allow_pickle=False) as saved:
        if not np.array_equal(saved["ids"], ids) or not np.array_equal(saved["partitions"], output_partitions_array):
            raise RuntimeError("fresh partition bank reload mismatch")
    saved_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if set(saved_checkpoint["states"]) != set(checkpoint_payload):
        raise RuntimeError("fresh checkpoint bank reload mismatch")

    manifest = {
        "schema": "night22a-junction-bank-v1",
        "lane": args.lane,
        "k": args.k,
        "seed": args.seed,
        "device": str(device),
        "representation_source": args.representation_source,
        "start_candidate": args.start_candidate,
        "start_partition_sha256": array_sha(start.astype(np.int32)),
        "ordered_ids_sha256": array_sha(ids),
        "carrier_path": str(carrier_path),
        "carrier_sha256": file_sha(carrier_path),
        "embedding_path": str(embedding_path),
        "embedding_file_sha256": file_sha(embedding_path),
        "embedding_array_sha256": array_sha(x),
        "parent_bank_path": str(parent_bank_path),
        "parent_bank_sha256": file_sha(parent_bank_path),
        "parent_manifest_sha256": file_sha(parent_manifest_path),
        "start_bank_path": str(start_bank_path),
        "start_bank_sha256": file_sha(start_bank_path),
        "registry_path": str(registry_path),
        "registry_sha256": file_sha(registry_path),
        "core_source_path": str(core_source),
        "core_source_sha256": file_sha(core_source),
        "producer_source_sha256": file_sha(Path(__file__).resolve()),
        "graph_names": graph_names,
        "graph_edges": {name: int(sp.triu(graph, k=1).nnz) for name, graph in named_graphs.items()},
        "candidate_ids": output_ids,
        "candidate_partition_sha256": {
            name: array_sha(partition) for name, partition in zip(output_ids, output_partitions_array)
        },
        "partition_bank_sha256": file_sha(output),
        "checkpoint_bank_sha256": file_sha(checkpoint),
        "candidate_diagnostics": diagnostics,
        "discovered_carrier_keys": discovered_carrier_keys,
        "accessed_carrier_keys": [
            "ids", "view1", "view2", "graph0__data", "graph0__indices", "graph0__indptr", "graph0__shape"
        ],
        "discovered_embedding_keys": discovered_embedding_keys,
        "accessed_embedding_keys": ["ids", args.embedding_key],
        "labels_read": 0,
        "wall_seconds": time.time() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
