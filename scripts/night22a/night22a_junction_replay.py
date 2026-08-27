"""Fresh-process strict replay for a locked Night-22A junction bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night22a_geometry import array_sha
from SpaLORA.night22a_junction import GeometryJunction, JunctionConfig, graph_bank, hard_partition, upper_edges


def load_graph(archive, prefix="graph0"):
    return sp.csr_matrix(
        (archive[prefix + "__data"], archive[prefix + "__indices"], archive[prefix + "__indptr"]),
        shape=tuple(int(x) for x in archive[prefix + "__shape"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--embedding-key", default="representation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bank_path = Path(args.bank)
    manifest = json.loads(bank_path.with_suffix(".json").read_text(encoding="utf-8"))
    checkpoint = torch.load(bank_path.with_suffix(".pt"), map_location="cpu", weights_only=False)
    with np.load(bank_path, allow_pickle=False) as bank:
        ids = np.asarray(bank["ids"])
        candidate_ids = np.asarray(bank["candidate_ids"]).astype(str)
        partitions = np.asarray(bank["partitions"], dtype=np.int32)
    with np.load(args.carrier, allow_pickle=False) as carrier:
        carrier_ids = np.asarray(carrier["ids"])
        view1 = np.asarray(carrier["view1"], dtype=np.float32)
        view2 = np.asarray(carrier["view2"], dtype=np.float32)
        spatial = load_graph(carrier)
    with np.load(args.embedding, allow_pickle=False) as embedding:
        embedding_ids = np.asarray(embedding["ids"])
        x = np.asarray(embedding[args.embedding_key], dtype=np.float32)
    if not np.array_equal(ids, carrier_ids) or not np.array_equal(ids, embedding_ids):
        raise RuntimeError("replay ID mismatch")
    graph_map = graph_bank(x, view1, view2, spatial, neighbors=int(checkpoint["registry"]["graph_neighbors"]))
    edges = [upper_edges(graph_map[name]) for name in checkpoint["graph_names"]]
    start = partitions[0]
    checked = []
    for index, candidate_id in enumerate(candidate_ids[1:], start=1):
        profile_id, arm, _ = candidate_id.split("__")
        profile = next(row for row in checkpoint["registry"]["profiles"] if row["profile_id"] == profile_id)
        model = GeometryJunction(x, start, manifest["k"], edges, JunctionConfig(**profile["config"]), arm)
        model.load_state_dict(checkpoint["states"][candidate_id], strict=True)
        observed, _, _ = hard_partition(model)
        if not np.array_equal(observed, partitions[index]):
            raise RuntimeError(f"partition replay mismatch: {candidate_id}")
        if array_sha(observed) != manifest["candidate_partition_sha256"][candidate_id]:
            raise RuntimeError(f"partition SHA mismatch: {candidate_id}")
        checked.append(candidate_id)
    Path(args.output).write_text(
        json.dumps({"status": "PASS", "checked": checked, "count": len(checked)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
