"""Create an annotation-free Night-22B adapter around a locked carrier."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from SpaLORA.night22a_geometry import array_sha


def file_sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def ordered_id_sha(ids: np.ndarray) -> str:
    return hashlib.sha256(b"\0".join(str(item).encode("utf-8") for item in ids)).hexdigest()


def night16h_partition_sha(partition: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(partition, dtype=np.int32))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    spec = contract["lanes"][args.lane]
    carrier_path = Path(spec["carrier"])
    carrier_manifest_path = carrier_path.with_suffix(".carrier.json")
    if file_sha(carrier_path) != spec["carrier_sha256"]:
        raise RuntimeError("carrier SHA differs from frozen contract")
    carrier_manifest = json.loads(carrier_manifest_path.read_text(encoding="utf-8"))
    if carrier_manifest["n"] != spec["n"] or carrier_manifest["k"] != spec["k"]:
        raise RuntimeError("carrier manifest N/K mismatch")
    if carrier_manifest["ordered_id_sha256"] != spec["ordered_ids_sha256"]:
        raise RuntimeError("carrier manifest ordered-ID SHA mismatch")

    allowed = {
        "ids", "view1", "view2", "retained", "start_ids", "start_partitions",
        "graph0__data", "graph0__indices", "graph0__indptr", "graph0__shape",
        "graph1__data", "graph1__indices", "graph1__indptr", "graph1__shape",
        "graph2__data", "graph2__indices", "graph2__indptr", "graph2__shape",
    }
    accessed = [
        "ids", "retained", "view1", "view2",
        "graph0__data", "graph0__indices", "graph0__indptr", "graph0__shape",
    ]
    with np.load(carrier_path, allow_pickle=False) as carrier:
        discovered = list(carrier.files)
        if set(discovered) - allowed:
            raise RuntimeError(f"unexpected carrier keys: {sorted(set(discovered) - allowed)}")
        if any(re.search(r"label|truth|annotation|cell.?type", key, re.I) for key in discovered):
            raise RuntimeError("annotation-like key discovered in numeric carrier")
        ids = np.asarray(carrier["ids"])
        retained = np.asarray(carrier["retained"], dtype=np.float32)
        view1 = np.asarray(carrier["view1"], dtype=np.float32)
        view2 = np.asarray(carrier["view2"], dtype=np.float32)
        graph0_shape = tuple(int(x) for x in carrier["graph0__shape"])
        graph0_nnz = int(len(carrier["graph0__data"]))
    if len(ids) != spec["n"] or retained.shape[0] != len(ids):
        raise RuntimeError("carrier observation count mismatch")
    if graph0_shape != (len(ids), len(ids)) or graph0_nnz <= 0:
        raise RuntimeError("registered graph shape/nnz invalid")
    if ordered_id_sha(ids) != spec["ordered_ids_sha256"]:
        raise RuntimeError("ordered-ID array SHA differs from frozen contract")
    if not all(np.isfinite(value).all() for value in (retained, view1, view2)):
        raise RuntimeError("non-finite numeric carrier view")

    night16h_path = Path(spec["night16h_partition"])
    with np.load(night16h_path, allow_pickle=False) as archive:
        night16h_keys = list(archive.files)
        night16h_ids = np.asarray(archive["ids"])
        night16h_partition = np.asarray(archive["partition"], dtype=np.int32)
        selected_candidate = str(np.asarray(archive["selected_candidate_id"]).item())
    if not np.array_equal(ids, night16h_ids):
        raise RuntimeError("Night-16H selected partition ordered IDs mismatch")
    if night16h_partition_sha(night16h_partition) != spec["night16h_partition_sha256"]:
        raise RuntimeError("Night-16H selected partition SHA mismatch")
    if np.unique(night16h_partition).size != spec["k"]:
        raise RuntimeError("Night-16H selected partition is not exact K")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / f"{args.lane}__RETAINED_CARRIER.npz"
    parent_path = output_dir / f"{args.lane}__TRANSFER_PARENT_AUTHORITY.npz"
    night16h_copy = output_dir / f"{args.lane}__NIGHT16H_FROZEN_SELECTOR.npz"
    np.savez_compressed(embedding_path, ids=ids, representation=retained)
    np.savez_compressed(parent_path, ids=ids)
    np.savez_compressed(
        night16h_copy,
        ids=ids,
        partition=night16h_partition,
        selected_candidate_id=np.asarray(selected_candidate),
    )
    with np.load(embedding_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["ids"], ids) or not np.array_equal(saved["representation"], retained):
            raise RuntimeError("embedding adapter reload mismatch")
    with np.load(night16h_copy, allow_pickle=False) as saved:
        if not np.array_equal(saved["partition"], night16h_partition):
            raise RuntimeError("Night-16H start adapter reload mismatch")

    parent_manifest = {
        "schema": "night22b-transfer-parent-authority-v1",
        "lane": args.lane,
        "k": spec["k"],
        "representation_source": contract["representation_source"],
        "carrier_path": str(carrier_path),
        "carrier_sha256": file_sha(carrier_path),
        "carrier_manifest_path": str(carrier_manifest_path),
        "carrier_manifest_sha256": file_sha(carrier_manifest_path),
        "embedding_path": str(embedding_path),
        "embedding_file_sha256": file_sha(embedding_path),
        "embedding_array_sha256": array_sha(retained),
        "ordered_ids_sha256": array_sha(ids),
        "authority_ordered_id_sha256": ordered_id_sha(ids),
        "night16h_partition_path": str(night16h_copy),
        "night16h_partition_file_sha256": file_sha(night16h_copy),
        "night16h_authority_partition_sha256": night16h_partition_sha(night16h_partition),
        "night22a_array_sha256_of_night16h_partition": array_sha(night16h_partition),
        "night16h_selected_candidate_id": selected_candidate,
        "view1_shape": list(view1.shape),
        "view2_shape": list(view2.shape),
        "retained_shape": list(retained.shape),
        "graph0_shape": list(graph0_shape),
        "graph0_nnz": graph0_nnz,
        "discovered_carrier_keys": discovered,
        "accessed_carrier_keys": accessed,
        "night16h_discovered_keys": night16h_keys,
        "annotation_arrays_accessed": 0,
        "labels_read": 0,
        "contract_path": str(contract_path),
        "contract_sha256": file_sha(contract_path),
    }
    parent_path.with_suffix(".json").write_text(
        json.dumps(parent_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(parent_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
