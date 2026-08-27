"""Build the preregistered primary/sensitivity start bank without labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from SpaLORA.night22a_geometry import array_sha


def file_sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def night16h_partition_sha(partition: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(partition, dtype=np.int32))
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-bank", required=True)
    parser.add_argument("--night16h-start", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    spec = contract["lanes"][args.lane]
    geometry_path = Path(args.geometry_bank)
    geometry_manifest = json.loads(geometry_path.with_suffix(".json").read_text(encoding="utf-8"))
    if geometry_manifest["lane"] != args.lane:
        raise RuntimeError("geometry lane mismatch")
    with np.load(geometry_path, allow_pickle=False) as geometry:
        ids = np.asarray(geometry["ids"])
        candidate_ids = np.asarray(geometry["candidate_ids"]).astype(str)
        partitions = np.asarray(geometry["partitions"], dtype=np.int32)
    if len(candidate_ids) != len(set(candidate_ids.tolist())):
        raise RuntimeError("duplicate geometry candidate ID")

    requested = [contract["primary_start"], "GEOM_FEATURE_NCUT_K24"]
    aliases = geometry_manifest["geometry_diagnostics"].get("candidate_aliases", {})
    selected_ids = []
    selected_partitions = []
    missing_preregistered = []
    for candidate in requested:
        resolved = aliases.get(candidate, candidate)
        found = np.flatnonzero(candidate_ids == resolved)
        if len(found) != 1:
            if candidate == contract["primary_start"]:
                missing_preregistered.append(candidate)
                continue
            raise RuntimeError(f"required frozen geometry sensitivity unavailable: {candidate} -> {resolved}")
        partition = partitions[int(found[0])]
        if np.unique(partition).size != spec["k"]:
            raise RuntimeError(f"geometry start is not exact K: {candidate}")
        selected_ids.append(candidate)
        selected_partitions.append(partition)

    night16h_path = Path(args.night16h_start)
    with np.load(night16h_path, allow_pickle=False) as saved:
        other_ids = np.asarray(saved["ids"])
        night16h_partition = np.asarray(saved["partition"], dtype=np.int32)
        selected_candidate = str(np.asarray(saved["selected_candidate_id"]).item())
    if not np.array_equal(ids, other_ids):
        raise RuntimeError("Night-16H start ID mismatch")
    if night16h_partition_sha(night16h_partition) != spec["night16h_partition_sha256"]:
        raise RuntimeError("Night-16H start partition SHA mismatch")
    selected_ids.append("NIGHT16H_FROZEN_SELECTOR")
    selected_partitions.append(night16h_partition)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.stack(selected_partitions).astype(np.int32)
    np.savez_compressed(output, ids=ids, candidate_ids=np.asarray(selected_ids, dtype="U64"), partitions=stacked)
    with np.load(output, allow_pickle=False) as saved:
        if not np.array_equal(saved["ids"], ids) or not np.array_equal(saved["partitions"], stacked):
            raise RuntimeError("frozen start bank reload mismatch")
    manifest = {
        "schema": "night22b-frozen-start-bank-v1",
        "lane": args.lane,
        "k": spec["k"],
        "candidate_ids": selected_ids,
        "geometry_alias_resolution": {
            candidate: aliases.get(candidate, candidate) for candidate in requested
        },
        "missing_preregistered_starts": missing_preregistered,
        "primary_start_available": contract["primary_start"] in selected_ids,
        "candidate_partition_sha256": {
            name: array_sha(partition) for name, partition in zip(selected_ids, stacked)
        },
        "geometry_bank_path": str(geometry_path),
        "geometry_bank_sha256": file_sha(geometry_path),
        "geometry_manifest_sha256": file_sha(geometry_path.with_suffix(".json")),
        "night16h_start_path": str(night16h_path),
        "night16h_start_file_sha256": file_sha(night16h_path),
        "night16h_authority_partition_sha256": night16h_partition_sha(night16h_partition),
        "night16h_selected_candidate_id": selected_candidate,
        "ordered_ids_sha256": array_sha(ids),
        "start_bank_sha256": file_sha(output),
        "labels_read": 0,
        "selection_semantics": "mechanically preregistered primary and sensitivity starts; no label comparison",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
