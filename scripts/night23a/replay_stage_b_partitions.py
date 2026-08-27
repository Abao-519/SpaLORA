"""Fresh-process exact replay of every locked Stage-B partition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from SpaLORA.night23a_xbed import (
    STAGE_B_ARMS,
    array_sha,
    file_sha,
    spectral_exact_k_partition,
    stage_b_arm_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if file_sha(args.bank) != manifest["bank_sha256"]:
        raise RuntimeError("partition bank SHA mismatch")
    with np.load(args.features, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        rows = np.asarray(archive["rows"], dtype=np.int32)
        cols = np.asarray(archive["cols"], dtype=np.int32)
        ids = np.asarray(archive["ids"]).astype("U")
    with np.load(args.prediction, allow_pickle=False) as archive:
        prediction = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(args.bank, allow_pickle=False) as archive:
        stored_ids = np.asarray(archive["ids"]).astype("U")
        candidate_ids = np.asarray(archive["candidate_ids"]).astype("U")
        stored = np.asarray(archive["partitions"], dtype=np.int32)
    if not np.array_equal(ids, stored_ids) or candidate_ids.tolist() != list(STAGE_B_ARMS):
        raise RuntimeError("fresh replay authority mismatch")
    arm_weights = stage_b_arm_weights(features, prediction)
    replayed = []
    for index, arm in enumerate(STAGE_B_ARMS):
        partition = spectral_exact_k_partition(len(ids), rows, cols, arm_weights[arm], args.k, seed=23)
        if not np.array_equal(partition, stored[index]):
            raise RuntimeError(f"fresh partition replay mismatch: {arm}")
        replayed.append({"candidate_id": arm, "partition_sha256": array_sha(partition)})
    output = {
        "schema": "night23a-stage-b-fresh-replay-v1",
        "status": "PASS",
        "lane": manifest["lane"],
        "candidate_count": len(replayed),
        "replayed": replayed,
        "bank_sha256": file_sha(args.bank),
        "labels_read": 0,
        "teacher_files_read": 0,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "PASS", "lane": manifest["lane"], "candidate_count": len(replayed)}, indent=2))


if __name__ == "__main__":
    main()
