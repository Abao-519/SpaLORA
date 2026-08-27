"""Fresh-process byte-exact replay of one locked Night-23B Stage-A oracle bank."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from SpaLORA.night23a_xbed import file_sha


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--producer", required=True)
    p.add_argument("--lane", required=True)
    p.add_argument("--feature", required=True)
    p.add_argument("--feature-manifest", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--teacher-manifest", required=True)
    p.add_argument("--teacher-partition", required=True)
    p.add_argument("--carrier", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--authority-bank", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="night23b_replay_") as td:
        td = Path(td)
        replay_csv = td / "oracle.csv"
        replay_bank = td / "oracle.npz"
        command = [sys.executable, args.producer, "--lane", args.lane,
                   "--feature", args.feature, "--feature-manifest", args.feature_manifest,
                   "--teacher", args.teacher, "--teacher-manifest", args.teacher_manifest,
                   "--teacher-partition", args.teacher_partition, "--carrier", args.carrier,
                   "--contract", args.contract, "--output", str(replay_csv),
                   "--output-bank", str(replay_bank)]
        subprocess.run(command, check=True)
        with np.load(args.authority_bank, allow_pickle=False) as a, np.load(replay_bank, allow_pickle=False) as b:
            exact = all(np.array_equal(a[k], b[k]) for k in ("ids", "candidate_ids", "partitions"))
        result = {
            "schema": "night23b-oracle-fresh-process-replay-v1",
            "lane": args.lane,
            "fresh_process": True,
            "ids_candidate_ids_partitions_exact": bool(exact),
            "authority_bank_sha256": file_sha(args.authority_bank),
            "replay_bank_sha256": file_sha(replay_bank),
        }
        if not exact:
            raise RuntimeError("fresh-process oracle bank replay mismatch")
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
