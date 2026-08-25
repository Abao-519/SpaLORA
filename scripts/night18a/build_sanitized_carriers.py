#!/usr/bin/env python3
"""Build compact numeric-only carriers from existing authority assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> None:
    kit_path, retained_path, output = Path(args.kit), Path(args.retained), Path(args.output)
    with np.load(kit_path, allow_pickle=False) as kit, np.load(retained_path, allow_pickle=False) as retained:
        ids = kit["ids"].astype("U")
        retained_key = f"{args.prefix}__retained_embedding"
        numeric = {
            "ids": ids,
            "view1": np.asarray(kit["view1"], dtype=np.float32),
            "view2": np.asarray(kit["view2"], dtype=np.float32),
            "retained": np.asarray(retained[retained_key], dtype=np.float32),
        }
        for out_prefix, kit_prefix in (("graph0", "operator4"), ("graph1", "graph"), ("graph2", "operator18")):
            for suffix in ("data", "indices", "indptr", "shape"):
                numeric[f"{out_prefix}__{suffix}"] = kit[f"{kit_prefix}__{suffix}"]
    if not (len(numeric["ids"]) == len(numeric["view1"]) == len(numeric["view2"]) == len(numeric["retained"])):
        raise RuntimeError("sanitized carrier shape mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **numeric)
    with np.load(output, allow_pickle=False) as replay:
        if replay.files != list(numeric) or not all(np.array_equal(replay[key], value) for key, value in numeric.items()):
            raise RuntimeError("sanitized carrier byte-array replay failed")
    output.with_suffix(".json").write_text(json.dumps({
        "schema": "night18a-sanitized-numeric-carrier-v1", "lane": args.lane,
        "source_kit_sha256": sha(kit_path), "source_retained_sha256": sha(retained_path),
        "carrier_sha256": sha(output), "keys": list(numeric), "annotation_keys_copied": 0,
        "n": len(numeric["ids"]), "view1_shape": list(numeric["view1"].shape),
        "view2_shape": list(numeric["view2"].shape), "retained_shape": list(numeric["retained"].shape),
    }, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--kit", required=True); parser.add_argument("--retained", required=True)
    parser.add_argument("--prefix", required=True); parser.add_argument("--lane", required=True); parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
