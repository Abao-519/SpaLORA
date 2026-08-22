#!/usr/bin/env python3
"""Fresh-process verifier for a Night-11B real smoke artifact."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))
from SpaLORA.night11b_discordance import array_sha, atomic_json, file_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = Path(manifest["artifact"]).resolve()
    if file_sha(artifact) != manifest["artifact_sha256"]:
        raise RuntimeError("fresh reload artifact SHA mismatch")
    checks = {}
    with np.load(str(artifact), allow_pickle=False) as payload:
        if set(payload.files) != set(manifest["arrays"]):
            raise RuntimeError("fresh reload array set mismatch")
        for name, expected in manifest["arrays"].items():
            value = payload[name]
            row = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_sha(value)}
            if row["shape"] != expected["shape"] or row["dtype"] != expected["dtype"] or row["sha256"] != expected["sha256"]:
                raise RuntimeError("fresh reload array mismatch: %s" % name)
            row["pass"] = True
            checks[name] = row
    required = {"ordered_ids_utf8", "feature_ids_utf8", "coordinates", "folds", "shared_prediction", "residual", "combined"}
    if set(checks) != required:
        raise RuntimeError("fresh reload required array contract mismatch")
    audit = {
        "schema": "spalora.night11b.fresh_reload.v1", "status": "PASS",
        "manifest": str(manifest_path), "artifact": str(artifact),
        "artifact_sha256": manifest["artifact_sha256"], "checks": checks,
        "shape_dtype_id_byte_exact": True, "label_values_opened": False,
        "process_id": os.getpid(),
    }
    atomic_json(manifest_path.parent / "fresh_reload_audit.json", audit)
    print(json.dumps({"event": "fresh_reload_pass", "dataset": manifest["dataset"]}, sort_keys=True))


if __name__ == "__main__":
    main()
