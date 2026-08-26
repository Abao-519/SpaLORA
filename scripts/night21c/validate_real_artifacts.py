"""Fail-closed real-artifact validation after Night-21C formal runs."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

WORK=Path("/root/night21c_working")
replays=sorted((WORK/"replays").glob("*.replay.json"))
if len(replays)!=18: raise RuntimeError(f"expected 18 formal replays, got {len(replays)}")
for path in replays:
    data=json.loads(path.read_text())
    if not all(data[key] for key in ("ids_exact","representation_exact","partition_exact")): raise RuntimeError(f"non-exact replay {path}")
banks=sorted((WORK/"endpoint_banks").glob("*.npz"))
if len(banks)!=26: raise RuntimeError(f"expected 26 locked banks, got {len(banks)}")
for path in banks:
    manifest=json.loads(path.with_suffix(".json").read_text())
    if not manifest["common_endpoint_matches_input_partition"]: raise RuntimeError(f"common endpoint mismatch {path}")
    with np.load(path,allow_pickle=False) as z:
        if z["partitions"].shape[0]!=8 or len(set(z["candidate_ids"].tolist()))!=8: raise RuntimeError(f"invalid bank {path}")
formal=sorted((WORK/"formal").glob("*.npz"))
if len(formal)!=18: raise RuntimeError(f"expected 18 formal representations, got {len(formal)}")
summary={"schema":"night21c-real-artifact-validation-v1","formal_representations":len(formal),"fresh_process_exact_replays":len(replays),"locked_endpoint_banks":len(banks),"locked_partitions":len(banks)*8,"status":"PASS"}
(WORK/"real_artifact_validation.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
print(json.dumps(summary,indent=2,sort_keys=True))
