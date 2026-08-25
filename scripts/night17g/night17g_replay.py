#!/usr/bin/env python3
"""Fresh-process exact checkpoint/representation/partition replay."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from SpaLORA.night17g_csbo import CSBOConfig,endpoint_partition,reload_core,sha256_array
def main():
 p=argparse.ArgumentParser(); p.add_argument("--carrier",required=True); p.add_argument("--candidate-bank",required=True); p.add_argument("--producer-dir",required=True); p.add_argument("--device",default="cuda"); p.add_argument("--output",required=True); a=p.parse_args()
 d=Path(a.producer_dir); manifest=json.loads((d/"producer.json").read_text()); ck=torch.load(d/"checkpoint.pt",map_location=a.device,weights_only=False)
 with np.load(a.carrier,allow_pickle=False) as z: c={k:np.asarray(z[k]) for k in z.files}
 with np.load(a.candidate_bank,allow_pickle=False) as z: b={k:np.asarray(z[k]) for k in z.files}
 hit=np.flatnonzero(b["candidate_ids"].astype("U")==manifest["strong_candidate_id"]); initial=b["partitions"][int(hit[0])]
 expected={x["run_id"]:x for x in manifest["rows"]}; rows=[]
 for run_id,state in ck["states"].items():
  meta=expected[run_id]; config=CSBOConfig(**meta["config"]); rep=reload_core(state,c["view1"],c["view2"],c["retained"],initial,config,a.device); part=endpoint_partition(rep,initial,manifest["k"])
  ok=sha256_array(rep)==meta["representation_sha256"] and sha256_array(part)==meta["partition_sha256"]
  if not ok: raise RuntimeError(f"replay mismatch {run_id}")
  rows.append({"run_id":run_id,"representation_sha256":sha256_array(rep),"partition_sha256":sha256_array(part),"exact":True})
 Path(a.output).write_text(json.dumps({"schema":"night17g-fresh-process-replay-v1","lane":manifest["lane"],"count":len(rows),"all_exact":True,"rows":rows},indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
