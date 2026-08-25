#!/usr/bin/env python3
"""Label-closed producer for Night-17G CSBO Stage A."""

from __future__ import annotations

import argparse, hashlib, json, os, resource, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import scipy.sparse as sp
import torch
from threadpoolctl import threadpool_limits

from SpaLORA.night17g_csbo import (CSBOConfig, build_edge_states, endpoint_partition,
    permute_edge_states, reload_core, row_normalize, sha256_array, standardize, train_core)

CONFIGS = (
    CSBOConfig("C01_CONSERVATIVE", 32, 0.08, 8e-4, 35, 0.5, 4.0, 0.25, 0.5, 0.20),
    CSBOConfig("C02_BALANCED", 32, 0.15, 1e-3, 45, 0.5, 2.0, 0.25, 1.0, 0.20),
)
ARMS = ("BACKBONE_NO_CSBO", "FULL_CSBO", "PERMUTED_EDGE_STATES", "UNSIGNED_ONLY", "BOUNDARY_TO_ABSTAIN", "CONFLICT_AS_POSITIVE")

def file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def csr_sha(graph: sp.csr_matrix) -> str:
    h=hashlib.sha256()
    for value in (np.asarray(graph.shape,dtype=np.int64),graph.indptr.astype(np.int64),graph.indices.astype(np.int64),graph.data):
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()

def load_graph(carrier: dict[str,np.ndarray], index: int) -> sp.csr_matrix:
    p=f"graph{index}__"
    return sp.csr_matrix((carrier[p+"data"],carrier[p+"indices"],carrier[p+"indptr"]),shape=tuple(carrier[p+"shape"]))

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--lane",required=True); p.add_argument("--carrier",required=True); p.add_argument("--candidate-bank",required=True)
    p.add_argument("--strong-candidate-id",required=True); p.add_argument("--k",type=int,required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--training-seed",type=int,default=0); p.add_argument("--device",default="cuda"); p.add_argument("--configs",default="ALL")
    args=p.parse_args(); started=time.time(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    with np.load(args.carrier,allow_pickle=False) as z: carrier={k:np.asarray(z[k]) for k in z.files}
    with np.load(args.candidate_bank,allow_pickle=False) as z: bank={k:np.asarray(z[k]) for k in z.files}
    if not np.array_equal(carrier["ids"].astype("U"),bank["ids"].astype("U")): raise ValueError("carrier/bank ordered IDs differ")
    if bank["partitions"].shape!=(89,len(carrier["ids"])) or len(np.unique(bank["candidate_ids"].astype("U")))!=89: raise ValueError("candidate bank authority must be unique 89-by-N")
    hit=np.flatnonzero(bank["candidate_ids"].astype("U")==args.strong_candidate_id)
    if len(hit)!=1: raise ValueError("strong candidate must resolve uniquely")
    initial=np.asarray(bank["partitions"][int(hit[0])],dtype=np.int32)
    if np.unique(initial).size!=args.k: raise ValueError("strong start K mismatch")
    graph=load_graph(carrier,0)
    states=build_edge_states(carrier["view1"],carrier["view2"],graph,carrier["ids"])
    permuted=permute_edge_states(states,carrier["ids"])
    for channel in ("attraction","boundary","conflict"):
        target=float(np.sum(states.base_weight.astype(np.float64)*getattr(states,channel).astype(np.float64)))
        current=float(np.sum(states.base_weight.astype(np.float64)*getattr(permuted,channel).astype(np.float64)))
        if not np.isclose(target,current,rtol=1e-7,atol=1e-8): raise RuntimeError(f"permutation changed base-weighted {channel} mass")
    configs=CONFIGS if args.configs=="ALL" else tuple(x for x in CONFIGS if x.config_id in args.configs.split(","))
    if not configs: raise ValueError("no config selected")
    run_ids=[]; partitions=[]; representations=[]; rows=[]; checkpoints={}
    def add(run_id,arm,config_id,partition,representation,diag):
        partition=np.asarray(partition,dtype=np.int32); representation=np.asarray(representation,dtype=np.float32)
        sizes=np.bincount(partition,minlength=args.k)
        if np.unique(partition).size!=args.k or np.any(sizes<=0): raise RuntimeError("exact K/no-empty violation")
        run_ids.append(run_id); partitions.append(partition); representations.append(representation)
        rows.append({"run_id":run_id,"arm":arm,"config_id":config_id,"training_seed":args.training_seed,
                     "partition_sha256":sha256_array(partition),"representation_sha256":sha256_array(representation),
                     "cluster_sizes":sizes.astype(int).tolist(),"min_cluster_size":int(sizes.min()),**diag})
    base=row_normalize(standardize(carrier["retained"]))
    add("INPUT_STRONG_START","INPUT_STRONG_START","BASELINE",initial,base,{"actual_optimizer_steps":0,"changed_from_strong_start":0})
    frozen=endpoint_partition(base,initial,args.k)
    add("FROZEN_RETAINED_SAME_HEAD","FROZEN_RETAINED_SAME_HEAD","BASELINE",frozen,base,{"actual_optimizer_steps":0,"changed_from_strong_start":int(np.sum(frozen!=initial))})
    device=args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
    with threadpool_limits(limits=1):
        for config in configs:
            for arm in ARMS:
                active=permuted if arm=="PERMUTED_EDGE_STATES" else states
                representation,partition,state,diag=train_core(carrier["view1"],carrier["view2"],carrier["retained"],initial,active,config,arm,args.training_seed,device)
                run_id=f"{config.config_id}__{arm}__S{args.training_seed}"
                replay=reload_core(state,carrier["view1"],carrier["view2"],carrier["retained"],initial,config,device)
                if not np.array_equal(replay,representation): raise RuntimeError("strict in-process representation reload mismatch")
                replay_partition=endpoint_partition(replay,initial,args.k)
                if not np.array_equal(replay_partition,partition): raise RuntimeError("strict in-process partition reload mismatch")
                checkpoints[run_id]=state
                add(run_id,arm,config.config_id,partition,representation,{**diag,"actual_optimizer_steps":config.steps,
                    "changed_from_strong_start":int(np.sum(partition!=initial)),"config":config.to_dict(),"edge_state_sha256":active.state_sha256})
    torch.save({"schema":"night17g-csbo-checkpoint-v1","lane":args.lane,"training_seed":args.training_seed,
                "strong_candidate_id":args.strong_candidate_id,"configs":[x.to_dict() for x in configs],"states":checkpoints},out/"checkpoint.pt")
    np.savez_compressed(out/"producer.npz",ids=carrier["ids"],run_ids=np.asarray(run_ids,dtype="U"),partitions=np.stack(partitions),
                        representations=np.stack(representations).astype(np.float32))
    with np.load(out/"producer.npz",allow_pickle=False) as z:
        if not np.array_equal(z["partitions"],np.stack(partitions)) or not np.array_equal(z["representations"],np.stack(representations)): raise RuntimeError("artifact reload failed")
    manifest={"schema":"night17g-csbo-producer-v1","lane":args.lane,"k":args.k,"n":len(initial),
        "shapes":{"view1":list(carrier["view1"].shape),"view2":list(carrier["view2"].shape),"retained":list(carrier["retained"].shape),"graph0":[*graph.shape,graph.nnz]},
        "authority":{"carrier_path":str(Path(args.carrier)),"carrier_sha256":file_sha(Path(args.carrier)),"candidate_bank_path":str(Path(args.candidate_bank)),
                     "candidate_bank_sha256":file_sha(Path(args.candidate_bank)),"candidate_count":89,"strong_candidate_id":args.strong_candidate_id,
                     "strong_candidate_unique_match":True,"ordered_ids_sha256":sha256_array(carrier["ids"]),"graph0_sha256":csr_sha(graph)},
        "strong_candidate_id":args.strong_candidate_id,"strong_partition_sha256":sha256_array(initial),"edge_state":dict(states.diagnostics),
        "edge_state_sha256":states.state_sha256,"permuted_edge_state_sha256":permuted.state_sha256,
        "label_values_accessed_or_used":False,"producer_label_reads":0,"candidate_count":len(rows),"rows":rows,
        "checkpoint_sha256":file_sha(out/"checkpoint.pt"),"artifact_sha256":file_sha(out/"producer.npz"),"artifact_reload":"PASS",
        "wall_seconds":time.time()-started,"peak_gpu_mib":float(torch.cuda.max_memory_allocated()/(1024**2)) if torch.cuda.is_available() else 0.0,
        "peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0}
    (out/"producer.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")

if __name__=="__main__": main()
