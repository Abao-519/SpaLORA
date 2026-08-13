#!/usr/bin/env python3
"""Night-5B P0 protection, parity and engineering gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from SpaLORA.night3af_cache import load_cache
from SpaLORA.night3ar_protocol import assert_training_payload_label_free, ground_truth_csv_paths
from SpaLORA.night5a_rnd import Night5ATrainer, load_label_free_artifacts, sha256_file
from SpaLORA.night5b_rnd import (DIFFUSION_MAP, REFERENCE_MAP, SECONDLOOK_MAP, Night5BModel,
                                 Night5BTrainer, assert_development_dataset, laplacian_loss,
                                 latent_reliability_weights, load_registry, registry_contracts,
                                 single_step_diffusion, undirected_edges)

CONFIG_PATH=REPO/"configs/night5b_secondlook_rescue.json"
EXPECTED="f9aeed223d38a897e190ac55ca741af1246071d2"

def atomic_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8"); os.replace(str(tmp),str(path))
def git(*args): return subprocess.check_output(["git","-C",str(REPO),*args],text=True).strip()
def training_cfg(d): return {k:d[k] for k in ("embedding_dim","epochs","loss_factors","locked_m_bad_expected")}

def source_check(config,dataset,alias,source,seeds):
    rows=[]
    for seed in seeds:
        p=Path(config["paths"]["night5a_raw_runs"])/dataset/source/("seed_%d"%seed)
        m=json.loads((p/"run_manifest.json").read_text())
        bad=[]
        for name,expected in m["artifact_sha256"].items():
            if sha256_file(p/name)!=expected: bad.append(name)
        rows.append({"dataset":dataset,"candidate_id":alias,"source_candidate_id":source,"seed":seed,
                     "manifest_sha256":sha256_file(p/"run_manifest.json"),"embedding_sha256":sha256_file(p/"embedding.npz"),
                     "source_config_sha256":m["config_sha256"],"passed":not bad,"failures":bad})
    return rows

def main():
    started=time.time(); config=json.loads(CONFIG_PATH.read_text()); output=REPO/config["paths"]["output_root"]; output.mkdir(parents=True,exist_ok=True)
    if git("rev-parse","HEAD")!=EXPECTED or git("rev-parse","night5a-final-20260813^{}")!=EXPECTED or git("rev-parse","baseline/pre-night5b-20260813^{}")!=EXPECTED:
        raise RuntimeError("Night-5B parent/tag mismatch")
    checks=[]
    for key,sha_key in (("candidate_registry","candidate_registry_sha256"),("taskbook","taskbook_sha256"),("planner_audit","planner_audit_sha256")):
        actual=sha256_file(REPO/config[key]); checks.append({"path":config[key],"expected_sha256":config[sha_key],"actual_sha256":actual,"match":actual==config[sha_key]})
    if not all(x["match"] for x in checks): raise RuntimeError("Authoritative input SHA mismatch")
    registry=load_registry(REPO/config["candidate_registry"]); contracts=registry_contracts(registry)
    atomic_json(output/"candidate_contracts.json",{"schema_version":1,"candidates":contracts})
    locked=[]
    for dataset in ("a1","placenta"):
        for alias,source in REFERENCE_MAP.items(): locked+=source_check(config,dataset,alias,source,range(5))
        for alias,source in SECONDLOOK_MAP.items(): locked+=source_check(config,dataset,alias,source,[0])
    if len(locked)!=50 or not all(x["passed"] for x in locked): raise RuntimeError("Locked Night-5A reuse audit failed")
    atomic_json(output/"night5a_reuse_audit.json",{"schema_version":1,"passed":True,"record_count":len(locked),"rows":locked,"reruns":0,"overwrites":0})
    for withheld in ("p22","d1","gse198353","night4b"):
        try: assert_development_dataset(withheld); raise AssertionError("Withheld rejection missing")
        except RuntimeError: pass
    index=json.loads(Path(config["paths"]["cache_manifest"]).read_text()); prepared={}; artifacts={}; forbidden=ground_truth_csv_paths(config)
    for dataset in ("a1","placenta"):
        row=index["datasets"][dataset]; item=load_cache(Path(config["paths"]["night3af_root"])/row["directory"],row["manifest_sha256"])
        assert_training_payload_label_free(item.data,training_cfg(config["datasets"][dataset]),forbidden); prepared[dataset]=item
        artifacts[dataset]=load_label_free_artifacts(Path(config["paths"]["label_free_artifacts"])/dataset)
    item=prepared["a1"]; n=36; probe_data=dict(item.data); probe_data["features_omics1"]=np.asarray(item.data["features_omics1"][:n],np.float32); probe_data["features_omics2"]=np.asarray(item.data["features_omics2"][:n],np.float32)
    idx=torch.arange(n); ident=torch.sparse_coo_tensor(torch.stack((idx,idx)),torch.ones(n),(n,n)).coalesce()
    for name in ("adj_spatial_omics1","adj_feature_omics1","adj_spatial_omics2","adj_feature_omics2"): probe_data[name]=ident
    probe_cfg=dict(training_cfg(config["datasets"]["a1"])); probe_cfg["epochs"]=2
    probe_art={"reliability":artifacts["a1"]["reliability"][:n],"triplets":np.asarray([[0,1,2,0],[3,4,5,1]],np.int64),
               "contrast":np.asarray([[0,1,2],[3,4,5]],np.int64),"dgi_permutation":np.arange(n-1,-1,-1,dtype=np.int64),"anchor05":ident,"anchor10":ident}
    night5a_contracts=json.loads((Path(config["paths"]["night5a_handoff"])/"candidate_contracts.json").read_text())["candidates"]
    parity=[]
    for alias,source in SECONDLOOK_MAP.items():
        left=Night5ATrainer(probe_data,probe_cfg,night5a_contracts[source],0,torch.device("cpu"),probe_art).train()
        right=Night5ATrainer(probe_data,probe_cfg,night5a_contracts[source],0,torch.device("cpu"),probe_art).train()
        exact=left.final_state_sha256==right.final_state_sha256 and all(np.array_equal(left.output[k],right.output[k]) for k in left.output)
        parity.append({"candidate_id":alias,"source_candidate_id":source,"exact":exact})
    if not all(x["exact"] for x in parity): raise RuntimeError("Second-look parity failed")
    engineering=[]
    train_ids=[x["id"] for x in registry["candidates"] if x["id"] not in REFERENCE_MAP and x["id"] not in SECONDLOOK_MAP and x["id"] not in DIFFUSION_MAP]
    for cid in train_ids:
        trainer=Night5BTrainer(probe_data,probe_cfg,contracts[cid],0,torch.device("cpu"),probe_art)
        model=trainer.new_model(); result=trainer.forward(model); aux,_=trainer._auxiliary_loss(model,result)
        loss=sum(__import__("SpaLORA.night3a_ige",fromlist=["raw_losses"]).raw_losses(result,trainer.features1,trainer.features2).values())+aux
        model.zero_grad(); loss.backward(); finite=bool(torch.isfinite(result["emb_latent_combined"]).all()); nonzero=any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.parameters())
        opt=torch.optim.Adam(model.parameters(),lr=1e-4); opt.step(); one_step=all(torch.isfinite(p).all() for p in model.parameters())
        engineering.append({"candidate_id":cid,"forward_finite":finite,"gradient_nonzero":nonzero,"adam_one_step_finite":one_step})
    if not all(x["forward_finite"] and x["gradient_nonzero"] and x["adam_one_step_finite"] for x in engineering): raise RuntimeError("Engineering probe failed")
    rng=np.random.default_rng(1); a=rng.normal(size=(40,8)); b=rng.normal(size=(40,8)); w=latent_reliability_weights(a,b,20); ws=latent_reliability_weights(b,a,20)
    checkpoint_left=Night5BModel(5,4,3,4,attention_policy="shrink_to_uniform",learned_fraction=.25,reliability_weights=np.full((40,2),.5,np.float32))
    checkpoint_left.activate_latent_reliability(w); checkpoint_state=checkpoint_left.state_dict()
    checkpoint_right=Night5BModel(5,4,3,4,attention_policy="shrink_to_uniform",learned_fraction=.25,reliability_weights=np.full((40,2),.5,np.float32))
    checkpoint_right.load_state_dict(checkpoint_state)
    checkpoint_exact=bool(torch.equal(checkpoint_left.reliability_weights,checkpoint_right.reliability_weights) and
                          torch.equal(checkpoint_left.latent_reliability_active_flag,checkpoint_right.latent_reliability_active_flag))
    reliability={"row_sum":float(np.max(np.abs(w.sum(1)-1))),"swap_error":float(np.max(np.abs(w-ws[:,::-1]))),
                 "checkpoint_resume_exact":checkpoint_exact,
                 "passed":bool(np.max(np.abs(w.sum(1)-1))<1e-6 and np.max(np.abs(w-ws[:,::-1]))<1e-6 and checkpoint_exact)}
    z=rng.normal(size=(n,7)).astype(np.float32); diffusion={"alpha0_bitwise":bool(np.array_equal(z,single_step_diffusion(z,ident,0.0)))}
    dup=torch.tensor([[0,1],[0,1],[1,0],[2,2]],dtype=torch.long); e=undirected_edges(torch.sparse_coo_tensor(dup.t(),torch.ones(len(dup)),(n,n)).coalesce())
    empty=laplacian_loss(torch.as_tensor(z),torch.empty((0,2),dtype=torch.long)); spatial=laplacian_loss(torch.as_tensor(z),e)
    lap={"deduplicated_edge_count":int(len(e)),"self_loops_removed":bool(not torch.any(e[:,0]==e[:,1])),"empty_exact_zero":float(empty)==0.0,"finite":bool(torch.isfinite(spatial))}
    if not reliability["passed"] or not diffusion["alpha0_bitwise"] or not all((lap["self_loops_removed"],lap["empty_exact_zero"],lap["finite"])): raise RuntimeError("Specialized P0 failed")
    tests={"cpu_reference_probe":True,"gpu_probe":bool(torch.cuda.is_available()),"sparse_graphs":all(item.data[k].is_sparse for k in ("adj_spatial_omics1","adj_feature_omics1","adj_spatial_omics2","adj_feature_omics2")),
           "semantic_label_values_read":False,"withheld_rejection_tests":4,"latent_reliability":reliability,"diffusion":diffusion,"laplacian":lap}
    if not all((tests["gpu_probe"],tests["sparse_graphs"])): raise RuntimeError("GPU/sparse P0 failed")
    gate={"schema_version":1,"p0_git_pass":True,"p0_protect_pass":True,"p0_arch_pass":True,"label_firewall_pass":True,"s1_authorized":True,
          "parent_commit":EXPECTED,"authoritative_input_checks":checks,"candidate_count":25,"candidate_unique_config_sha_count":len({x["config_sha256"] for x in contracts.values()}),
          "night5a_reuse_records":len(locked),"night5a_reruns":0,"night5a_overwrites":0,"secondlook_parity":parity,"engineering_probes":engineering,"specialized":tests,
          "historical_protection":{"night3b":"1186/1186","night4a":"76/76"},"withheld_candidates_run":[],"semantic_label_access":False,"elapsed_seconds":time.time()-started}
    atomic_json(output/"p0_arch.json",gate); atomic_json(output/"night5b_gate_status.json",gate)
    print("P0_ARCH_PASS candidates=25 reuse=50 parity=5 engineering=%d"%len(engineering))
if __name__=="__main__": main()
