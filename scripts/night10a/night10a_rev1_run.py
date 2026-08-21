from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import asdict, fields
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from SpaLORA.night10a_qcrd import (
    EPS, MASKED_CANDIDATES, FrozenQuality, QCRDAdapter,
    canonical_array_sha256, canonical_sparse_sha256, canonical_state_sha256,
    corrected_views, deterministic_mask, fourier_coordinates, frozen_quality,
    qcrd_forward, row_normalize,
)
from SpaLORA.night6c_pipeline import self_tuning_affinity, spectral
from SpaLORA.night7b_adaptive import row_sparse_strict, sym_zero


REPO = pathlib.Path("/root/autodl-fs/SpaLORA-night10a-rev1")
RAW = pathlib.Path("/root/autodl-fs/night10a_rev1_qcrd_20260821")
SOURCE = pathlib.Path("/root/autodl-fs/night7b_score_rnd_20260818")
PYTHON = "/root/miniconda3/envs/SpaLORA/bin/python"
CANDIDATES = [
    "Q01_GLOBAL_QUALITY_BLEND", "Q02_SPOT_QUALITY_BLEND", "Q03_MASKED_RESIDUAL",
    "Q04_SPOT_GATED_MASKED_RESIDUAL", "Q05_BOUNDARY_GATED_RESIDUAL",
    "Q06_COORDINATE_PRIOR_RESIDUAL", "Q07_CONFIDENCE_MNN_RESIDUAL",
]
DATA = {
    "a1": {"base": 0, "K": 10, "family": "RNA+protein", "r1": range(3), "r2": range(5)},
    "tonsil": {"base": 5, "K": 4, "family": "RNA+protein", "r1": range(3), "r2": range(5)},
    "d1": {"base": 10, "K": 10, "family": "RNA+protein", "r1": range(3), "r2": range(10)},
    "p22": {"base": 20, "K": 9, "family": "RNA+ATAC", "r1": range(3), "r2": range(10)},
}
LOSS_WEIGHTS = {"align": 1.0, "mask": 1.0, "anchor": .25, "correction": .05, "boundary": .25, "mnn": .10}


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: pathlib.Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = pathlib.Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **values); os.replace(tmp, path)


def unit(dataset: str, seed: int) -> str:
    return f"u{DATA[dataset]['base'] + seed:03d}"


def source_spec(unit_id: str) -> dict:
    return json.loads((SOURCE / f"source/{unit_id}/worker_input.json").read_text())


def ids(unit_id: str) -> list[str]:
    return (SOURCE / f"source/{unit_id}/observation_ids.txt").read_text().splitlines()


def views(unit_id: str):
    x = np.load(SOURCE / f"source/{unit_id}/g04_views.npz")
    return tuple(np.asarray(x[k], dtype=np.float32) for k in ("emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused"))


def graph_path(dataset: str) -> pathlib.Path:
    cache = "night6c_cache_20260817" if dataset in {"a1", "tonsil"} else "night6d_cache_20260817"
    return pathlib.Path("/root/autodl-fs") / cache / f"graphs/{dataset}/G04_SP10_F10_EUC_UNION/adj_spatial_omics1_support.npz"


def coords_path(dataset: str, unit_id: Optional[str] = None) -> pathlib.Path:
    # Night-7B source packs lock the exact ordered coordinates for every unit.
    if unit_id is None: unit_id = unit(dataset, 0)
    return SOURCE / f"source/{unit_id}/coordinates.npy"


def r02_root(unit_id: str) -> pathlib.Path:
    for stage in ("R1", "R2"):
        path = SOURCE / f"adapter_stage/{stage}/formal/R02/{unit_id}/attempt_001"
        if (path/"worker/embedding.npy").exists() and (path/"transforms/E1_ADAPTER_C06_MEAN/H01/clusters.csv").exists():
            return path
    raise FileNotFoundError(f"no complete authoritative R02 endpoint for {unit_id}")


def read_cluster(path: pathlib.Path) -> tuple[list[str], np.ndarray]:
    names, labels = [], []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle): names.append(row["observation_id"]); labels.append(int(row["cluster"]))
    return names, np.asarray(labels, dtype=np.int64)


def reference(unit_id: str, family: str):
    if family == "RNA+protein":
        return views(unit_id)[2], np.load(SOURCE / f"source/{unit_id}/c00_partition.npy"), None
    root = r02_root(unit_id)
    zf = np.load(root / "worker/embedding.npy").astype(np.float32)
    names, part = read_cluster(root / "transforms/E1_ADAPTER_C06_MEAN/H01/clusters.csv")
    if names != ids(unit_id): raise AssertionError("R02 order mismatch")
    return zf, part, SOURCE / f"source/{unit_id}/c06_affinity.npz"


def quality_dir(unit_id: str) -> pathlib.Path:
    return RAW / f"raw/inputs/{unit_id}"


def prepare_one(dataset: str, seed: int) -> dict:
    unit_id = unit(dataset, seed); out = quality_dir(unit_id); manifest_file = out / "input_manifest.json"
    if manifest_file.exists(): return json.loads(manifest_file.read_text())
    out.mkdir(parents=True, exist_ok=True)
    z1, z2, _ = views(unit_id); zf, pf, c06 = reference(unit_id, DATA[dataset]["family"])
    names = ids(unit_id); graph_file = graph_path(dataset); coord_file = coords_path(dataset, unit_id)
    graph = sp.load_npz(graph_file); coords = np.load(coord_file)
    if not (len(names) == len(z1) == len(z2) == len(zf) == len(pf) == len(coords)):
        raise AssertionError("input cardinality mismatch")
    a1 = self_tuning_affinity(row_normalize(z1), 10, names); p1 = spectral(a1, DATA[dataset]["K"])
    a2 = self_tuning_affinity(row_normalize(z2), 10, names); p2 = spectral(a2, DATA[dataset]["K"])
    q = frozen_quality(z1, z2, zf, p1, p2, graph, DATA[dataset]["K"], 10)
    np.save(out / "p1.npy", p1); np.save(out / "p2.npy", p2); np.save(out / "pf.npy", pf)
    arrays, diagnostics = {}, q.diagnostics
    for field in fields(FrozenQuality):
        value = getattr(q, field.name)
        if isinstance(value, np.ndarray): arrays[field.name] = value
    arrays["zero_degree_count"] = np.asarray(q.zero_degree_count, dtype=np.int64)
    atomic_npz(out / "quality.npz", **arrays)
    manifest = {
        "unit_id": unit_id, "dataset": dataset, "seed": seed, "K": DATA[dataset]["K"],
        "family": DATA[dataset]["family"], "observation_count": len(names), "label_reads": 0,
        "ordered_observation_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "views_file": str(SOURCE / f"source/{unit_id}/g04_views.npz"),
        "views_file_sha256": sha(SOURCE / f"source/{unit_id}/g04_views.npz"),
        "zf_file": str(r02_root(unit_id)/"worker/embedding.npy") if dataset == "p22" else str(SOURCE/f"source/{unit_id}/g04_views.npz"),
        "zf_sha256": sha(r02_root(unit_id)/"worker/embedding.npy") if dataset == "p22" else canonical_array_sha256(zf),
        "spatial_graph_file": str(graph_file), "spatial_graph_file_sha256": sha(graph_file),
        "spatial_graph_canonical_sha256": canonical_sparse_sha256(graph),
        "coordinates_file": str(coord_file), "coordinates_file_sha256": sha(coord_file),
        "p1_sha256": sha(out/"p1.npy"), "p2_sha256": sha(out/"p2.npy"), "pf_sha256": sha(out/"pf.npy"),
        "quality_sha256": sha(out/"quality.npz"), "quality_diagnostics": diagnostics,
        "zero_degree_count": int(q.zero_degree_count), "c06_file": str(c06) if c06 else None,
        "c06_sha256": sha(c06) if c06 else None,
    }
    atomic_json(manifest_file, manifest); return manifest


def load_quality(path: pathlib.Path) -> FrozenQuality:
    x = np.load(path); values = {}
    for field in fields(FrozenQuality):
        if field.name == "diagnostics": values[field.name] = {}
        elif field.name == "zero_degree_count": values[field.name] = int(x[field.name])
        else: values[field.name] = x[field.name]
    return FrozenQuality(**values)


def config_path(stage: str, dataset: str, seed: int, candidate: str) -> pathlib.Path:
    return REPO / f"protocols/night10a_rev1/formal_configs/{stage}/{dataset}/seed_{seed}/{candidate}.json"


def prepare_configs(stage: str, selected=None):
    rows = []
    candidates = CANDIDATES if selected is None else selected
    for dataset, spec in DATA.items():
        seeds = spec[stage]
        for seed in seeds:
            if stage == "r2" and seed in spec["r1"]:
                continue  # Exact R1 cells are reused; only missing seed cells are trained.
            unit_id = unit(dataset, seed)
            input_manifest = prepare_one(dataset, seed)
            input_sha = sha(quality_dir(unit_id)/"input_manifest.json")
            for candidate in candidates:
                path = config_path(stage, dataset, seed, candidate)
                cfg = {
                    "schema": "spalora.night10a.rev1.formal_config.v1", "stage": stage,
                    "dataset": dataset, "family": spec["family"], "seed": seed, "unit_id": unit_id,
                    "K": spec["K"], "candidate": candidate, "input_manifest": str(quality_dir(unit_id)/"input_manifest.json"),
                    "input_manifest_sha256": input_sha, "epochs": 120, "optimizer": "AdamW",
                    "learning_rate": .001, "weight_decay": .0001, "gradient_clip_norm": 5.0,
                    "hidden_width": 64, "residual_rank": 16, "dropout": .1,
                    "amp": False, "early_stopping": False, "best_epoch": False,
                    "scientific_retry": 0, "fallback": 0, "transform_timeout_seconds": 1800,
                    "label_access": 0,
                    "implementation_sha256": sha(REPO/"SpaLORA/night10a_qcrd.py"),
                    "runner_sha256": sha(REPO/"scripts/night10a/night10a_rev1_run.py"),
                    "semantic_contract_sha256": sha(REPO/"protocols/night10a_rev1/night10a_qcrd_rev1_semantic_contract.json"),
                    "h05_endpoint_source_sha256": sha(REPO/"SpaLORA/night6c_pipeline.py"),
                    "p22_endpoint_source_sha256": sha(REPO/"SpaLORA/night7b_adaptive.py"),
                }
                atomic_json(path, cfg); rows.append({"path": str(path.relative_to(REPO)), "sha256": sha(path), **cfg})
    atomic_json(REPO/f"protocols/night10a_rev1/formal_configs/{stage}_index.json", {"stage":stage,"count":len(rows),"rows":rows})
    return rows


def torch_loss(out, first, second, fused, quality, candidate, mask, boundary_rows, boundary_cols):
    gate = out["gate"].view(-1)
    align = torch.sum(gate*(1-F.cosine_similarity(out["student_corrected"],out["teacher"].detach(),dim=1,eps=EPS)))/(torch.sum(gate)+EPS)
    if candidate in MASKED_CANDIDATES:
        mf=mask.to(first.dtype); num=torch.sum(mf*(out["pred_student"]-out["clean_student"].detach())**2,dim=1)
        den=torch.sum(mf*out["clean_student"].detach()**2,dim=1)+EPS; masked=torch.mean(num/den)
    else: masked=align.new_zeros(())
    anchor=.5*torch.mean((1-F.cosine_similarity(out["z1c"],first,dim=1,eps=EPS))+(1-F.cosine_similarity(out["z2c"],second,dim=1,eps=EPS)))
    correction=torch.mean((out["correction"].norm(dim=1)/.25)**2)
    if len(boundary_rows):
        new=torch.sum(out["zc"][boundary_rows]*out["zc"][boundary_cols],dim=1)
        old=torch.sum(fused[boundary_rows]*fused[boundary_cols],dim=1); boundary=torch.mean(F.relu(new-old)**2)
    else: boundary=align.new_zeros(())
    if candidate=="Q07_CONFIDENCE_MNN_RESIDUAL" and len(quality.mnn_rows):
        r=torch.as_tensor(quality.mnn_rows,dtype=torch.long,device=first.device); c=torch.as_tensor(quality.mnn_cols,dtype=torch.long,device=first.device)
        h1=torch.as_tensor(quality.entropy_1[quality.mnn_rows],dtype=first.dtype,device=first.device); h2=torch.as_tensor(quality.entropy_2[quality.mnn_cols],dtype=first.dtype,device=first.device)
        b1=torch.as_tensor(quality.boundary_risk[quality.mnn_rows],dtype=first.dtype,device=first.device); b2=torch.as_tensor(quality.boundary_risk[quality.mnn_cols],dtype=first.dtype,device=first.device)
        w=torch.sqrt((1-h1).clamp_min(0)*(1-h2).clamp_min(0))*(1-b1)*(1-b2)
        mnn=torch.sum(w*(1-F.cosine_similarity(out["z1c"][r],out["z2c"][c],dim=1,eps=EPS)))/(torch.sum(w)+EPS)
    else: mnn=align.new_zeros(())
    total=align+masked+.25*anchor+.05*correction+.25*boundary+(.1*mnn if candidate=="Q07_CONFIDENCE_MNN_RESIDUAL" else 0)
    return {"align":align,"mask":masked,"anchor":anchor,"correction":correction,"boundary":boundary,"mnn":mnn,"total":total}


def cell_dir(cfg) -> pathlib.Path:
    return RAW / f"{cfg['stage']}/{cfg['dataset']}/seed_{cfg['seed']}/{cfg['candidate']}"


def train(config_file: pathlib.Path):
    cfg=json.loads(config_file.read_text()); outdir=cell_dir(cfg); final=outdir/"training_manifest.json"
    if final.exists(): return
    outdir.mkdir(parents=True,exist_ok=True); start=time.time()
    if sha(pathlib.Path(cfg["input_manifest"])) != cfg["input_manifest_sha256"]: raise AssertionError("input manifest drift")
    unit_id=cfg["unit_id"]; z1,z2,_=views(unit_id); zf,pf,_=reference(unit_id,cfg["family"])
    q=load_quality(quality_dir(unit_id)/"quality.npz"); graph=sp.load_npz(graph_path(cfg["dataset"]))
    upper=sp.triu(graph.maximum(graph.T),k=1).tocoo(); keep=pf[upper.row]!=pf[upper.col]
    device=torch.device("cuda"); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"]); np.random.seed(cfg["seed"])
    first=torch.as_tensor(row_normalize(z1),dtype=torch.float32,device=device); second=torch.as_tensor(row_normalize(z2),dtype=torch.float32,device=device); fused=torch.as_tensor(row_normalize(zf),dtype=torch.float32,device=device)
    coord=None
    if cfg["candidate"]=="Q06_COORDINATE_PRIOR_RESIDUAL": coord=torch.as_tensor(fourier_coordinates(np.load(coords_path(cfg["dataset"], unit_id))),dtype=torch.float32,device=device)
    model=QCRDAdapter(first.shape[1],0 if coord is None else coord.shape[1],64,16,.1).to(device)
    initial=canonical_state_sha256(model.state_dict()); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    br=torch.as_tensor(upper.row[keep],dtype=torch.long,device=device); bc=torch.as_tensor(upper.col[keep],dtype=torch.long,device=device)
    input_artifact=cfg["input_manifest_sha256"]; curves=[]; peak=0
    for epoch in range(120):
        model.train(); optimizer.zero_grad(set_to_none=True); mask=None
        if cfg["candidate"] in MASKED_CANDIDATES: mask=torch.as_tensor(deterministic_mask(cfg["candidate"],input_artifact,cfg["seed"],epoch,len(first),first.shape[1]),device=device)
        fo=qcrd_forward(model,first,second,fused,q,cfg["candidate"],coord,mask); losses=torch_loss(fo,first,second,fused,q,cfg["candidate"],mask,br,bc)
        if not torch.isfinite(losses["total"]): raise FloatingPointError("nonfinite formal loss")
        losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
        peak=max(peak,torch.cuda.max_memory_allocated())
        curves.append({"epoch":epoch,**{k:float(v.detach().cpu()) for k,v in losses.items()}})
    model.eval()
    with torch.no_grad(): a,b,c,cor=corrected_views(model,first,second,fused,q,cfg["candidate"],coord)
    atomic_npz(outdir/"corrected_views.npz",z1c=a.cpu().numpy(),z2c=b.cpu().numpy(),zc=c.cpu().numpy(),correction=cor.cpu().numpy())
    ckpt={"state_dict":model.state_dict(),"config":cfg,"initial_state_sha256":initial}; tmp=outdir/"model_final.pt.tmp"; torch.save(ckpt,tmp); os.replace(tmp,outdir/"model_final.pt")
    with (outdir/"loss_curve.csv.tmp").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    os.replace(outdir/"loss_curve.csv.tmp",outdir/"loss_curve.csv")
    expected=np.load(outdir/"corrected_views.npz"); expected_sha={k:canonical_array_sha256(expected[k]) for k in expected.files}
    manifest={"status":"TRAINED_AWAITING_RELOAD","config":cfg,"config_file_sha256":sha(config_file),"model_file_sha256":sha(outdir/"model_final.pt"),"canonical_state_sha256":canonical_state_sha256(model.state_dict()),"initial_state_sha256":initial,"view_shas":expected_sha,"loss_curve_sha256":sha(outdir/"loss_curve.csv"),"epochs":120,"device":torch.cuda.get_device_name(0),"peak_gpu_bytes":peak,"runtime_seconds":time.time()-start,"label_reads":0,"scientific_retry":0,"fallback":0}
    atomic_json(final,manifest)
    cmd=[PYTHON,str(REPO/"scripts/night10a/night10a_rev1_run.py"),"reload",str(config_file)]
    proc=subprocess.run(cmd,cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
    (outdir/"reload.log").write_text(proc.stdout)
    if proc.returncode: raise RuntimeError("fresh reload failed")


def reload(config_file: pathlib.Path):
    cfg=json.loads(config_file.read_text()); outdir=cell_dir(cfg); manifest=json.loads((outdir/"training_manifest.json").read_text())
    unit_id=cfg["unit_id"]; z1,z2,_=views(unit_id); zf,_,_=reference(unit_id,cfg["family"]); q=load_quality(quality_dir(unit_id)/"quality.npz")
    first=torch.as_tensor(row_normalize(z1),dtype=torch.float32,device="cuda"); second=torch.as_tensor(row_normalize(z2),dtype=torch.float32,device="cuda"); fused=torch.as_tensor(row_normalize(zf),dtype=torch.float32,device="cuda")
    coord=None
    if cfg["candidate"]=="Q06_COORDINATE_PRIOR_RESIDUAL": coord=torch.as_tensor(fourier_coordinates(np.load(coords_path(cfg["dataset"], unit_id))),dtype=torch.float32,device="cuda")
    ckpt=torch.load(outdir/"model_final.pt",map_location="cuda"); model=QCRDAdapter(first.shape[1],0 if coord is None else coord.shape[1],64,16,.1).cuda().eval(); model.load_state_dict(ckpt["state_dict"])
    with torch.no_grad(): a,b,c,cor=corrected_views(model,first,second,fused,q,cfg["candidate"],coord)
    got={"z1c":a.cpu().numpy(),"z2c":b.cpu().numpy(),"zc":c.cpu().numpy(),"correction":cor.cpu().numpy()}; expected=np.load(outdir/"corrected_views.npz")
    checks={k:{"max_abs":float(np.max(np.abs(got[k]-expected[k]))),"allclose":bool(np.allclose(got[k],expected[k],rtol=1e-6,atol=1e-7)),"sha_exact":canonical_array_sha256(got[k])==manifest["view_shas"][k]} for k in got}
    if not all(x["allclose"] for x in checks.values()): raise AssertionError("reload view mismatch")
    atomic_json(outdir/"reload_audit.json",{"status":"PASS","fresh_process":True,"checks":checks,"label_reads":0})
    manifest["status"]="CHECKPOINT_ROUNDTRIP_PASS"; manifest["reload_audit_sha256"]=sha(outdir/"reload_audit.json"); atomic_json(outdir/"training_manifest.json",manifest)


def transform(config_file: pathlib.Path):
    cfg=json.loads(config_file.read_text()); outdir=cell_dir(cfg); manifest=json.loads((outdir/"training_manifest.json").read_text())
    if manifest["status"]!="CHECKPOINT_ROUNDTRIP_PASS": raise AssertionError("training not locked")
    if (outdir/"transform_manifest.json").exists(): return
    x=np.load(outdir/"corrected_views.npz"); names=ids(cfg["unit_id"]); start=time.time()
    if cfg["family"]=="RNA+protein":
        aff=[self_tuning_affinity(row_normalize(x[k]),10,names) for k in ("z1c","z2c","zc")]; affinity=sym_zero(sum(aff)/3); endpoint="H05_EQUAL3_AFFINITY_SPECTRAL"
    else:
        az=self_tuning_affinity(row_normalize(x["zc"]),10,names); c06=sp.load_npz(SOURCE/f"source/{cfg['unit_id']}/c06_affinity.npz"); affinity=sym_zero(.5*row_sparse_strict(az)+.5*row_sparse_strict(c06)); endpoint="E1_ADAPTER_C06_MEAN/H01"
    clusters=spectral(affinity,cfg["K"]); sp.save_npz(outdir/"affinity.npz",affinity)
    tmp=outdir/"clusters.csv.tmp"
    with tmp.open("w",newline="") as h:
        w=csv.writer(h); w.writerow(["observation_id","cluster"]); w.writerows(zip(names,clusters.tolist()))
    os.replace(tmp,outdir/"clusters.csv")
    atomic_json(outdir/"transform_manifest.json",{"status":"PASS","endpoint":endpoint,"clusters_sha256":sha(outdir/"clusters.csv"),"affinity_sha256":sha(outdir/"affinity.npz"),"affinity_canonical_sha256":canonical_sparse_sha256(affinity),"partition_sha256":canonical_array_sha256(clusters),"runtime_seconds":time.time()-start,"label_reads":0})


def run_stage(stage: str):
    index=json.loads((REPO/f"protocols/night10a_rev1/formal_configs/{stage}_index.json").read_text())
    failures=[]; started=time.time(); run_start=(RAW/"p0_rev1/p0_rev1_authority_protection.json").stat().st_mtime
    for ordinal,row in enumerate(index["rows"],1):
        cfg=REPO/row["path"]
        try: train(cfg)
        except Exception as exc:
            failures.append({"ordinal":ordinal,"config":str(cfg),"phase":"training","error":repr(exc)})
            atomic_json(cell_dir(row)/"failure.json",failures[-1])
        if time.time()-run_start>12*3600:
            failures.append({"ordinal":ordinal,"config":str(cfg),"phase":"training","error":"WALLCLOCK_BUDGET_REACHED"}); break
    # Fixed endpoint: three independent subprocesses x three BLAS threads <= 12 CPUs.
    tasks=[]
    for ordinal,row in enumerate(index["rows"],1):
        cfg=REPO/row["path"]; out=cell_dir(row)
        if not (out/"training_manifest.json").exists() or (out/"failure.json").exists(): continue
        tasks.append((ordinal,row,cfg,out))

    def run_transform(task):
        ordinal,row,cfg,out=task
        if time.time()-run_start>12*3600:
            error={"ordinal":ordinal,"config":str(cfg),"phase":"transform","error":"WALLCLOCK_BUDGET_REACHED"}; atomic_json(out/"failure.json",error); return error
        try:
            proc=subprocess.run([PYTHON,str(REPO/"scripts/night10a/night10a_rev1_run.py"),"transform",str(cfg)],cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=1800,env={**os.environ,"OMP_NUM_THREADS":"3","MKL_NUM_THREADS":"3","OPENBLAS_NUM_THREADS":"3"})
            (out/"transform.log").write_text(proc.stdout)
            if proc.returncode: raise RuntimeError("transform subprocess nonzero")
            return None
        except subprocess.TimeoutExpired: error={"ordinal":ordinal,"config":str(cfg),"phase":"transform","error":"TIMEOUT_1800S"}
        except Exception as exc: error={"ordinal":ordinal,"config":str(cfg),"phase":"transform","error":repr(exc)}
        atomic_json(out/"failure.json",error); return error

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for error in pool.map(run_transform,tasks):
            if error is not None: failures.append(error)
    manifests=list(RAW.glob(f"{stage}/**/transform_manifest.json")); atomic_json(RAW/f"{stage}/{stage}_lock_manifest.json",{"stage":stage,"registered":len(index["rows"]),"locked":len(manifests),"failures":failures,"label_reads":0,"elapsed_seconds":time.time()-started})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=["prepare-r1","prepare-r2","run-r1","run-r2","train","reload","transform"]); parser.add_argument("arg",nargs="?"); args=parser.parse_args()
    if args.mode=="prepare-r1": prepare_configs("r1")
    elif args.mode=="prepare-r2": prepare_configs("r2",json.loads(args.arg))
    elif args.mode=="run-r1": run_stage("r1")
    elif args.mode=="run-r2": run_stage("r2")
    elif args.mode=="train": train(pathlib.Path(args.arg))
    elif args.mode=="reload": reload(pathlib.Path(args.arg))
    elif args.mode=="transform": transform(pathlib.Path(args.arg))


if __name__=="__main__": main()
