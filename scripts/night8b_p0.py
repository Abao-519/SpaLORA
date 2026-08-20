#!/usr/bin/env python3
"""Night-8B authority, provenance, firewall, cache, parity and CUDA gate."""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
import torch

REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from SpaLORA.night3af_cache import sha256_file
from SpaLORA.night6c_pipeline import array_sha, atomic_json, sparse_sha
from SpaLORA.night7a_consensus import build_base_affinities, candidate_affinity
from SpaLORA.night7b_adaptive import VIEWS
from SpaLORA.night8b_pipeline import (
    ANN, G00, G04, OUT, RAW, REGISTRY, SOURCE, build_caches, file_md5,
    load_official_counts, observation_sha,
)

TASK=REPO/"protocols/night8b/SpaLORA_Night8B_MISAR_Family_Policy_External_Confirmation_Taskbook_2026-08-20.md"
EXPECTED_TASK="068d9e7b0953dc600cd1cc809978a12037afb007bc894a52a8429cbd7f0bedcf"
EXPECTED_REG="4ea482e37fcfdb17ccc8981a2895892c76b6c13a7f5019eae0976b408f2036c7"
PARENT="b2eb62d8ddb1b80b14f916e4dcbbcb407581c6ca"
BRANCH="revision/q2-night8b-misar-family-policy-external-20260820"
PROTECT="baseline/pre-night8b-misar-family-policy-external-20260820"


def git(*args): return subprocess.check_output(["git",*args],cwd=REPO,text=True).strip()


def verify_authority(reg):
    checks={"taskbook":{"expected":EXPECTED_TASK,"actual":sha256_file(TASK)},
            "registry":{"expected":EXPECTED_REG,"actual":sha256_file(REGISTRY)}}
    for value in checks.values(): value["match"]=value["expected"]==value["actual"]
    if not all(x["match"] for x in checks.values()): raise RuntimeError("authority SHA mismatch")
    if git("rev-parse","HEAD") != PARENT or git("branch","--show-current") != BRANCH:
        raise RuntimeError("parent/branch mismatch")
    if git("rev-parse",PROTECT+"^{}") != PARENT or git("rev-parse","night8a-eval-recovery-final-20260820^{}") != PARENT:
        raise RuntimeError("tag chain mismatch")
    files=[]
    for row in reg["dataset"]["training_inputs"]:
        path=SOURCE/row["name"]
        actual={"name":row["name"],"bytes":path.stat().st_size,"md5":file_md5(path),"sha256":sha256_file(path)}
        actual["match"]=actual["bytes"]==row["bytes"] and actual["md5"]==row["md5"] and actual["sha256"]==row["sha256"]
        files.append(actual)
    if not all(x["match"] for x in files): raise RuntimeError("official training file mismatch")
    return checks,files


def provenance_and_mapping(reg):
    archive=ANN/"MISAR_seq_mouse_E15_brain_data.zip"
    carrier=ANN/"MISAR_seq_mouse_E15_brain_ATAC_data.h5"
    ann=reg["dataset"]["annotation_carrier"]
    if archive.stat().st_size!=ann["bytes"] or file_md5(archive)!=ann["md5"]:
        raise RuntimeError("annotation carrier archive mismatch")
    ids,coords,rna,atac,genes,peaks,tissue=load_official_counts()
    with h5py.File(carrier,"r") as h:
        keys=sorted(h.keys()); y_shape=list(h["Y"].shape); y_dtype=str(h["Y"].dtype)
        cells=h["cell"][:].astype("U"); carrier_pos=np.asarray(h["pos"][:])
    # Y was deliberately not sliced or converted above.
    if y_shape != [1949] or len(set(cells))!=1949 or set(cells)!=set(ids):
        raise RuntimeError("annotation cell/K prelock contract mismatch")
    carrier_index={x:i for i,x in enumerate(cells)}
    order=np.asarray([carrier_index[x] for x in ids],np.int64)
    if not np.array_equal(carrier_pos[order].astype(np.float32),coords):
        raise RuntimeError("independent coordinate mapping mismatch")
    mapping=OUT/"prelabel_observation_mapping.csv"; mapping.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"observation_id":ids,"carrier_row":order,"array_col":coords[:,0].astype(int),
                  "array_row":coords[:,1].astype(int)}).to_csv(mapping,index=False)
    corrected={
        "status":"PASS_CORRECTED","official_accession":"OEP003285",
        "valid_raw_read_cross_reference":"SRP491963","official_processed_record":"Zenodo 7480069",
        "annotation_carrier":"Figshare 21623148 v5 file 42520831",
        "forbidden_incorrect_cross_reference":"GSE213264",
        "correction":"GSE213264 is unrelated spatial-CITE-seq and is not a MISAR cross-reference",
        "training_uses_annotation_expression_matrix":False,"Y_values_read":False,
        "raw_shapes":{"RNA":list(rna.shape),"ATAC":list(atac.shape)},
        "mapping_rows":len(order),"mapping_sha256":sha256_file(mapping),
        "carrier":{"archive_sha256":sha256_file(archive),"archive_md5":file_md5(archive),
                   "hdf5_keys":keys,"Y_shape":y_shape,"Y_dtype":y_dtype,
                   "Y_value_deserialized":False},
    }
    atomic_json(OUT/"corrected_misar_provenance_lock.json",corrected)
    return corrected


def p22_code_parity():
    source=Path("/root/autodl-fs/night7b_score_rnd_20260818/source/u020")
    with np.load(source/"g00_views.npz",allow_pickle=False) as z: v00={k:np.asarray(z[k]) for k in VIEWS}
    with np.load(source/"g04_views.npz",allow_pickle=False) as z: v04={k:np.asarray(z[k]) for k in VIEWS}
    ids=[x for x in (source/"observation_ids.txt").read_text().splitlines() if x]
    coords=np.load(source/"coordinates.npy",allow_pickle=False)
    base=build_base_affinities(v00,v04,ids,coords)
    c06,_=candidate_affinity("C06_DUAL_ROW_STOCHASTIC_MEAN",base,ids)
    expected=sp.load_npz(source/"c06_affinity.npz")
    result={"status":"PASS","role":"read_only_code_semantics_parity_only",
            "p22_artifacts_in_misar_model_dag":False,
            "s00_exact":sparse_sha(base["S_G00"])==sparse_sha(sp.load_npz(source/"s00.npz")),
            "s04_exact":sparse_sha(base["S_G04"])==sparse_sha(sp.load_npz(source/"s04.npz")),
            "c06_exact":sparse_sha(c06)==sparse_sha(expected),"label_access":False}
    if not all(result[x] for x in ("s00_exact","s04_exact","c06_exact")):
        result["status"]="FAIL"; raise RuntimeError("P22 code-only parity mismatch")
    atomic_json(OUT/"p0_p22_code_parity.json",result); return result


def cuda_probe():
    if not torch.cuda.is_available(): raise RuntimeError("BLOCKED_DEPLOYABILITY: CUDA unavailable")
    torch.cuda.reset_peak_memory_stats(); x=torch.randn(32,32,device="cuda",requires_grad=True)
    loss=(x@x.T).square().mean(); loss.backward()
    layer=torch.nn.Linear(32,8).cuda(); opt=torch.optim.Adam(layer.parameters(),lr=1e-4)
    opt.zero_grad(); z=layer(x.detach()).square().mean(); z.backward(); opt.step()
    result={"status":"PASS","cuda_available":True,"tensor_device":str(x.device),
            "parameter_device":str(next(layer.parameters()).device),"loss_device":str(z.device),
            "backward_finite":bool(torch.isfinite(x.grad).all()),
            "gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"torch_cuda":torch.version.cuda,
            "driver":subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).strip(),
            "peak_gpu_mib":torch.cuda.max_memory_allocated()/1048576.0}
    atomic_json(OUT/"p0_cuda_probe.json",result); return result


def smoke():
    root=RAW/"smoke"; base=root/"base"; adapter=root/"adapter"; transforms=root/"transforms"
    if root.exists(): raise RuntimeError("smoke root already exists; explicit audit required")
    py=sys.executable; runner=str(REPO/"scripts/night8b_train.py")
    env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    commands=[
        ["timeout","3600",py,runner,"base","--seed","0","--root",str(base),"--epochs","2"],
        ["timeout","3600",py,runner,"reload-base","--seed","0","--root",str(base)],
        ["timeout","3600",py,runner,"adapter","--seed","0","--base-root",str(base),"--adapter-root",str(adapter),"--smoke"],
        ["timeout","1200",py,runner,"transform","--seed","0","--method","U00","--base-root",str(base),"--adapter-root",str(adapter),"--transform-root",str(transforms)],
        ["timeout","1200",py,runner,"transform","--seed","0","--method","F00","--base-root",str(base),"--adapter-root",str(adapter),"--transform-root",str(transforms)],
    ]
    rows=[]
    for command in commands:
        started=time.perf_counter(); rc=subprocess.run(command,cwd=REPO,env=env).returncode
        rows.append({"command":" ".join(command[2:5]),"returncode":rc,"seconds":time.perf_counter()-started})
        if rc: raise RuntimeError("P0 smoke command failed")
    result={"status":"PASS","scientific_training_units":0,"formal_transforms":0,
            "label_access":False,"rows":rows,"U00_and_F00_transform_smoke":True}
    atomic_json(OUT/"p0_smoke_audit.json",result); return result


def main():
    started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(OUT/"p0_attempt1_infrastructure_correction.json", {
        "status":"CORRECTED_BEFORE_FORMAL_SCIENCE",
        "failure":"Scanpy 1.9.1 highly_variable boolean marked 2001 genes although only ranks 0..1999 were finite",
        "correction":"use the unchanged Seurat-v3 ranks and select exactly the finite locked ranks 0..1999",
        "scope":"global_prelabel_infrastructure_correction_1_of_4",
        "scientific_training_units":0,"formal_transforms":0,"Y_values_read":False,
        "failed_attempt_preserved_in_log":"/root/autodl-fs/night8b_raw_runs_20260820/logs/p0.log",
        "parameter_changed":False,"candidate_changed":False,"threshold_changed":False,
    })
    atomic_json(OUT/"p0_attempt2_infrastructure_correction.json", {
        "status":"CORRECTED_BEFORE_FORMAL_SCIENCE",
        "failure":"fresh-process CUDA forward differed by at most 5.960464477539063e-08 while the first implementation incorrectly required bitwise equality",
        "correction":"apply the already-established Night-6C checkpoint view tolerance atol=1e-6, rtol=1e-5 while retaining exact canonical tensor-state SHA",
        "scope":"global_prelabel_infrastructure_correction_2_of_4",
        "scientific_training_units":0,"formal_transforms":0,"Y_values_read":False,
        "failed_attempt_preserved_in_log":"/root/autodl-fs/night8b_raw_runs_20260820/logs/p0_attempt2.log",
        "parameter_changed":False,"candidate_changed":False,"threshold_changed":False,
    })
    atomic_json(OUT/"p0_attempt3_infrastructure_correction.json", {
        "status":"CORRECTED_BEFORE_FORMAL_SCIENCE",
        "failure":"Python 3.8 pathlib.Path.write_text does not accept the newline keyword",
        "correction":"write the identical UTF-8 LF-delimited observation contract through Path.open(newline='\\n')",
        "scope":"global_prelabel_infrastructure_correction_3_of_4",
        "scientific_training_units":0,"formal_transforms":0,"Y_values_read":False,
        "failed_attempt_preserved_in_log":"/root/autodl-fs/night8b_raw_runs_20260820/logs/p0_attempt3.log",
        "parameter_changed":False,"candidate_changed":False,"threshold_changed":False,
    })
    reg=json.loads(REGISTRY.read_text()); checks,files=verify_authority(reg)
    provenance=provenance_and_mapping(reg); caches=build_caches(); parity=p22_code_parity(); gpu=cuda_probe()
    smoke_result=smoke()
    environment={"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
                 "sklearn":sklearn.__version__,"torch":torch.__version__,"cpu_count":os.cpu_count(),
                 "memory_kib":int(dict(line.split(":",1) for line in Path('/proc/meminfo').read_text().splitlines())["MemTotal"].split()[0])}
    audit={"status":"P0_PASS","authority":checks,"official_files":files,"provenance":provenance,
           "cache_lock":caches,"p22_code_parity":parity,"cuda":gpu,"smoke":smoke_result,
           "environment":environment,"label_access":False,"autodl_api_called":False,
           "elapsed_seconds":time.perf_counter()-started}
    atomic_json(OUT/"p0_authority_provenance_deployability.json",audit)
    print(json.dumps({"status":"P0_PASS","cache":caches,"elapsed":audit["elapsed_seconds"]},sort_keys=True))


if __name__=="__main__": main()
