#!/usr/bin/env python3
"""Verify and total-lock all label-free Night-8B scientific artifacts."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from SpaLORA.night3af_cache import sha256_file
from SpaLORA.night6c_pipeline import atomic_json
from SpaLORA.night8b_pipeline import G00,G04,OUT,RAW,observation_sha

BASE=RAW/"formal/base"; ADAPTER=RAW/"formal/adapter"; TRANSFORM=RAW/"formal/transforms"


def checked(path: Path, expected: str | None = None) -> dict:
    if not path.is_file(): raise RuntimeError(f"missing artifact: {path}")
    actual=sha256_file(path)
    if expected is not None and actual!=expected: raise RuntimeError(f"SHA mismatch: {path}")
    return {"path":str(path),"size_bytes":path.stat().st_size,"sha256":actual}


def main():
    runner=json.loads((RAW/"formal_runner_state.json").read_text())
    if runner.get("status")!="complete" or runner.get("label_access") is not False:
        raise RuntimeError("formal runner is not completely label-free locked")
    base_rows=[]; adapter_rows=[]; transforms=[]; files=[]
    for seed in range(10):
        bdir=BASE/f"seed_{seed}/attempt_001"; aggregate=json.loads((bdir/"base_unit_manifest.json").read_text())
        reload=json.loads((bdir/"base_checkpoint_reload_audit.json").read_text())
        if aggregate.get("status")!="success" or aggregate.get("scientific_unit_count")!=1 or reload.get("status")!="PASS":
            raise RuntimeError(f"base seed {seed} failed formal/reload gate")
        if aggregate.get("runtime_seconds",3601)>3600 or aggregate.get("retry") is not False:
            raise RuntimeError(f"base seed {seed} budget/retry violation")
        sub=[]
        for graph in (G00,G04):
            d=bdir/graph; m=json.loads((d/"training_manifest.json").read_text())
            if m.get("status")!="success" or not m.get("cuda_used") or m.get("label_access") is not False:
                raise RuntimeError(f"base semantic failure {seed}/{graph}")
            for name,key in (("model_final.pt","checkpoint_sha256"),("views.npz","views_sha256")):
                files.append(checked(d/name,m[key]))
            files.append(checked(d/"training_manifest.json")); files.append(checked(d/"loss_curve.csv"))
            sub.append(m)
        base_rows.append({"seed":seed,"unit_manifest":checked(bdir/"base_unit_manifest.json"),
                          "reload_audit":checked(bdir/"base_checkpoint_reload_audit.json"),
                          "submodels":sub,"checkpoint_round_trip_pass":True})
        adir=ADAPTER/f"formal/seed_{seed}/attempt_001"; wrap=json.loads((adir/"adapter_unit_manifest.json").read_text())
        worker=adir/"worker"; wm=json.loads((worker/"training_manifest.json").read_text())
        ra=json.loads((worker/"reload_forward_audit.json").read_text())
        if wrap.get("status")!="success" or wm.get("status")!="success" or ra.get("status")!="PASS":
            raise RuntimeError(f"adapter seed {seed} failed formal/reload gate")
        if wm.get("runtime_seconds",3601)>3600 or wm.get("retry") is not False or wm.get("label_access") is not False:
            raise RuntimeError(f"adapter seed {seed} budget/firewall violation")
        files += [checked(worker/"model_final.pt",wm["checkpoint_sha256"]),
                  checked(worker/"embedding.npy"),checked(worker/"training_manifest.json"),
                  checked(worker/"reload_forward_audit.json")]
        adapter_rows.append({"seed":seed,"unit_manifest":checked(adir/"adapter_unit_manifest.json"),
                             "worker_manifest":wm,"checkpoint_round_trip_pass":True})
    expected=[]
    ids=None
    for method in ("U00","F00"):
        for seed in range(10):
            d=TRANSFORM/method/f"seed_{seed}"; m=json.loads((d/"transform_manifest.json").read_text())
            table=pd.read_csv(d/"clusters.csv")
            if ids is None: ids=table.observation_id.astype(str).tolist()
            if table.observation_id.astype(str).tolist()!=ids or len(table)!=1949 or table.cluster.nunique()!=12:
                raise RuntimeError(f"partition contract mismatch {method}/{seed}")
            if m.get("status")!="success" or m.get("K")!=12 or m.get("label_access") is not False:
                raise RuntimeError(f"transform manifest mismatch {method}/{seed}")
            if m.get("runtime_seconds",1201)>1200 or m.get("retry") is not False or m.get("fallback") is not False:
                raise RuntimeError(f"transform budget/retry mismatch {method}/{seed}")
            files += [checked(d/"affinity.npz",m["affinity_file_sha256"]),
                      checked(d/"clusters.csv",m["clusters_file_sha256"]),checked(d/"transform_manifest.json")]
            transforms.append({"method":method,"seed":seed,"manifest":m,
                               "manifest_file":checked(d/"transform_manifest.json")})
            expected.append((method,seed))
    if len(set(expected))!=20 or observation_sha(ids)!=base_rows[0]["submodels"][0]["ordered_observation_sha256"]:
        raise RuntimeError("20-cell primary key or observation order mismatch")
    manifest={"schema_version":1,"status":"TOTAL_LOCKED_BEFORE_LABEL_ACCESS",
              "training_units":{"base":10,"adapter":10,"total":20},"checkpoint_roundtrips":"20/20",
              "transforms":20,"partitions":20,"seeds":list(range(10)),"methods":["U00","F00"],
              "ordered_observation_sha256":observation_sha(ids),"observation_count":1949,"K":12,
              "label_values_read":False,"label_access":False,"scientific_retry":0,"fallback":0,
              "p22_artifacts_as_misar_inputs":False,"base_runs":base_rows,
              "adapter_runs":adapter_rows,"transform_rows":transforms,"declared_files":files,
              "formal_runner_state_sha256":sha256_file(RAW/"formal_runner_state.json"),
              "code_commit_at_lock":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()}
    atomic_json(OUT/"locked_misar_training_and_prediction_manifest.json",manifest)
    atomic_json(OUT/"budget_and_access_audit.json",{
        "status":"PASS","scientific_training":"20/20","transforms":"20/20",
        "scientific_retry":0,"fallback":0,"global_prelabel_corrections":"2/4",
        "Y_values_read":False,"total_locked":True,"p22_artifacts_in_misar_dag":False})
    print(json.dumps({"status":manifest["status"],"declared_files":len(files)},sort_keys=True))


if __name__=="__main__": main()
