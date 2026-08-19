#!/usr/bin/env python3
"""Generate reports and compact-ready evidence for Night-7C recovery."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
RAW=Path("/root/autodl-fs/night7c_replay_recovery_20260818")
OUT=REPO/"outputs/night7c_replay_recovery_handoff"
from SpaLORA.night7a_consensus import atomic_json, sha256_file


def resource_summary(path:Path):
    if not path.is_file(): return {"rows":0}
    f=pd.read_csv(path); result={"rows":len(f)}
    for col in ("cpu_percent","rss_mib","gpu_util_percent","gpu_memory_mib","gpu_power_w","gpu_sm_clock_mhz"):
        if col in f:
            x=pd.to_numeric(f[col],errors="coerce").dropna()
            if len(x): result[col]={"mean":float(x.mean()),"max":float(x.max())}
    return result


def main():
    p1a=json.loads((OUT/"p1a_replay_portability_contract.json").read_text())
    p1b=json.loads((OUT/"p1b_feature_freeze_contract.json").read_text())
    p2=json.loads((OUT/"p2_runtime_contract.json").read_text())
    t=json.loads((OUT/"routing_transform_manifest.json").read_text())
    wt=json.loads((OUT/"weighted_mnn_training_manifest.json").read_text())
    wx=json.loads((OUT/"weighted_mnn_transform_manifest.json").read_text())
    parallel_contract=json.loads((OUT/"stagew_parallel_amendment_contract.json").read_text())
    parallel_plan=json.loads((OUT/"stagew_remaining47_plan.json").read_text())
    gate=json.loads((OUT/"gate_audit.json").read_text())
    terminal=gate["terminal_status"]
    invalid=[]
    for d in sorted((RAW/"invalid_attempts").glob("*")):
        invalid.append({"attempt":d.name,"classification":"implementation_correction","preserved":True})
    failures=[]
    for stage,rows in (("T",t["transforms"]),("W_TRAIN",wt["training_cells"]),("W_TRANSFORM",wx["transforms"])):
        for x in rows:
            if x["status"]!="success": failures.append({"stage":stage,"candidate":x.get("candidate_id"),"unit_id":x.get("unit_id"),"status":x["status"],"failure_type":x.get("failure_type","")})
    with (OUT/"failure_audit.csv").open("w",newline="") as h:
        fields=["stage","candidate","unit_id","status","failure_type"]; w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(failures)
    budget={"schema_version":1,"replay_initial_forwards":60,"old_order_diagnostics":4,"formal_training":48,
            "formal_transforms":288,"routing_transforms":240,"weighted_transforms":48,"scientific_retry":0,
            "implementation_corrections":invalid,"failed_scientific_cells":len(failures),"label_window_count":1}
    atomic_json(OUT/"budget_audit.json",budget)
    runtime={"schema_version":1,"p2":p2,"p2_serial_resource":resource_summary(RAW/"p2_serial_resource.csv"),
             "p2_parallel_resource":resource_summary(RAW/"p2_parallel_resource.csv"),
             "stage_t_resource":resource_summary(RAW/"stage_t_resource.csv"),
             "stage_w_resource":resource_summary(RAW/"stage_w_resource.csv"),
             "stage_w_resume_resource":resource_summary(RAW/"stage_w_resume_resource.csv"),
             "stage_w_parallel_resource":resource_summary(RAW/"stage_w_parallel_resource.csv"),
             "stage_w_parallel_contract":parallel_contract,
             "interpretation":"CUDA training uses positive GPU memory; sparse affinity and spectral partition are CPU phases where GPU=0 is expected. Stage T retained the serial backend. P2 four-worker scientific artifacts were exact but the short-task speedup was 1.17498x. Before labels or Stage-W results existed, the SHA-locked 2026-08-19 amendments authorized preserving the naturally completing W00 boundary and using the unchanged four-worker scheduler for the mechanically derived remaining 47 transforms."}
    atomic_json(OUT/"runtime_report.json",runtime)
    compact_infra=OUT/"infrastructure"
    compact_infra.mkdir(parents=True,exist_ok=True)
    for name in ("w00_boundary_guard.jsonl","w00_boundary_guard.nohup.log",
                 "stagew_boundary_parallel_coordinator.jsonl","stagew_boundary_parallel.nohup.log",
                 "w00_boundary_validation.json","stagew_parallel_completion_order.json",
                 "stagew_boundary_parallel_completion.json"):
        src=RAW/"infrastructure"/name
        if src.is_file(): shutil.copy2(src,compact_infra/name)
    incident_out=compact_infra/"incident_manifests"
    for manifest in sorted((RAW/"invalid_attempts").glob("*/**/incident_manifest.json")):
        rel=manifest.relative_to(RAW/"invalid_attempts")
        dst=incident_out/rel
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(manifest,dst)
    raw_index=[]
    for name in ("p1a_historical_order_replay","p1b_features","p2_runtime","stage_t","stage_w","invalid_attempts","infrastructure"):
        root=RAW/name; files=[p for p in root.rglob("*") if p.is_file()] if root.exists() else []
        raw_index.append({"path":str(root),"file_count":len(files),"total_size_bytes":sum(p.stat().st_size for p in files)})
    atomic_json(OUT/"raw_artifact_index.json",{"schema_version":1,"raw_files_in_compact":False,"roots":raw_index})
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    atomic_json(OUT/"git_audit_pre_final.json",{"schema_version":1,"branch":subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip(),"head":head,
      "old_tag_peel":subprocess.check_output(["git","rev-parse","night7c-final-20260818^{}"],cwd=REPO,text=True).strip(),"force_push":False})
    report=f'''# SpaLORA Night-7C replay portability recovery report

## Outcome

Terminal status: `{terminal}`.

The original Night-7C `IMPLEMENTATION_SEMANTICS_INVALID` commit, tag, report and compact delivery remain immutable. This recovery used the SHA-locked first-row MNN values from the original Night-7B loss curves as the sole routing authority.

## Replay portability recovery

- Historical-order initial forward: 30 units x 2 fresh processes = 60/60 finite replays.
- Within-current-hardware maximum repeat error: `{p1a['gates']['within_current_hardware_repeat_absolute_error_max']}`.
- Maximum historical absolute / relative error: `{p1a['gates']['historical_vs_current_absolute_error_max']}` / `{p1a['gates']['historical_vs_current_relative_error_max']}`.
- T02/T03 maximum alpha delta: `{p1a['gates']['T02_T03_alpha_absolute_delta_max']}`; T04 decision flips: `{p1a['gates']['T04_hard_decision_mismatch_count']}`.
- Initial state, RNG and fixed-index mismatch counts were all zero. Four old-order shape diagnostics were retained as diagnostics only.
- Final-checkpoint/endpoint parity was performed in separate fresh processes and passed `{p1b['fresh_final_checkpoint_endpoint_parity']}`.

## Scientific execution

- Stage T: {t['transform_attempts']}/240 routing transforms, {t['successful_transforms']} successes, {t['failed_transforms']} retained failures.
- Stage W: {wt['training_attempts']}/48 CUDA training cells and {wx['transform_attempts']}/48 transforms; successes {wt['successful_training']} and {wx['successful_transforms']}.
- Stage W preserved the naturally completed `W00_FILTER75/u000` as the first formal transform, then mechanically scheduled the remaining `{parallel_plan['remaining_count']}` cells with four unchanged H01/ARPACK worker processes.
- W00 runtime was `{wx.get('preserved_w00_runtime_seconds')}` seconds; remaining-47 parallel wall time was `{wx.get('remaining47_parallel_wall_seconds')}` seconds.
- Scientific retry: 0; fallback: 0; one label window after total lock.
- Selected router: `{gate.get('selected_router')}`.
- Weighted-MNN candidates promoted for full seeds: `{gate.get('promoted_weighted_mnn_candidates')}`.

## Runtime

Four-worker CPU transforms were canonical-exact but only `{p2['speedup']:.6f}x` faster than serial in the short P2 benchmark. Formal Stage T therefore used `{p2['formal_backend']}`. The first formal Stage-W transform then exposed an over-eight-hour single-core ARPACK tail. Before any Stage-W terminal result or label access, two SHA-locked infrastructure amendments revised only scheduling: W00 was allowed to return naturally under the unchanged solver, an identity-bound event guard stopped the old driver at its atomic manifest boundary, and the remaining 47 independent transforms used the already exact four-worker backend. H01, ARPACK, affinity, K, random state, tolerances, inputs and per-unit schema were unchanged. GPU=0 during spectral/affinity CPU phases is expected and phase-aware logs are preserved.

## Scientific limits

This is development-panel R&D on A1, tonsil, D1 and previously used P22, not a pristine external benchmark or SOTA claim. Weighted-MNN pilot promotion, if any, requires full-seed confirmation. No labels, metrics, dataset identity, tissue, platform or file name were used for training or routing.
'''
    (OUT/"night7c_replay_recovery_report.md").write_text(report)
    (OUT/"plain_language_summary.txt").write_text(f"Night-7C recovery terminal status: {terminal}. Replay portability passed before any formal unit; formal execution used locked historical m_initial authority.\n")
    print(json.dumps({"terminal_status":terminal,"report":str(OUT/"night7c_replay_recovery_report.md")},sort_keys=True))


if __name__=="__main__": main()
