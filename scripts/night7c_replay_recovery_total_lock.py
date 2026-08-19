#!/usr/bin/env python3
"""Create the pre-label total lock after Stage T and Stage W."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
OUT=REPO/"outputs/night7c_replay_recovery_handoff"
from SpaLORA.night7a_consensus import atomic_json, sha256_file


def main():
    files=["p0_recovery_authority.json","p1a_replay_portability_contract.json","p1b_feature_freeze_contract.json",
           "routing_feature_manifest.csv","p2_runtime_contract.json","routing_transform_manifest.json",
           "routing_weights_manifest.csv","weighted_mnn_training_manifest.json","weighted_mnn_transform_manifest.json",
           "stagew_immediate_semantic_triage.json","stagew_immediate_triage_affinity_manifest.json",
           "stagew_immediate_semantic_triage_cells.csv","stagew_resource_bounded_plan_and_eligibility.json",
           "stagew_worker_threading_resource_disclosure.json"]
    rows=[]
    for name in files:
        path=OUT/name
        if not path.is_file(): raise RuntimeError("total-lock input missing: "+name)
        rows.append({"path":name,"sha256":sha256_file(path),"size_bytes":path.stat().st_size})
    t=json.loads((OUT/"routing_transform_manifest.json").read_text())
    wt=json.loads((OUT/"weighted_mnn_training_manifest.json").read_text())
    wx=json.loads((OUT/"weighted_mnn_transform_manifest.json").read_text())
    triage=json.loads((OUT/"stagew_immediate_semantic_triage.json").read_text())
    plan=json.loads((OUT/"stagew_resource_bounded_plan_and_eligibility.json").read_text())
    if len(t["transforms"])!=240 or len(wt["training_cells"])!=48 or len(wx["transforms"])!=48:
        raise RuntimeError("total-lock cardinality mismatch")
    if any(x.get("label_access") for x in t["transforms"]+wt["training_cells"]+wx["transforms"]):
        raise RuntimeError("pre-lock label access detected")
    if triage.get("status")!="PASS_W01_W05_BOUNDED_CONTINUATION_AUTHORIZED" or triage.get("label_access") is not False:
        raise RuntimeError("Stage-W semantic triage authority mismatch")
    if plan.get("status")!="LOCKED_PRE_LABEL" or plan.get("label_access") is not False or len(plan.get("all_cells",[]))!=48:
        raise RuntimeError("Stage-W resource plan mismatch")
    expected_counts={"RESOURCE_CENSORED_USER_STOP_AFTER_EXTREME_LONGTAIL":1,
                     "SKIPPED_CANDIDATE_RESOURCE_CIRCUIT_BREAKER":7,"success":40}
    if wx.get("status_counts")!=expected_counts:
        raise RuntimeError("Stage-W resource outcomes changed")
    expected_eligible=["W01_QUALITY_SOFT","W02_CONFLICT_RANK","W03_QUALITY_CONFLICT",
                       "W04_QUALITY_SHARED","W05_QUALITY_CONFLICT_SHARED"]
    if plan.get("eligible_weighted_mnn_candidates")!=expected_eligible or wx.get("eligible_weighted_mnn_candidates")!=expected_eligible:
        raise RuntimeError("Stage-W eligible candidate set mismatch")
    lock={"schema_version":1,"status":"LOCKED_PRE_LABEL","label_access":False,"label_window_authorized":True,
          "routing_transform_attempts":240,"weighted_training_attempts":48,"weighted_transform_attempts":48,
          "weighted_physical_success":40,"weighted_resource_censored":1,"weighted_resource_skipped":7,
          "eligible_weighted_mnn_candidates":expected_eligible,
          "excluded_weighted_mnn_candidates":[{"candidate_id":"W00_FILTER75","reason":"INELIGIBLE_RESOURCE_CENSORED"}],
          "scientific_retry":0,"fallback_count":sum(x.get("fallback",False) for x in t["transforms"]+wt["training_cells"]+wx["transforms"]),
          "files":rows}
    atomic_json(OUT/"total_prelabel_lock.json",lock)
    print(json.dumps(lock,sort_keys=True))


if __name__=="__main__": main()
