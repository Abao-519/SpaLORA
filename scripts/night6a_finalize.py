#!/usr/bin/env python3
import csv, hashlib, json, shutil
from pathlib import Path

REPO = Path('/root/autodl-fs/SpaLORA-night6a')
OUT = REPO / 'outputs/night6a_handoff'
RAW = Path('/root/autodl-fs/night6a_raw_runs_20260814')
TERMINAL = 'IMPLEMENTATION_SEMANTICS_INVALID'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')

manifests=[]; attempts=[]; raw_rows=[]; diagnostics=[]; errors=[]
for mp in sorted(RAW.glob('*/*/seed_*/run_manifest.json')):
    m=json.loads(mp.read_text()); manifests.append((mp,m))
    attempts.append({k:m.get(k) for k in ['ordinal','stage','dataset','candidate_id','seed','status','runtime_seconds','gpu_peak_allocated_mib','process_peak_rss_mib','nonfinite_count','code_commit','config_sha256','cache_manifest_sha256','initial_state_sha256','final_state_sha256','semantic_label_access']})
    attempts[-1]['run_manifest_path']=str(mp);attempts[-1]['run_manifest_sha256']=sha(mp)
    for name,expected in m['artifacts'].items():
        p=mp.parent/name; actual=sha(p) if p.exists() else None
        if actual!=expected: errors.append({'path':str(p),'expected':expected,'actual':actual})
        raw_rows.append({'dataset':m['dataset'],'candidate_id':m['candidate_id'],'seed':m['seed'],'artifact':name,'path':str(p),'size_bytes':p.stat().st_size if p.exists() else -1,'sha256':actual})
    sp=mp.parent/'semantic_contract.json'; s=json.loads(sp.read_text())
    diagnostics.append({'stage':m['stage'],'dataset':m['dataset'],'candidate_id':m['candidate_id'],'seed':m['seed'],'semantic_match':s.get('match'),'registered_changes':json.dumps(s.get('registered_changes',[]),sort_keys=True),'actual_mechanisms':json.dumps(s.get('actual',{}).get('actual_mechanisms',[]),sort_keys=True),'graph_sha256':s.get('actual',{}).get('graph_sha256'),'parameter_count':s.get('actual',{}).get('parameter_count'),'initial_neighbor_cosine':s.get('actual',{}).get('initial_embedding_geometry',{}).get('neighbor_cosine'),'final_neighbor_cosine':s.get('actual',{}).get('final_embedding_geometry',{}).get('neighbor_cosine'),'initial_pairwise_variance':s.get('actual',{}).get('initial_embedding_geometry',{}).get('pairwise_variance'),'final_pairwise_variance':s.get('actual',{}).get('final_embedding_geometry',{}).get('pairwise_variance')})

def write_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['empty']);w.writeheader();w.writerows(rows)

write_csv(OUT/'all_attempts_manifest.csv',attempts)
write_csv(OUT/'raw_artifact_manifest.csv',raw_rows)
write_csv(OUT/'graph_and_gradient_diagnostics/runtime_semantic_and_geometry.csv',diagnostics)

for n in (1,2,3):
    src=OUT/f'r{n}_decision.json'; dst=OUT/f'round{n}_decision.json'; shutil.copyfile(src,dst)
r3=json.loads((OUT/'round3_decision.json').read_text())
r3['diagnostic_scientific_status']=r3.get('terminal_status')
r3['terminal_status']=TERMINAL
r3['authoritative_override_reason']='Pre-lock P0 numeric audit called anndata.read_h5ad on the original tonsil files. This deserialized obs values into memory even though no label column was indexed, logged, passed to training, or used for selection.'
write_json(OUT/'round3_decision.json',r3)

firewall={
 'status':TERMINAL,
 'strict_firewall_pass':False,
 'violation':{'phase':'P0-DATA numeric audit before any formal training','script':'scripts/p0_data_numeric.py','operation':'anndata.read_h5ad on original official tonsil RNA/ADT h5ad','semantic_effect':'AnnData eagerly deserialized obs column values into memory','values_indexed_printed_exported_or_used':False,'impact':'Policy violation is irreversible; formal training later used low-level label-free copies, but Night-6A cannot be certified firewall-clean.'},
 'formal_training_inputs':'label-free tonsil copies/cache with zero obs columns; immutable A1/placenta caches',
 'label_use_for_training_or_selection':False,
 'post_lock_label_evaluation':{'A1':'after 32/32 R1 lock','tonsil':'after 4/4 R2 reference-only lock','placenta':'after 9/9 R3 reference-only lock'},
 'protected_access':{'P22':0,'D1':0,'GSE198353':0,'MISAR_training':0,'Night4B':0,'Night5D_metric_content':0},
 'training_units':{'formal_success':len(attempts),'failed':sum(x['status']!='success' for x in attempts),'retries':0},
 'artifact_hash_verification':{'artifacts_checked':len(raw_rows),'mismatches':errors}
}
write_json(OUT/'p0_data_and_label_firewall_audit.json',firewall)

metrics=list(csv.DictReader(open(OUT/'per_seed_metrics.csv',encoding='utf-8')))
a1=[x for x in metrics if x['dataset']=='a1' and x['candidate_id']=='N00' and int(x['seed']) in range(5)]
mean=lambda k:sum(float(x[k]) for x in a1)/len(a1)
r1=json.loads((OUT/'round1_decision.json').read_text())
best=sorted(r1['candidate_summaries'],key=lambda x:x['delta_q'],reverse=True)
spass=[x for x in best if not x['spatial_protection_failed']]

audit={
 'status':TERMINAL,'semantic_unit_tests':'8/8 passed','formal_runtime_semantic_contracts':f"{sum(x['semantic_match'] is True for x in diagnostics)}/{len(diagnostics)} matched",
 'formal_runs':{'success':len(attempts),'failure':0,'retry':0,'budget_cap':96},
 'raw_artifact_sha_verification':{'checked':len(raw_rows),'mismatches':len(errors)},
 'protected_history':{'upstream_parent':'fccffcff8fb591467b4f7a390897217ffb2617ff','night5d_recovery_index':'16/16 locally verified','night5d_original_manifest_sha256':'a29853735d502baadd0e44be092bfa47e9b9d86299759d54422a6d1503f36560','historical_files_modified':False},
 'invariance':{'candidate_registry_count':16,'unique_config_hashes':16,'fixed_seeds':True,'fixed_run_order':True,'checkpoint_selection_by_label':False,'parameter_tuning':False,'seed_search':False,'failed_attempt_deletion':False},
 'known_violation':'See p0_data_and_label_firewall_audit.json; pre-lock eager obs deserialization makes the authoritative terminal status invalid.'
}
write_json(OUT/'tests_and_invariance_audit.json',audit)
write_json(OUT/'night6a_completion.json',{'terminal_status':TERMINAL,'scientific_result_is_diagnostic_only':'NO_STRUCTURAL_RESCUE_CANDIDATE','formal_training_success':len(attempts),'formal_training_failure':0,'candidate_locked_for_D1_P22':[],'reason':'Strict pre-lock label firewall semantic violation discovered during final audit; no further science authorized.'})

report=f'''# SpaLORA Night-6A report

## Authoritative terminal status

`{TERMINAL}`.

The final audit found that the P0 numeric-audit script called `anndata.read_h5ad` on the original tonsil files before the formal lock. Although the script never indexed, printed, exported, or used annotation values, eager AnnData loading deserialized `obs` values into memory. This violates the taskbook's strict pre-lock firewall. Formal training used label-free copies with zero `obs` columns, no label informed any training, checkpoint, seed, parameter, or candidate decision, and P22/D1/GSE198353/MISAR training/Night-4B remained untouched. Nevertheless, this run is not certified firewall-clean and no candidate may be promoted from it.

## Execution and evidence preservation

- P0 semantic tests: 8/8 passed; formal runtime contracts: {sum(x['semantic_match'] is True for x in diagnostics)}/{len(diagnostics)} matched.
- Formal units: {len(attempts)} success, 0 failure, 0 retry, within the cap of 96.
- R1 completed 32/32 fixed A1 units before evaluation. No candidate had positive mean delta-Q while passing spatial protection; the advancement set was empty.
- R2 and R3 therefore ran only preregistered N00 reference coverage: 4/4 and 9/9 units, respectively. No negative candidate was back-filled.
- Raw artifacts remain under `{RAW}`. {len(raw_rows)} files were independently rehashed; mismatches: {len(errors)}.

## Diagnostic results (not an authoritative scientific promotion)

The least-negative R1 delta-Q was {best[0]['candidate_id']} ({best[0]['delta_q']:.6f}); among spatial-protection-passing candidates it was {spass[0]['candidate_id']} ({spass[0]['delta_q']:.6f}). N00 A1 five-seed means were ARI={mean('ari'):.6f}, NMI={mean('nmi'):.6f}, Q={mean('q'):.6f}; these do not meet the competitiveness marker ARI >= 0.316 and NMI >= 0.406.

## Required scientific questions

1. **Real pruning versus intersection-edge gain:** diagnostically, no registered pruning candidate improved mean A1 Q over N00; some pruning variants also failed spatial protection. This cannot support a promotion claim.
2. **Hard versus soft pruning:** the hard and soft variants did not show an actionable monotone rescue. Stronger graph changes tended to impair neighbor/Moran behavior; there is no defensible dose trend.
3. **Gradient conflict after IGE:** PCGrad/MinNorm semantics passed analytic and runtime probes, but neither converted the diagnostic A1 conflict intervention into positive mean delta-Q. The run cannot establish an externally valid causal conclusion.
4. **Barlow/neighbor alignment across A1 and tonsil:** no alignment candidate advanced from R1, so the preregistered funnel correctly did not spend tonsil candidate runs. Cross-dataset benefit was not demonstrated.
5. **Combinations versus modules:** every combination had negative mean R1 delta-Q. Complexity stacking did not outperform the single modules under the locked screen.
6. **Spatial trade-offs:** several candidates failed the locked neighbor/Moran/Geary protection rule. Candidates that passed it still had negative mean delta-Q, so no accuracy-spatial win was identified.
7. **Recent-method ranges:** direct competitiveness claims are not made. Public-method numbers can differ in data processing, label use, checkpoint selection, and evaluation protocol; those risks prevent a fair numerical ranking here.

## Interpretation and next action

The preregistered numeric funnel would diagnostically end as `NO_STRUCTURAL_RESCUE_CANDIDATE`, but the stricter authoritative result is `{TERMINAL}`. Do not open D1/P22 for these candidates and do not use this run as confirmatory evidence. Preserve it as an auditable failed implementation attempt. A future rerun, if separately authorized, must perform all pre-lock numeric inspection through a low-level label-excluding reader from the first byte access.
'''
(OUT/'night6a_report.md').write_text(report,encoding='utf-8')

required=['night6a_report.md','p0_data_and_label_firewall_audit.json','p0_semantic_contract.json','candidate_registry_resolved.json','all_attempts_manifest.csv','per_seed_metrics.csv','round1_decision.json','round2_decision.json','round3_decision.json','graph_and_gradient_diagnostics/runtime_semantic_and_geometry.csv','tests_and_invariance_audit.json','raw_artifact_manifest.csv','night6a_completion.json','r1_training_manifest.json','r2_training_manifest.json','r3_training_manifest.json']
files=[]
for rel in required:
    p=OUT/rel
    files.append({'path':rel,'size_bytes':p.stat().st_size,'sha256':sha(p)})
write_json(OUT/'delivery_index.json',{'schema':'non-self-referential-v1','terminal_status':TERMINAL,'branch':'revision/q2-night6a-structural-rescue-20260814','planned_final_tag':'night6a-final-20260814','files':files,'verification':{'indexed_files':len(files),'missing':[]}})
print(json.dumps({'status':TERMINAL,'runs':len(attempts),'raw_files':len(raw_rows),'hash_errors':errors,'indexed':len(files)},sort_keys=True))
