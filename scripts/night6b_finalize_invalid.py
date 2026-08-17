#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO = Path('/root/autodl-fs/SpaLORA-night6b')
OUT = REPO / 'outputs/night6b_handoff'
REG_PATH = REPO / 'protocols/night6b/SpaLORA_Night6B_Candidate_Registry_2026-08-17.json'
TERMINAL = 'IMPLEMENTATION_SEMANTICS_INVALID'


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write_json(rel, payload):
    p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8')


def config_sha(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


reg=json.loads(REG_PATH.read_text())
graphs=[{'id':x['id'],'config_sha256':config_sha(x),'resolved':x} for x in reg['graph_candidates']]
heads=[{'id':x['id'],'config_sha256':config_sha(x),'resolved':x} for x in reg['head_candidates']]
assert len(graphs)==9 and len(heads)==12
assert len({x['config_sha256'] for x in graphs})==9
assert len({x['config_sha256'] for x in heads})==12
write_json('candidate_registry_resolved.json',{
 'status':'PARSE_PASS_BUT_EXECUTION_BLOCKED','registry_sha256':sha(REG_PATH),
 'graph_candidates':graphs,'head_candidates':heads,'graph_count':9,'head_count':12,
 'seeds':reg['seeds'],'formal_candidate_outputs_produced':False})

access=[]
for line in (OUT/'firewall/data_steward_access.jsonl').read_text().splitlines():
    if line.strip():access.append(json.loads(line))
write_json('firewall/data_role_and_access_audit.json',{
 'status':'FIREWALL_CLEAN_AT_STOP','terminal_status':TERMINAL,
 'roles':{
  'data_steward':{'deserialized_into_memory':True,'explicitly_indexed_or_observed':True,'used_for_training_or_selection':False,'authorized_role':'data_steward','purpose':'locked ontology and label-free copy'},
  'trainer_transformer':{'deserialized_original_obs_into_memory':False,'explicitly_indexed_or_observed':False,'used_for_training_or_selection':False,'authorized_role':'trainer_transformer','process_started':False},
  'evaluator':{'deserialized_original_obs_into_memory':False,'explicitly_indexed_or_observed':False,'used_for_training_or_selection':False,'authorized_role':'evaluator','process_started':False}},
 'data_steward_access_log':access,
 'protected_dataset_access':{'D1':0,'P22':0,'GSE198353':0,'Night4B':0,'night5d_metric_content':0,'night6a_raw_runs':0},
 'night6a_invalid_checkpoint_embedding_metric_use':False,
 'negative_tests':'9/9 passed',
 'label_firewall_breach':False})

ref=json.loads((OUT/'reference_reuse_and_multiview_parity.json').read_text())
write_json('p0_semantic_contract.json',{
 'status':'FAIL_REQUIRED_A1_REFERENCE_REPLAY','terminal_status':TERMINAL,
 'registry_parse':{'graphs':'9/9','heads':'12/12','unique_graph_config_sha':'9/9','unique_head_config_sha':'12/12'},
 'firewall_negative_tests':'9/9 passed',
 'ontology_contract':'PASS; final_annot K=4',
 'reference_replay':{'status':ref['status'],'valid_checkpoint_available':False,'forward_replay_executed':False,'private_views_derived':False,'artifact_hash_mismatches':len(ref['artifact_hash_mismatches'])},
 'formal_training_started':False,'formal_head_transforms_started':False,
 'failed_gate':'Taskbook 5.2.3/5.2.4 and section 6 require valid Night-5 C04/B01 final checkpoints for read-only forward replay. Night-5A never saved them.',
 'forbidden_workarounds_rejected':['retrain C04/B01','substitute Night-3AF model state','substitute Night-6A invalid model state','claim fused embedding is a loadable checkpoint']})

write_json('graph_cache_manifest_index.json',{'status':'NOT_STARTED_DUE_P0_SEMANTIC_HARD_STOP','entries':[],'entry_count':0,'note':'Label-free source copies were created, but no candidate graph cache or ASR/Moran preprocessing was built after the checkpoint gate failed.'})
for stage in ('r1','r2'):
    write_json(f'{stage}_training_manifest.json',{'stage':stage.upper(),'status':'NOT_STARTED_P0_SEMANTIC_HARD_STOP','locked_before_label_access':True,'planned_units':34 if stage=='r1' else 27,'attempted_units':0,'success_count':0,'failure_count':0,'runs':[]})
    write_json(f'{stage}_transform_manifest.json',{'stage':stage.upper(),'status':'NOT_STARTED_P0_SEMANTIC_HARD_STOP','locked_before_label_access':True,'planned_max_transforms':432 if stage=='r1' else 120,'attempted_transforms':0,'success_count':0,'failure_count':0,'transforms':[]})
write_json('r1_decision.json',{'stage':'R1','status':'NOT_REACHED','advanced_graphs':[],'advanced_heads':[],'reason':'P0 semantic reference replay failed before formal training.'})

for name,header in {
 'per_seed_metrics.csv':['stage','dataset','graph_id','head_id','seed','ari','nmi','q','neighbor_agreement','moran_i','geary_c','boundary_disagreement'],
 'graph_head_five_seed_summary.csv':['graph_id','head_id','macro_delta_q','worst_dataset_delta_q','paired_q_wins','spatial_gate_a1','spatial_gate_tonsil','status'],
 'balanced_and_accuracy_frontiers.csv':['track','graph_id','head_id','macro_delta_q','worst_dataset_delta_q','paired_q_wins','spatial_gate','decision'],
}.items():
    with (OUT/name).open('w',newline='',encoding='utf-8') as f: csv.writer(f).writerow(header)

write_json('night6b_decision.json',{
 'terminal_status':TERMINAL,'candidate_lock_status':'NONE','balanced_candidate':None,'accuracy_frontier_candidate':None,
 'ontology_result':{'status':'PASS','target_column':'final_annot','known_k':4},
 'formal_scientific_training_units':0,'training_retries':0,'head_transforms':0,'transform_corrections':0,
 'scientific_result_available':False,
 'hard_stop_reason':'The required valid Night-5 C04/B01 final checkpoints do not exist because the authoritative Night-5A runner did not save model state. Therefore exact read-only forward replay and private-view derivation cannot be performed.',
 'protected_access':{'D1':0,'P22':0,'GSE198353':0,'Night4B':0},
 'next_authority_needed':'A new planning decision must either preregister a clean C04/B01 baseline retraining as new evidence or provide a genuinely SHA-verified historical checkpoint. This run cannot make that change.'})

write_json('tests_and_invariance_audit.json',{
 'terminal_status':TERMINAL,'firewall_tests':{'passed':9,'failed':0},
 'night5_c04_artifact_hashes':{'checked':30,'mismatches':0},
 'registry':{'graphs':9,'heads':12,'all_config_hashes_unique':True},
 'budgets':{'scientific_training':0,'scientific_training_cap':61,'retries':0,'retry_cap':12,'head_transforms':0,'head_transform_cap':552},
 'fixed_seeds_preserved':[0,1,2,3,4],'seed_search':False,'parameter_search_after_results':False,'label_checkpoint_selection':False,
 'history_mutation':False,'force_push':False,'night6a_invalid_artifact_use':False,
 'ontology_authorization':'Only data_steward read final_annot/lab/lab_lynn/src; known K=4 was exported without per-spot labels.',
 'formal_evaluator_started':False})

report='''# SpaLORA Night-6B report

## Authoritative terminal status

`IMPLEMENTATION_SEMANTICS_INVALID`

Night-6B stopped at P0-SEMANTIC before any formal training or cluster-head transform. The taskbook requires A1 G00/H00 seeds 0-4 to be created by loading the valid historical Night-5 C04/B01 final checkpoints, replaying forward on the valid cache, reproducing fused clusters and ARI/NMI exactly, and deriving the previously unsaved private views.

That historical model state does not exist. All five authoritative C04 seed manifests and their 30 declared artifacts rehashed without mismatch, but each run contains only `embedding.npz`, `attention.npz`, `clusters.csv`, `observation_ids.csv`, `loss_trajectory.csv`, and `coefficient_probe.json`. The Night-5A runner's own required-artifact list excluded model state and contains no `torch.save`. Remote raw storage and the local Night-5A compact/full archives contain no C04/B01 `.pt`, `.pth`, `.ckpt`, `model_final`, or checkpoint artifact. Historical Night-3AF model states are semantically different and were not substituted.

Retraining C04/B01 would be new evidence, not read-only replay, and is explicitly forbidden by this taskbook. Consequently exact private-view derivation is impossible and the hard gate correctly fails.

## Completed valid preflight evidence

- Four authoritative input SHA-256 values matched locally and remotely.
- Night-6A compact delivery index independently verified 16/16; its status remains `IMPLEMENTATION_SEMANTICS_INVALID` and none of its checkpoints, embeddings, or candidate metrics was used.
- Git parent/tag/protection tag matched `7f204a56690768f22bd06e0dac1b5785c97c4c70`.
- The independent data steward verified the official tonsil RNA/ADT source hashes.
- The preregistered `final_annot` column has actual nonmissing `K=4`: connective & epithelial tissue 731, germinal center 183, lymphoid follicle 834, tonsillar parenchyma 2578. RNA and ADT annotations match spot-by-spot.
- New label-free RNA/ADT files contain zero `obs` columns and share exact ordered barcodes.
- Data-steward label access was authorized and explicitly recorded as deserialized and observed, but never used for training or selection.
- The trainer/transformer and evaluator never started. D1, P22, GSE198353, Night-4B, Night-5D metric content, and Night-6A raw runs were not accessed.
- Nine fail-closed firewall tests passed.

## Budget and scientific interpretation

- Formal scientific training: 0/61.
- Training retry attempts: 0/12.
- Head transforms: 0/552.
- Transform corrections: 0/48.
- No ARI/NMI, graph comparison, head comparison, balanced frontier, or accuracy frontier was produced.

This is an infrastructure-of-evidence/implementation-contract failure, not a negative scientific result. It cannot be reported as `NO_GRAPH_OR_CLUSTER_RESCUE_CANDIDATE`.

## Required next decision

A future authority document must choose one of two auditable routes: provide genuinely SHA-verified historical C04/B01 checkpoints, or preregister a clean baseline retraining and count it as new Night-6B evidence. The present taskbook does not authorize either substitution, so this run stops without guessing.
'''
(OUT/'night6b_report.md').write_text(report,encoding='utf-8')

required=[
 'night6b_report.md','ontology/tonsil_ontology_contract.json','firewall/data_role_and_access_audit.json','firewall/source_to_label_free_manifest.json',
 'p0_protect_audit.json','p0_semantic_contract.json','candidate_registry_resolved.json','graph_cache_manifest_index.json','reference_reuse_and_multiview_parity.json',
 'r1_training_manifest.json','r1_transform_manifest.json','r1_decision.json','r2_training_manifest.json','r2_transform_manifest.json','per_seed_metrics.csv',
 'graph_head_five_seed_summary.csv','balanced_and_accuracy_frontiers.csv','night6b_decision.json','tests_and_invariance_audit.json','night5a_checkpoint_local_archive_audit.json',
 'failures/p0_protect_attempt1.json']
files=[]
for rel in required:
    p=OUT/rel
    files.append({'path':rel,'size_bytes':p.stat().st_size,'sha256':sha(p)})
for p in [REPO/'SpaLORA/night6b_firewall.py',REPO/'scripts/night6b_p0_protect.py',REPO/'scripts/night6b_data_steward.py',REPO/'scripts/night6b_reference_reuse_audit.py',REPO/'scripts/night6b_finalize_invalid.py',REPO/'tests/test_night6b_firewall.py']:
    files.append({'path':str(p.relative_to(REPO)),'size_bytes':p.stat().st_size,'sha256':sha(p),'root':'repo'})
for p in sorted((REPO/'protocols/night6b').iterdir()):
    files.append({'path':str(p.relative_to(REPO)),'size_bytes':p.stat().st_size,'sha256':sha(p),'root':'repo'})
write_json('delivery_index.json',{
 'schema':'non-self-referential-v1','terminal_status':TERMINAL,'branch':'revision/q2-night6b-graph-affinity-rescue-20260817',
 'planned_final_tag':'night6b-final-20260817','files':files,'internal_output_root':'outputs/night6b_handoff','repo_root_marker':'root=repo'})
print(json.dumps({'status':TERMINAL,'indexed_files':len(files),'formal_training':0,'head_transforms':0},sort_keys=True))
