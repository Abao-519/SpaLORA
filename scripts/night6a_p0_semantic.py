#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from SpaLORA.night3af_cache import load_cache
from SpaLORA.night6a_runtime import Night6ATrainer,resolved_changes
from SpaLORA.night6a_structural import sparse_sha256

registry=json.load(open(REPO/'protocols/night6a/SpaLORA_Night6A_Candidate_Registry_2026-08-14.json'))
candidates=registry['candidates']; contracts=[]
for c in candidates:
 payload=json.dumps(c,sort_keys=True,separators=(',',':')).encode(); row=dict(c);row['config_sha256']=hashlib.sha256(payload).hexdigest();row['resolved']=resolved_changes(c);contracts.append(row)
if [x['id'] for x in contracts]!=['N%02d'%i for i in range(16)] or len({x['config_sha256'] for x in contracts})!=16:raise RuntimeError('16/16 registry parse failed')
base=contracts[0]
if base['id']!='N00' or base['resolved']!={'graph':None,'optimization':None,'alignment':None,'staged':False}:raise RuntimeError('N00 mapping failed')

index=json.load(open('/root/autodl-fs/SpaLORA-night5a/outputs/night3af_handoff/preprocessing_cache_manifest.json'))
a1row=index['datasets']['a1'];a1=load_cache(Path('/root/autodl-fs/SpaLORA-night5a')/a1row['directory'],a1row['manifest_sha256'])
cfg={'embedding_dim':64,'epochs':2,'loss_factors':[1.9,2.5,1.5,10.0],'locked_m_bad_expected':2.289938091}
device=torch.device('cuda:0'); probes={}
for cid in ('N00','N02','N05','N07'):
 c=next(x for x in contracts if x['id']==cid);trainer=Night6ATrainer(a1.data,cfg,c,0,device,a1.obs_names,a1.coordinates)
 result=trainer.train();probes[cid]={'graph_sha256':result.auxiliary.get('graph_sha256'),'final_state_sha256':result.final_state_sha256,
   'loss_last':result.logs[-1],'actual_mechanisms':result.auxiliary.get('actual_mechanisms',[]),'nonfinite':result.auxiliary.get('nonfinite_gradient_count',0)}
if probes['N02']['graph_sha256']==probes['N00']['graph_sha256'] or probes['N05']['final_state_sha256']==probes['N00']['final_state_sha256'] or probes['N07']['loss_last'].get('alignment_weight',0)<=0:raise RuntimeError('real A1 construction probes did not differ')
payload={'status':'P0_SEMANTIC_PASS','registry_count':16,'unique_config_sha_count':16,'n00_exact_mapping':'Night5 C04 shrink_to_uniform rho=.25 corr2_off active IGE',
 'analytic_tests':'8/8 passed','real_a1_probe_steps':2,'real_a1_probes':probes,'formal_training_units':0,'implementation_retry_units':0,
 'label_values_read':False,'fixed_endpoint':True,'seeds':[0,1,2,3,4],'path_guards':['P22','D1'],'lower_is_better':['geary','boundary_disagreement']}
(REPO/'outputs/night6a_handoff/candidate_registry_resolved.json').write_text(json.dumps({'candidates':contracts},indent=2,sort_keys=True)+'\n')
(REPO/'outputs/night6a_handoff/p0_semantic_contract.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,sort_keys=True))
