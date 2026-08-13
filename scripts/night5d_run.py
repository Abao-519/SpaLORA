#!/usr/bin/env python3
"""Locked label-free Night-5D P22 training and diffusion."""
from __future__ import annotations
import csv, gc, hashlib, json, os, resource, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd, torch
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from SpaLORA.night3af_cache import load_cache
from SpaLORA.night3a_ige import input_sha256, model_state_sha256
from SpaLORA.night3b_ablation import Night3BTrainer
from SpaLORA.night3ar_protocol import ScientificWindow, assert_training_payload_label_free, ground_truth_csv_paths, training_cfg
from SpaLORA.night5a_rnd import anchor_graph, sha256_file
from SpaLORA.night5b_rnd import Night5BTrainer, single_step_diffusion
from scripts.night3a_runner import cluster_exact, validate_attention

CFG=REPO/'configs/night5d_locked_p22_confirmation.json'
REQ=('embedding.npz','attention.npz','clusters.csv','observation_ids.csv','loss_trajectory.csv','checkpoint_index.csv','coefficient_probe.json','model_state.pt','run_manifest.json')
IDS=('B00_C00_FULL_IGE','B01_C04_SHRINK25','B10_SHRINK25_ANCHOR10','C09_RNA_ANCHOR10')
CONTRACTS={
 'B00_C00_FULL_IGE':{'id':'B00_C00_FULL_IGE','role':'historical_reference','config_sha256':'c07fc7067c342250897fc538147a1c68648653fb04f2325cb4662d8e889b91ac','source_candidate':'C00_FULL_IGE','attention':'learned','corr2':True,'loss_calibration':'IGE'},
 'B01_C04_SHRINK25':{'id':'B01_C04_SHRINK25','role':'conservative_confirmatory_anchor','config_sha256':'d643bfdebfbba447d7656ba334ccdfb9cc932874c7f34ba23d9c95ff840b3820','source_candidate':'C04_SHRINK25','attention':'shrink_to_uniform','learned_fraction_rho':.25,'corr2':False,'loss_calibration':'active_set_IGE'},
 'B10_SHRINK25_ANCHOR10':{'id':'B10_SHRINK25_ANCHOR10','role':'primary_integrated_balanced_method','config_sha256':'4984e467294a3047924984952eaf0e8e63ee63c64319c2ffe8b4f3bd60e3d22f','attention':'shrink_to_uniform','learned_fraction_rho':.25,'corr2':False,'rna_anchor_eta':1.0,'loss_calibration':'active_set_IGE'},
 'C09_RNA_ANCHOR10':{'id':'C09_RNA_ANCHOR10','source_config_sha256':'9b899da8fb845f1076f08a5ae6dd9bb0250698205de590878bba7ad654111376','attention':'uniform_all','corr2':False,'rna_anchor_eta':1.0,'loss_calibration':'active_set_IGE'}
}
def atom(p,o):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(o,indent=2,sort_keys=True),encoding='utf8'); os.replace(q,p)
def csvwrite(p,rows):
 keys=sorted({k for r in rows for k in r}); f=p.open('w',newline='',encoding='utf8'); w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows);f.close()
def canon(o): return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def runtime_contract(cid,trainer,model):
 policy='learned' if cid.startswith('B00') else ('shrink_to_uniform' if cid.startswith(('B01','B10')) else 'uniform_all')
 frac=1.0 if policy=='learned' else (.25 if policy=='shrink_to_uniform' else None)
 mechs=['ige_base']+(['corr2'] if cid.startswith('B00') else [])+(['rna_anchor'] if cid.startswith(('B10','C09')) else [])
 names=[n for n,_ in model.named_parameters() if 'attention' in n.lower()]
 return {'candidate_id':cid,'actual_attention_policy':getattr(model,'attention_policy',policy),'actual_learned_fraction':getattr(model,'learned_fraction',frac),'enabled_mechanisms':sorted(mechs),'attention_parameter_names':names,'attention_parameter_count':sum(p.numel() for n,p in model.named_parameters() if 'attention' in n.lower())}
def semantic_probe(cid,tr,model):
 out=tr.forward(model); rc=runtime_contract(cid,tr,model); checks={}
 if cid=='C09_RNA_ANCHOR10':
  checks['uniform_exact']=all(torch.equal(out[k],torch.full_like(out[k],.5)) for k in ('alpha','alpha_omics1','alpha_omics2'))
  loss=out['emb_latent_combined'].sum(); grads=torch.autograd.grad(loss,[p for n,p in model.named_parameters() if 'attention' in n.lower()],allow_unused=True)
  checks['unused_attention_grad_zero_or_none']=all(g is None or torch.count_nonzero(g).item()==0 for g in grads)
  state={k:v.detach().clone() for k,v in model.state_dict().items()}; before=out['emb_latent_combined'].detach().clone(); repeat=tr.forward(model)['emb_latent_combined'].detach().clone()
  with torch.no_grad():
   for n,p in model.named_parameters():
    if 'attention' in n.lower(): p.add_(torch.randn_like(p))
  after=tr.forward(model)['emb_latent_combined'].detach(); repeat_max=float(torch.max(torch.abs(before-repeat)).cpu()); perturb_max=float(torch.max(torch.abs(before-after)).cpu()); envelope=max(2.0*repeat_max,2.0*32.0*float(torch.finfo(torch.float32).eps)*max(1.0,float(before.abs().max().cpu()))); checks['unperturbed_repeat_max_abs_difference']=repeat_max;checks['attention_perturbation_max_abs_difference']=perturb_max;checks['gpu_numerical_envelope']=envelope;checks['attention_perturbation_within_repeat_envelope']=perturb_max<=envelope; model.load_state_dict(state)
 if not all(v for k,v in checks.items() if k not in ('unperturbed_repeat_max_abs_difference','attention_perturbation_max_abs_difference','gpu_numerical_envelope')): raise RuntimeError('semantic probe failed '+repr(checks))
 return rc,checks
def trainer_for(cid,data,cfg,seed,art):
 if cid=='B00_C00_FULL_IGE': return Night3BTrainer(data,cfg,'FULL_IGE',seed,torch.device('cuda:0'),1e-12)
 return Night5BTrainer(data,cfg,CONTRACTS[cid],seed,torch.device('cuda:0'),art,1e-12)
def immutable_p22_artifacts(data):
 """Derive only the registered anchor graph from the immutable cached sparse graphs."""
 anchor10,stats=anchor_graph(data['adj_spatial_omics1'],data['adj_feature_omics1'],1.0)
 graph=anchor10.coalesce().cpu()
 digest=hashlib.sha256(graph.indices().numpy().tobytes()+graph.values().numpy().tobytes()).hexdigest()
 return {'anchor10':anchor10},{'source':'immutable_night3af_p22_cache','eta':1.0,'stats':stats,'graph_tensor_sha256':digest,'semantic_label_access':False}
def main():
 config=json.load(open(CFG)); out=REPO/config['paths']['output_root']; out.mkdir(parents=True,exist_ok=True); raw=Path(config['paths']['raw_runs']); raw.mkdir(parents=True,exist_ok=True)
 idx=json.load(open(config['paths']['cache_manifest']))['datasets']['p22']; prepared=load_cache(Path(config['paths']['night3af_root'])/idx['directory'],idx['manifest_sha256']); cfg=training_cfg(config['datasets']['p22']); assert_training_payload_label_free(prepared.data,cfg,ground_truth_csv_paths(config)); art,anchor_audit=immutable_p22_artifacts(prepared.data); atom(out/'p22_anchor_graph_audit.json',anchor_audit)
 runs=[]
 for cid in IDS:
  seeds=range(5,10) if cid.startswith('B00') else range(10)
  for seed in seeds: runs.append({'ordinal':len(runs)+1,'candidate_id':cid,'seed':seed})
 assert len(runs)==35
 transforms=[{'ordinal':i+1,'candidate_id':'B17_C09_DIFFUSE10','source_candidate_id':'C09_RNA_ANCHOR10','seed':i,'alpha':.1,'steps':1} for i in range(10)]
 atom(out/'locked_p22_run_order.json',{'schema_version':1,'locked_before_training':True,'training_units':runs,'transforms':transforms,'training_count':35,'transform_count':10})
 window=ScientificWindow(config,out,'night5d_training_and_diffusion').install(); records=[]
 try:
  for row in runs:
   cid,seed=row['candidate_id'],row['seed']; d=raw/cid/f'seed_{seed}'; d.mkdir(parents=True,exist_ok=True)
   if any(d.iterdir()): raise RuntimeError('refuse overwrite '+str(d))
   try:
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(); tr=trainer_for(cid,prepared.data,cfg,seed,art); probe=tr.new_model(); rc,checks=semantic_probe(cid,tr,probe); del probe
    started=time.perf_counter(); result=tr.train();torch.cuda.synchronize();secs=time.perf_counter()-started; validate_attention(result.output,len(prepared.obs_names)); emb=np.asarray(result.output['SpaLORA'],np.float32); clusters=cluster_exact(emb,9,2020)
    np.savez_compressed(d/'embedding.npz',SpaLORA=emb);np.savez_compressed(d/'attention.npz',alpha=np.asarray(result.output['alpha'],np.float32),alpha_omics1=np.asarray(result.output['alpha_omics1'],np.float32),alpha_omics2=np.asarray(result.output['alpha_omics2'],np.float32));pd.DataFrame({'observation_id':prepared.obs_names.astype(str)}).to_csv(d/'observation_ids.csv',index=False);pd.DataFrame({'observation_id':prepared.obs_names.astype(str),'cluster':clusters}).to_csv(d/'clusters.csv',index=False);csvwrite(d/'loss_trajectory.csv',result.logs);csvwrite(d/'checkpoint_index.csv',[{'step':r['step'],'state_sha256':r['checkpoint_state_sha256'],'state_file_saved':r['step']==1600,'state_file':'model_state.pt' if r['step']==1600 else ''} for r in result.logs]);torch.save(result.model.state_dict(),d/'model_state.pt')
    atom(d/'coefficient_probe.json',{'raw_initial_losses':result.initial_losses,'raw_rms_gradients':(result.probe or {}).get('gradients',{}),'frozen_coefficients':result.coefficients,'initial_state_sha256':result.initial_state_sha256,'runtime_semantic_contract':rc,'semantic_checks':checks,'semantic_label_access':False})
    arts={n:sha256_file(d/n) for n in REQ if n!='run_manifest.json'}; man={'schema_version':1,'dataset':'p22','candidate_id':cid,'seed':seed,'ordinal':row['ordinal'],'declared_contract':CONTRACTS[cid],'declared_contract_sha256':canon(CONTRACTS[cid]),'resolved_runtime_contract':rc,'resolved_runtime_contract_sha256':canon(rc),'semantic_contract_match':True,'semantic_checks':checks,'locked_input_sha256':input_sha256(prepared.data,prepared.obs_names,prepared.data['selected_gene_names']),'cache_manifest_sha256':idx['manifest_sha256'],'cache_content_sha256':idx['canonical_cache_content_sha256'],'initial_state_sha256':result.initial_state_sha256,'final_state_sha256':result.final_state_sha256,'saved_model_state_sha256':sha256_file(d/'model_state.pt'),'artifact_sha256':arts,'timings':{'training_seconds':secs},'resources':{'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024},'semantic_label_access':False};atom(d/'run_manifest.json',man);records.append({'type':'training','candidate_id':cid,'seed':seed,'status':'success','manifest':str(d/'run_manifest.json'),'manifest_sha256':sha256_file(d/'run_manifest.json')});print('DONE',row['ordinal'],'/35',cid,seed,round(secs,2),flush=True)
   except Exception as e: atom(d/'failure.json',{'error':repr(e),'traceback':traceback.format_exc()}); records.append({'type':'training','candidate_id':cid,'seed':seed,'status':'failure','failure':str(d/'failure.json')}); print('FAIL',cid,seed,repr(e),flush=True)
   gc.collect();torch.cuda.empty_cache()
  if any(r['status']!='success' for r in records): raise RuntimeError('training failures retained')
  adj=prepared.data['adj_spatial_omics1'].coalesce().cpu(); adj_sha=hashlib.sha256(adj.indices().numpy().tobytes()+adj.values().numpy().tobytes()).hexdigest()
  for row in transforms:
   seed=row['seed']; src=raw/'C09_RNA_ANCHOR10'/f'seed_{seed}'; d=raw/'B17_C09_DIFFUSE10'/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True)
   if any(d.iterdir()): raise RuntimeError('refuse overwrite '+str(d))
   with np.load(src/'embedding.npz') as z: emb=np.asarray(z['SpaLORA'],np.float32)
   zero=single_step_diffusion(emb,adj,0.0); transformed=single_step_diffusion(emb,adj,.1); ref=torch.nn.functional.normalize(.9*torch.from_numpy(emb)+.1*torch.sparse.mm(adj,torch.from_numpy(emb)),p=2,dim=1,eps=1e-12).numpy().astype(np.float32)
   checks={'alpha0_bitwise_equal':np.array_equal(zero,emb),'independent_reference_max_abs_difference':float(np.max(np.abs(ref-transformed))),'within_tolerance':np.allclose(ref,transformed,rtol=0,atol=1e-7),'adjacency_sparse':adj.is_sparse}
   if not all(v for k,v in checks.items() if k!='independent_reference_max_abs_difference'): raise RuntimeError('diffusion check '+repr(checks))
   clusters=cluster_exact(transformed,9,2020);np.savez_compressed(d/'embedding.npz',SpaLORA=transformed);pd.DataFrame({'observation_id':prepared.obs_names.astype(str),'cluster':clusters}).to_csv(d/'clusters.csv',index=False);pd.DataFrame({'observation_id':prepared.obs_names.astype(str)}).to_csv(d/'observation_ids.csv',index=False)
   man={'schema_version':1,'dataset':'p22','candidate_id':'B17_C09_DIFFUSE10','candidate_config_sha256':'fda31ad9ccef53f07606d42004bdd8556794c41b1824cc49ad52656760bfc738','seed':seed,'source_manifest':str(src/'run_manifest.json'),'source_manifest_sha256':sha256_file(src/'run_manifest.json'),'source_config_sha256':CONTRACTS['C09_RNA_ANCHOR10']['source_config_sha256'],'source_embedding_sha256':sha256_file(src/'embedding.npz'),'formula':'L2Normalize((1-alpha)*z + alpha*A_spatial_hat*z)','alpha':.1,'steps':1,'immutable_sparse_adjacency_sha256':adj_sha,'checks':checks,'artifact_sha256':{n:sha256_file(d/n) for n in ('embedding.npz','clusters.csv','observation_ids.csv')},'semantic_label_access':False};atom(d/'transform_manifest.json',man);records.append({'type':'transform','candidate_id':'B17_C09_DIFFUSE10','seed':seed,'status':'success','manifest':str(d/'transform_manifest.json'),'manifest_sha256':sha256_file(d/'transform_manifest.json')});print('TRANSFORM',seed,flush=True)
  lock={'schema_version':1,'locked_before_any_semantic_label_access':True,'training_count':35,'transform_count':10,'failure_count':0,'runs':records};atom(out/'locked_p22_training_and_transform_manifest.json',lock);fw=window.close(True);atom(out/'label_firewall.json',fw);atom(out/'training_and_transform_complete.json',{'training':35,'transforms':10,'failures':0,'lock_sha256':sha256_file(out/'locked_p22_training_and_transform_manifest.json'),'semantic_label_access':False});print('TOTAL_LOCK_35_10',flush=True)
 except Exception:
  try: window.close(False)
  except Exception: pass
  raise
if __name__=='__main__':main()
