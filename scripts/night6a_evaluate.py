#!/usr/bin/env python3
import argparse,csv,json,os,sys
from pathlib import Path
import anndata as ad
import numpy as np,pandas as pd
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from SpaLORA.night1_evaluation import evaluate,load_evaluation_labels
from SpaLORA.night3af_cache import load_cache,sha256_file
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary,symmetric_knn_adjacency
OUT=REPO/'outputs/night6a_handoff';RAW=Path('/root/autodl-fs/night6a_raw_runs_20260814')
REG=json.load(open(REPO/'protocols/night6a/SpaLORA_Night6A_Candidate_Registry_2026-08-14.json'));CON={x['id']:x for x in REG['candidates']}
CFG={'a1':{'rna':'/root/autodl-fs/Human lymph node/A1/humanlymphnode_rna.h5ad','ground_truth':'/root/autodl-fs/Human lymph node/A1/A1_groundtruth.csv','ground_truth_id_column':'Barcode','ground_truth_label_column':'manual-anno','ground_truth_id_rule':'strip_s1_prefix','spatial_neighbors':18},
'tonsil':{'rna':'/root/autodl-fs/datasets/human_tonsil_official/section1/s1_adata_rna.h5ad','ground_truth':'obs[final_annot]','ground_truth_label_column':'final_annot','spatial_neighbors':18},
'placenta':{'rna':'/root/autodl-fs/Human placenta architecture/humanplacenta_rna.h5ad','ground_truth':'obs[cell_type]','ground_truth_label_column':'cell_type','spatial_neighbors':1}}
CACHE={'a1':'/root/autodl-fs/SpaLORA-night5a/outputs/night3af_handoff/preprocessing_cache/a1','tonsil':'/root/autodl-fs/night6a_preprocessing_cache_20260814/tonsil','placenta':'/root/autodl-fs/SpaLORA-night5a/outputs/night3af_handoff/preprocessing_cache/placenta'}
MET=['ari','nmi','q','spatial_neighbor_agreement','spatial_cluster_moran_mean','spatial_cluster_geary_mean','boundary_disagreement']
def atom(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def labels(dataset,ids):
 if dataset=='a1':return load_evaluation_labels('a1',CFG['a1'],ids)
 src=ad.read_h5ad(CFG[dataset]['rna'],backed='r');series=src.obs[CFG[dataset]['ground_truth_label_column']].astype(str);mp=pd.Series(series.values,index=src.obs_names.astype(str));src.file.close();v=ids.isin(mp.index);return np.flatnonzero(v),mp.reindex(ids[v]).to_numpy(str)
def spatial_fail(row):return (row['delta_neighbor']<-.03 and row['delta_moran']<-.03) or (row['delta_geary']>.03 and (row['delta_neighbor']<-.03 or row['delta_moran']<-.03))
def evaluate_stage(stage):
 lock=json.load(open(OUT/(stage.lower()+'_training_manifest.json')));assert lock['locked_before_label_access']; rows=[]
 for rec in lock['runs']:
  if rec['status']!='success':continue
  d,c,s=rec['dataset'],rec['candidate_id'],int(rec['seed']);directory=RAW/d/c/('seed_%d'%s);prepared=load_cache(Path(CACHE[d]));ids=pd.Index(pd.read_csv(directory/'observation_ids.csv').observation_id.astype(str));assert ids.equals(prepared.obs_names)
  pos,true=labels(d,ids);pred=pd.read_csv(directory/'clusters.csv').cluster.to_numpy();emb=np.load(directory/'embedding.npz')['SpaLORA'];coords=prepared.coordinates
  m=evaluate(true,pred[pos],pred,emb,coords,CFG[d]['spatial_neighbors']);graph=symmetric_knn_adjacency(coords,CFG[d]['spatial_neighbors']);geary,_=mean_one_vs_rest_geary(pred,graph)
  man=json.load(open(directory/'run_manifest.json'));rows.append({'stage':stage,'dataset':d,'candidate_id':c,'seed':s,'ari':m['ari'],'nmi':m['nmi'],'q':(m['ari']+m['nmi'])/2,
   'spatial_neighbor_agreement':m['spatial_neighbor_agreement'],'spatial_cluster_moran_mean':m['spatial_cluster_moran_mean'],'spatial_cluster_geary_mean':geary,
   'boundary_disagreement':1-m['spatial_neighbor_agreement'],'runtime_seconds':man['runtime_seconds'],'gpu_peak_allocated_mib':man['gpu_peak_allocated_mib'],'run_manifest_sha256':sha256_file(directory/'run_manifest.json')})
 old=[];p=OUT/'per_seed_metrics.csv'
 if p.exists():old=pd.read_csv(p).to_dict('records')
 keyed={(x['dataset'],x['candidate_id'],int(x['seed'])):x for x in old}
 for x in rows:keyed[(x['dataset'],x['candidate_id'],int(x['seed']))]=x
 allrows=sorted(keyed.values(),key=lambda x:(x['dataset'],x['candidate_id'],int(x['seed'])));pd.DataFrame(allrows).to_csv(p,index=False)
 return pd.DataFrame(allrows),lock
def summaries(frame,candidates,datasets,seeds):
 result=[]
 for c in candidates:
  deltas=[]
  for d in datasets:
   for s in seeds:
    x=frame[(frame.dataset==d)&(frame.candidate_id==c)&(frame.seed==s)].iloc[0];r=frame[(frame.dataset==d)&(frame.candidate_id=='N00')&(frame.seed==s)].iloc[0]
    deltas.append({'dataset':d,'seed':s,**{'delta_'+m:float(x[m]-r[m]) for m in MET},'runtime_ratio':float(x.runtime_seconds/r.runtime_seconds),'gpu_ratio':float(x.gpu_peak_allocated_mib/r.gpu_peak_allocated_mib)})
  z=pd.DataFrame(deltas);by={d:float(g.delta_q.mean()) for d,g in z.groupby('dataset')};rec={'candidate_id':c,'family':CON[c]['family'],'delta_ari':float(z.delta_ari.mean()),'delta_nmi':float(z.delta_nmi.mean()),'delta_q':float(z.delta_q.mean()),
   'dataset_delta_q':by,'paired_q_wins':int((z.delta_q>0).sum()),'paired_q_total':len(z),'delta_neighbor':float(z.delta_spatial_neighbor_agreement.mean()),'delta_moran':float(z.delta_spatial_cluster_moran_mean.mean()),'delta_geary':float(z.delta_spatial_cluster_geary_mean.mean()),'delta_boundary':float(z.delta_boundary_disagreement.mean()),
   'runtime_ratio':float(z.runtime_ratio.mean()),'gpu_ratio':float(z.gpu_ratio.max()),'heterogeneous':bool((z.delta_q>0).any() and (z.delta_q<0).any()),'per_cell':deltas};rec['spatial_protection_failed']=spatial_fail(rec);result.append(rec)
 return result
def r1(frame):
 sums=summaries(frame,['N%02d'%i for i in range(1,16)],['a1'],[0,1]);pool=[x for x in sums if x['delta_q']>0 and not x['spatial_protection_failed']]
 selected=[]
 for fam in ('graph','optimization','alignment'):
  rows=sorted([x for x in pool if x['family']==fam],key=lambda x:(-x['delta_q'],x['runtime_ratio'],x['candidate_id']))
  if rows:selected.append(rows[0]['candidate_id'])
 for x in sorted(pool,key=lambda x:(-x['delta_q'],x['runtime_ratio'],x['candidate_id'])):
  if x['candidate_id'] not in selected and len(selected)<6:selected.append(x['candidate_id'])
 for x in sums:x['advanced']=x['candidate_id'] in selected
 return {'stage':'R1','status':'LOCKED','candidate_summaries':sums,'advanced_candidates':selected,'family_rescue_applied':True,'label_access_after_lock':True,'parameter_tuning':False,'seed_search':False,'withheld_access':False}
def r2(frame):
 candidates=[]
 sums=summaries(frame,candidates,['a1','tonsil'],[0,1,2]) if candidates else []
 return {'stage':'R2','status':'REFERENCE_ONLY_NO_R1_CANDIDATE','candidate_summaries':sums,'advanced_candidates':[],
  'q_core_definition':'0.60*Q_A1+0.40*Q_tonsil','label_access_after_lock':True,'parameter_tuning':False,'seed_search':False,'withheld_access':False}
def r3(frame):
 return {'stage':'R3','status':'NO_STRUCTURAL_RESCUE_CANDIDATE','terminal_status':'NO_STRUCTURAL_RESCUE_CANDIDATE',
  'candidate_summaries':[],'locked_candidates':[],'reference_five_seed_complete':True,
  'label_access_after_lock':True,'parameter_tuning':False,'seed_search':False,'withheld_access':False}
def main():
 a=argparse.ArgumentParser();a.add_argument('--stage',required=True);x=a.parse_args();frame,lock=evaluate_stage(x.stage)
 if x.stage=='R1':decision=r1(frame)
 elif x.stage=='R2':decision=r2(frame)
 elif x.stage=='R3':decision=r3(frame)
 atom(OUT/(x.stage.lower()+'_decision.json'),decision);print(json.dumps({'stage':x.stage,'advanced':decision['advanced_candidates'],'summaries':[{k:v for k,v in r.items() if k!='per_cell'} for r in decision['candidate_summaries']]},sort_keys=True))
if __name__=='__main__':main()
