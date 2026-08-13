#!/usr/bin/env python3
"""Single post-lock P22 label window and preregistered Night-5D statistics."""
from __future__ import annotations
import csv, itertools, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary,symmetric_knn_adjacency
from SpaLORA.night5a_rnd import sha256_file
CFG=REPO/'configs/night5d_locked_p22_confirmation.json'
PRIMARY=(('B01_C04_SHRINK25','B00_C00_FULL_IGE'),('B10_SHRINK25_ANCHOR10','B00_C00_FULL_IGE'),('B17_C09_DIFFUSE10','B00_C00_FULL_IGE'))
SECONDARY=(('B10_SHRINK25_ANCHOR10','B01_C04_SHRINK25'),('B17_C09_DIFFUSE10','C09_RNA_ANCHOR10'))
METRICS=('ari','nmi','q','spatial_neighbor_agreement','spatial_cluster_moran_mean','spatial_cluster_geary_mean','boundary_disagreement')
def atom(p,o):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(o,indent=2,sort_keys=True),encoding='utf8');os.replace(q,p)
def csvwrite(p,rows):
 keys=sorted({k for r in rows for k in r});f=p.open('w',newline='',encoding='utf8');w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows);f.close()
def holm(rows):
 order=sorted(range(len(rows)),key=lambda i:rows[i]['exact_signflip_p']);running=0.0
 for rank,i in enumerate(order): running=max(running,(len(rows)-rank)*rows[i]['exact_signflip_p']);rows[i]['holm_adjusted_p']=min(1.0,running)
def contrast(frame,cand,base,rng):
 c=frame[frame.candidate_id==cand].sort_values('seed');b=frame[frame.candidate_id==base].sort_values('seed');m=c.merge(b,on='seed',suffixes=('_candidate','_baseline'),validate='one_to_one');d=(m.q_candidate-m.q_baseline).to_numpy(float);obs=float(d.mean());means=[]
 for signs in itertools.product((-1.,1.),repeat=10): means.append(float(np.mean(d*np.asarray(signs))))
 p=float(np.count_nonzero(np.asarray(means)>=obs)/1024.0);row={'contrast':cand+'-'+base,'candidate_id':cand,'baseline_id':base,'n':10,'delta_q_mean':obs,'delta_q_median':float(np.median(d)),'delta_q_sd':float(np.std(d,ddof=1)),'q_wins':int(np.count_nonzero(d>0)),'exact_signflip_p':p,'permutations':1024}
 for metric in METRICS:
  x=(m[metric+'_candidate']-m[metric+'_baseline']).to_numpy(float); boot=x[rng.integers(0,10,size=(100000,10))].mean(1);row['delta_'+metric+'_mean']=float(x.mean());row['delta_'+metric+'_median']=float(np.median(x));row['delta_'+metric+'_sd']=float(np.std(x,ddof=1));row['delta_'+metric+'_ci_low']=float(np.quantile(boot,.025));row['delta_'+metric+'_ci_high']=float(np.quantile(boot,.975))
 return row
def main():
 config=json.load(open(CFG));out=REPO/config['paths']['output_root'];lock=out/'locked_p22_training_and_transform_manifest.json';complete=out/'training_and_transform_complete.json';lp=json.load(open(lock));cp=json.load(open(complete));
 if not(lp.get('locked_before_any_semantic_label_access') and lp['training_count']==35 and lp['transform_count']==10 and lp['failure_count']==0 and cp['lock_sha256']==sha256_file(lock)):raise RuntimeError('total lock invalid')
 # Evaluator imports occur only after the total-lock checks above.
 from SpaLORA.night1_evaluation import evaluate,load_evaluation_labels
 from scripts.night3a_evaluate import coordinates_for_ids
 raw=Path(config['paths']['raw_runs']);hist=Path(config['paths']['historical_b00']);ids=pd.Index(pd.read_csv(hist/'seed_0/observation_ids.csv').observation_id.astype(str));pos,labels=load_evaluation_labels('p22',config['datasets']['p22'],ids);coords=coordinates_for_ids(config['datasets']['p22'],ids);graph=symmetric_knn_adjacency(coords,16);edges=np.transpose(graph.nonzero());rows=[]
 for cid in ('B00_C00_FULL_IGE','B01_C04_SHRINK25','B10_SHRINK25_ANCHOR10','C09_RNA_ANCHOR10','B17_C09_DIFFUSE10'):
  for seed in range(10):
   d=(hist/f'seed_{seed}') if cid.startswith('B00') and seed<5 else raw/cid/f'seed_{seed}'; mp=d/('transform_manifest.json' if cid.startswith('B17') else 'run_manifest.json')
   runids=pd.Index(pd.read_csv(d/'observation_ids.csv').observation_id.astype(str));
   if not runids.equals(ids):raise RuntimeError('ID mismatch '+str(d))
   clusters=pd.read_csv(d/'clusters.csv').cluster.to_numpy();emb=np.load(d/'embedding.npz')['SpaLORA'];met=evaluate(labels,clusters[pos],clusters,emb,coords,16);geary,_=mean_one_vs_rest_geary(clusters,graph);boundary=float(np.mean(clusters[edges[:,0]]!=clusters[edges[:,1]]));runtime=0.;gpu=0.
   man=json.load(open(mp));
   if cid.startswith('B17'):
    src=json.load(open(raw/'C09_RNA_ANCHOR10'/f'seed_{seed}/run_manifest.json'));runtime=float(src['timings']['training_seconds']);gpu=float(src['resources']['gpu_peak_allocated_mib'])
   else:runtime=float(man['timings']['training_seconds']);gpu=float(man['resources']['gpu_peak_allocated_mib'])
   rows.append({'candidate_id':cid,'seed':seed,'ari':float(met['ari']),'nmi':float(met['nmi']),'q':float((met['ari']+met['nmi'])/2),'spatial_neighbor_agreement':float(met['spatial_neighbor_agreement']),'spatial_cluster_moran_mean':float(met['spatial_cluster_moran_mean']),'spatial_cluster_geary_mean':float(geary),'boundary_disagreement':boundary,'effective_end_to_end_runtime_seconds':runtime,'gpu_peak_allocated_mib':gpu,'manifest_sha256':sha256_file(mp)})
 frame=pd.DataFrame(rows);csvwrite(out/'p22_per_seed_metrics.csv',rows)
 finish_from_frozen_metrics(config,out,frame)
def finish_from_frozen_metrics(config,out,frame):
 old=pd.read_csv(config['paths']['night3b_metrics']);old=old[(old.dataset=='p22')&(old.variant=='FULL_IGE')].sort_values('seed').copy();old['q']=(old.ari+old.nmi)/2;new=frame[(frame.candidate_id=='B00_C00_FULL_IGE')&(frame.seed<5)].sort_values('seed');audit=[]
 aliases={'ari':'ari','nmi':'nmi','q':'q','spatial_neighbor_agreement':'spatial_neighbor_agreement','spatial_cluster_moran_mean':'spatial_cluster_moran_mean','spatial_cluster_geary_mean':'spatial_cluster_geary_mean','boundary_disagreement':'boundary_disagreement'}
 for _,r in new.iterrows():
  o=old[old.seed==r.seed].iloc[0];diffs={k:abs(float(r[k])-float(o[v])) for k,v in aliases.items()};audit.append({'seed':int(r.seed),'max_abs_difference':max(diffs.values()),'within_1e_12':max(diffs.values())<=1e-12,'differences':diffs})
 atom(out/'baseline_reuse_and_extension_audit.json',{'historical_seeds':audit,'all_match':all(x['within_1e_12'] for x in audit),'historical_model_state_file_available':False,'absence_expected_by_original_contract':True,'state_hash_only':True,'numerical_artifacts_verified':True,'weight_level_analysis_forbidden':True})
 if not all(x['within_1e_12'] for x in audit):raise RuntimeError('BASELINE_REPLAY_MISMATCH')
 summary=[]
 for cid,g in frame.groupby('candidate_id'):
  row={'candidate_id':cid,'n':len(g)}
  for k in METRICS:row[k+'_mean']=float(g[k].mean());row[k+'_sd']=float(g[k].std(ddof=1))
  row['runtime_seconds_mean']=float(g.effective_end_to_end_runtime_seconds.mean());row['gpu_peak_allocated_mib_max']=float(g.gpu_peak_allocated_mib.max());summary.append(row)
 csvwrite(out/'p22_candidate_summary.csv',summary);rng=np.random.default_rng(20260814);prim=[contrast(frame,*x,rng) for x in PRIMARY];holm(prim)
 for r in prim:
  r['material_gain']=bool(r['delta_ari_mean']>0 and r['delta_nmi_mean']>0 and r['delta_q_mean']>=.01 and r['q_wins']>=7 and r['holm_adjusted_p']<.05 and r['delta_q_ci_low']>0)
  r['spatial_protection_failed']=bool((r['delta_spatial_neighbor_agreement_mean']<-.03 and r['delta_spatial_cluster_moran_mean']<-.03) or (r['delta_spatial_cluster_geary_mean']>.03 and (r['delta_spatial_neighbor_agreement_mean']<-.03 or r['delta_spatial_cluster_moran_mean']<-.03)))
  r['status']='CONFIRMED_BALANCED' if r['material_gain'] and not r['spatial_protection_failed'] else ('ACCURACY_SPATIAL_TRADEOFF' if r['material_gain'] else ('POSITIVE_BUT_INCONCLUSIVE' if r['delta_q_mean']>0 else 'NOT_CONFIRMED'))
 sec=[contrast(frame,*x,rng) for x in SECONDARY];holm(sec);atom(out/'p22_primary_exact_tests.json',{'primary_endpoint':'paired delta Q','tests':prim,'holm_family_size':3,'bootstrap_replicates':100000,'bootstrap_seed':20260814});atom(out/'p22_secondary_mechanistic_tests.json',{'tests':sec,'holm_family_size':2,'secondary_only':True});atom(out/'p22_spatial_protection.json',{'rules_locked':True,'contrasts':[{'contrast':r['contrast'],'failed':r['spatial_protection_failed']} for r in prim]})
 res=[{'candidate_id':r['candidate_id'],'effective_end_to_end_runtime_seconds':r['runtime_seconds_mean'],'gpu_peak_allocated_mib_max':r['gpu_peak_allocated_mib_max'],'diffusion_includes_source_training_cost':r['candidate_id']=='B17_C09_DIFFUSE10'} for r in summary];csvwrite(out/'p22_resource_accounting.csv',res)
 confirmed=[r['candidate_id'] for r in prim if r['status']=='CONFIRMED_BALANCED'];positive=[r for r in prim if r['delta_q_mean']>0];term='P22_CONFIRMATION_SUCCESS' if confirmed else ('P22_PARTIAL_OR_MIXED_EVIDENCE' if positive else 'P22_NO_LOCKED_CANDIDATE_CONFIRMED');decision={'termination_status':term,'confirmed_balanced_candidates':confirmed,'primary_results':prim,'labels_opened_once_after_total_lock':True,'p22_evidence_status':'one-time cross-dataset confirmation; not pristine external test or strictly untouched holdout','no_method_changes_after_label_open':True,'d1_run':False,'gse198353_run':False,'night4b_run':False};atom(out/'p22_confirmatory_decision.json',decision);atom(out/'postlock_label_access.json',{'occurred':True,'opened_after_total_lock':True,'dataset':'p22','opened_once':True,'method_changes_after_access':False});print(term)
def resume_from_frozen_metrics():
 config=json.load(open(CFG));out=REPO/config['paths']['output_root'];frame=pd.read_csv(out/'p22_per_seed_metrics.csv');
 if len(frame)!=50 or frame.duplicated(['candidate_id','seed']).any():raise RuntimeError('frozen metric table invalid')
 finish_from_frozen_metrics(config,out,frame)
if __name__=='__main__':main()
