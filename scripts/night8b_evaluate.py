#!/usr/bin/env python3
"""Single authorized post-lock MISAR label evaluation and preregistered decision."""
from __future__ import annotations
import itertools,json,subprocess,sys
from pathlib import Path
import h5py,numpy as np,pandas as pd,scipy.sparse as sp
from sklearn.metrics import (adjusted_mutual_info_score,adjusted_rand_score,completeness_score,
    fowlkes_mallows_score,homogeneity_score,normalized_mutual_info_score,v_measure_score)
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from SpaLORA.night1_evaluation import _mean_cluster_moran
from SpaLORA.night3af_cache import load_cache,sha256_file
from SpaLORA.night3b_metrics import mean_one_vs_rest_geary,symmetric_knn_adjacency
from SpaLORA.night6c_pipeline import atomic_json
from SpaLORA.night8b_pipeline import ANN,CACHE,OUT,RAW
TRANSFORM=RAW/'formal/transforms'; SNAP=RAW/'evaluation/authorized_metric_snapshot.npz'

def metric(true,pred,graph):
    r,c=graph.nonzero(); nei=float(np.mean(pred[r]==pred[c])); ari=float(adjusted_rand_score(true,pred)); nmi=float(normalized_mutual_info_score(true,pred))
    geary,_=mean_one_vs_rest_geary(pred,graph)
    return {'ari':ari,'nmi':nmi,'q':(ari+nmi)/2,'ami':float(adjusted_mutual_info_score(true,pred)),
      'fmi':float(fowlkes_mallows_score(true,pred)),'homogeneity':float(homogeneity_score(true,pred)),
      'completeness':float(completeness_score(true,pred)),'v_measure':float(v_measure_score(true,pred)),
      'neighbor_agreement':nei,'moran_i':float(_mean_cluster_moran(pred,graph)),
      'geary_c':float(geary),'boundary_disagreement':1-nei}
def signflip(v):
    obs=float(np.mean(v)); vals=[float(np.mean(v*np.asarray(s))) for s in itertools.product((-1.,1.),repeat=10)]
    count=int(np.sum(np.asarray(vals)>=obs-1e-15)); return {'observed_mean':obs,'enumerations':1024,'tail_count':count,'p_one_sided':count/1024}
def bootstrap(v):
    rng=np.random.default_rng(20260820); idx=rng.integers(0,10,size=(100000,10)); means=v[idx].mean(1); lo,hi=np.percentile(means,[2.5,97.5])
    return {'replicates':100000,'seed':20260820,'mean':float(np.mean(v)),'ci_lower':float(lo),'ci_upper':float(hi)}
def main():
    lock=OUT/'locked_misar_training_and_prediction_manifest.json'; push=OUT/'prelabel_push_audit.json'
    if json.loads(lock.read_text()).get('status')!='TOTAL_LOCKED_BEFORE_LABEL_ACCESS' or json.loads(push.read_text()).get('status')!='PASS': raise RuntimeError('post-lock push gate absent')
    mapping=pd.read_csv(OUT/'prelabel_observation_mapping.csv'); prepared=load_cache(CACHE/'base',sha256_file(CACHE/'base/manifest.json'))
    ids=prepared.obs_names.astype(str).to_numpy()
    if mapping.observation_id.astype(str).tolist()!=ids.tolist(): raise RuntimeError('mapping order drift')
    carrier=ANN/'MISAR_seq_mouse_E15_brain_ATAC_data.h5'
    with h5py.File(carrier,'r') as h: raw=np.asarray(h['Y'][:])
    decoded=np.asarray([x.decode() if isinstance(x,bytes) else str(x) for x in raw]); truth=decoded[mapping.carrier_row.to_numpy(int)]
    if len(truth)!=1949 or len(np.unique(truth))!=12: raise RuntimeError('locked K=12 annotation contract failed')
    graph=symmetric_knn_adjacency(prepared.coordinates,18).tocsr(); rows=[]; predictions={}
    locked=json.loads(lock.read_text()); base={int(x['seed']):x for x in locked['base_runs']}; adapters={int(x['seed']):x for x in locked['adapter_runs']}
    for method in ('U00','F00'):
      for seed in range(10):
        d=TRANSFORM/method/f'seed_{seed}'; table=pd.read_csv(d/'clusters.csv'); pred=table.cluster.to_numpy(np.int64); predictions[f'{method}_{seed}']=pred
        values=metric(truth,pred,graph); b=base[seed]; g04=next(x for x in b['submodels'] if x['graph_id'].startswith('G04')); g00=next(x for x in b['submodels'] if x['graph_id'].startswith('G00'))
        tm=json.loads((d/'transform_manifest.json').read_text()); aw=adapters[seed]['worker_manifest']
        if method=='U00': train_s=float(g04['runtime_seconds']); peak=float(g04['peak_gpu_mib']); adapter_s=0.
        else: train_s=float(g00['runtime_seconds'])+float(g04['runtime_seconds'])+float(aw['runtime_seconds']); peak=max(float(g00['peak_gpu_mib']),float(g04['peak_gpu_mib']),float(aw['peak_gpu_mib'])); adapter_s=float(aw['runtime_seconds'])
        rows.append({'method':method,'seed':seed,**values,'base_training_seconds':train_s-adapter_s,'adapter_increment_seconds':adapter_s,
                     'transform_seconds':float(tm['runtime_seconds']),'end_to_end_seconds':train_s+float(tm['runtime_seconds']),'peak_gpu_mib':peak,
                     'clusters_sha256':sha256_file(d/'clusters.csv')})
    frame=pd.DataFrame(rows).sort_values(['method','seed']); frame.to_csv(OUT/'misar_20row_metrics.csv',index=False)
    paired=[]
    for seed in range(10):
      u=frame[(frame.method=='U00')&(frame.seed==seed)].iloc[0]; f=frame[(frame.method=='F00')&(frame.seed==seed)].iloc[0]
      paired.append({'seed':seed,**{f'delta_{k}':float(f[k]-u[k]) for k in ('ari','nmi','q','neighbor_agreement','moran_i','geary_c','boundary_disagreement','end_to_end_seconds','peak_gpu_mib')}})
    pair=pd.DataFrame(paired); pair.to_csv(OUT/'misar_paired_deltas.csv',index=False); q=pair.delta_q.to_numpy(float)
    stats={'mean_delta_ari':float(pair.delta_ari.mean()),'mean_delta_nmi':float(pair.delta_nmi.mean()),'mean_delta_q':float(q.mean()),
      'q_wins':int((q>0).sum()),'exact_sign_flip':signflip(q),'bootstrap_delta_q':bootstrap(q),'per_seed':paired,
      'inference_note':'10 seeds measure algorithmic stability, not independent biological replication'}
    atomic_json(OUT/'misar_paired_statistics.json',stats)
    spatial={'mean_delta_neighbor':float(pair.delta_neighbor_agreement.mean()),'mean_delta_moran':float(pair.delta_moran_i.mean()),
      'mean_delta_geary':float(pair.delta_geary_c.mean()),'mean_delta_boundary':float(pair.delta_boundary_disagreement.mean())}
    spatial['pass']=spatial['mean_delta_neighbor']>=-.01 and spatial['mean_delta_moran']>=-.02 and spatial['mean_delta_geary']<=.02 and spatial['mean_delta_boundary']<=.01
    atomic_json(OUT/'misar_spatial_protection.json',spatial)
    u=frame[frame.method=='U00']; f=frame[frame.method=='F00']; resource={'U00_mean_end_to_end_seconds':float(u.end_to_end_seconds.mean()),'F00_mean_end_to_end_seconds':float(f.end_to_end_seconds.mean()),
      'runtime_ratio':float(f.end_to_end_seconds.mean()/u.end_to_end_seconds.mean()),'U00_mean_peak_gpu_mib':float(u.peak_gpu_mib.mean()),'F00_mean_peak_gpu_mib':float(f.peak_gpu_mib.mean()),
      'peak_gpu_ratio':float(f.peak_gpu_mib.mean()/u.peak_gpu_mib.mean()),'runtime_ratio_max':1.5,'peak_gpu_ratio_max':1.25}
    resource['pass']=resource['runtime_ratio']<=1.5 and resource['peak_gpu_ratio']<=1.25; atomic_json(OUT/'misar_resource_audit.json',resource)
    SNAP.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(SNAP,truth=truth,g_data=graph.data,g_indices=graph.indices,g_indptr=graph.indptr,g_shape=np.asarray(graph.shape),**predictions)
    independent=OUT/'misar_independent_recalculation.json'; subprocess.run([sys.executable,str(REPO/'scripts/night8b_independent_metrics.py'),'--snapshot',str(SNAP),'--output',str(independent)],check=True,cwd=REPO)
    indep=json.loads(independent.read_text()); diffs=[]
    for row in rows:
      other=next(x for x in indep['rows'] if x['method']==row['method'] and x['seed']==row['seed'])
      for key in ('ari','nmi','q','ami','fmi','homogeneity','completeness','v_measure','neighbor_agreement','moran_i','geary_c','boundary_disagreement'): diffs.append(abs(row[key]-other[key]))
    for row in paired:
      other=next(x for x in indep['paired'] if x['seed']==row['seed'])
      for key in ('ari','nmi','q','neighbor_agreement','moran_i','geary_c','boundary_disagreement'):
        diffs.append(abs(row[f'delta_{key}']-other[f'delta_{key}']))
    maxdiff=float(max(diffs)); indep['maximum_absolute_error_vs_primary']=maxdiff; indep['status']='PASS' if maxdiff<=1e-12 else 'FAIL'; atomic_json(independent,indep)
    if maxdiff>1e-12: terminal='IMPLEMENTATION_SEMANTICS_INVALID'
    else:
      science=(stats['mean_delta_q']>=.01 and stats['mean_delta_ari']>=0 and stats['mean_delta_nmi']>=0 and stats['q_wins']>=8 and stats['exact_sign_flip']['p_one_sided']<.05 and stats['bootstrap_delta_q']['ci_lower']>0 and spatial['pass'])
      terminal=('NIGHT8B_MISAR_FAMILY_POLICY_BALANCED_CONFIRMED' if science and resource['pass'] else 'NIGHT8B_MISAR_FAMILY_POLICY_ACCURACY_CONFIRMED_WITH_COMPLEXITY_COST' if science else 'NIGHT8B_MISAR_PARTIAL_OR_MIXED_EVIDENCE' if stats['mean_delta_q']>0 else 'NIGHT8B_MISAR_FAMILY_POLICY_NOT_GENERALIZED')
    atomic_json(OUT/'label_window_audit.json',{'status':'PASS','one_authorized_window':True,'Y_values_read_once':True,'Y_rows':1949,'K':12,'lock_sha256':sha256_file(lock),'prelabel_push_commit':json.loads(push.read_text())['commit'],'return_to_training_transform_or_clustering':False})
    atomic_json(OUT/'night8b_decision.json',{'terminal_status':terminal,'science_gate_pass':bool(science) if maxdiff<=1e-12 else False,'resource_gate_pass':resource['pass'],'candidate_search':False,'label_opened_post_total_lock_and_push':True,'third_party_benchmark_run':False,'claim_sota':False,'p22_artifacts_as_inputs':False})
    print(json.dumps({'terminal_status':terminal,'mean_delta_q':stats['mean_delta_q'],'q_wins':stats['q_wins'],'independent_max_error':maxdiff},sort_keys=True))
if __name__=='__main__': main()
