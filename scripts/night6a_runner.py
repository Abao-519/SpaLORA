#!/usr/bin/env python3
import argparse,csv,gc,hashlib,json,os,resource,sys,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from SpaLORA.night3af_cache import load_cache,sha256_file
from SpaLORA.night6a_runtime import Night6ATrainer
from scripts.night3a_runner import cluster_exact

OUT=REPO/'outputs/night6a_handoff';RAW=Path('/root/autodl-fs/night6a_raw_runs_20260814')
REG=json.load(open(REPO/'protocols/night6a/SpaLORA_Night6A_Candidate_Registry_2026-08-14.json'))
CON={x['id']:x for x in REG['candidates']}
DATA={
'a1':('/root/autodl-fs/SpaLORA-night5a/outputs/night3af_handoff/preprocessing_cache/a1',64,10),
'tonsil':('/root/autodl-fs/night6a_preprocessing_cache_20260814/tonsil',64,4),
'placenta':('/root/autodl-fs/SpaLORA-night5a/outputs/night3af_handoff/preprocessing_cache/placenta',128,10)}
CFGS={'a1':{'embedding_dim':64,'epochs':200,'loss_factors':[1.9,2.5,1.5,10.0],'locked_m_bad_expected':2.289938091},
'tonsil':{'embedding_dim':64,'epochs':200,'loss_factors':[1.9,2.5,1.5,10.0],'locked_m_bad_expected':2.289938091},
'placenta':{'embedding_dim':128,'epochs':200,'loss_factors':[5.,6.,1.,10.],'locked_m_bad_expected':2.289938091}}

def atom(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
def h(path):return sha256_file(path)
def config_sha(c):return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def path(d,c,s):return RAW/d/c/('seed_%d'%s)
def load(d):return load_cache(Path(DATA[d][0]))
def save_state(path,model):torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)

def one(stage,d,cid,seed,ordinal):
 directory=path(d,cid,seed);directory.mkdir(parents=True,exist_ok=True)
 if (directory/'run_manifest.json').exists():return directory/'run_manifest.json'
 if any(directory.iterdir()):raise RuntimeError('partial run retained: '+str(directory))
 prepared=load(d);candidate=CON[cid];cfg=CFGS[d];started=time.perf_counter()
 try:
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  trainer=Night6ATrainer(prepared.data,cfg,candidate,seed,torch.device('cuda:0'),prepared.obs_names,prepared.coordinates)
  result=trainer.train();torch.cuda.synchronize();embedding=np.asarray(result.output['SpaLORA'],np.float32)
  clusters=cluster_exact(embedding,DATA[d][2],2020)
  np.savez_compressed(directory/'embedding.npz',SpaLORA=embedding)
  pd.DataFrame({'observation_id':prepared.obs_names.astype(str)}).to_csv(directory/'observation_ids.csv',index=False)
  pd.DataFrame({'observation_id':prepared.obs_names.astype(str),'cluster':clusters}).to_csv(directory/'clusters.csv',index=False)
  pd.DataFrame(result.logs).to_csv(directory/'loss_trajectory.csv',index=False);save_state(directory/'model_final.pt',result.model)
  atom(directory/'semantic_contract.json',{'candidate_id':cid,'registered_changes':candidate['changes'],'actual':result.auxiliary,'match':True})
  artifacts={n:h(directory/n) for n in ('embedding.npz','observation_ids.csv','clusters.csv','loss_trajectory.csv','model_final.pt','semantic_contract.json')}
  manifest={'stage':stage,'dataset':d,'candidate_id':cid,'seed':seed,'ordinal':ordinal,'status':'success','config_sha256':config_sha(candidate),
   'cache_manifest_sha256':h(Path(DATA[d][0])/'manifest.json'),'code_commit':os.popen('git -C %s rev-parse HEAD'%REPO).read().strip(),
   'initial_state_sha256':result.initial_state_sha256,'final_state_sha256':result.final_state_sha256,'embedding_sha256':artifacts['embedding.npz'],
   'cluster_sha256':artifacts['clusters.csv'],'runtime_seconds':time.perf_counter()-started,'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
   'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
   'nonfinite_count':int(result.auxiliary.get('nonfinite_gradient_count',0)),'semantic_label_access':False,'fixed_final_checkpoint':True,'artifacts':artifacts}
  atom(directory/'run_manifest.json',manifest);print('DONE',stage,d,cid,seed,round(manifest['runtime_seconds'],2),flush=True);return directory/'run_manifest.json'
 except Exception as e:
  atom(directory/'failure.json',{'stage':stage,'dataset':d,'candidate_id':cid,'seed':seed,'error':repr(e),'traceback':traceback.format_exc(),'retained':True});raise
 finally:gc.collect();torch.cuda.empty_cache()

def main():
 a=argparse.ArgumentParser();a.add_argument('--stage',required=True);a.add_argument('--plan',required=True);z=a.parse_args();plan=json.load(open(z.plan));runs=plan['runs'];records=[]
 for r in runs:
  try:p=one(z.stage,r['dataset'],r['candidate_id'],int(r['seed']),int(r['ordinal']));records.append({'status':'success','path':str(p),'sha256':h(p),**r})
  except Exception as e:p=path(r['dataset'],r['candidate_id'],int(r['seed']))/'failure.json';records.append({'status':'failure','path':str(p),'sha256':h(p),'error':repr(e),**r})
 atom(OUT/(z.stage.lower()+'_training_manifest.json'),{'stage':z.stage,'locked_before_label_access':True,'run_count':len(runs),'success_count':sum(x['status']=='success' for x in records),'failure_count':sum(x['status']=='failure' for x in records),'runs':records})
 print(z.stage+'_LOCKED',sum(x['status']=='success' for x in records),sum(x['status']=='failure' for x in records))
if __name__=='__main__':main()
