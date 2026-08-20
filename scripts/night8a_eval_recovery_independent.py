#!/usr/bin/env python3
"""Independent contingency-table and vectorized-spatial recovery evaluator."""
from __future__ import annotations

import hashlib, json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path(os.environ.get("NIGHT8A_RECOVERY_REPO", "/root/autodl-fs/SpaLORA-night8a-eval-recovery"))
OUT = REPO / "outputs/night8a_eval_recovery"; RAW = Path("/root/autodl-fs/night8a_eval_recovery_20260820")
LABEL_ROOT = Path("/root/autodl-fs/night7a_consensus_20260818/evaluation_label_snapshots")

def sha(path: Path) -> str:
    h=hashlib.sha256();
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def load_pred(row):
    t=pd.read_csv(row["cluster_path"]); ids=t.observation_id.astype(str).to_numpy()
    pred=np.load(row["partition_path"],allow_pickle=False).astype(np.int64) if str(row["partition_path"]).endswith('.npy') else t.cluster.to_numpy(np.int64)
    return ids,pred

def contingency_metrics(y, pred):
    _, yi=np.unique(y,return_inverse=True); _, pi=np.unique(pred,return_inverse=True)
    c=np.zeros((yi.max()+1,pi.max()+1),dtype=np.int64); np.add.at(c,(yi,pi),1)
    n=int(c.sum()); comb=lambda x: x*(x-1)//2
    sum_c=int(np.sum(c*(c-1)//2)); a=c.sum(1); b=c.sum(0); sum_a=int(np.sum(a*(a-1)//2)); sum_b=int(np.sum(b*(b-1)//2))
    total=comb(n); expected=sum_a*sum_b/total if total else 0.; maximum=.5*(sum_a+sum_b)
    ari=(sum_c-expected)/(maximum-expected) if maximum!=expected else 1.0
    pxy=c/n; px=a/n; py=b/n; nz=c>0
    ii,jj=np.nonzero(nz); mi=float(np.sum(pxy[ii,jj]*np.log(pxy[ii,jj]/(px[ii]*py[jj]))))
    hx=float(-np.sum(px[px>0]*np.log(px[px>0]))); hy=float(-np.sum(py[py>0]*np.log(py[py>0])))
    nmi=mi/(.5*(hx+hy)) if (hx+hy)>0 else 1.0
    return float(ari),float(nmi)

def spatial(pred,a):
    a=a.tocsr().astype(np.float64); a.setdiag(0); a.eliminate_zeros(); a.sum_duplicates(); a.sort_indices()
    rows,cols=a.nonzero(); neighbor=float(np.mean(pred[rows]==pred[cols])); n=len(pred); s0=float(a.sum())
    morans=[]; gearys=[]
    weights=np.asarray(a[rows,cols]).reshape(-1)
    for level in np.unique(pred):
        raw=(pred==level).astype(np.float64); z=raw-raw.mean(); den=float(z@z)
        if den>0:
            morans.append((n/s0)*float(z@a.dot(z))/den)
            num=float(np.sum(weights*(raw[rows]-raw[cols])**2))
            gearys.append((n-1)*num/(2*s0*den))
    return neighbor,float(np.mean(morans)),float(np.mean(gearys)),1-neighbor

def main():
    view=json.loads((OUT/'evaluation_view_manifest.json').read_text()); bykey={(x['stage'],x['config_id'],x['dataset'],int(x['seed'])):x for x in view['rows']}
    primary=pd.concat([pd.read_csv(RAW/'primary_r1.csv'),pd.read_csv(RAW/'primary_r2.csv')],ignore_index=True)
    labels={}
    audit={}
    for ds in ('a1','tonsil','d1','p22'):
        p=LABEL_ROOT/f'{ds}_labels_locked.npz'; z=np.load(p,allow_pickle=False); labels[ds]=(z['observation_id'].astype(str),z['label'].astype(str)); audit[ds]={"file_sha256":sha(p),"authorized_role":"night8a_eval_recovery_independent","used_for_training_or_transform":False}
    diffs=[]; success=primary[primary.evaluation_status=='SUCCESS']
    for _,p in success.iterrows():
        row=bykey[(p.stage,p.config_id,p.dataset,int(p.seed))]; ids,pred=load_pred(row); lid,y=labels[p.dataset]
        if not np.array_equal(ids,lid): raise RuntimeError('independent order mismatch')
        ari,nmi=contingency_metrics(y,pred); q=.5*(ari+nmi); na,mi,gc,bd=spatial(pred,sp.load_npz(RAW/'adjacency'/f'{p.dataset}_k18.npz'))
        for metric,value in {'ari':ari,'nmi':nmi,'q':q,'neighbor_agreement':na,'moran_i':mi,'geary_c':gc,'boundary_disagreement':bd}.items():
            diffs.append({"stage":p.stage,"config_id":p.config_id,"dataset":p.dataset,"seed":int(p.seed),"metric":metric,"primary":float(p[metric]),"independent":value,"abs_error":abs(float(p[metric])-value)})
    maximum=max(x['abs_error'] for x in diffs); status='PASS' if maximum<=1e-12 and len(success)==127 else 'RECOVERY_SEMANTICS_INVALID'
    payload={"schema_version":"night8a-eval-recovery-independent-v1","status":status,"successful_rows":len(success),"metric_comparisons":len(diffs),"max_abs_error":maximum,"tolerance":1e-12,"scope":["ARI contingency-table","NMI contingency-table","Q","neighbor agreement","Moran I","Geary C","boundary disagreement"],"label_audit":audit,"worst":sorted(diffs,key=lambda x:x['abs_error'],reverse=True)[:20]}
    (RAW/'independent_recompute_audit.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps(payload,sort_keys=True));
    if status!='PASS': raise SystemExit(2)

if __name__=='__main__': main()
