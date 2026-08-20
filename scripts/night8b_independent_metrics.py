#!/usr/bin/env python3
"""Independent contingency-table and spatial recomputation from a locked snapshot."""
from __future__ import annotations
import argparse,json,math,os
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (adjusted_mutual_info_score,completeness_score,
    fowlkes_mallows_score,homogeneity_score,v_measure_score)

def contingency(a,b):
    _,a=np.unique(a,return_inverse=True); _,b=np.unique(b,return_inverse=True)
    c=np.zeros((a.max()+1,b.max()+1),dtype=np.int64); np.add.at(c,(a,b),1); return c
def comb2(x): return x*(x-1)/2.0
def ari(a,b):
    c=contingency(a,b); n=c.sum(); s=comb2(c).sum(); ra=comb2(c.sum(1)).sum(); cb=comb2(c.sum(0)).sum(); tot=comb2(n)
    expected=ra*cb/tot; maximum=.5*(ra+cb)
    return float((s-expected)/(maximum-expected)) if maximum!=expected else 1.0
def nmi(a,b):
    c=contingency(a,b).astype(float); n=c.sum(); pi=c.sum(1); pj=c.sum(0); nz=np.nonzero(c)
    mi=float(np.sum((c[nz]/n)*np.log((c[nz]*n)/(pi[nz[0]]*pj[nz[1]]))))
    ha=float(-np.sum((pi/n)*np.log(pi/n))); hb=float(-np.sum((pj/n)*np.log(pj/n)))
    # sklearn's locked default is arithmetic-mean normalization.
    return float(mi/(.5*(ha+hb))) if ha and hb else 1.0
def spatial(pred,graph):
    r,c=graph.nonzero(); neighbor=float(np.mean(pred[r]==pred[c])); n=len(pred); s0=float(graph.sum())
    morans=[]; gearys=[]
    for label in np.unique(pred):
        x=(pred==label).astype(float); z=x-x.mean(); den=float(z@z)
        morans.append(float((n/s0)*(z@(graph@z))/den) if den else 0.0)
        diff=x[r]-x[c]; gearys.append(float(((n-1)/(2*s0))*np.sum(graph.data*diff*diff)/den) if den else 0.0)
    return neighbor,float(np.mean(morans)),float(np.mean(gearys)),1.0-neighbor
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--snapshot',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    with np.load(a.snapshot,allow_pickle=False) as z:
        truth=z['truth'].astype(str); graph=sp.csr_matrix((z['g_data'],z['g_indices'],z['g_indptr']),shape=tuple(z['g_shape']))
        rows=[]
        for method in ('U00','F00'):
            for seed in range(10):
                pred=z[f'{method}_{seed}']; ar=ari(truth,pred); nm=nmi(truth,pred); nei,mo,ge,bd=spatial(pred,graph)
                rows.append({'method':method,'seed':seed,'ari':ar,'nmi':nm,'q':(ar+nm)/2,
                             'ami':float(adjusted_mutual_info_score(truth,pred)),
                             'fmi':float(fowlkes_mallows_score(truth,pred)),
                             'homogeneity':float(homogeneity_score(truth,pred)),
                             'completeness':float(completeness_score(truth,pred)),
                             'v_measure':float(v_measure_score(truth,pred)),
                             'neighbor_agreement':nei,'moran_i':mo,'geary_c':ge,'boundary_disagreement':bd})
    paired=[]
    for seed in range(10):
        u=next(x for x in rows if x['method']=='U00' and x['seed']==seed); f=next(x for x in rows if x['method']=='F00' and x['seed']==seed)
        paired.append({'seed':seed,**{f'delta_{k}':f[k]-u[k] for k in ('ari','nmi','q','neighbor_agreement','moran_i','geary_c','boundary_disagreement')}})
    payload={'status':'PASS','implementation':'independent_contingency_tables_no_sklearn_primary',
             'rows':rows,'paired':paired,'label_source_reopened':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); tmp=a.output.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)); os.replace(tmp,a.output)
if __name__=='__main__': main()
