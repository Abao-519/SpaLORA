"""Independent label-opening evaluator for locked Night-21B artifacts."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score


def categorical_spatial(part, graph):
    graph=sp.csr_matrix(graph).maximum(sp.csr_matrix(graph).T); graph.setdiag(0); graph.eliminate_zeros()
    rows,cols=graph.nonzero(); valid=rows<cols; rows=rows[valid]; cols=cols[valid]
    agree=float(np.mean(part[rows]==part[cols])) if len(rows) else float("nan")
    morans=[]; gearys=[]
    degree=np.asarray(graph.sum(axis=1)).ravel(); w=float(graph.sum())
    for label in np.unique(part):
        x=(part==label).astype(float); centered=x-x.mean(); denom=float(np.sum(centered**2))
        if denom<=0 or w<=0: continue
        morans.append(len(x)/w*float(centered@(graph@centered))/denom)
        diff=x[rows]-x[cols]; upper=sp.triu(graph,k=1).tocoo(); gearys.append((len(x)-1)/(2*float(upper.data.sum()))*float(np.sum(upper.data*(x[upper.row]-x[upper.col])**2))/denom)
    return agree,float(np.mean(morans)),float(np.mean(gearys))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--artifact",required=True); p.add_argument("--authority",required=True)
    p.add_argument("--reference-h5ad")
    p.add_argument("--output",required=True); a=p.parse_args()
    with np.load(a.artifact,allow_pickle=False) as z: ids=np.asarray(z["ids"]); pred=np.asarray(z["partition"])
    with np.load(a.authority,allow_pickle=False) as z:
        auth_ids=np.asarray(z["ids"])
        if a.reference_h5ad:
            graph=sp.csr_matrix((z["graph0__data"],z["graph0__indices"],z["graph0__indptr"]),shape=tuple(z["graph0__shape"]))
        else:
            labels=np.asarray(z["labels_primary"]); mask=np.asarray(z["label_mask"],dtype=bool)
            graph=sp.csr_matrix((z["operator4__data"],z["operator4__indices"],z["operator4__indptr"]),shape=tuple(z["operator4__shape"]))
    if not np.array_equal(ids,auth_ids): raise RuntimeError("ordered ID mismatch")
    if a.reference_h5ad:
        import anndata as ad
        reference=ad.read_h5ad(a.reference_h5ad)
        reference_ids=np.asarray(reference.obs_names.astype(str),dtype=ids.dtype)
        if not np.array_equal(ids,reference_ids): raise RuntimeError("reference ordered ID mismatch")
        labels=np.asarray(reference.obs["cell_type"].astype(str))
        mask=np.ones(len(labels),dtype=bool)
    if mask.sum()==0: raise RuntimeError("empty evaluation mask")
    truth=labels[mask]; prediction=pred[mask]
    sizes=np.bincount(pred,minlength=int(pred.max())+1)
    agreement,moran,geary=categorical_spatial(pred,graph)
    result={"ari":adjusted_rand_score(truth,prediction),"nmi":normalized_mutual_info_score(truth,prediction),
            "ami":adjusted_mutual_info_score(truth,prediction),"fmi":fowlkes_mallows_score(truth,prediction),
            "n_total":len(pred),"n_eval":int(mask.sum()),"observed_k":int(np.unique(pred).size),
            "min_cluster_size":int(sizes.min()),"cluster_sizes":json.dumps(sizes.tolist()),
            "neighbor_agreement":agreement,"moran_indicator_macro":moran,"geary_indicator_macro":geary,
            "authority_labels_read_by":"independent_evaluator_only"}
    Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")
if __name__=="__main__": main()
