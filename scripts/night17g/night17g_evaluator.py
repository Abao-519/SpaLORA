#!/usr/bin/env python3
"""Independent label-opening evaluator for locked Night-17G partitions."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_mutual_info_score,adjusted_rand_score,fowlkes_mallows_score,homogeneity_score,normalized_mutual_info_score,v_measure_score

def encode(x): return np.unique(np.asarray(x).astype(str),return_inverse=True)[1].astype(np.int32)
def load_graph(path):
 with np.load(path,allow_pickle=False) as z:
  p="graph1__"; return sp.csr_matrix((z[p+"data"],z[p+"indices"],z[p+"indptr"]),shape=tuple(z[p+"shape"]),dtype=np.float64)
def spatial_metrics(partition,graph):
 partition=encode(partition); graph=sp.csr_matrix(graph,dtype=np.float64).maximum(sp.csr_matrix(graph,dtype=np.float64).T).tocsr(); graph.setdiag(0); graph.eliminate_zeros(); total=float(graph.sum()); n=len(partition); upper=sp.triu(graph,k=1,format="coo"); moran=[]; geary=[]
 for group in range(int(partition.max())+1):
  x=(partition==group).astype(float); centered=x-x.mean(); denominator=float(np.sum(centered**2))
  if denominator<=0: continue
  moran.append(float(n/total*(centered@(graph@centered))/denominator)); numerator=float(np.sum(upper.data*(x[upper.row]-x[upper.col])**2)); geary.append(float((n-1)/total*numerator/denominator))
 rows=np.repeat(np.arange(n),np.diff(graph.indptr)); agreement=float(np.sum(graph.data*(partition[rows]==partition[graph.indices]))/total)
 return float(np.mean(moran)),float(np.mean(geary)),agreement
def load_reference(args):
    if args.reference_kind=="npz":
        with np.load(args.reference,allow_pickle=False) as z: ids=z[args.reference_id_key].astype("U"); labels=z[args.label_key].astype("U"); mask=z[args.mask_key].astype(bool)
    else:
        value=ad.read_h5ad(args.reference); ids=np.asarray(value.obs_names.astype(str)).astype("U"); s=value.obs[args.label_key]; mask=~np.asarray(s.isna()); labels=np.asarray(s.astype(str)).astype("U")
    return ids,labels,mask
def main():
    p=argparse.ArgumentParser(); p.add_argument("--lane",required=True); p.add_argument("--producer-dir",required=True); p.add_argument("--carrier",required=True); p.add_argument("--reference",required=True)
    p.add_argument("--reference-kind",choices=["npz","h5ad"],required=True); p.add_argument("--reference-id-key",default="ids"); p.add_argument("--label-key",required=True); p.add_argument("--mask-key",default="label_mask"); p.add_argument("--k",type=int,required=True); p.add_argument("--output",required=True); args=p.parse_args()
    d=Path(args.producer_dir); manifest=json.loads((d/"producer.json").read_text())
    with np.load(d/"producer.npz",allow_pickle=False) as z: ids=z["ids"].astype("U"); run_ids=z["run_ids"].astype("U"); partitions=z["partitions"].astype(np.int32)
    rid,labels,mask=load_reference(args); lookup={x:i for i,x in enumerate(rid)}
    if not set(ids).issubset(lookup): raise ValueError("producer IDs absent from reference")
    order=np.asarray([lookup[x] for x in ids]); labels=labels[order]; mask=mask[order]
    truth=labels[mask]
    if np.unique(truth).size!=args.k: raise ValueError("reference K mismatch")
    source={x["run_id"]:x for x in manifest["rows"]}; rows=[]; graph=load_graph(args.carrier)
    for run_id,partition in zip(run_ids,partitions):
        meta=source[str(run_id)]; pred=partition[mask]; sizes=np.bincount(encode(partition),minlength=args.k)
        if np.unique(partition).size!=args.k or np.any(sizes<=0): raise RuntimeError("locked partition violates exact K")
        moran,geary,agreement=spatial_metrics(partition,graph)
        rows.append({"lane":args.lane,"candidate_id":str(run_id),"arm":meta["arm"],"config_id":meta["config_id"],"training_seed":meta["training_seed"],
          "partition_sha256":meta["partition_sha256"],"representation_sha256":meta["representation_sha256"],"absolute_ari":adjusted_rand_score(truth,pred),"absolute_nmi":normalized_mutual_info_score(truth,pred),
          "ami":adjusted_mutual_info_score(truth,pred),"fmi":fowlkes_mallows_score(truth,pred),"homogeneity":homogeneity_score(truth,pred),"v_measure":v_measure_score(truth,pred),"morans_i_macro":moran,"gearys_c_macro":geary,"neighbor_agreement":agreement,
          "n_total":len(partition),"n_evaluated":int(mask.sum()),"k":args.k,"min_cluster_size":int(sizes.min()),"cluster_sizes":json.dumps(sizes.tolist()),"changed_from_strong_start":meta.get("changed_from_strong_start",0),"wall_seconds":manifest["wall_seconds"],"peak_gpu_mib":manifest["peak_gpu_mib"],"peak_rss_mib":manifest["peak_rss_mib"]})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); fields=list(rows[0])
    with out.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    payload=b"\0".join(f"{i}\t{l}".encode() for i,l,m in zip(ids,labels,mask) if m)
    out.with_suffix(".json").write_text(json.dumps({"schema":"night17g-independent-evaluator-v1","lane":args.lane,"partition_lock_sha256":manifest["artifact_sha256"],"ordered_label_sha256":hashlib.sha256(payload).hexdigest(),"labels_loaded_after_partition_lock":True,"producer_label_reads":0,"evaluator_label_reads":1},indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
