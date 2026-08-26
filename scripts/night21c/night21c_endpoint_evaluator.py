"""Independent label-opening evaluator for a locked Night-21C endpoint bank."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score,
                             homogeneity_score, normalized_mutual_info_score, v_measure_score)


def array_sha(value):
    value=np.ascontiguousarray(value); h=hashlib.sha256(); h.update(str(value.dtype).encode()); h.update(json.dumps(list(value.shape)).encode()); h.update(value.tobytes()); return h.hexdigest()


def graph_from(archive, prefix):
    return sp.csr_matrix((archive[prefix+"__data"],archive[prefix+"__indices"],archive[prefix+"__indptr"]),shape=tuple(int(x) for x in archive[prefix+"__shape"]))


def categorical_spatial(partition, graph):
    graph=sp.csr_matrix(graph,dtype=np.float64).maximum(sp.csr_matrix(graph,dtype=np.float64).T); graph.setdiag(0); graph.eliminate_zeros()
    upper=sp.triu(graph,k=1).tocoo(); rows,cols=upper.row,upper.col
    agreement=float(np.mean(partition[rows]==partition[cols])) if len(rows) else float("nan")
    total=float(graph.sum()); moran=[]; geary=[]
    for label in np.unique(partition):
        x=(partition==label).astype(float); centered=x-x.mean(); denom=float(np.sum(centered**2))
        if denom<=0 or total<=0 or upper.data.sum()<=0: continue
        moran.append(len(x)/total*float(centered @ (graph @ centered))/denom)
        geary.append((len(x)-1)/(2*float(upper.data.sum()))*float(np.sum(upper.data*np.square(x[rows]-x[cols])))/denom)
    return agreement,float(np.mean(moran)),float(np.mean(geary))


def load_truth(authority_path, ids, reference_h5ad=None):
    with np.load(authority_path,allow_pickle=False) as z:
        auth_ids=np.asarray(z["ids"])
        if reference_h5ad:
            graph=graph_from(z,"graph0")
        else:
            labels=np.asarray(z["labels_primary"]); mask=np.asarray(z["label_mask"],dtype=bool); graph=graph_from(z,"operator4")
    if not np.array_equal(ids,auth_ids): raise RuntimeError("authority ordered IDs mismatch")
    if reference_h5ad:
        import anndata as ad
        ref=ad.read_h5ad(reference_h5ad)
        ref_ids=np.asarray(ref.obs_names.astype(str),dtype=ids.dtype)
        if not np.array_equal(ids,ref_ids): raise RuntimeError("reference ordered IDs mismatch")
        raw=ref.obs["cell_type"]
        mask=np.asarray(raw.notna(),dtype=bool); labels=np.asarray(raw.astype(str))
    if not mask.any(): raise RuntimeError("empty evaluation mask")
    return labels,mask,graph


def main():
    p=argparse.ArgumentParser(); p.add_argument("--bank",required=True); p.add_argument("--authority",required=True); p.add_argument("--reference-h5ad"); p.add_argument("--output",required=True); a=p.parse_args()
    bank_path=Path(a.bank); manifest=json.loads(bank_path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(bank_path,allow_pickle=False) as z: ids=np.asarray(z["ids"]); candidate_ids=np.asarray(z["candidate_ids"]); partitions=np.asarray(z["partitions"])
    if len(candidate_ids)!=len(set(candidate_ids.tolist())) or partitions.shape!=(len(candidate_ids),len(ids)): raise RuntimeError("invalid candidate bank")
    expected=manifest["candidate_partition_sha256"]
    for name,part in zip(candidate_ids.tolist(),partitions):
        if expected.get(name)!=array_sha(part): raise RuntimeError("partition SHA mismatch")
    labels,mask,graph=load_truth(a.authority,ids,a.reference_h5ad)
    rows=[]
    for name,part in zip(candidate_ids.tolist(),partitions):
        truth,pred=labels[mask],part[mask]; sizes_full=np.unique(part,return_counts=True)[1]; sizes_eval=np.unique(pred,return_counts=True)[1]
        agreement,moran,geary=categorical_spatial(part,graph)
        rows.append({"lane":manifest["lane"],"representation_source":manifest["representation_source"],"candidate_id":name,"candidate_partition_sha256":array_sha(part),
                     "ari":adjusted_rand_score(truth,pred),"nmi":normalized_mutual_info_score(truth,pred),"ami":adjusted_mutual_info_score(truth,pred),"fmi":fowlkes_mallows_score(truth,pred),
                     "homogeneity":homogeneity_score(truth,pred),"v_measure":v_measure_score(truth,pred),"n_total":len(part),"n_eval":int(mask.sum()),"observed_k_full":int(np.unique(part).size),
                     "min_cluster_size_full":int(sizes_full.min()),"cluster_sizes_full":"|".join(map(str,sorted(sizes_full.tolist()))),"min_cluster_size_eval":int(sizes_eval.min()),"cluster_sizes_eval":"|".join(map(str,sorted(sizes_eval.tolist()))),
                     "neighbor_agreement":agreement,"moran_indicator_macro":moran,"geary_indicator_macro":geary,"label_access":"independent_evaluator_after_bank_lock"})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps({"schema":"night21c-endpoint-evaluation-v1","bank_sha256":hashlib.sha256(bank_path.read_bytes()).hexdigest(),"candidate_count":len(rows),"ground_truth_loaded_after_bank_lock":True},indent=2,sort_keys=True),encoding="utf-8")


if __name__=="__main__": main()
