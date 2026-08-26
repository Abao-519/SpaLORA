#!/usr/bin/env python3
"""Independent evaluator for locked Night-18E partition artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from scripts.night18e.night18e_producer import load_csr, sha_array, sha_file


def encode(value: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(value).astype(str), return_inverse=True)[1].astype(np.int32)


def categorical_spatial(partition: np.ndarray, graph: sp.csr_matrix) -> tuple[float, float, float]:
    partition = encode(partition)
    graph = sp.csr_matrix(graph, dtype=np.float64).maximum(sp.csr_matrix(graph, dtype=np.float64).T).tocsr()
    graph.setdiag(0); graph.eliminate_zeros()
    n = len(partition); total = float(graph.sum())
    if total <= 0: raise ValueError("graph has no positive weight")
    upper = sp.triu(graph, k=1, format="coo")
    moran=[]; geary=[]
    for group in range(int(partition.max())+1):
        indicator=(partition==group).astype(np.float64); centered=indicator-indicator.mean(); denominator=float(np.sum(centered**2))
        if denominator<=0: continue
        moran.append(float(n/total*(centered @ (graph @ centered))/denominator))
        numerator=float(np.sum(upper.data*(indicator[upper.row]-indicator[upper.col])**2))
        geary.append(float((n-1)/total*numerator/denominator))
    rows=np.repeat(np.arange(n),np.diff(graph.indptr))
    agreement=float(np.sum(graph.data*(partition[rows]==partition[graph.indices]))/total)
    return float(np.mean(moran)),float(np.mean(geary)),agreement


def load_reference(args):
    if args.reference_kind == "npz":
        with np.load(args.reference, allow_pickle=False) as z:
            ids=np.asarray(z[args.reference_id_key]).astype("U"); labels=np.asarray(z[args.label_key]).astype("U"); mask=np.asarray(z[args.mask_key],dtype=bool)
    elif args.reference_kind == "h5ad":
        value=ad.read_h5ad(args.reference); ids=np.asarray(value.obs_names.astype(str)).astype("U"); series=value.obs[args.label_key]; mask=~np.asarray(series.isna(),dtype=bool); labels=np.asarray(series.astype(str)).astype("U")
    else:
        with Path(args.reference).open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle,delimiter="\t"))
        ids=np.asarray([row[args.reference_id_key] for row in rows]).astype("U"); labels=np.asarray([row[args.label_key] for row in rows]).astype("U"); mask=np.ones(len(ids),dtype=bool)
    if not (len(ids)==len(labels)==len(mask)) or len(np.unique(ids))!=len(ids): raise ValueError("reference contract invalid")
    return ids,labels,mask


def run(args):
    producer=json.loads(Path(args.producer_json).read_text())
    if int(producer.get("producer_label_reads",-1)) != 0: raise RuntimeError("producer label firewall failed")
    with np.load(args.partition_bank,allow_pickle=False) as bank:
        bank_ids=np.asarray(bank["ids"]).astype("U"); candidate_ids=np.asarray(bank["candidate_ids"]).astype("U"); partitions=np.asarray(bank["partitions"],dtype=np.int32)
    with np.load(args.carrier,allow_pickle=False) as carrier:
        carrier_ids=np.asarray(carrier["ids"]).astype("U"); graph=load_csr(carrier,"graph1")
    if not np.array_equal(bank_ids,carrier_ids): raise ValueError("bank/carrier ID mismatch")
    if len(candidate_ids)!=len(producer["rows"]) or any(candidate_ids[i]!=producer["rows"][i]["candidate_id"] for i in range(len(candidate_ids))): raise ValueError("candidate ID/manifest mismatch")
    ref_ids,ref_labels,ref_mask=load_reference(args); lookup={value:index for index,value in enumerate(ref_ids)}
    if not set(carrier_ids).issubset(lookup): raise ValueError("carrier IDs missing in reference")
    order=np.asarray([lookup[x] for x in carrier_ids],dtype=np.int64); labels=ref_labels[order]; mask=ref_mask[order]
    truth=labels[mask]
    if len(np.unique(truth))!=int(args.k): raise ValueError("reference K mismatch")
    results=[]
    for index,source in enumerate(producer["rows"]):
        row={key:source.get(key,"") for key in ("candidate_id","start_id","start_index","arm","certificate_config_id","status","failure","partition_sha256","initial_partition_sha256","changed_from_initial","wall_seconds")}
        if source["status"]=="PASS":
            partition=partitions[index]
            if sha_array(partition)!=source["partition_sha256"]: raise ValueError("partition SHA mismatch")
            encoded=encode(partition); sizes=np.bincount(encoded,minlength=int(args.k)); eval_sizes=np.bincount(encoded[mask],minlength=int(args.k))
            if len(np.unique(encoded))!=int(args.k) or np.any(sizes<=0): raise ValueError("partition exact-K/no-empty failed")
            moran,geary,agreement=categorical_spatial(partition,graph); prediction=partition[mask]
            row.update(absolute_ari=float(adjusted_rand_score(truth,prediction)),absolute_nmi=float(normalized_mutual_info_score(truth,prediction)),ami=float(adjusted_mutual_info_score(truth,prediction)),fmi=float(fowlkes_mallows_score(truth,prediction)),homogeneity=float(homogeneity_score(truth,prediction)),v_measure=float(v_measure_score(truth,prediction)),morans_i_macro=moran,gearys_c_macro=geary,neighbor_agreement=agreement,n_total=len(partition),n_evaluated=int(mask.sum()),k=int(args.k),cluster_sizes_full=json.dumps([int(x) for x in sizes]),cluster_sizes_eval=json.dumps([int(x) for x in eval_sizes]),min_cluster_size_full=int(sizes.min()),min_cluster_size_eval=int(eval_sizes.min()),evaluator_label_reads=1)
        results.append(row)
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); columns=sorted({key for row in results for key in row})
    with output.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=columns); writer.writeheader(); writer.writerows(results)
    audit={"schema":"night18e-independent-evaluator-v1","lane":args.lane,"reference_path":str(Path(args.reference).resolve()),"reference_sha256":sha_file(args.reference),"label_key":args.label_key,"mask_key":args.mask_key,"n_reference":len(ref_ids),"n_total":len(carrier_ids),"n_evaluated":int(mask.sum()),"reference_k":len(np.unique(truth)),"reference_counts":{str(x):int(np.sum(truth==x)) for x in sorted(np.unique(truth))},"ordered_label_sha256":hashlib.sha256(b"\0".join(f"{i}\t{l}".encode() for i,l,m in zip(carrier_ids,labels,mask) if m)).hexdigest(),"locked_partition_artifact_sha256":sha_file(args.partition_bank),"producer_label_reads":0,"evaluator_label_reads":1,"rows":len(results)}
    output.with_suffix(".evaluator.json").write_text(json.dumps(audit,indent=2,sort_keys=True))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--carrier",required=True); p.add_argument("--partition-bank",required=True); p.add_argument("--producer-json",required=True); p.add_argument("--reference",required=True); p.add_argument("--reference-kind",choices=("npz","h5ad","tsv"),required=True); p.add_argument("--reference-id-key",default="ids"); p.add_argument("--label-key",required=True); p.add_argument("--mask-key",default="label_mask"); p.add_argument("--lane",required=True); p.add_argument("--k",type=int,required=True); p.add_argument("--output",required=True); run(p.parse_args())


if __name__=="__main__": main()
