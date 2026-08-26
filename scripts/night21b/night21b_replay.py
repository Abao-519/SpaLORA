"""Fresh-process producer artifact/checkpoint replay."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from SpaLORA.night21b_msrd import MSRDConfig, common_kmeans_endpoint, reload_msrd, robust_standardize, sha256_array


def load_graph(z):
    return sp.csr_matrix((z["graph0__data"], z["graph0__indices"], z["graph0__indptr"]), shape=tuple(z["graph0__shape"]))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--artifact",required=True); p.add_argument("--carrier",required=True)
    p.add_argument("--checkpoint"); p.add_argument("--output",required=True); a=p.parse_args()
    artifact=np.load(a.artifact,allow_pickle=False); expected_repr=np.asarray(artifact["representation"]); expected_part=np.asarray(artifact["partition"])
    ids=np.asarray(artifact["ids"]); artifact.close()
    with np.load(a.carrier,allow_pickle=False) as z:
        if any(token in key.lower() for key in z.files for token in ("label","truth","annot")):
            raise RuntimeError("annotation-like carrier key")
        v1=np.asarray(z["view1"],dtype=np.float32); v2=np.asarray(z["view2"],dtype=np.float32)
        retained=np.asarray(z["retained"],dtype=np.float32); graph=load_graph(z)
    if a.checkpoint:
        try: payload=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
        except TypeError: payload=torch.load(a.checkpoint,map_location="cpu")
        config=MSRDConfig(**payload["config"])
        representation=reload_msrd(v1,v2,retained,graph,config,int(payload["training_seed"]),payload["state_dict"],"cpu")
    else:
        representation=robust_standardize(retained)
    partition=common_kmeans_endpoint(representation,len(np.unique(expected_part)),0)
    result={"schema":"night21b-fresh-replay-v1","ids_exact":bool(np.array_equal(ids,np.load(a.artifact)["ids"])),
            "representation_max_abs":float(np.max(np.abs(representation-expected_repr))),
            "representation_close":bool(np.allclose(representation,expected_repr,rtol=2e-5,atol=2e-5)),
            "partition_exact":bool(np.array_equal(partition,expected_part)),
            "partition_sha256":sha256_array(partition)}
    if not (result["representation_close"] and result["partition_exact"]): raise RuntimeError(result)
    Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")
if __name__=="__main__": main()

