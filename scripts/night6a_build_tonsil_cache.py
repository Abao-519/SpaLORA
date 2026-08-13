#!/usr/bin/env python3
"""Create label-free tonsil AnnData and deterministic SpaLORA cache without reading obs semantics."""
import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from SpaLORA.night1_pipeline import prepare_corrected
from SpaLORA.night3af_cache import save_cache, sha256_file

SOURCE = Path("/root/autodl-fs/datasets/human_tonsil_official/section1")
LABEL_FREE = Path("/root/autodl-fs/night6a_tonsil_label_free_20260814")
CACHE = Path("/root/autodl-fs/night6a_preprocessing_cache_20260814/tonsil")
REPO = Path("/root/autodl-fs/SpaLORA-night6a")

def decode(values):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]

def read_index(group):
    key = group.attrs.get("_index", "_index")
    if isinstance(key, bytes): key = key.decode()
    return decode(group[key][...])

def read_x(handle):
    x = handle["X"]
    if isinstance(x, h5py.Dataset): return np.asarray(x)
    shape = tuple(map(int, x.attrs["shape"]))
    encoding = x.attrs.get("encoding-type", b"")
    if isinstance(encoding, bytes): encoding = encoding.decode()
    cls = sp.csc_matrix if encoding == "csc_matrix" else sp.csr_matrix
    return cls((x["data"][...], x["indices"][...], x["indptr"][...]), shape=shape)

def make(source, target):
    if target.exists():
        existing = ad.read_h5ad(target, backed="r")
        if len(existing.obs.columns) != 0 or "spatial" not in existing.obsm:
            raise RuntimeError("existing label-free h5ad contract mismatch")
        return {"path": str(target), "shape": list(map(int, existing.shape)), "obs_columns": list(existing.obs.columns),
                "spatial_shape": list(map(int, existing.obsm["spatial"].shape)), "sha256": sha256_file(target),
                "size": target.stat().st_size, "reused_verified": True}
    with h5py.File(source, "r") as h:
        obs = read_index(h["obs"]); var = read_index(h["var"])
        x = read_x(h); spatial = np.asarray(h["obsm/spatial"])
    obj = ad.AnnData(X=x, obs=pd.DataFrame(index=pd.Index(obs)), var=pd.DataFrame(index=pd.Index(var)))
    obj.obsm["spatial"] = spatial
    obj.write_h5ad(target, compression="gzip")
    return {"path": str(target), "shape": list(map(int, obj.shape)), "obs_columns": list(obj.obs.columns),
            "spatial_shape": list(map(int, spatial.shape)), "sha256": sha256_file(target), "size": target.stat().st_size}

LABEL_FREE.mkdir(parents=True, exist_ok=True)
rna_path, adt_path = LABEL_FREE/"tonsil_s1_rna_label_free.h5ad", LABEL_FREE/"tonsil_s1_adt_label_free.h5ad"
created = {
    "rna": make(SOURCE/"s1_adata_rna.h5ad", rna_path),
    "adt": make(SOURCE/"s1_adata_adt.h5ad", adt_path),
}
cfg = {"rna": str(rna_path), "modality2": str(adt_path), "hvg": 3000, "spatial_neighbors": 18}
pre = {"min_cells":10,"rna_target_sum":10000.0,"feature_graph":{"k":20,"metric":"correlation"},
       "pca_svd_solver":"randomized","pca_random_state":0,"deterministic_pca":True,
       "alpha":1.0,"rescue_non_hvg":1000,"moran_shrinkage_tau":20.0}
prepared = prepare_corrected("tonsil", cfg, pre, "corrected_unweighted")
cache_protocol = {"pca_svd_solver":"randomized","pca_random_state":0}
manifest = save_cache(CACHE, "tonsil", prepared, cache_protocol)
payload = {"status":"P0_TONSIL_LABEL_FREE_CACHE_PASS","source_obs_semantic_values_read":False,
           "created":created,"cache_manifest_sha256":sha256_file(CACHE/"manifest.json"),
           "canonical_model_input_sha256":manifest["canonical_model_input_sha256"],
           "cache_content_sha256":manifest["canonical_cache_content_sha256"],
           "global_rule":"same RNA+ADT preprocessing/training configuration as A1",
           "tonsil_training_cfg":{"n_clusters":4,"hvg":3000,"spatial_neighbors":18,"embedding_dim":64,
                                  "epochs":200,"loss_factors":[1.9,2.5,1.5,10.0],"locked_m_bad_expected":2.289938091}}
(REPO/"outputs/night6a_handoff/p0_tonsil_cache_audit.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,sort_keys=True))
