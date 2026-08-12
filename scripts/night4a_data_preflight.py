#!/usr/bin/env python3
"""Night-4A data provenance and label-firewall preflight."""
import csv, hashlib, json, os
from collections import Counter
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pandas as pd

OUT=Path('/root/autodl-fs/SpaLORA-night4a/outputs/night4a_handoff')
RAW=Path('/root/autodl-fs/night4a_external_data/raw')
EXT=Path('/root/autodl-fs/night4a_external_data/extracted')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''): h.update(b)
 return h.hexdigest()
def dump(name,x):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(t,p);return sha(p)
def h5ad_schema(p):
 a=ad.read_h5ad(p,backed='r'); ids=pd.Index(a.obs_names.astype(str));
 result={'path':str(p),'sha256':sha(p),'n_obs':a.n_obs,'n_vars':a.n_vars,'duplicate_obs_ids':int(ids.duplicated().sum()),'obs_columns_names_only':list(map(str,a.obs.columns)),'obsm_keys':list(a.obsm.keys()),'layers':list(a.layers.keys()),'x_storage':type(a.X).__name__,'nonfinite_scan':'deferred_chunked_cache_build'}
 a.file.close();return result,ids
datasets=[];align=[]
# held-out local D1 and official tonsils
pairs=[('gse263617_d1_ln',Path('/root/autodl-fs/Human lymph node/D1/humanlymphnode_rna.h5ad'),Path('/root/autodl-fs/Human lymph node/D1/humanlymphnode_adt.h5ad'),'held_out_within_study'),('gse263617_a1_tonsil',EXT/'GSE263617/GSM8195495_A1_TNSL.h5ad',EXT/'GSE263617/GSM8195499_A1_TNSL_Protein.h5ad','held_out_within_study'),('gse263617_d1_tonsil',EXT/'GSE263617/GSM8195497_D1_TNSL.h5ad',EXT/'GSE263617/GSM8195501_D1_TNSL_Protein.h5ad','held_out_within_study')]
for name,rna,mod2,role in pairs:
 rs,ri=h5ad_schema(rna);ms,mi=h5ad_schema(mod2); exact_set=set(ri)==set(mi); exact_order=ri.equals(mi)
 datasets.append({'dataset':name,'accession':'GSE263617','role':role,'modality_pair':'RNA+protein','source':'NCBI GEO plus preexisting local D1','rna':rs,'modality2':ms,'coordinates_available':bool(rs['obsm_keys']),'semantic_values_read':False,'license_terms':'NCBI GEO public record; cite source study'})
 align.append({'dataset':name,'rna_n':len(ri),'modality2_n':len(mi),'exact_id_set':exact_set,'exact_id_order':exact_order,'unambiguous_alignment':exact_set})
# independent SPOTS matrices: feature type names are technical metadata, not labels
for rep in (1,2):
 hp=RAW/f'GSE198353/GSE198353_spleen_rep_{rep}_filtered_feature_bc_matrix.h5'
 with h5py.File(hp,'r') as h:
  bar=np.asarray(h['matrix/barcodes']).astype('U'); feat=np.asarray(h['matrix/features/feature_type']).astype('U'); shape=list(map(int,h['matrix/shape'][:]))
 pos=pd.read_csv(EXT/f'GSE198353/rep{rep}/spatial/tissue_positions_list.csv',header=None); pids=pos.iloc[:,0].astype(str)
 types=dict(Counter(feat)); paired=('Gene Expression' in types and 'Antibody Capture' in types); exact=set(bar)==set(pids)
 datasets.append({'dataset':f'gse198353_spleen_rep{rep}','accession':'GSE198353','role':'truly_independent','modality_pair':'RNA+protein','matrix_path':str(hp),'matrix_sha256':sha(hp),'shape':shape,'feature_type_counts':types,'n_barcodes':len(bar),'duplicate_obs_ids':int(pd.Index(bar).duplicated().sum()),'coordinates_available':True,'semantic_values_read':False,'license_terms':'NCBI GEO public record; cite SPOTS study'})
 align.append({'dataset':f'gse198353_spleen_rep{rep}','rna_n':len(bar),'modality2_n':len(bar) if paired else 0,'coordinate_n':len(pids),'exact_id_set':exact,'exact_id_order':pd.Index(bar).equals(pd.Index(pids)),'unambiguous_alignment':paired and exact})
raw=[]
for p in sorted(RAW.rglob('*')):
 if p.is_file() and '.chunks' not in str(p) and 'single_connection' not in p.name: raw.append({'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha(p)})
registry={'schema_version':1,'created_before_semantic_label_values':True,'datasets':datasets}
OUT.mkdir(parents=True,exist_ok=True)
pd.DataFrame([{'dataset':d['dataset'],'accession':d['accession'],'role':d['role'],'modalities':d['modality_pair'],'coordinates':d['coordinates_available'],'semantic_values_read':False} for d in datasets]).to_csv(OUT/'data_source_registry.csv',index=False)
dump('data_source_registry.json',registry);dump('raw_file_manifest.json',{'files':raw});dump('observation_alignment_report.json',{'datasets':align})
lock={'schema_version':1,'locked_before_semantic_label_values':True,'datasets':[d['dataset'] for d in datasets],'qualification_rules':['paired observation IDs','reproducible coordinates','auditable source and feature semantics','independent/manual labels for accuracy only','at least two nonempty domains','no outcome-driven inclusion'],'preprocessing':{'RNA':'method-native count-compatible preprocessing; fixed before labels','protein':'method-native CLR or official preprocessing; fixed before labels','common_observation_set':'exact intersection fixed before labels','seeds':[0,1,2,3,4]},'candidate_matrix_roles':{'GSE263617':'held-out within-study only','GSE198353':'truly independent family; accuracy eligibility pending post-lock label audit','GSE213264':'independent candidate; paired IDs/coordinates unresolved from GEO files'}}
locksha=dump('data_preprocessing_lock.json',lock)
dump('label_free_cache_manifest.json',{'status':'SCHEMA_AND_INPUT_LOCK_COMPLETE','immutable_cache_built':False,'reason':'Night-4A preflight only; no model-ready cache needed before method-specific compatibility','data_preprocessing_lock_sha256':locksha,'semantic_columns_disabled':True})
(OUT/'semantic_label_access_log.jsonl').write_text(json.dumps({'event':'LABEL_FREE_LOCK_CREATED','lock_sha256':locksha,'semantic_label_values_read':False})+'\n')
print('DATA_LABEL_FREE_LOCK_PASS',locksha)
