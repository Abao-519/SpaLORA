from __future__ import annotations

import hashlib,json,os,pathlib,shutil,subprocess,time

REPO=pathlib.Path('/root/autodl-fs/SpaLORA-night10a-rev1')
RAW=pathlib.Path('/root/autodl-fs/night10a_rev1_qcrd_20260821')
OUT=REPO/'outputs/night10a_rev1_handoff'; STAGE=RAW/'official_compact'
PARENT='1e576b68938fa194dcdd53ee58915767b7a78325'; BRANCH='revision/q2-night10a-rev1-qcrd-execution-20260821'; TAG='night10a-rev1-final-20260821'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def atomic(p,x):
 p=pathlib.Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8'); os.replace(q,p)

def run(*args,check=True):return subprocess.run(list(args),cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=check)

def build_evidence():
 trainings=sorted(RAW.glob('r1/**/training_manifest.json')); reloads=sorted(RAW.glob('r1/**/reload_audit.json')); transforms=sorted(RAW.glob('r1/**/transform_manifest.json')); failures=sorted(RAW.glob('r1/**/failure.json'))
 failure_rows=[]
 for p in failures:
  x=json.loads(p.read_text()); failure_rows.append({'relative_path':str(p.relative_to(RAW)),'sha256':sha(p),**x})
 partial=[]
 for d in sorted(RAW.glob('r1/**/*')):
  if d.is_dir() and (d/'training_manifest.json').exists() and not (d/'transform_manifest.json').exists() and not (d/'failure.json').exists():
   files=[{'relative_path':str(p.relative_to(RAW)),'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(d.glob('*')) if p.is_file()]
   partial.append({'cell':str(d.relative_to(RAW)),'files':files,'status':'INVALID_PARTIAL_TRANSFORM_STOPPED_AFTER_SYSTEMIC_SEMANTIC_FAILURE'})
 capture=RAW/'logs/formal_semantic_stop_capture.json'
 audit={'schema':'spalora.night10a.rev1.retrospective_invalid.v1','terminal_status':'IMPLEMENTATION_SEMANTICS_INVALID','p0_rev1_original_status':'PASS','p0_rev1_retrospective_status':'FALSE_PASS_REAL_FORWARD_COVERAGE_GAP','root_cause':{'private_view_dimension':128,'p22_r02_reference_dimension':64,'adapter_declared_input_dimension':384,'actual_concatenated_dimension':320,'q06_actual_concatenated_dimension':336,'q06_declared_input_dimension':400,'error':'mat1 and mat2 shapes cannot be multiplied','affected_dataset':'p22','affected_registered_cells':21},'why_not_corrected':'The scientific implementation and 63 formal outputs were already locked. Adding a reference projection or changing the adapter shape now would alter the preregistered trainable architecture after scientific execution began.','successful_training_cells':len(trainings),'checkpoint_roundtrip_pass_cells':len(reloads),'complete_transform_cells':len(transforms),'formal_failure_cells':len(failures),'p22_failures_before_optimizer_step':len(failure_rows),'r1_label_reads':0,'stage_m_runs':0,'r2_training_cells':0,'scientific_retry':0,'fallback':0,'stop_capture_sha256':sha(capture),'partial_transforms':partial,'failure_rows':failure_rows}
 atomic(OUT/'p0_rev1_retrospective_false_pass_audit.json',audit)
 atomic(OUT/'r1_failure_audit.json',{'count':len(failure_rows),'rows':failure_rows})
 raw=[]
 for pattern in ('r1/**/training_manifest.json','r1/**/reload_audit.json','r1/**/transform_manifest.json','r1/**/failure.json','r1/**/model_final.pt','r1/**/corrected_views.npz','r1/**/clusters.csv','r1/**/affinity.npz','raw/inputs/**/*','logs/*'):
  for p in sorted(RAW.glob(pattern)):
   if p.is_file():raw.append({'relative_path':str(p.relative_to(RAW)),'size':p.stat().st_size,'sha256':sha(p)})
 seen={x['relative_path']:x for x in raw}; import csv
 with (OUT/'raw_artifact_manifest.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['relative_path','size','sha256']);w.writeheader();w.writerows(seen.values())
 resource=[]
 for p in trainings:
  x=json.loads(p.read_text()); c=x['config'];resource.append({'dataset':c['dataset'],'seed':c['seed'],'candidate':c['candidate'],'runtime_seconds':x['runtime_seconds'],'peak_gpu_mib':x['peak_gpu_bytes']/(1024**2),'device':x['device'],'checkpoint_roundtrip':(p.parent/'reload_audit.json').exists()})
 with (OUT/'resource_per_training_cell.csv').open('w',newline='') as f:
  fields=list(resource[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(resource)
 atomic(OUT/'resource_audit.json',{'successful_cuda_trainings':len(resource),'complete_transforms':len(transforms),'peak_gpu_mib':max(x['peak_gpu_mib'] for x in resource),'training_runtime_seconds':sum(x['runtime_seconds'] for x in resource),'all_cuda':all('4080' in x['device'] for x in resource),'transform_timeout_seconds':1800,'overall_wallclock_hours':12,'semantic_stop_before_budget':True})
 atomic(OUT/'label_read_and_firewall_audit.json',{'p0_rev1_label_reads':0,'r1_label_reads':0,'misar_y_reads':0,'e18_5_reads':0,'stage_m':0,'labels_deserialized_after_formal_start':False,'label_firewall':'PASS'})
 atomic(OUT/'tests_and_semantics_audit.json',{'p0_unit_tests':'12 passed','p0_real_endpoint_parity':'PASS','retrospective_real_p22_adapter_forward':'FAIL','systemic_semantic_status':'IMPLEMENTATION_SEMANTICS_INVALID','scientific_retry':0,'implementation_not_modified_after_formal_lock':True})
 atomic(OUT/'frontier_registry.json',{'status':'NOT_EVALUATED_SEMANTIC_INVALID','accuracy':None,'balanced':None,'spatial':None,'r2_candidates':[]})
 atomic(OUT/'night10a_decision.json',{'terminal_status':'IMPLEMENTATION_SEMANTICS_INVALID','r1_scientific_conclusion':None,'r2_ran':False,'stage_m_ran':False,'labels_read':0,'successful_training_cells_preserved':len(trainings),'failed_p22_cells_preserved':len(failures),'original_night10a_preserved':True})
 (OUT/'per_seed_metrics_long.csv').write_text('status,dataset,candidate,seed\nNOT_EVALUATED_SEMANTIC_INVALID,,,\n')
 (OUT/'metric_backfill_long.csv').write_text('status,dataset,method,metric,value\nNOT_RUN_SEMANTIC_INVALID,,,,\n')
 report=f'''# Night-10A REV1 report\n\nTerminal status: `IMPLEMENTATION_SEMANTICS_INVALID`.\n\n## Plain-language result\n\nNo ARI, NMI or Q result was opened. The label firewall remained closed, so there is no valid claim that A1, tonsil, D1 or P22 improved or declined.\n\nThe registered unified adapter worked dimensionally on the three RNA+protein datasets, producing {len(trainings)} locked CUDA trainings and {len(transforms)} completed fixed-endpoint transforms before the systemic issue was recognized. On P22, every one of the 21 registered trainable cells failed at its first forward: the G04 private views have 128 columns while the authoritative R02 reference embedding has 64. The adapter concatenated 128+128+64=320 columns but its frozen layer expected 384 (Q06: 336 actual versus 400 expected).\n\nThis also reveals a retrospective P0-REV1 coverage defect: P0 checked the real P22 endpoint and a synthetic CUDA adapter round-trip separately, but never ran the adapter on the real mixed-dimensional P22 inputs. Therefore the earlier P0 PASS is retained as history but marked a false pass.\n\n## Why execution stopped\n\nAdding a projection for the 64-dimensional reference or changing input dimensions would modify the trainable architecture after formal scientific outputs had already begun. The taskbook prohibits that. The run therefore stopped with 0 label reads, 0 Stage M, 0 R2, 0 scientific retries and 0 fallback. Existing successful, failed and partial artifacts are preserved and SHA-indexed; none is used as scientific evidence.\n\n## Required future repair\n\nA new authority revision must specify a single cross-family dimension contract, for example a preregistered fixed/non-trainable reference projection or a family-independent adapter that accepts an explicitly registered reference dimension. It must add real A1 and real P22 forward/loss/checkpoint tests before any formal training. This cannot be repaired inside Night-10A REV1.\n'''
 (OUT/'night10a_report.md').write_text(report,encoding='utf-8');(OUT/'plain_language_summary.md').write_text('Night-10A REV1 did not produce a score conclusion. P22 exposed a 128-versus-64 input contract error after 63 protein-family trainings; labels stayed closed, results were preserved, and the run stopped without changing the locked model.\n',encoding='utf-8')
 atomic(OUT/'shutdown_dispatch_prepared.json',{'prepared':True,'command':'/usr/bin/shutdown','dispatched':False,'note':'local post-dispatch sidecar will record SSH result'})
 return audit

def index_repo():
 rows=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='delivery_index.json':rows.append({'relative_path':str(p.relative_to(REPO)).replace('\\','/'),'size':p.stat().st_size,'sha256':sha(p)})
 atomic(OUT/'delivery_index.json',{'schema':'spalora.night10a.rev1.invalid.delivery.v1','count':len(rows),'files':rows})

def persist():
 run('git','add','outputs/night10a_rev1_handoff','scripts/night10a/night10a_rev1_finalize_invalid.py')
 run('git','commit','-m','night10a rev1: preserve semantic invalidity evidence')
 index_repo();run('git','add',str(OUT/'delivery_index.json'));run('git','commit','-m','night10a rev1: add final invalidity delivery index');run('git','push','origin',BRANCH)
 if run('git','rev-parse','-q','--verify','refs/tags/'+TAG,check=False).returncode==0:raise RuntimeError('final tag already exists')
 run('git','tag','-a',TAG,'-m','Night-10A REV1 semantic invalidity final 2026-08-21');run('git','push','origin',TAG)
 if STAGE.exists():shutil.rmtree(STAGE)
 shutil.copytree(OUT,STAGE/'handoff');(STAGE/'source').mkdir(parents=True)
 for p in (REPO/'SpaLORA/night10a_qcrd.py',REPO/'scripts/night10a/night10a_rev1_p0.py',REPO/'scripts/night10a/night10a_rev1_run.py',REPO/'scripts/night10a/night10a_rev1_evaluate.py',REPO/'scripts/night10a/night10a_rev1_finalize_invalid.py',REPO/'tests/night10a/test_night10a_qcrd_rev1.py',REPO/'tests/night10a/test_night10a_rev1_runner.py'):shutil.copy2(p,STAGE/'source'/p.name)
 shutil.copytree(REPO/'protocols/night10a_rev1',STAGE/'protocols/night10a_rev1')
 bundle=STAGE/'night10a_rev1_incremental_20260821.bundle';run('git','bundle','create',str(bundle),TAG,'^'+PARENT)
 rows=[]
 for p in sorted(STAGE.rglob('*')):
  if p.is_file() and p.name!='compact_delivery_index.json':rows.append({'relative_path':str(p.relative_to(STAGE)).replace('\\','/'),'size':p.stat().st_size,'sha256':sha(p)})
 atomic(STAGE/'compact_delivery_index.json',{'schema':'spalora.compact.root_relative.v1','root_rule':'relative to official_compact','count':len(rows),'files':rows,'git_commit':run('git','rev-parse','HEAD').stdout.strip(),'git_tag':TAG})
 total=sum(p.stat().st_size for p in STAGE.rglob('*') if p.is_file());assert total<100*1024*1024
 return {'commit':run('git','rev-parse','HEAD').stdout.strip(),'tag':TAG,'compact':str(STAGE),'compact_bytes':total,'compact_index_sha256':sha(STAGE/'compact_delivery_index.json'),'bundle_sha256':sha(bundle)}

def main():
 a=build_evidence();p=persist();atomic(RAW/'final_remote_summary.json',{'audit':a,'package':p});print(json.dumps(p,indent=2))
if __name__=='__main__':main()
