# Night-15B Windows local runner

The downloaded `local_compute_kit.tar.gz` is a reduced, replayable package.  It
contains seven dataset archives (P22, MISAR E15.5 S1, A1, D1 and tonsil s1–s3),
public evaluator labels/masks, reduced modality views, registered strong
embeddings, sparse CSR graphs and label-free partition banks.  It contains no
raw fragment table, checkpoint collection or dense N×N matrix.

Reference command used on the project Windows host:

```powershell
$env:PYTHONPATH='<workspace>\night15b_local_work\vendor'
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
python scripts\night15b\night15b_local_runner.py `
  --kit '<delivery>\working\local_compute_kit' `
  --output '<delivery>\working\head_hpo' `
  --datasets P22 MISAR_E15_5_S1 A1 tonsil_s1 D1 tonsil_s2 tonsil_s3 `
  --coarse-seeds 0 --fine-seeds 0,1 --top-n 6
```

The runner processes one dataset at a time, reuses each embedding transform
within a lane, writes a partial ledger after every completed dataset, and stores
only the selected partition bank.  Public labels are used only for metrics and
cross-run HPO ranking.  They are never passed to an embedding transform,
clustering input, SAPR model input, prototype target or unsupervised loss.
