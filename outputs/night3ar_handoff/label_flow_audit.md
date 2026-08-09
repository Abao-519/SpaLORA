# Night-3A-R label flow audit

1. Before the scientific window, immutable files are opened only as bytes to verify SHA-256; no content is returned.
2. P0A-R alone parses only the ground-truth identifier column to verify spot order/set; `label_values_read=false`.
3. The scientific-window hook and `pandas.read_csv` guard are installed before preprocessing/model code.
4. A1/P22 ground-truth CSV opens, label parsers, and evaluator imports are forbidden until `locked_60_run_manifest.json` is fsynced.
5. Placenta RNA h5ad is a required input, but `_load_label_free` removes all `obs` columns before preprocessing output enters the trainer; `cell_type` is never accessed.
6. The independent evaluator starts only after 60/60 completion and a PASS training firewall.
