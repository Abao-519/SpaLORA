# Night-3A-F label flow audit

Raw input and ground-truth files are byte-hashed before each scientific window. The two deterministic builders read only paired h5ad training inputs; `prepare_corrected` drops every `obs` column before any feature operation. No label CSV is parsed. The published cache contains IDs, genes, arrays, sparse graphs, coordinates and PCA metadata, but no labels. P0B-F and all 60 runs load only that cache. The independent evaluator is forbidden until the locked 60-run manifest is fsynced.
