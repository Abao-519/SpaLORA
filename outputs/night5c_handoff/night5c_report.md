# SpaLORA Night-5C report

## Outcome

Status: `NIGHT5C_CORRECTION_COMPLETE_CANDIDATES_LOCKED_FOR_P22`.

Future locked P22 candidates: `B01_C04_SHRINK25`, `B10_SHRINK25_ANCHOR10`, `B17_C09_DIFFUSE10`. No P22 run was performed.

## Runtime semantic correction

All 24 corrected units used actual `uniform_all`: each manifest contains declared and resolved contracts, both contract SHAs, `semantic_contract_match=true`, exact within/cross alpha `[0.5,0.5]`, and zero learnable attention parameters. The eight preflight construction probes covered four candidates x two datasets at seed 0; the negative shrink regression was rejected before training.

The 24 old B21-B24 units remain permanently invalid and untouched. `supersession_map.json` links them one-to-one to the new corrected manifests. All 84 unaffected Night-5B training manifests were independently re-hashed; reruns/overwrites: 0/0.

## Corrected three-seed results

```csv
candidate_id,dataset,ari,nmi,q,spatial_neighbor_agreement,spatial_cluster_moran_mean,spatial_cluster_geary_mean,boundary_disagreement,runtime_seconds,gpu_peak_allocated_mib
B21_C09_LAPLACIAN005,a1,0.2233340029665063,0.37104914157708097,0.29719157227179366,0.6951832716758091,0.6296281630267042,0.37897650513341335,0.31535201772127514,4.057821510980527,704.724609375
B21_C09_LAPLACIAN005,placenta,0.664139869049289,0.6960757704592971,0.680107819754293,0.5148415563578018,0.5108551224297266,0.6510058963509501,0.630461018585566,2.8451406244809427,404.05419921875
B22_C09_LAPLACIAN010,a1,0.20758102866792594,0.3603240574003437,0.28395254303413475,0.7490645065271931,0.7008490791274772,0.30615873691556256,0.26071462968313586,4.047642799094319,704.724609375
B22_C09_LAPLACIAN010,placenta,0.49692708371824795,0.5748503800272841,0.535888731872766,0.5930605695948656,0.5831457267706832,0.6094628452352998,0.5855660149650013,2.906236703818043,404.05419921875
B23_C10_LAPLACIAN005,a1,0.19832030475881735,0.3486247725035458,0.2734725386311816,0.7032146957520092,0.6515315259489242,0.3578218436451919,0.3078493691611287,4.433118640445173,706.08154296875
B23_C10_LAPLACIAN005,placenta,0.6686922405338004,0.7167475566164602,0.6927198985751303,0.47131969514640987,0.47449657601650314,0.6632745887191379,0.6555636012551291,3.2170162039498487,404.130859375
B24_C10_LAPLACIAN010,a1,0.2020071933147625,0.3555626412289257,0.2787849172718441,0.7419579453161543,0.6865271156620246,0.3225607844584138,0.26923817779061926,4.431799440334241,706.08154296875
B24_C10_LAPLACIAN010,placenta,0.46128911007340356,0.5663928724108604,0.513840991242132,0.592860008022463,0.573985390117491,0.605380441140036,0.573256094617427,3.3394948917751512,404.130859375
```

## S1 replay and conditional stage

Old top-up set: `B09_SHRINK25_ANCHOR05`, `B10_SHRINK25_ANCHOR10`, `B06_SECONDLOOK_RNA_ANCHOR05`, `B15_LATENT_RELIABILITY50`.
Recomputed set: `B09_SHRINK25_ANCHOR05`, `B10_SHRINK25_ANCHOR10`, `B06_SECONDLOOK_RNA_ANCHOR05`, `B15_LATENT_RELIABILITY50`.
Added/removed: none. No Laplacian candidate newly entered, so conditional seeds 3/4 ran 0 units.

## Eligibility and frontiers

Canonical five-seed: `B01_C04_SHRINK25`, `B02_C09_RNA_ANCHOR10`, `B03_C10_MNN_TRIPLET01`, `B17_C09_DIFFUSE10`, `B18_C09_DIFFUSE25`, `B19_C10_DIFFUSE10`, `B20_C10_DIFFUSE25`, `B09_SHRINK25_ANCHOR05`, `B10_SHRINK25_ANCHOR10`, `B06_SECONDLOOK_RNA_ANCHOR05`, `B15_LATENT_RELIABILITY50`.
Exploratory five-seed: none.
B10 is canonical and the strict balanced choice above B01. B17 is canonical and the accuracy-frontier leader. B21-B24 remain valid corrected three-seed development evidence but are noncanonical because they were not selected for top-up.

Balanced frontier: `B10_SHRINK25_ANCHOR10`, `B17_C09_DIFFUSE10`, `B01_C04_SHRINK25`, `B09_SHRINK25_ANCHOR05`, `B19_C10_DIFFUSE10`, `B18_C09_DIFFUSE25`.
Accuracy frontier: `B17_C09_DIFFUSE10`, `B02_C09_RNA_ANCHOR10`, `B06_SECONDLOOK_RNA_ANCHOR05`, `B19_C10_DIFFUSE10`, `B03_C10_MNN_TRIPLET01`.
All candidate summaries contain deltas against historical B00 and explicit `relative_b01` fields. Diffusion gating uses source training cost plus preserved historical incremental zero; the zero is not interpreted as end-to-end cost.

## Protocol and budget

Training: 24 mandatory + 0 conditional = 24 scientific units; failures 0; same-tuple retries 0; total attempts 24, within 40/44 caps. P22, D1, GSE198353, and Night-4B access/run counts are all zero.

## Engineering and delivery

Tests, Git commit/tag/push, compact D-drive delivery hashes, and shutdown dispatch are finalized after this report draft; machine shutdown status must only claim command dispatch exit status, not console power state.
