# SpaLORA Night-15A 多模态贡献与分数稳定性报告

## 我现在需要知道的三件事

1. **问题**：Night-14B 的 P22/MISAR 高分究竟由坐标、RNA+ATAC 分子信息，还是聚类 head 产生；本轮用完全相同的 K、mask、endpoint seed 和 head 做了逐层拆分。
2. **实际动作与流水线位置**：先在冻结表示/聚类端做 coordinate-only、两条单模态、融合、去坐标、去滤波、去 refinement 对照；再从真实 feature-level RNA/ATAC 经 HVG/TF-IDF/LSI 或 Moran feature bank 训练统一 cross-reconstruction/MCDF 核心，并把 3–5 个机制候选冻结后用 backbone seeds 3–7 复核。所有 checkpoint 都做 fresh-process 数值回放。
3. **论文含义**：主终态为 `NO_ADDED_METHOD_SIGNAL`，次级信号为 `STABLE_HEAD_SIGNAL`。这不是 SOTA、confirmed milestone 或论文结论；它只回答现有分数来源，并决定 MCDF 是否值得继续作为主方法。

## 结果分类

- 主终态：`NO_ADDED_METHOD_SIGNAL`
- 次级信号：`STABLE_HEAD_SIGNAL`
- 聚类语义：公开 benchmark、标签后置 evaluator 的 known-K/protocol-K 无监督聚类；标签没有进入模型输入、loss、gradient 或单次训练 checkpoint selection。

## 相同 head 的分数来源主表

| dataset | cluster_k | candidate_id | endpoint_role | control_id | ari_best | ari_median | ari_mean | ari_min | nmi_best | nmi_median | nmi_mean | nmi_min | ami_mean | fmi_mean | morans_i_mean | gearys_c_mean | wall_seconds | gpu_seconds | peak_gpu_mib | peak_rss_mib |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MISAR_E15_5_S1 | 7 | F32_MISAR_K7_MAX_ARI | nan | FUSED_WITHOUT_GRAPH_FILTER | 0.5143 | 0.4256 | 0.4230 | 0.3350 | 0.6017 | 0.5630 | 0.5611 | 0.5162 | 0.5588 | 0.5349 | 0.9161 | 0.0842 | 4.8876 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 7 | F32_MISAR_K7_MAX_ARI | nan | ATAC_PLUS_COORDINATES | 0.4724 | 0.4237 | 0.4190 | 0.3375 | 0.6016 | 0.5575 | 0.5548 | 0.5075 | 0.5524 | 0.5314 | 0.9291 | 0.0714 | 5.1639 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 7 | F32_MISAR_K7_MAX_ARI | nan | FUSED_FULL | 0.5099 | 0.4032 | 0.4040 | 0.3007 | 0.6290 | 0.5493 | 0.5511 | 0.4726 | 0.5487 | 0.5200 | 0.9284 | 0.0719 | 5.7354 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 7 | F32_MISAR_K7_MAX_ARI | nan | NIGHT14B_FROZEN_BEST | 0.5099 | 0.4032 | 0.4040 | 0.3007 | 0.6290 | 0.5493 | 0.5511 | 0.4726 | 0.5487 | 0.5200 | 0.9284 | 0.0719 | 0.0000 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 7 | F32_MISAR_K7_MAX_ARI | nan | FUSED_WITHOUT_SPATIAL_REFINEMENT | 0.5041 | 0.4028 | 0.4029 | 0.2981 | 0.6248 | 0.5526 | 0.5520 | 0.4745 | 0.5496 | 0.5192 | 0.9248 | 0.0755 | 4.6237 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 7 | F32_MISAR_K7_MAX_ARI | nan | COORDINATE_ONLY | 0.3861 | 0.3776 | 0.3784 | 0.3769 | 0.5042 | 0.4913 | 0.4924 | 0.4905 | 0.4896 | 0.5005 | 0.9367 | 0.0604 | 0.9256 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 7 | F32_MISAR_K7_MAX_ARI | nan | RNA_PLUS_COORDINATES | 0.4262 | 0.3740 | 0.3703 | 0.2129 | 0.5710 | 0.5193 | 0.5200 | 0.3831 | 0.5175 | 0.4928 | 0.9236 | 0.0758 | 5.2792 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 7 | F32_MISAR_K7_MAX_ARI | nan | RNA_ONLY | 0.4048 | 0.2763 | 0.2568 | 0.1294 | 0.5095 | 0.4486 | 0.4440 | 0.3568 | 0.4409 | 0.4133 | 0.9029 | 0.0973 | 5.6697 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 7 | F32_MISAR_K7_MAX_ARI | nan | FUSED_WITHOUT_COORDINATES | 0.4849 | 0.2664 | 0.2647 | 0.1104 | 0.5264 | 0.4305 | 0.4308 | 0.2889 | 0.4277 | 0.4240 | 0.9003 | 0.0998 | 5.4534 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 7 | F32_MISAR_K7_MAX_ARI | nan | ATAC_ONLY | 0.4287 | 0.2315 | 0.2476 | 0.1145 | 0.5330 | 0.4089 | 0.4102 | 0.3022 | 0.4070 | 0.4028 | 0.8985 | 0.1014 | 5.1509 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 12 | F33_MISAR_K12_MAX_ARI | nan | ATAC_PLUS_COORDINATES | 0.4199 | 0.2724 | 0.2787 | 0.1932 | 0.5332 | 0.4707 | 0.4745 | 0.4072 | 0.4697 | 0.4140 | 0.8991 | 0.1017 | 4.6213 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 12 | F33_MISAR_K12_MAX_ARI | nan | FUSED_FULL | 0.4143 | 0.2650 | 0.2742 | 0.1905 | 0.5329 | 0.4773 | 0.4754 | 0.4118 | 0.4707 | 0.4060 | 0.9000 | 0.1020 | 4.5160 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 12 | F33_MISAR_K12_MAX_ARI | nan | NIGHT14B_FROZEN_BEST | 0.4143 | 0.2650 | 0.2742 | 0.1905 | 0.5329 | 0.4773 | 0.4754 | 0.4118 | 0.4707 | 0.4060 | 0.9000 | 0.1020 | 0.0000 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 12 | F33_MISAR_K12_MAX_ARI | nan | FUSED_WITHOUT_SPATIAL_REFINEMENT | 0.3779 | 0.2623 | 0.2697 | 0.1977 | 0.5446 | 0.4958 | 0.4894 | 0.4350 | 0.4849 | 0.4003 | 0.8981 | 0.1043 | 3.4534 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 12 | F33_MISAR_K12_MAX_ARI | nan | COORDINATE_ONLY | 0.2410 | 0.2262 | 0.2270 | 0.2177 | 0.4712 | 0.4658 | 0.4643 | 0.4509 | 0.4598 | 0.3572 | 0.9220 | 0.0785 | 0.7561 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 12 | F33_MISAR_K12_MAX_ARI | nan | RNA_PLUS_COORDINATES | 0.2787 | 0.2046 | 0.2154 | 0.1674 | 0.4721 | 0.4246 | 0.4274 | 0.3881 | 0.4222 | 0.3527 | 0.8943 | 0.1053 | 4.6734 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 12 | F33_MISAR_K12_MAX_ARI | nan | FUSED_WITHOUT_GRAPH_FILTER | 0.3176 | 0.2028 | 0.2044 | 0.1394 | 0.5098 | 0.4176 | 0.4176 | 0.3447 | 0.4121 | 0.3549 | 0.8226 | 0.1797 | 4.3763 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 12 | F33_MISAR_K12_MAX_ARI | nan | ATAC_ONLY | 0.3499 | 0.1732 | 0.1706 | 0.0534 | 0.4514 | 0.3964 | 0.3929 | 0.3224 | 0.3866 | 0.3859 | 0.8634 | 0.1369 | 5.1153 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 12 | F33_MISAR_K12_MAX_ARI | nan | FUSED_WITHOUT_COORDINATES | 0.3287 | 0.1579 | 0.1682 | 0.0697 | 0.4631 | 0.3805 | 0.3843 | 0.3387 | 0.3781 | 0.3684 | 0.8687 | 0.1332 | 5.2784 | 0.0000 | 0.0000 | 2008.8477 |
| MISAR_E15_5_S1 | 12 | F33_MISAR_K12_MAX_ARI | nan | RNA_ONLY | 0.2146 | 0.1271 | 0.1321 | 0.0748 | 0.4075 | 0.3809 | 0.3808 | 0.3500 | 0.3748 | 0.3184 | 0.8739 | 0.1273 | 4.5383 | 0.0000 | 0.0000 | 2008.8477 |
| P22 | 9 | F30_P22_K9_MAX_ARI | nan | FUSED_WITHOUT_GRAPH_FILTER | 0.5192 | 0.4908 | 0.4936 | 0.4800 | 0.6548 | 0.6335 | 0.6350 | 0.6144 | 0.6344 | 0.5697 | 0.9320 | 0.0702 | 24.1124 | 0.0000 | 0.0000 | 2008.8477 |
| P22 | 9 | F30_P22_K9_MAX_ARI | nan | FUSED_FULL | 0.5691 | 0.4831 | 0.4960 | 0.4330 | 0.6851 | 0.6189 | 0.6247 | 0.5667 | 0.6241 | 0.5722 | 0.9303 | 0.0726 | 18.5718 | 0.0000 | 0.0000 | 2008.8477 |
| P22 | 9 | F30_P22_K9_MAX_ARI | nan | NIGHT14B_FROZEN_BEST | 0.5691 | 0.4831 | 0.4960 | 0.4330 | 0.6851 | 0.6189 | 0.6247 | 0.5667 | 0.6241 | 0.5722 | 0.9303 | 0.0726 | 0.0000 | 0.0000 | 0.0000 | 2008.8477 |
| P22 | 9 | F30_P22_K9_MAX_ARI | nan | RNA_PLUS_COORDINATES | 0.5390 | 0.4824 | 0.4819 | 0.4162 | 0.6555 | 0.6066 | 0.6091 | 0.5708 | 0.6084 | 0.5588 | 0.9353 | 0.0672 | 19.4236 | 0.0000 | 0.0000 | 2008.8477 |
| P22 | 9 | F30_P22_K9_MAX_ARI | nan | ATAC_PLUS_COORDINATES | 0.5533 | 0.4810 | 0.4806 | 0.4100 | 0.6567 | 0.6199 | 0.6189 | 0.5646 | 0.6182 | 0.5586 | 0.9323 | 0.0707 | 18.1550 | 0.0000 | 0.0000 | 2008.8477 |
| P22 | 9 | F30_P22_K9_MAX_ARI | nan | FUSED_WITHOUT_SPATIAL_REFINEMENT | 0.5610 | 0.4798 | 0.4916 | 0.4336 | 0.6750 | 0.6116 | 0.6191 | 0.5670 | 0.6185 | 0.5683 | 0.9230 | 0.0805 | 14.1099 | 0.0000 | 0.0000 | 2008.8477 |
| P22 | 9 | F30_P22_K9_MAX_ARI | nan | FUSED_WITHOUT_COORDINATES | 0.5060 | 0.4485 | 0.4438 | 0.3546 | 0.6405 | 0.6042 | 0.5965 | 0.5305 | 0.5958 | 0.5317 | 0.9175 | 0.0863 | 18.6044 | 0.0000 | 0.0000 | 2008.8477 |
| P22 | 9 | F30_P22_K9_MAX_ARI | nan | ATAC_ONLY | 0.5197 | 0.4476 | 0.4533 | 0.3824 | 0.6424 | 0.5962 | 0.5987 | 0.5404 | 0.5980 | 0.5405 | 0.9115 | 0.0924 | 18.1350 | 0.0000 | 0.0000 | 2008.8477 |
| P22 | 9 | F30_P22_K9_MAX_ARI | nan | RNA_ONLY | 0.4725 | 0.4347 | 0.4346 | 0.4068 | 0.6391 | 0.5900 | 0.5916 | 0.5669 | 0.5909 | 0.5211 | 0.9277 | 0.0753 | 19.8589 | 0.0000 | 0.0000 | 2008.8477 |
| P22 | 9 | F30_P22_K9_MAX_ARI | nan | COORDINATE_ONLY | 0.2855 | 0.2846 | 0.2845 | 0.2827 | 0.4642 | 0.4628 | 0.4630 | 0.4619 | 0.4621 | 0.3821 | 0.9662 | 0.0338 | 4.1447 | 0.0000 | 0.0000 | 2008.8477 |

## 冻结后 seeds 3–7 稳定性与 exact protocol context 主表

| dataset | cluster_k | candidate_id | endpoint_role | control_id | ari_best | ari_median | ari_mean | ari_min | nmi_best | nmi_median | nmi_mean | nmi_min | ami_mean | fmi_mean | morans_i_mean | gearys_c_mean | wall_seconds | gpu_seconds | peak_gpu_mib | peak_rss_mib |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MISAR_E15_5_S1 | 7 | R40_MCDF_MASKED_MODALITY | LABEL_FREE_PARTITION_MEDOID | z1 | 0.4534 | 0.4295 | 0.4182 | 0.3774 | 0.6057 | 0.5760 | 0.5683 | 0.5314 | 0.5660 | 0.5317 | 0.9209 | 0.0795 | 120.6725 | 118.7012 | 354.7144 | 939.0547 |
| MISAR_E15_5_S1 | 7 | R40_MCDF_MASKED_MODALITY | BASE_RANDOM_START | z1 | 0.4872 | 0.4083 | 0.4034 | 0.2611 | 0.6191 | 0.5558 | 0.5557 | 0.4602 | 0.5533 | 0.5194 | 0.9190 | 0.0813 | 125.9264 | 118.7012 | 354.7144 | 939.0547 |
| MISAR_E15_5_S1 | 7 | R21_MCDF_STRONG_CONTRIBUTION | LABEL_FREE_PARTITION_MEDOID | z1 | 0.4377 | 0.4082 | 0.3965 | 0.3411 | 0.5609 | 0.5253 | 0.5251 | 0.4934 | 0.5225 | 0.5148 | 0.8381 | 0.1646 | 132.6826 | 131.3729 | 354.7500 | 933.9141 |
| MISAR_E15_5_S1 | 7 | R21_MCDF_STRONG_CONTRIBUTION | BASE_RANDOM_START | z1 | 0.4402 | 0.3753 | 0.3762 | 0.3091 | 0.5632 | 0.5114 | 0.5153 | 0.4747 | 0.5128 | 0.4969 | 0.8348 | 0.1669 | 138.1803 | 131.3729 | 354.7500 | 933.9141 |
| MISAR_E15_5_S1 | 7 | R10_FIXED_MULTISCALE | LABEL_FREE_PARTITION_MEDOID | z2 | 0.3117 | 0.2992 | 0.2839 | 0.2228 | 0.5004 | 0.4872 | 0.4727 | 0.4274 | 0.4699 | 0.4239 | 0.8962 | 0.1043 | 1488.8230 | 1476.6567 | 261.7847 | 933.5625 |
| MISAR_E15_5_S1 | 7 | R10_FIXED_MULTISCALE | BASE_RANDOM_START | z2 | 0.3741 | 0.2857 | 0.2872 | 0.1873 | 0.5419 | 0.4744 | 0.4712 | 0.4034 | 0.4683 | 0.4288 | 0.8929 | 0.1081 | 1493.4658 | 1476.6567 | 261.7847 | 933.5625 |
| MISAR_E15_5_S1 | 12 | R40_MCDF_MASKED_MODALITY | LABEL_FREE_PARTITION_MEDOID | z2 | 0.3455 | 0.3350 | 0.3372 | 0.3341 | 0.5680 | 0.5594 | 0.5597 | 0.5474 | 0.5558 | 0.4600 | 0.8254 | 0.1777 | 120.6725 | 118.7012 | 354.7144 | 939.0547 |
| MISAR_E15_5_S1 | 12 | R40_MCDF_MASKED_MODALITY | BASE_RANDOM_START | z2 | 0.3580 | 0.3343 | 0.3284 | 0.2511 | 0.5736 | 0.5527 | 0.5513 | 0.4866 | 0.5474 | 0.4520 | 0.8266 | 0.1760 | 127.1952 | 118.7012 | 354.7144 | 939.0547 |
| MISAR_E15_5_S1 | 12 | R21_MCDF_STRONG_CONTRIBUTION | BASE_RANDOM_START | z2 | 0.3737 | 0.3073 | 0.3078 | 0.2269 | 0.5890 | 0.5425 | 0.5445 | 0.4771 | 0.5406 | 0.4334 | 0.8879 | 0.1120 | 139.5074 | 131.3729 | 354.7500 | 933.9141 |
| MISAR_E15_5_S1 | 12 | R10_FIXED_MULTISCALE | BASE_RANDOM_START | z2 | 0.3559 | 0.2994 | 0.2956 | 0.2330 | 0.5887 | 0.5382 | 0.5385 | 0.4956 | 0.5346 | 0.4222 | 0.8898 | 0.1103 | 1495.2914 | 1476.6567 | 261.7847 | 933.5625 |
| MISAR_E15_5_S1 | 12 | R10_FIXED_MULTISCALE | LABEL_FREE_PARTITION_MEDOID | z2 | 0.3369 | 0.2988 | 0.2945 | 0.2462 | 0.5726 | 0.5363 | 0.5388 | 0.5114 | 0.5348 | 0.4213 | 0.8937 | 0.1062 | 1488.8230 | 1476.6567 | 261.7847 | 933.5625 |
| MISAR_E15_5_S1 | 12 | R21_MCDF_STRONG_CONTRIBUTION | LABEL_FREE_PARTITION_MEDOID | z2 | 0.3367 | 0.2893 | 0.2980 | 0.2794 | 0.5681 | 0.5341 | 0.5412 | 0.5284 | 0.5372 | 0.4243 | 0.8844 | 0.1152 | 132.6826 | 131.3729 | 354.7500 | 933.9141 |
| P22 | 9 | R21_MCDF_STRONG_CONTRIBUTION | BASE_RANDOM_START | mcdf | 0.4939 | 0.4487 | 0.4546 | 0.4237 | 0.6326 | 0.6002 | 0.6081 | 0.5843 | 0.6074 | 0.5364 | 0.9101 | 0.0926 | 685.3857 | 652.0762 | 1591.2900 | 960.8008 |
| P22 | 9 | R21_MCDF_STRONG_CONTRIBUTION | LABEL_FREE_PARTITION_MEDOID | mcdf | 0.4935 | 0.4486 | 0.4538 | 0.4237 | 0.6323 | 0.6002 | 0.6077 | 0.5848 | 0.6070 | 0.5357 | 0.9103 | 0.0925 | 656.4659 | 652.0762 | 1591.2900 | 960.8008 |
| P22 | 9 | R40_MCDF_MASKED_MODALITY | BASE_RANDOM_START | z2 | 0.4742 | 0.4255 | 0.4312 | 0.3822 | 0.6174 | 0.5880 | 0.5902 | 0.5679 | 0.5895 | 0.5170 | 0.9172 | 0.0856 | 226.4648 | 196.2034 | 1541.4355 | 1019.4258 |
| P22 | 9 | R40_MCDF_MASKED_MODALITY | LABEL_FREE_PARTITION_MEDOID | z2 | 0.4724 | 0.4240 | 0.4264 | 0.3920 | 0.6065 | 0.5883 | 0.5886 | 0.5749 | 0.5879 | 0.5132 | 0.9169 | 0.0860 | 201.1337 | 196.2034 | 1541.4355 | 1019.4258 |
| P22 | 9 | R10_FIXED_MULTISCALE | LABEL_FREE_PARTITION_MEDOID | mcdf | 0.4487 | 0.4209 | 0.4253 | 0.4127 | 0.6156 | 0.5941 | 0.6011 | 0.5896 | 0.6004 | 0.5076 | 0.9444 | 0.0564 | 1898.8295 | 1835.4368 | 1137.5098 | 951.2070 |
| P22 | 9 | R10_FIXED_MULTISCALE | BASE_RANDOM_START | mcdf | 0.4934 | 0.4151 | 0.4159 | 0.3566 | 0.6271 | 0.5942 | 0.5944 | 0.5652 | 0.5937 | 0.4995 | 0.9444 | 0.0563 | 1922.8306 | 1835.4368 | 1137.5098 | 951.2070 |
| P22_3DOT_K18 | 18 | LOCKED_C15_W02_REFERENCE | nan | z2 | 0.6122 | 0.6122 | 0.6122 | 0.6122 | 0.7233 | 0.7233 | 0.7233 | 0.7233 | 0.7216 | 0.6533 | 0.8605 | 0.1434 | 0.6082 | nan | nan | nan |

完整逐行结果见 `main_results_table.csv`、`raw_feature_and_mcdf_results.csv` 和 `all_run_ledger.csv`。BEST、median、mean、min 分开保留，任何坏 seed、失败或提前停止路线均未删除。

### 直接解释

- P22 K=9 的 fused mean ARI 0.4960，高于 RNA+coordinates 0.4819 与 ATAC+coordinates 0.4806，但优势小且 backbone seeds 3–7 未维持：最好统一候选 R21 的 ARI 为 best 0.4939、median 0.4487、mean 0.4546。
- MISAR K=7 的 fused mean ARI 0.4040 低于 ATAC+coordinates 0.4190；K=12 的 fused mean 0.2742 也略低于 ATAC+coordinates 0.2787，因此不能登记稳定多模态贡献。
- label-free partition medoid 对 R40/MISAR K=7 把 ARI min 从 0.2611 提到 0.3774、median 从 0.4083 提到 0.4295；它属于稳分 head 设施，不是 MCDF 方法增益。
- MISAR K=12 的冻结候选最高 ARI 仅 0.3737，远低于 0.50；P22 K=18 的 0.6122/0.7233 来自锁定参考表示对官方作者 18-state assignment 的 exact-protocol context，不是本轮新模型成绩。

## 协议边界

- P22 K=9 使用项目 9,196/9,196 exact ground truth。
- P22 K=18 使用 3d-OT 官方 h5ad 中作者提供的 18-state domain assignment；它与项目 IDs byte-exact 同序，**不是**把 K=9 人工拆分，但也不是独立专家 ground truth，因此只作 protocol context。
- MISAR K=7 使用七类公开 Y；SEPAR K=12 lane 忠实保留“聚成 12 类、对七类 Y 评价”的官方 Tutorial4 语义。
- GSE213264 Spatial-CITE-seq tonsil 与 canonical Zenodo tonsil 三切片不是重复资产：accession/platform/spot 数均不同，三个 exact ID 交集均为 0。

## 最重要的失败与限制

- exact SEPAR 第一次完成 100 次迭代后，在最终 clustering 处因 upstream AnnData ArrayView 兼容性失败；该失败完整保留。v2 只改变数组 materialization，但为避免外部方法挤占主模型预算而提前停止，因此本轮没有把 SEPAR 外部实测分数写入 own-method evidence。
- 2,500+2,500 Moran feature bank 因 RNA 有限可识别特征不足而 fail-closed；仅进行一次透明平台数值修订到 1,500+1,500。
- raw-feature trainable candidates 若只在开发 best run 上好看、但 seeds 3–7 的 median/mean 未保持，则不登记方法信号。
- P22 K=18 与 MISAR K=12 都是特定论文协议，不与 K=9/K=7 数字伪装成同一任务的直接胜负。
- A1 与 tonsil s1 完成统一模型工程训练/回放；由于两条主 RNA+ATAC lane 均无新增方法信号，D1 与 tonsil s2/s3 未继续消耗预算，状态透明保留为 NOT_RUN_AFTER_PRIMARY_NEGATIVE。

## 5–8 句导师汇报版

Night-15A 把 Night-14B 的高分拆成坐标、单模态、融合和聚类 head 四层。坐标很重要，但 coordinate-only 并不能解释全部 P22/MISAR 得分。P22 的融合在开发最佳值上仍可超过单模态，但 MISAR 尤其 K=12 没有形成稳定的融合优势。我们从真实 feature-level RNA/ATAC 训练了同一个统一核心，并用未参与 HPO 的 backbone seeds 3–7 检查，而不是重复最好 seed。P22 K=18 与 SEPAR K=12 的官方协议已闭合并单独报告。GSE213264 tonsil 已证实不是现有 canonical tonsil 的重复切片。最终分类是 `NO_ADDED_METHOD_SIGNAL`（次级 `STABLE_HEAD_SIGNAL`），所以本轮不能写成新模块已成立或 SOTA。下一步只能围绕真正保留下来的层继续，不能再包装无独立增量的 MCDF 门控。

## 技术附录

- fresh-process checkpoint numerical round-trip：formal unseen 30/30；active evidence 62/62。另有 1 条早期 superseded 失败原样保留。
- 训练总 GPU 秒：11367.20；峰值 GPU：1929.24 MiB；峰值训练 RSS：1023.18 MiB。
- training label reads=0；labels in loss/gradient/checkpoint selection=false；dense N×N=0。
- 最终 commit/tag、bundle 与 compact index 在交付 manifest 中记录；本报告自身不循环嵌入最终 commit hash。
