# Night-18D Human Placenta Frozen Structured-Energy External Transfer

## 我现在需要知道的三件事

1. **冻结迁移没有成功。** Night-15F 的 RNA+chromatin 家族中心配置在 1,662 个 human placenta 细胞、K=10 协议上的正式无标签输出只有 ARI/NMI **0.079067/0.132742**；最强匹配对照分别达到 ARI 0.370597 和 NMI 0.533451。主分类是 **SCIENTIFIC_NEGATIVE**。
2. **问题发生在完整结构能量消费者，而不是数据闭合。** no-op medoid 为 0.351753/0.533451，L2 low-pass 为 0.370597/0.526850；full 相对同起点 no-op 在 13/13 start 中没有一次 ARI/NMI 双升，且正式 full medoid 恰好是一个未移动的 ATAC-start 分区，不能归因于 decoder。
3. **负结果可信且可复算。** 官方 ATAC 文件与现有容器虽然容器 SHA 不同，但 X、ordered IDs、63 个 TF 相关调控特征、obs 和坐标数值全部闭合；formal partition 与三张指标表 fresh-process byte-exact，5 个 targeted tests 通过。AutoDL 保持开机，未派发 shutdown。

## 本轮实际改了哪一层

本轮没有训练新 backbone，也没有增加 selector。我们把 Night-15F 的完整结构化能量——动态多模态 prototype unary、三尺度稀疏空间图、冲突感知 pairwise、rejected-mass self-return 与 alpha-expansion——作为一个冻结消费者，接在新 human placenta 的 label-free RNA/ATAC 数值表示与 13 个机械 start 上。RNA 和官方 processed ATAC-derived TF-associated regulatory features 分别降维；两个独立 PCA 坐标系先用同位点正交 Procrustes 对齐，再融合。K=10 来自公开 annotation 协议；逐点 cell_type 只在 130 个 candidate partitions 全部保存并哈希后由独立 evaluator 打开。

## 绝对指标主表（family-frozen primary profile）

数据协议：human placenta，N/eval=1662/1662，K=10，all-cell original-author cell-type annotation。

| Arm | ARI | NMI | AMI | FMI | Moran macro | Geary macro | min cluster | changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NO_OP_START | 0.351753 | 0.533451 | 0.527461 | 0.455774 | 0.185354 | 0.821487 | 63 | 0 |
| L2_LOWPASS_MATCHED | 0.370597 | 0.526850 | 0.520703 | 0.473341 | 0.280484 | 0.734014 | 52 | 518 |
| FULL_FROZEN_ENERGY | 0.079067 | 0.132742 | 0.121888 | 0.209290 | 0.029652 | 0.969985 | 113 | 0 |
| REGISTERED_SCALE_ONLY | 0.309905 | 0.484815 | 0.478264 | 0.418387 | 0.204347 | 0.795766 | 85 | 104 |
| NO_SELF_RETURN_STAY | 0.010557 | 0.150762 | 0.134573 | 0.375718 | 0.480195 | 0.502178 | 4 | 1356 |
| PAIRWISE_ZERO_KEEP_STAY | 0.348180 | 0.508719 | 0.502431 | 0.452444 | 0.217835 | 0.793143 | 55 | 76 |
| PURE_DYNAMIC_UNARY | 0.316972 | 0.451548 | 0.444553 | 0.424800 | 0.248878 | 0.762454 | 54 | 237 |
| SINGLE_SITE_SAME_ENERGY | 0.109972 | 0.193994 | 0.183547 | 0.247496 | 0.686007 | 0.321946 | 52 | 716 |

Full 相对最强 ARI 对照差值为 **-0.291530**，相对最强 NMI 对照差值为 **-0.400709**。这些“最强”按每个指标分别计算，没有把两个不同对照伪装成同一分区。label-assisted locked-candidate oracle 仅作开发附表，不能替代无标签 medoid；它也没有挽救 family-default full。

## 贡献归因

- **上游 start/简单平滑有可用信号：** no-op 与 L2 low-pass 显著高于 full；锁定候选中的 label-assisted L2 best 仅作为 benchmark development context。
- **完整 frozen energy 无独立贡献：** full selected ARI/NMI 同时低于 no-op、L2、registered-scale、pairwise-zero 和 pure-unary 等关键对照；同起点配对双升为 0/13。
- **self-return 是保护项但不是成功机制：** 关闭 self-return 后出现 min cluster=4 且分数进一步下降；这说明 stay cost 防止崩坏，却不能证明完整能量可迁移。
- **solver 不是唯一原因：** matched single-site 也低分，且 full alpha-expansion 本身未超过强对照。
- **不作原创性扩张：** prototype、邻域统计、多尺度图、Potts/CRF 与 alpha-expansion 都有明确先例；本轮只检验既有组合的外部迁移，结果为负。

## 数据与标签语义

- RNA: 1662×36601 sparse counts；ATAC: 1662×63 **official processed ATAC-derived TF-associated regulatory features**，不是 raw peaks 或 LSI。
- 官方 repo ATAC SHA-256: `f53ef887ec5b96a79d19ded1ac21a7e4ef44ae0d07ab572706bc076350f1460d`；现有容器 SHA-256: `ed3ead8662a1dbeae01bec9b3b0a7e1a9f6b9a23825672b7e1c9b22887d1999e`。数值/ID/feature/obs/coordinates exact audit 均 PASS。
- source H5AD 的 obs 元数据在文件级随 AnnData 载入，但 carrier 计算没有索引 annotation 列；producer 只读 sanitized numeric carrier。authority JSON 在 producer 前只消费 `status` 与 `authority_gap` 两个 allow-list 字段。
- evaluator 在 formal artifact SHA `b27e3b7b210d755b43af4dc28585e3c14523701ac34ded9975bab1b7dd691074` 锁定后读取 `cell_type`；ordered label hash 为 `2913bec86ce371b52d70de8faf29ad3e9f6a3f79e8e0751bd58738afb91594ff`。

## 失败、限制与下一步

1. family-center 参数来自 P22 K9 与 MISAR K7 的数值几何中心，外部 placenta 的细胞级坐标图、类别不均衡与 63 维调控特征响应明显不同；本轮结果说明这种参数中心不能直接当作可迁移方法。
2. label-free medoid 是预注册消费者，不保证选择指标最优 start；但 full 的逐 start 配对也 0/13 双升，因此负结论不由 medoid 单独造成。
3. placenta 是细胞类型恢复协议，不应与组织域 benchmark 数字混表；本轮不声称 SOTA、paper-ready 或 confirmed milestone。
4. 如果继续该数据，优先研究 family-level calibration 或更强 raw-feature representation；不应在本轮结果后按 placenta 标签回调旧能量。

## 导师汇报版

我们第一次把 Night-15F 的完整结构能量原封不动迁移到一个此前未参与开发的高质量 human placenta RNA+ATAC 单元。数据权威、1662 个细胞的 ID、坐标、63 个 ATAC 衍生调控特征和 K=10 注释协议均已闭合。冻结 family profile 的正式无标签输出只有 ARI/NMI 0.079067/0.132742，明显低于 no-op 和简单 L2 low-pass。13 个相同起点中 full 没有一次实现 ARI/NMI 双升，说明失败不是 selector 偶然选错一个候选。关闭 self-return 会更差，表明它有防崩作用，但完整 decoder 仍不具备跨研究迁移性。因此本轮是高价值的科学负结果：它阻止我们把 P22/MISAR 上的开发高分误写成普适方法。工程上 formal 与 fresh-process replay 完全一致，交付可复算；下一步若继续，应转向可解释的 family calibration 或更强 representation，而不是继续包装旧能量。

## 技术附录摘要

- carrier SHA-256: `08eaa461858eb3de987fc69803a5a124df834cf978a05a2f7d2b087a80b67dd5`；13 starts；图 nnz=[8308, 16196, 35890]
- family config SHA-256: `b4e68265f6e5854b183f152922b3aebab7ba382c519c3311880520d26ac4e4c6`；source P22/MISAR hashes={"MISAR_E15_5_S1": "b2a34f991f31270a4fe747d64ecfb2c0762ac7b73d0ee8b60eacb947745331fa", "P22": "d60a73221abf6c6e975fbc778153fdd3885dd28979ed57b1155232389854b5e2"}
- formal candidates: 130；labels read by producer: 0；dense N×N: 0
- replay: partition artifact byte-exact；all/selected/paired CSV byte-exact
- shutdown_dispatched=false；AutoDL action=`KEEP_ON_FOR_WORKER2_REVIEW`
