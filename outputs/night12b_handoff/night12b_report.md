# SpaLORA Night-12B 执行报告

## 我现在需要知道的三件事

1. 本轮原本要检验：两个 discovery repeats 中“正确 gene link 相对全部错误 links 的优势、空间块稳定性和重复一致性”，能否预测第三个完全 held-out repeat 的 link utility。
2. 实际已经在 feature-level RNA/ATAC/ADT 后、任何聚类或候选神经模型之前，闭合了 6/6 真实预处理、空间块、OLS、穷尽 decoy 和 64 次 block bootstrap；但独立 leave-one-replicate-out（LORO，留一重复验证）汇总在第一次 Spearman 调用处因冻结代码与当前 SciPy 返回字段不兼容而停止。
3. 因唯一一次全局 correction cycle 已在 formal 前的数学等价 OLS refreeze 中用完，本轮不能再修第二次。因此结论是实现失败，不是科学正结果，也不是科学负结果；不能据此决定 RNA+ATAC 或 RNA+protein 的 link evidence 是否可识别。

## 终态与分类

- 终态：`NIGHT12B_IMPLEMENTATION_SEMANTICS_INVALID`
- 分类：`IMPLEMENTATION_FAILURE`
- 直接原因：当前 SciPy 的 `spearmanr` 返回对象提供 `.correlation`，冻结汇总代码读取 `.statistic`，首次正式 LORO 汇总因此抛出 `AttributeError`。
- 停机依据：formal freeze 后最多允许一次全局工程修正；该额度已使用。没有第二次修复、没有局部补跑、没有独立手算绕过 runner。

## 输入 panel 与真实路径

P10 三重复的 `status=unique` 权威交集为登记的 83 genes，末尾换行文本 SHA-256 与合约一致。随后只按六个 RNA feature identifiers 和 Ensembl release 79 GRCm38 唯一 interval 做结构交集，得到最终 `m=80`。`Jaml`、`Lilrb4a`、`Vsir` 因不是唯一 Ensembl79 interval 被结构删除，没有替补。P5 lexical-first 256 未进入科学路径。

| 家族 | 单元 | RNA shape | target shape | non-identifiable genes | 预检 reload | formal unit |
|---|---:|---:|---:|---:|---:|---:|
| RNA+ATAC | P5S1 | 7794 × 80 | 7794 × 80 | 22 | 通过 | finite |
| RNA+ATAC | P5S2 | 28545 × 80 | 28545 × 80 | 13 | 通过 | finite |
| RNA+ATAC | P5S3 | 9426 × 80 | 9426 × 80 | 17 | 通过 | finite |
| RNA+protein | P10S1 | 7447 × 80 | 7447 × 80 | 19 | 通过 | finite |
| RNA+protein | P10S2 | 41289 × 80 | 41289 × 80 | 18 | 通过 | finite |
| RNA+protein | P10S3 | 5845 × 80 | 5845 × 80 | 20 | 通过 | finite |

常数或零特征按合约保留并标记；没有静默删除。

## 科学主表为何没有可报告值

| 家族 | held-out fold | median p_link | link-only Spearman | bootstrap-only Spearman | replicate-only Spearman | combined Spearman | gate |
|---|---|---:|---:|---:|---:|---:|---|
| RNA+ATAC | P5S1 / P5S2 / P5S3 | 未形成 | 未形成 | 未形成 | 未形成 | 未形成 | 未评价 |
| RNA+protein | P10S1 / P10S2 / P10S3 | 未形成 | 未形成 | 未形成 | 未形成 | 未形成 | 未评价 |

六单元的 per-candidate utility、true-link rank tail 和 block-bootstrap 工件已经 finite 写出；汇总器也在失败前写出了逐候选与逐 feature 中间表。但 LORO 第一个 fold 的第一次 Spearman 就停止，因此没有完成任何 held-out fold，没有 pooled correlation/95% CI，没有 combined-minus-best-single CI，也没有 family gate。这里“未形成”不能写成 0，也不能解释为科学门失败。

## 资源与边界

- 预检最慢单元：P5S3，1702.7 秒。
- 正式六单元并行计算最慢单元：P10S2，9.34 秒。
- 从 raw baseline 到 formal 失败锁定：约 3047 秒。
- 进程 peak RAM：796.81 MiB。
- Night-12B CUDA allocation：0 MiB；设备观测基线为 1 MiB。
- 8/8 针对性测试在 refreeze 前通过。
- 6/6 真实预检、6/6 fresh-process reload、6/6 formal unit finite。
- Night-12A raw metadata fingerprint 前后 byte-exact 一致：114 files，6,603,242,533 bytes。
- 新下载、标签读取、ARI/NMI/AMI/FMI/Q、annotation spatial metric、聚类 endpoint、候选神经模型训练、第三方 benchmark、QCRD、dataset routing、dense N×N、scientific retry/fallback、跨重复 spot alignment 和历史 raw 修改全部为 0。

## 对论文意味着什么

Night-12B 没有产生可用的无标签可识别性 go/no-go。已经闭合的数据 panel、六单元真实路径和 per-unit evidence 工件可证明输入与工程下层可运行，但不能替代 LORO held-out 统计门。它既不授权下一轮 gated-fusion layer，也不能被写成这条方法方向的科学负结果。若项目治理允许未来新 revision，需要重新预注册并从全部六单元和所有汇总完整重跑；本轮本身保持失败封口。

## 导师汇报版

Night-12B 想检验同一 gene 的跨模态 link evidence 能否跨真实重复泛化。我们先从登记的 83 个 P10 genes 出发，经六个 RNA identifiers 与唯一 Ensembl79 interval 机械交集得到 80 个 genes，没有按表达或结果选 feature。六个真实重复都完成了 feature-level 预处理、空间块 OLS、穷尽错误 links 和 64 次 block bootstrap，fresh-process reload 也全部通过。正式汇总在第一次 Spearman 调用时遇到 SciPy 返回字段兼容错误，因此 held-out 相关、置信区间和严格 gate 均未形成。由于唯一 correction cycle 已经用完，我们没有再修代码或绕过 runner 手算。这个终态是实现失败，不是 local signal，也不是 scientific negative。当前不能据此启动小型 gated-fusion layer；后续若要继续，必须作为新的、完整重跑的预注册 revision 决定。

## 技术审计摘要

- Parent：`ed7c93d979a85eb8b907058ea463d93ba65f6956` / `night12a-final-20260822`
- 保护 tag：`baseline/pre-night12b-replicate-calibrated-link-identifiability-20260822`
- correction cycles：1/1
- formal scientific retries：0
- final commit/tag、compact index 和 bundle SHA 在最终交付 manifest 中登记。
