# Night-8A 独立审计与恢复决定

日期：2026-08-20  
结论：Night-8A 的 fail-closed 判定正确；但现有证据支持先做 evaluation-only recovery，而不是重训 116 个模型。

## 1. 已独立核验

- Windows compact 实际包含 70 个文件、2,331,737 bytes；`compact_delivery_index.json` 索引其余 69 个文件。
- 69/69 索引文件存在且 SHA-256 匹配。
- compact index SHA-256：`e74202074a3b4d882670d960bf66c3c1e8667ac3a53baf811ce0658c0ca0d4b6`。
- compact root SHA：`e09aec6882a99af7c74d294b6d68a766efc9580e91bb25d0c4ae84424255e265`。
- report、decision、planner tar、incremental bundle 的本地 SHA 与交接陈述一致。
- bundle `list-heads` 中 branch 为 `d09aa00b5e25269e66712dd47d01064b7c9422cf`；final tag 是 annotated tag object。
- R1 为 32/32 terminal cells：29 次真实训练、3 个 no-op alias。
- R2 为 96/96 terminal cells：87 次真实训练、9 个 no-op alias。
- 合计 116 次真实 CUDA 训练、12 个 alias；1 个固定 transform failure 为 B03/P22/seed1。

服务器现已关机，因此本审计没有重新连接，也没有实时重哈希 `/root/autodl-fs/night8a_raw_runs_20260820`。远端原始文件的再次核验必须是恢复任务的 P0 硬门。

## 2. 错误范围

错误发生在 `night8a_transform.py` 的空模块 RNA_PROTEIN 基线路径：

- A00/B00 应直接复用 Night-7B source 中 SHA-locked C00 affinity 和 partition；
- 实际旧运行却从 module-off embeddings 重新构造 equal-view affinity 并重新 spectral clustering；
- 9 个 R2 protein B00 cells 与锁定 C00 的 partition ARI 为 `0.879618–0.982742`，最大指标偏差 `0.009087943105`；
- P22 B00/R02 三个 pilot seeds 指标误差不超过 `4.44e-16`。

因此原 DEV_WINDOW_1 的 fail-closed 是正确的，原 Night-8A 仍永久保持 `IMPLEMENTATION_SEMANTICS_INVALID`。

## 3. 为什么候选训练大概率可救回

从本地 R1/R2 stage manifests 逐格检查：

- 排除 A00/B00 以及设计上 no-op 的 A05 RNA_PROTEIN alias 后，共检查 109 个候选单元；
- 109/109 的训练 reference path 均不指向 Night-8A 的错误 A00/B00 输出；
- RNA_PROTEIN 候选由自己的三路新 embedding 构造 H05 equal-3 affinity；
- RNA_EPIGENOME 候选由自己的 fused embedding 与锁定 C06 affinity 构造 R02 endpoint；
- B05 的 RNA_PROTEIN cells 合法 alias 到 B04，不依赖 B00；
- 唯一不可评价单元仍是 B03/P22/seed1 的原始 cluster-K numerical failure。

所以目前最有证据的判断是：错误污染了 comparator transform，而没有污染 B01–B07 的训练、checkpoint、embedding 或候选 transform。这个判断在恢复任务中仍须通过远端全文件 SHA 与依赖图再次确认；若任何候选依赖 B00，立即停止。

## 4. 新发现的 shortlist 逻辑问题

原 `shortlist()` 把 `B00_FAMILY_REFERENCE` 与新候选放在同一个 eligible table 中，没有显式排除 comparator。这样在新模块无增益时，B00 本身可能被选为“unified/frontier finalist”。

恢复规则必须在重新打开标签之前冻结：

- B00 只用于 comparator parity 和 delta 计算；
- B00 不得进入新候选 shortlist、Pareto new-candidate set 或 R3 finalist；
- B03 因 11/12 而不是 12/12 complete，不得晋级，但保留描述性结果；
- 其他门、权重、seeds、tie-break 和阈值全部沿用原 Night-8A registry，不作事后调整。

这是与任何候选分数无关的语义修复，不是看结果后改门。

## 5. 推荐下一步

先运行一个独立、只评价的恢复任务：

1. 原 Night-8A 目录与文件保持 byte-identical；
2. 0 training、0 transform、0 checkpoint rewrite、0 external benchmark；
3. comparator 从锁定 C00/R02 partition 构造只读 evaluation view；
4. 新候选只读取已锁定 clusters/embeddings；
5. 一次授权开发标签窗口，双实现独立复算并要求 `1e-12` 一致；
6. 产出最多三个 shortlist ID 后停止，不启动 R3；
7. 规划对话先审查真实 R1/R2 分数，再决定是否花 GPU 成本补 R3 seeds。

若恢复成功，116 次训练不会白费。若 P0 发现候选文件 SHA、依赖隔离或 locked comparator 不成立，终止并报告，不重训。

## 6. 外部数据顺序

MISAR E15.5 S1 的无标签锁定是有效阶段成果，但现在不能抢跑 external benchmark。正确顺序是：

1. 恢复并审查 Night-8A pilot；
2. 对最多三个固定候选补 R3 seeds；
3. 冻结最终 family candidate；
4. 才在 MISAR 做一次性外部 RNA+ATAC confirmation。
