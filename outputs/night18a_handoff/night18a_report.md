# Night-18A report — deep backbone portability

## 我现在需要知道的三件事

1. **问题**：我们把表示层和聚类决策层彻底拆开，真实训练了同一套 RNA+protein / RNA+ATAC 稀疏双模态 backbone，再在每个完全相同的 embedding 上比较 common endpoint、可行分区 medoid 和 Night-16H 固定结构解码。
2. **实际改变的层**：模型包含双模态稀疏空间/feature-kNN 编码、遮蔽重构、轻量 DEC 分离和 retained residual anchor；参数训练 80 步且真实变化。结构解码不重训表示，只在同一个 15-candidate bank 上应用固定 exact-K/no-singleton/internal-edge 可行域和分子/空间仲裁。
3. **论文意义与分类**：P22 的深度表示相对 frozen carrier 的 common KMeans 有小幅双升，但 A1 没有；结构解码在两个学习后 embedding 上均未产生独立双指标增益。因此分类是 **LOCAL_SIGNAL**：保留 P22 representation 线索，否定“当前 Night-16H decoder 已可迁移到任意深度 backbone”。

## 绝对指标与分层归因

| Lane | 结果层 | 配置 / endpoint | ARI | NMI | 对 matched 参考 ΔARI/ΔNMI | 解释 |
|---|---|---|---:|---:|---:|---|
| A1 K10 | frozen representation | frozen common KMeans | 0.235268 | 0.387639 | reference | 同 endpoint 表示参考 |
| A1 K10 | best learned common | C02 no-graph common KMeans | 0.232410 | 0.388158 | -0.002858 / +0.000519 | 无双升，表示信号失败 |
| A1 K10 | same learned embedding decoder | C02 fixed structured | 0.233167 | 0.387609 | +0.000757 / -0.000549 vs common | 解码无双升 |
| A1 K10 | decoder-only context | frozen fixed structured | 0.245050 | 0.390472 | +0.009782 / +0.002833 vs frozen common | 只说明 frozen candidate bank 内的 head 效应，不是深度 backbone 迁移 |
| P22 K9 | frozen representation | frozen common KMeans | 0.470897 | 0.599897 | reference | 同 endpoint 表示参考 |
| P22 K9 | best learned common | C03 no-anchor common KMeans | 0.478639 | 0.606982 | +0.007742 / +0.007086 | P22 局部表示双升 |
| P22 K9 | best learned medoid | C03 feasible medoid | 0.482802 | 0.612864 | +0.004163 / +0.005882 vs same common | medoid 有局部 head 增益 |
| P22 K9 | fixed structured NMI profile | C01 fixed structured | 0.474764 | 0.624991 | -0.003283 / +0.018537 vs same common | NMI/ARI 取舍，不是双升 |
| P22 K9 | decoder-only context | frozen fixed structured | 0.499500 | 0.622887 | +0.028603 / +0.022990 vs frozen common | 仍低于历史强完整管线，且没有迁移到 learned embedding |

历史公开开发前沿为 A1 0.276003/0.421740、P22 0.595552/0.717931；本轮所有结果均未刷新它们。本轮主问题不是追逐这些历史 pipeline 数字，而是做同 embedding、同候选预算的表示/解码归因。

## 机制结论

- **表示层**：P22 在 common endpoint 上获得小幅、可重放的双指标改善；去 anchor 的 C03 略优，说明增益不是 anchor 正则单独解释。但 A1 同公式没有双升，因此还不是跨模态家族的 backbone 信号。
- **结构决策层**：fixed structured decoder 在 frozen embedding 上可以有效，但移到学习后 embedding 时，A1 与 P22 都没有相对 same-embedding common KMeans 的双指标增益；它也没有稳定超过 same-embedding feasible medoid。这直接否定了本轮的 decoder portability 假设。
- **工程路径**：A1 为 3484×30 + 3484×30，P22 为 9196×30 + 9196×50；两条路径均使用真实稀疏图、真实 optimizer steps、strict checkpoint load 和 fresh-process representation/partition exact replay。工程闭合不等于方法成功。

## 最重要失败与限制

1. A1 的图传播 full 配置比 frozen/common 更差；no-graph 控制反而更好，说明当前 sparse graph reconstruction 对 RNA+protein 还会过平滑或错配。
2. P22 learned representation 的局部增益远低于历史完整 pipeline，高分不能包装成 score frontier。
3. candidate generator 是统一、预算受限的 15-candidate bank，不等同于 Night-16H 历史 89-candidate authority bank；这里检验的是 fixed rule 的可移植语义，而不是复刻历史最高分。
4. 配置比较使用公开 annotation 的 post-lock evaluator，属于透明 benchmark development，不是盲测。

## 导师汇报版

我们这轮第一次把表示层和结构解码层放在完全匹配的实验里。A1 和 P22 都真实训练了同一套稀疏双模态网络，参数、seed、checkpoint 和 fresh replay 全部闭合。P22 的深度表示相对 frozen carrier 小幅双升，说明 RNA+ATAC 表示仍有可开发信号。A1 没有双升，而且 no-graph 比 full 更好，提示当前图传播对 RNA+protein 可能过平滑。Night-16H 固定结构解码在 frozen embedding 上有效，但迁移到学习后 embedding 后，两个数据都没有独立双指标增益。于是本轮不是统一方法里程碑，而是一个 P22 局部表示信号加一个明确的 decoder portability 负结果。后续应优先改进原生 backbone/graph alignment，再重新校准结构证据，而不是直接把旧 selector 当 backbone-agnostic 模块。所有结果均为公开 benchmark development，不能称 SOTA 或 paper-ready。

## 技术状态

完整 32 行绝对指标、matched contribution、源码碰撞、label flow、P0 registry、资源、失败 ledger 和 exact replay 与本报告同目录。服务器保持开机，`shutdown_dispatched=false`。
