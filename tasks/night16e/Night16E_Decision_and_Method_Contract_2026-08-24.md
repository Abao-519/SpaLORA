# Night-16E 决策与方法合约

日期：2026-08-24  
工作名：`TSRE`（Tri-State Relation Energy，三态关系能量；只是开发代号，不是论文最终命名）

## 一、决策

Night-16D 已以实现正确的科学负结果封口。它说明：把 support、boundary、conflict 做成强表示之上的可训练残差，不能独立超过输入强起点。下一轮不再扩大这类神经残差网格。

当前最可靠的方法基础是 Night-15F 的稀疏多尺度直接聚类能量：它在九条开发协议上均取得过 ARI/NMI 双升，并且多尺度图、拒绝质量 stay cost 和部分 alpha-expansion 大邻域移动有 matched ablation 支持。Night-16E 的问题是：Night-16C 的三态跨模态关系，若不修改表示，而是直接约束“哪些边应平滑、哪些边不应合并、冲突时应信任哪个私有模态”，能否产生独立且可迁移的聚类收益。

这不是把多个论文的完整模型串起来。既有先例分别覆盖多尺度图、prototype unary、Potts/graph-cut、共享/私有表示和边界梯度；Night-16E 只保留一个待验证的新组合对象：**同一条稀疏空间边上的支持、共同边界和模态冲突被送入三种不同的聚类能量作用，而被拒绝的关系质量显式回到当前状态。** 是否够作论文贡献只由 matched ablation、家族冻结迁移和绝对分数共同决定。

## 二、统一公式

对节点标签 `y`、第 `t` 个外循环的当前标签 `y^t`，冻结当轮动态量后优化：

`E_t(y) = sum_i [U_content(i,y_i) + U_boundary(i,y_i|y^t) + U_private(i,y_i) + U_stay(i,y_i|y^t)] + lambda_s sum_(i,j) W_support(i,j) 1[y_i != y_j] + U_size(y)`

其中：

1. `U_content`：Night-15F 已验证的 retained/RNA/第二模态动态 prototype 距离与多尺度稀疏图特征；
2. `W_support >= 0`：两模态共同支持为域内边时才施加正的 Potts 平滑；
3. `U_boundary`：把高置信共同边界转成冻结邻域标签下的 boundary-exclusion unary，避免把边界两侧强行合并；不得使用会破坏二元 submodularity 的负 Potts 权重；
4. `U_private`：边被判断为模态冲突时，按节点自身、无标签的模态可靠度保留更可信的私有 prototype 证据，而不是先把两个模态平均掉；
5. `U_stay`：关系证据被拒绝或不可靠时，形成改变当前标签的 stay cost；
6. `U_size`：沿用 Night-15F 的可选、仅依赖当前预测计数和 K 的软约束；
7. support/boundary/conflict 的连续权重由同一内容公式产生，三者语义在两个模态家族完全相同；
8. 同一冻结外循环内，每个接受的 alpha move 必须降低原始浮点能量。动态量在外循环间重算，因此不宣称跨循环单一目标全局单调或全局最优。

允许 Codex2 在保持以上语义的前提下，自主改进可靠度、归一化、数值稳定性、候选移动顺序和稀疏实现；不要被伪代码束缚。如果发现一个更简洁且更可证明的等价实现，应优先采用并在报告中说明。

## 三、数据家族与发现/迁移划分

### RNA+protein

- discovery：A1 K=10、tonsil slice 1 K=4；
- frozen transfer：D1 K=10、tonsil slice 2/3 K=4；
- 无标签真实性：GSE213264 tonsil、SPOTS rep1/rep2、P10S1/S2/S3。

### RNA+chromatin

- discovery：canonical RNA+ATAC P22（N=9196）K=9；K=18 只作独立 sensitivity；
- frozen transfer：MISAR E15.5 的既有明确协议，以及本轮新闭合的 E18.5 reference protocol；若 E11.0/E13.5 同包可低成本闭合，也加入 transfer board；
- 无标签真实性：P5S1/S2/S3。

必须把项目 canonical P22（9196，RNA+ATAC）与 SMART 效率实验的 9752 RNA+H3K27me3 P22 分开。MISAR 每个发育阶段的 annotation 文件、字段、K、mask 和 hash 分别登记；项目既有 E15.5 K7/K12 与近期论文的其他 K 协议不能混表。

## 四、两张结果板

1. **论文方法板**：每个模态家族只在 discovery 单元选择一组数值 profile，随后冻结到 transfer；同一 architecture、energy 和参数含义，允许两个家族的数值 profile 不同。
2. **分数前沿板**：允许每条公开 benchmark 独立、透明、label-assisted HPO；候选 partition 先生成和哈希，独立 evaluator 后读公开 reference；max-ARI、max-NMI 和 balanced Pareto 分开报告。它用于摸清上限，不冒充自动部署策略。

不把数据集名称作为核心公式输入。工程 registry 可以按 lane 找文件和加载已经冻结的 profile，这不等于模型内按名称选择不同算法。

## 五、必要对照

- byte-exact strong start / no-op；
- Night-15F 同起点 direct energy；
- Night-16C post-hoc CMBF-TPR；
- TSRE full；
- support-only；
- boundary off；
- conflict/private off；
- rejected-mass stay off；
- single-site 与 alpha-expansion（只在 full 有信号后扩展）。

主张边界：只有 TSRE full 相对同起点 Night-15F 基础能量取得增益，且至少一个新增三态项在 matched ablation 中有独立作用，才能称方法信号。若只有 per-lane HPO 刷新分数，分类为 score-frontier/local signal；若只有 head 产生效果，明确写 head-only。

## 六、数据扩展优先路径

优先使用 Zenodo `10.5281/zenodo.14789361` 的 `spatial ATAC-RNA-seq MB.zip`（网页登记 268.2 MB，MD5 `bf4b68a7e55566e07820816a23198fca`），不要先下载 10.7 GB SMART 整包。先查远端缓存，再查 archive central directory，验证无路径穿越、文件名、大小、MD5/SHA-256，随后只读解压到新 provenance root。

若该包不包含所需 RNA/ATAC/坐标/reference annotation，不猜测、不伪造；依次检查 INSTINCT 官方 processed 资产、PRESENT 官方教程资产、SMART 官方数据。只有较小来源确实不足且磁盘/时间允许时，才下载 SMART 10.7 GB 包，并在报告中登记原因。

## 七、终态解释

- `IMPLEMENTATION_FAILURE`：真实路径或能量语义未成立；
- `SCIENTIFIC_NEGATIVE`：实现正确但新增能量无增益；
- `SCORE_FRONTIER_ADVANCE`：透明 per-lane HPO 刷新公开开发分数；
- `FAMILY_FROZEN_METHOD_SIGNAL`：冻结 family profile 在 transfer 形成可重复增益，且 matched ablation 支持新增机制；
- `CONFIRMED_MILESTONE`：还需跨独立 study、多个种子/确定性回放、非退化簇和机制对照；
- `PAPER_READY_EVIDENCE`：另需完整外部基线、生物学解释和复现包，本轮不预设能达到。

