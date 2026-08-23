# SpaLORA Night-13C：endpoint 稳健性与可训练统一核心重置任务书

日期：2026-08-23  
角色：Worker 2 统筹；Codex 2 在 AutoDL 执行  
父提交：`b97ea8f19d0fdb5352da61ada8475dda70826e91`  
父标签：`night13b-final-20260823`

## 一、本轮想解决什么

Night-13C 回答两个连续问题：

1. Night-13B 在 A1、P22 的提分，是稳定的表示改善，还是固定 KMeans 对微小扰动敏感造成的偶然跳变？
2. 如果局部图 residual 确有价值，能否把它降格为安全辅助层，同时建立一套真正接收两个模态、会训练、跨 RNA+protein 与 RNA+ATAC 共用的模型核心？

本轮不是外部 baseline 复现任务。完整外部比较等自己的候选达到稳定水准后再做。

## 二、已冻结的证据修订

Night-13B 原 compact 不改写，另建 errata/evidence-regrade ledger：

- B10 是 simple-concat PCA 上的确定性多尺度图后处理，不是已训练的端到端模型。
- `seed` 没有进入模型，KMeans 固定 seed 0；35 行实际只有 7 个独特结果。
- 15 个 candidate ID 是同一公式的超参数网格，只算一个机制家族。
- A1/P22 提升仅是 common endpoint 表示层信号；native 完整管线高水位未被超过。
- protein 全 study-block 汇总小幅为正；ATAC 的 P22/MISAR 等权汇总为负。
- ARISE 与 SpaBalance 在 `source_mechanism_audit.md` 的 frozen commit 各多了一个字符，登记为 metadata errata，不修改旧 compact。

后续报告禁止再写“B10 五 seed 稳定”“15 种机制”“训练参数为 1”。

## 三、执行原则：给 Codex 自由，但不让证据失真

Codex 2 可以自主决定正常调试、缓存、批量大小、训练轮数、学习率、候选淘汰、搜索预算和是否加深某条有希望路线；纯工程错误正常修复并重跑受影响单元，不设荒谬的单次 correction 上限。

最少底线只有：

1. 不伪造、不删失败、不隐藏坏 seed；
2. 不读取数据集名称后切换模型 backbone、loss 或后处理；
3. 标签可用于公开 benchmark 评价与跨运行 HPO，但不进入无监督训练 loss、gradient 或单次 run 的 checkpoint 选择；
4. common endpoint 比较使用相同 observations、mask、known K 与 endpoint seeds；
5. 历史 raw、旧 artifact、旧 tag 只读；
6. 超参数变体不能冒充不同机制；
7. 结果不夸大为盲测、SOTA 或 paper-ready。

## 四、远端约定

- workspace：`/root/autodl-fs/SpaLORA-night13c`
- branch：`revision/q2-night13c-endpoint-robustness-trainable-core-20260823`
- protection tag：`baseline/pre-night13c-endpoint-robustness-trainable-core-20260823`
- output/data root：`/root/autodl-fs/night13c_endpoint_robustness_trainable_core_20260823`
- final tag：`night13c-final-20260823`
- 普通 push；禁止 force；本轮不新增下载。

## 五、Stage A：低成本重验 B10

优先复用已经锁定的 embedding 和图，不重新训练 encoder。

### A1. 真实 endpoint 不确定性

对 identity 与 B10 至少运行固定登记的 KMeans seeds `0..29`，并建立 consensus partition。确定性 embedding 只登记一份，不复制成 model seed。

每个数据集报告：

- ARI/NMI/AMI/FMI 的 endpoint-seed 分布、均值、标准差、配对差值与胜率；
- consensus 绝对指标；
- inertia、centroid margin、partition change rate；
- embedding SHA 与每个 partition SHA。

### A2. 最小机制消融

至少区分：identity、P4-only、P18-only、B10 mixture 和固定 beta 路径。优先在 MISAR 的 `beta=0..0.08` 加密；其他数据按实际有效区间自适应取点。

若发现 endpoint 几何不一致是主因，再比较原 PCA、row-normalized、whitened PCA；若图语义是主因，再比较 union-kNN、mutual-kNN 或距离加权图。Codex 可提前淘汰明显无效组合，无需机械跑满笛卡尔积。

### A3. 强 backbone 附加检验

在 observations 和维度闭合时，将完全相同的 B10/最佳安全 residual 施加到：

- protein：C00 的 A1、D1、tonsil 锁定 embeddings；
- ATAC：F00 与 N02 的 P22 锁定 embeddings；
- MISAR 若没有可审计强 embedding，不得同名重构冒充复用。

同时保留两条板：

- `COMMON_HEAD_ROBUSTNESS`：相同 endpoint seeds 的表示层公平比较；
- `NATIVE_FULL_PIPELINE_CONTEXT`：C00/H05、F00、N02 的历史完整管线高水位。

### A4. Stage A 判定

若 A1/P22 的优势不能在大多数 endpoint seeds 与 consensus 中保留，或只存在于极窄 KMeans 盆地，登记 `B10_ENDPOINT_ARTIFACT_OR_FRAGILE`，结束围绕 B10 threshold 的搜索。

若信号稳健，B10 仍只晋级为辅助 head/ablation；它不会因 Stage A 通过而自动成为论文核心。

## 六、Stage B：真正的可训练统一 core

Stage B 的输入必须保留两个模态张量或两个模态独立 embeddings。模型必须有 optimizer steps，训练 seed 必须实际影响初始化/采样/训练轨迹，checkpoint reload 必须闭合。

### B1. 首选窄问题

优先检验临时候选“cross-modal consensus edge residual”（跨模态共识边残差）：

- 对每条稀疏空间边，分别计算 RNA 与第二模态的支持度；
- 用对称、连续的共识可靠度抑制冲突边，而不是用 RNA 单边锚定；
- 使用 identity-preserving residual，避免强迫所有点接受空间传播；
- protein 与 ATAC 共用同一 edge/core/loss 公式，只允许正常的 modality adapter。

该名称只是为了标记真实数学对象，不预先声称原创。必须记录它与 SpatialGlue、ARISE、PRAGA、SpaMV、MultiGATE 等方法的真实差异。

### B2. 机制多样性

除非 Stage A 已足以否定整条路线，至少比较三个真正不同的机制家族。建议池为：

1. 参数无关或弱参数的跨模态共识边 residual；
2. 小型可训练的对称 edge-reliability core，以 masked reconstruction、cross-modal agreement 和 identity/variance preservation 等无标签目标训练；
3. modality-dropout + cross-reconstruction 的共享 latent core，不依赖图 gate 决定是否融合；
4. Codex 从正式论文/官方源码审计后提出的其他可控候选。

同一公式的 threshold、width、learning-rate 或 loss-weight 变化只算一个机制家族。Codex 可调整候选数量，不必为凑数运行低价值组合。

### B3. 先做真实 P0

至少在 A1 与 P22 各跑一条：

`真实原始输入 -> preprocessing -> 两模态 forward -> loss -> finite gradient -> optimizer step -> checkpoint strict reload -> fresh-process 数值复现 -> fusion -> common endpoint`

必须列出 raw/processed tensor shape、真实 optimizer steps、可训练参数量、loss 分量、checkpoint SHA、运行时间、GPU/RAM。不能再用“endpoint 单独通过 + 合成 adapter 通过”代替真实混合路径。

### B4. 自适应开发集

为避免再次等到最后才发现 ATAC 翻转，初筛同时包含：

- protein：A1 与 tonsil s1，各代表一个 study block；
- ATAC：P22 与 MISAR E15.5，各代表一个 study block。

公开标签可用于 candidate/config 的跨运行排序。先用 seed 0 快筛，保留有希望机制后用真实 training seeds 0/1/2；最终候选用 0..4。所有 seed 结果保留。

性能优先级：

1. 两个 family 的 study-balanced ΔARI 与 ΔNMI；
2. 较弱 family 与最差 study，避免靠 P22 单点拉高 macro；
3. common endpoint 稳健性与 consensus；
4. native 完整管线高水位；
5. 时间、显存、RAM 的 Pareto 权衡。

若有一个机制显著领先，Codex 可把剩余预算集中于其邻域；若所有机制被强参考支配，可提前停止，不需要烧完整夜算力。

## 七、冻结后 internal confirmation

冻结唯一 core、全局配置、训练方案和 seeds 后，再打开：

- D1；
- tonsil s2/s3，分别报告但合成一个 tonsil block；

A1/D1 合成 lymph-node block，tonsil s1/s2/s3 合成 tonsil block。ATAC 以 P22 和 MISAR 各一票。当前没有 pristine labeled confirmation，因此只能称公开 benchmark 的开发与内部确认。

只有两家族均有正向或至少非劣信号时，才按资源选做 P5/P10/SPOTS 无标签重复；负结果后不要为了凑表耗尽算力。

## 八、进展门与终态

候选成为“可继续论文核心”的最低门：

- 真正训练语义闭合，optimizer steps 大于 0；
- 同一 core、同一全局规则覆盖两个模态家族；
- common endpoint 的多初始化与 consensus 不依赖单一 KMeans 盆地；
- protein 与 ATAC 的 study-balanced ARI/NMI 不再整体翻转；
- 相对历史 native 高水位的差距明确报告，不能只超过 simple。

终态先归类为：

- `IMPLEMENTATION FAILURE`；
- `INFRASTRUCTURE FAILURE`；
- `SCIENTIFIC NEGATIVE`；
- `LOCAL SIGNAL`；
- `CONFIRMED MILESTONE`。

允许的细化终态包括：

- `NIGHT13C_B10_ENDPOINT_FRAGILE`；
- `NIGHT13C_GRAPH_HEAD_ROBUST_LOCAL_SIGNAL`；
- `NIGHT13C_UNIFIED_TRAINABLE_CORE_LOCAL_SIGNAL`；
- `NIGHT13C_UNIFIED_TRAINABLE_CORE_MILESTONE`。

Night-13C 不授予 `PAPER-READY EVIDENCE`；外部强 baseline、公平协议、完整消融、生物解释和冻结后新数据确认仍属论文阶段要求。

## 九、交付与沟通

Windows compact 目标：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night13c_delivery_20260823/official_compact`

至少包含：Night-13B evidence-regrade/errata、B10 endpoint robustness board、consensus/perturbation结果、strong-backbone residual board、candidate family registry、search/failure ledger、真实 P0、absolute metrics、common/native 双板、study/family aggregation、机制碰撞矩阵、源码/config/checkpoint、测试、报告、通俗总结、decision JSON、bundle、root-relative compact index 与 Windows 独立复算。

主表必须区分 representation-level 与 full-pipeline，包含 total/evaluated、K、training seed、endpoint seed、absolute ARI/NMI、Δ、AMI/FMI、Moran/Geary、consensus、wins、wall/GPU/RAM。Q 只能作内部辅助量。

只在阶段完成、方向改变或出现真实阻塞时更新，不逐命令输出。最终先给“我现在需要知道的三件事”、明确分类、绝对指标表和论文含义；再给 5–8 句导师汇报版；Git/hash 放技术附录。

完成 compact 独立复算、普通 push 和远端 tag peel 后，将 `/usr/bin/shutdown` 作为最后一条远端命令派发，之后不重连。
