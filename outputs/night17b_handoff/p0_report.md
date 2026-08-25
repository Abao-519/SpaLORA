# Night-17B SFRD P0 报告

## 我现在需要知道的三件事

1. **问题**：Night-16H 的 89 个结构可行分区里确实有大量关系信息；本轮检验能否把这些软关系蒸馏进一个真正更新参数的 RNA+ATAC 表示，而不是继续选择现成分区。
2. **实际动作所在层**：在真实 view1、view2 与 retained carrier 上，用注册空间边加受限 feature-kNN 边形成稀疏 pair bank；所有结构可行候选先产生 co-cluster / separated / uncertainty 后验，再训练小型残差编码器，最终所有 arm 使用同一个 KMeans endpoint。标签只在所有 partition 与 hash 写出后由独立 evaluator 打开。
3. **论文含义**：严格门只通过 **1/3** 主 lane。P22 有真实的可训练表示局部增益；MISAR 没有，human 又明显不如非训练的 feasible-relation smooth control。终态为 **SCIENTIFIC_NEGATIVE**，不能把它写成跨研究方法信号，也不刷新 Night-16H 的总体分数前沿。

## 明确终态

- classification: `SCIENTIFIC_NEGATIVE`
- status: `NIGHT17B_SFRD_NOT_IDENTIFIABLE`
- 次级事实：`P22_REPRESENTATION_LOCAL_SIGNAL`
- 严格继续门：full learned representation 必须同时超过 frozen-retained 与 feasible-consensus 两个同-head control，并在至少 2/3 主 lane 双指标为正；实际为 **1/3**。

## 绝对指标与匹配贡献

所有数字均为同一个 KMeans head；“强参考”按 ARI、NMI 分别取 frozen 与 feasible-smooth 的较强值。

| 数据集 | Frozen ARI/NMI | Feasible smooth ARI/NMI | SFRD full ARI/NMI | Δ vs 强参考 ARI/NMI | Unbiased-bank ARI/NMI | Permuted-relation ARI/NMI | 严格门 |
|---|---:|---:|---:|---:|---:|---:|---|
| P22_K9 | 0.389321/0.578404 | 0.393799/0.584658 | 0.438681/0.588142 | 0.044882/0.003485 | 0.437351/0.585807 | 0.388124/0.577598 | PASS |
| MISAR_K7 | 0.361113/0.537172 | 0.360034/0.538317 | 0.359145/0.534573 | -0.001969/-0.003744 | 0.358301/0.531001 | 0.360093/0.533015 | FAIL |
| HUMAN_HIPPOCAMPUS_K7 | 0.128913/0.186938 | 0.195737/0.278234 | 0.149208/0.213257 | -0.046528/-0.064978 | 0.151279/0.218270 | 0.164453/0.213839 | FAIL |

次级 melanoma K2 安全 lane：frozen `0.001074/0.001300`，feasible smooth `0.980593/0.953369`，SFRD full `0.203735/0.210751`。它同样否定“可训练残差优于现成可行关系 head”。

## 高分来自哪一层

- **P22**：full 比强参考提高 `+0.044882 ARI / +0.003485 NMI`；unbiased bank 仍为 `0.437351/0.585807`，说明这条局部增益不完全依赖历史 authority 候选。但 unweighted 与 raw-view-adapter 对照非常接近 full，关系权重本身的净贡献仍弱。
- **Human hippocampus**：full 虽高于 frozen，却低于 feasible smooth `-0.046528/-0.064978`；permuted relation 与仅启用一个 raw-view adapter 的 arm 还高于 full，关系位置特异性不成立。
- **MISAR**：full 比强参考低 `-0.001969/-0.003744`，unbiased bank 进一步下降。
- `SINGLE_VIEW1/2` 只屏蔽对应 raw-view adapter；retained adapter 仍可能含双模态信息，所以这些 arm 只能解释为 **ONE_RAW_VIEW_ADAPTER_ABLATION**，不能称真正单模态模型。
- full bank 含历史 benchmark-HPO strong starts；因此本轮主张只能依赖 matched unbiased-bank sensitivity。P22 在该 sensitivity 下保持，另外两条不保持。

## 真实工程 P0

| lane | view1 | view2 | retained | sparse pair 数 | optimizer steps | 参数 L2 变化 | max grad | fresh replay |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P22_K9 | [9196, 30] | [9196, 50] | [9196, 64] | 47670 | 40 | 2.637128 | 0.025155 | PASS |
| MISAR_K7 | [1949, 30] | [1949, 50] | [1949, 64] | 10029 | 40 | 2.641227 | 0.037979 | PASS |
| HUMAN_HIPPOCAMPUS_K7 | [2500, 64] | [2500, 64] | [2500, 64] | 15608 | 40 | 2.842159 | 0.122999 | PASS |


三条主 lane 均实际产生非零梯度和参数变化；checkpoint strict load 后，GPU fresh-process representation 与 partition 均 byte-exact。候选关系与 feature-neighbour 只用稀疏 pair list，没有 dense N×N。

## 工程修正与作废范围

1. 首次 smoke 未设置仓库 `PYTHONPATH`，在导入前退出；设置正确路径后重跑，未生成科学结果。
2. 初次 fresh replay 在 CPU 上重算 GPU representation，浮点路径不同而未 byte-exact；改为相同 CUDA 数值路径后 3/3 seed0 与额外运行均 exact。
3. evaluator 最初把 training seed 写死为 0；改为从锁定 run IDs 解析，重新评价所有既有 partitions，分区未变。
4. **最重要修正**：首版 gate 只与 frozen retained 比较，错误忽略 feasible-consensus same-head control，曾误判 2/3 并提前运行 seeds 1/2。修正后严格门为 1/3；这些额外 seed 文件完整保留但标记 `SUPERSEDED_PREMATURE_CONFIRMATION_DUE_GATE_BUG`，不进入主科学判定。

## 标签与 HPO 边界

- 三个 config 与所有 control 的 partitions/hash 先写出，之后 evaluator 才读取公开 benchmark annotations。
- family config `F03_ANCHORED` 是透明的 post-lock label-assisted benchmark HPO；它不是盲测或冻结外部确认。
- 89-candidate bank 本身来自历史公开 benchmark 开发；本轮不能称原始数据端到端完全无标签教师。`UNBIASED_BANK_WEIGHTED` 剔除了 primary authority/medoid，只保留 robustness KMeans seeds 与 ordered continuation starts。

## 限制

- SFRD 仍是 retained-representation plug-in，不是 raw-fragment end-to-end 模型。
- P22 的局部分数 `0.438681/0.588142` 低于 Night-16H 已交付的总体分区高位，不能称 score-frontier advance。
- Human 与 melanoma 表明可行候选关系经过简单非训练 head 已可非常强，而当前残差训练会损失这些结构。
- 没有证据支持继续加 seed 或扩大同机制网格；下一对象应改变蒸馏目标/消费者，而不是修补本公式。

## 导师汇报版

这轮测试的是能否把 Night-16H 多个可行分区的共识关系学进一个新表示。模型确实在三套真实 RNA+ATAC 数据上完成了反向传播、参数更新和 checkpoint 精确重放。P22 上，同一 KMeans head 的 ARI 从强参考 0.3938 提到 0.4387，NMI 从 0.5847 提到 0.5881，说明有一条局部表示信号。可是 MISAR 轻微下降，human 也明显输给非训练的关系平滑 control；melanoma 的非训练 control 更远高于残差模型。严格科学门因此只有 1/3，通过不了。早先 2/3 是 gate 漏掉强 control 的实现错误，已经修正，额外 seeds 保留但不作证据。结论是科学负结果，不是 SOTA，也不是可投稿的方法核心。

## 技术附录

- source/config selection: `F03_ANCHORED`，seed0 为主 P0。
- targeted tests: 7/7 PASS。
- AutoDL 状态：保持开机，`shutdown_dispatched=false`。
