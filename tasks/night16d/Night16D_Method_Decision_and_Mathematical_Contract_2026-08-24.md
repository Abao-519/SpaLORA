# Night-16D 方法决策与数学合约

日期：2026-08-24  
工作名：CMBF-RL（Cross-Modal Boundary Field Residual Learning；仅为内部工作名，不锁定论文命名）

## 1. 本轮真正要解决的问题

Night-16C 已证明：根据两种模态对同一条空间边的变化程度，可以把边解释为“共同支持、共同边界、模态冲突”，并可在强起点上作极小的安全修复。但它仍是事后改分区，移动点数只有 1–10，部分数据关闭 boundary 或 conflict 后还略好；因此不能证明完整组合本身就是有效方法。

Night-16D 不再继续扩大事后 repair 网格。它要检验一个更直接的对象：**三种边状态能否在表示学习阶段执行三种不同运算，并在不破坏强起点的前提下形成更可分的空间多组学表示。**

## 2. 对“是不是把几篇论文拼起来”的独立判断

不是把 BANKSY、stLVG、PRAGA 等完整模型串联运行。Night-16C 借用了已经存在的思想类别：邻域梯度、方向信息、prototype 聚合；其可争取的新对象是：

1. 同一条真实空间边上，由两模态共同生成的 support / boundary / conflict 三态关系；
2. 三态关系不是三个权重，而是触发三种不同的表示运算；
3. 被拒绝的跨点信息通过 self-return/teacher anchor 留在本点，避免错误传播。

组合式创新可以发表，但只有在“新组合对应一个明确缺口、模块之间存在不可替代的数学关系、完整模型相对合理对照有稳定独立增益”时才成立。Night-16C 只满足前两项的一部分，第三项尚未满足。Night-16D 的任务就是补第三项；若补不上，不包装为主方法。

## 3. 统一模型对象

对任一 RNA+protein 或 RNA+chromatin 数据单元，输入为两视图低维表示 `H1, H2`、稀疏空间图 `E`、固定强起点表示 `T`。输入适配器可以随模态维度变化；下述核心公式和参数含义不随数据集名称变化。

### 3.1 三态边场

对每条有向稀疏空间边 `(i,j)`，分别从两个模态计算稳健的边变化秩 `r1_ij, r2_ij in [0,1]`：

```text
S_ij = (1-r1_ij)(1-r2_ij)                       # 两模态共同支持
B_ij = r1_ij r2_ij                              # 两模态共同边界
C_ij = r1_ij(1-r2_ij) + r2_ij(1-r1_ij)         # 模态冲突
S_ij + B_ij + C_ij = 1
```

实现前须复核 Night-16C 的全局边秩是否需要改为 node-local rank。两者都可作为候选，但必须在开发阶段明确登记，冻结后不得混用。

### 3.2 关系专属消息算子

对标准化后的表示 `H` 构造三个不同的稀疏消息，而不是把三态只当作一个连续权重：

```text
M_sup  = D_S^{-1} S H               # 域内低通/对齐
M_bnd  = H - D_B^{-1} B H           # 共同边界高通/差分
M_conf = [H1-H2, D_C^{-1} C H1, D_C^{-1} C H2]
                                      # 冲突信息保留在私有通道，不强迫对齐
```

零度节点必须精确 self-return，不允许 NaN 或隐式全零。

### 3.3 可信残差融合

共享的小型门控网络只读取 label-free node statistics（例如三态入/出质量、跨模态邻域重叠、multi-start stability、局部图尺度与表示范数），输出 `g_self, g_sup, g_bnd, g_conf`：

```text
g_i = softmax(G_phi(q_i))
R_i = g_sup M_sup + g_bnd M_bnd + g_conf P_conf(M_conf)
Z_i = LayerNorm(T_i + alpha_i * P_res([T_i, R_i]))
```

`alpha_i` 必须被 trust/self-return 约束：高可信强起点默认接近 0；只有边场证据充分时才放大残差。允许 Codex2 基于真实源码与数值稳定性改写为等价的更简单实现，但必须保留“三态对应三种运算 + 强起点残差锚定”这两个不可变语义。

### 3.4 无标签训练目标

建议的最小目标为：

```text
L = lambda_rec L_reconstruct
  + lambda_sup L_support_attraction
  + lambda_bnd L_boundary_margin
  + lambda_conf L_conflict_private_preservation
  + lambda_anchor L_teacher_anchor
  + lambda_var L_noncollapse
```

- `L_support_attraction` 只拉近 support 边。
- `L_boundary_margin` 只对共同边界执行 margin separation；不得把所有非邻居构成 dense N×N negatives。
- `L_conflict_private_preservation` 禁止冲突边被强制跨模态对齐，并要求至少能保留/重构各模态私有信息。
- `L_teacher_anchor` 以 label-free trust 加权，防止强表示被整体破坏。
- `L_noncollapse` 可用方差/协方差约束或等价稳定实现。

真实标签不得进入 forward、loss、gradient、early stopping 或单次训练内 checkpoint 选择。公开标签可以在候选训练完成且工件锁定后，用于透明的 family-level benchmark HPO。

## 4. 参数与数据集规则

模型大框架和参数语义完全统一。允许两个预先定义的模态家族各冻结一组数值：

- RNA+protein：A1 + tonsil s1 用于开发；冻结后只迁移到 D1、tonsil s2、tonsil s3。
- RNA+chromatin：P22 K=9 用于开发；冻结后只迁移到 MISAR K=7。P22 K=18、MISAR K=12 只作 secondary protocol。

禁止用 dataset name 在代码中选择模型分支。输入 adapter、特征维数、K、平台预处理和家族默认值可以显式配置；同一家族冻结后不得再根据确认集标签改数值。

同时保留一个“逐 lane benchmark-HPO score frontier”作为探索板，但必须与 family-frozen method evidence 分表，不能用逐 lane 最优冒充自动部署模型。

## 5. 必须修复的 Night-16C 语义缺口

1. `GENERIC_REPAIR_ONLY` 不能再用 `sweeps=0` 与 `STRONG_START_ONLY` 做成同一对象；必须调用 Night-16B 的真实 generic repair，或明确改名为 no-repair control。
2. `DIRECTIONAL_EDGE_ONLY` 目前仍包含 CMBF conductance/unary/pairwise，不是方向单项；必须实现真实方向单项或改名并准确披露。
3. chromatin 的 `directional_mix=0` 意味着方向模块实际未贡献，不能在主张中把它算作已验证组件。
4. 增加以下最小贡献对照：teacher only；generic signed/low-pass graph residual；support-only；support+boundary；support+conflict；full tri-state；full without trust/self-return；full with shuffled edge states。

## 6. 开发漏斗与晋级规则

### P0：真实端到端语义

至少 A1、P22 各 1 个 seed 走完：真实 feature/preprocessing -> H1/H2/T -> CMBF -> train -> checkpoint reload -> Z -> common clustering endpoint。必须确认参数和表示均发生有限、非零变化，并可 fresh-process 重放。

### 快速开发

先用每家族的开发单元、1–2 training seeds 搜索结构和粗参数。允许正常的工程修复与科学迭代；不再设置荒谬的一次 correction 上限。每次公式变化需登记版本，冻结后才进入确认。

### 冻结确认

仅当至少出现下列任一情况才扩为 3–5 seeds：

- 两个不同 study 的开发协议都有 ARI/NMI 双升；或
- 一个开发协议获得明显提升（建议绝对 ARI 或 NMI >= 0.01），另一个无明显灾难性下降；或
- family-balanced ΔQ（仅内部筛选量）稳定为正，并且最小簇/空间连续性无明显退化。

确认输出必须同时报告 best、median、mean、min；best-run 用于 score frontier，family-frozen mean/median 用于方法证据。不要因为一个 seed 未赢就自动否决，但不得隐藏任何已正式运行的 seed。

## 7. 成功与失败解释

- `REPRESENTATION_METHOD_SIGNAL`：full tri-state 在同起点、同 endpoint 下，相对 teacher 和 generic graph residual 有可重复独立增益，并至少跨两个真实 study。
- `FAMILY_LOCAL_SIGNAL`：只在一个家族/部分数据有效。
- `HEAD_OR_TEACHER_SIGNAL`：高分来自起点或 head，full representation 无独立贡献。
- `SCIENTIFIC_NEGATIVE`：实现正确但无独立增益。
- `IMPLEMENTATION_FAILURE`：真实语义或重放不成立。

任何单次新高都进入 score frontier，但只有符合上述贡献对照后才可归因于新模型。

## 8. 新颖性边界

论文可比较的关键区别应表述为：

- PRAGA：动态图与 prototype 对比；本方法关注同一空间边的跨模态三态语义及关系专属运算。
- SpaMV/SpaMode：节点/潜变量层面的 shared-private 分解；本方法在局部边层面先判定 support/boundary/conflict，再决定平滑、分离或私有保留。
- SpaBalance：训练梯度层面的全局模态冲突协调；本方法是空间边层面的局部冲突路由。
- ARISE：RNA 锚定的共享边交集；本方法不把 RNA 永久设为唯一权威，而由两视图共同判定边状态。
- CoMo/PRESENT/SpatialMOSI：通用跨模态或层级图对比；本方法的正负/解耦关系来自三态边场，不是通用 same-neighbor positives。

如果源码/论文检索发现已有方法已完整实现“跨模态三态边 + 三种关系专属算子 + self-return 残差锚定”，立即降级新颖性并报告，不得换名字规避碰撞。

