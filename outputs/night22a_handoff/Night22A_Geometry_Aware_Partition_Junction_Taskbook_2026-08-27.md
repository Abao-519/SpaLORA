# Night-22A：Geometry-Aware Partition Junction 架构发现与方法 P0

日期：2026-08-27  
性质：开发性架构发现 + 自研方法 P0，不是盲测、SOTA 或论文结论。  
工作名：`GAJ` 仅用于工程沟通；在贡献与碰撞审计成立前不得作为论文方法名。

## 1. 为什么现在做这一轮

Night-21C 已证明瓶颈不是单一层：

- A1 的表示仍有缺口；
- tonsil s1 的 endpoint 可带来较大恢复；
- P22 的 official graph-only 表示在标签后置的 KNN probe 上 balanced accuracy 达到约 0.941，但无监督 KMeans/GMM 仍远低于历史 frontier；
- placenta 的 AE-only 表示在线性 probe 上 balanced accuracy 约 0.874，GMM 明显优于 KMeans。

这说明至少在 P22 和 placenta，“局部或线性可分信息已经存在，但球形、同方差或单一密度假设的 endpoint 无法把它组织成全局分区”。下一步不应继续盲目更换 backbone，也不应把更大的 endpoint HPO 表冒充方法贡献。

## 2. 科学问题

能否在同一数学框架下，直接从锁定多模态表示和稀疏局部图学习 `N×K` 分区，使其同时适应非球形流形、大小不均的空间域及跨模态局部冲突，并在至少一个 RNA+protein 和一个 RNA+chromatin 单元上形成独立于输入 carrier 与简化臂的增益？

## 3. 不可混淆的贡献边界

1. 标签可以在候选产物锁定后由独立 evaluator 用于公开 benchmark HPO；必须保留完整候选表并如实称为 `label-assisted benchmark HPO`。
2. 标签不得进入模型输入、loss、gradient 或单次训练内部 checkpoint 选择。
3. `K` 按公开 benchmark annotation 协议给定。
4. RNA+protein 与 RNA+chromatin 使用同一模型方程、同一代码路径和同一参数语义；允许每个模态家族冻结一套数值配置，不允许按数据集名称切换算法分支。
5. GAJ 必须生成新分区。仅从既有 208 个 endpoint 中挑选、仅做共识或仅做 repair，只能登记为 head/selector signal。
6. 不要求官方外部方法整表复现。本轮重点是自身分区模块和分数；文献/源码工作只用于避免碰撞与指导实现。

## 4. Stage A：几何 ceiling 诊断

在四条开发 lane 上使用 Night-21C 锁定的 retained carrier、最有诊断价值的 official representation 及现有历史强起点：

- A1 K10
- tonsil s1 K4
- P22 K9
- placenta K10

补齐一个有预算上限的、可复算的 geometry board。至少覆盖：现有 KMeans/GMM、稀疏 self-tuning spectral/Ncut、扩散坐标后聚类、稀疏图 community/exact-K 方案，以及经源码核验后认为最有价值的非球形 endpoint。可以替换低价值候选，不要求机械跑满清单。

目的不是把最佳 head 写成创新，而是回答：

- 每条 lane 是球形、椭球、流形、密度还是空间边界问题；
- 哪种几何在同一 family 内可迁移；
- 历史 frontier 是否确实存在于可复用 carrier 中。

所有候选必须先生成、记录参数和分区哈希，再由独立 evaluator 读标签。保留失败配置，但无需为无意义的工程日志消耗大量交付篇幅。

## 5. Stage B：自研直接分区 junction

Codex 可根据 Stage A 证据自由选择最终实现，不预先把公式锁死。实现必须满足：

- 输入为锁定表示、空间图和模态内局部图；主要计算保持稀疏，避免无必要的 dense `N×N`；
- 直接学习软分区或等价的离散分区变量，而非先学 embedding 再无条件接普通 KMeans；
- 能表达非球形局部结构和不均衡 cluster mass，不强制等大小簇；
- 具有同一套跨家族计算图；节点/边/cluster 级自适应可以由数据内容产生；
- 参数真实更新，checkpoint/fresh-process 可重放；
- 最终给出 exact K、无空簇的硬分区；若含结构修复，必须单独对照其贡献；
- FULL 必须与 carrier、最强纯 head、最强单一原子臂和“关闭新 junction”版本匹配比较。

可重点考察但不强制照搬的方向：稀疏可微 normalized-cut、局部关系似然、图扰动稳定性、非均衡 cluster-volume barrier、cluster-conditioned 几何。若源码审计发现过于同构，应及时换方向。不要把 DEC、SwAV、P2OT、MDNC、S3RL/SEPAR、CRCT、DeepCut 或普通多视图图聚类的现成目标换名后称为自研。

开发阶段允许合理迭代、修 bug 和改公式；不设置人为的 correction-cycle 配额。每次实质变更保留简短 ledger，最终只冻结一个可解释版本。

## 6. HPO、冻结与确认

### 开发

- 先用 seed 0 做广搜索；只对前列结构做训练 seeds 0–2。
- family-HPO 分别以 A1+tonsil s1、P22+placenta 的 study-balanced ARI/NMI 为依据。
- 单数据集最佳结果可登记 score frontier，但不能冒充 family-default 方法证据。

### 进入冻结确认的最低门

满足以下任一条件即可进入，不要求所有数据集同时提高：

1. 四条开发 lane 中至少 2 条 ARI/NMI 双升，且覆盖两个模态家族，study-balanced macro ΔARI > 0.01；或
2. 至少两条 lane 刷新可信历史 frontier，且 FULL 在这些 lane 胜过所有匹配简化臂。

### 确认

冻结每个家族一套配置，再运行：D1、tonsil s2、tonsil s3、MISAR K7；不得重新按 lane 搜参。若某资产不可用，明确登记，不伪造替代。

确认里程碑建议门：至少一个 protein confirmation 与 MISAR 双升，四条确认 lane 的 study-balanced macro ARI/NMI 均不为负；同时开发增益不能由纯 head 或单一原子臂完全解释。

若只得到 geometry/head 提升，保留分数与代码，但终态写 `HEAD_ONLY_SIGNAL`。若自研 junction 未胜出，写 `NO_PARTITION_JUNCTION_SIGNAL`，不要追加无限 HPO 挽救。

## 7. 新数据低成本旁路审计

只做元数据、标签 provenance、K/mask、许可和下载体积审计，优先：GSE205055 mouse embryo RNA+epigenome、Stereo-CITE mouse thymus。方法 P0 未通过前不启动大规模下载。GSE213264 若仍没有闭合的全域 canonical ground truth，不得称专家标签 benchmark。

## 8. 工程与资源

- Night-21C 后根盘约 50 GiB、可用约 22.8 GiB。训练期间保持至少 15 GiB 可用；大型缓存和新数据优先放数据盘。
- 不重复复制历史 raw、conda cache 或官方仓库；先按尺寸列出候选清理项，只删除可重建缓存。
- 本地 CPU 能做的 endpoint/HPO/统计/索引尽量放 Windows；GPU 只承担训练或确有加速收益的任务。
- 运行开始前做四条真实 lane 的 preprocessing→forward/loss→checkpoint reload→hard partition P0；正式结果后做代表性 fresh-process replay。

## 9. Git

Night-21C 新私钥位于 `/root/.ssh/night21c_github_ed25519`，公钥必须先由用户注册到 GitHub。注册后只测试一次 `ssh -T` 和 `git ls-remote`；成功后普通 push。失败则停止反复尝试，交付可验证增量 bundle。严禁打印私钥或 token。

## 10. 交付

正文先给“我现在需要知道的三件事”，并给：

1. 绝对 ARI/NMI、差值、seed 分布、最小簇和资源主表；
2. geometry ceiling board；
3. FULL/atomic/head/carrier 匹配贡献表；
4. 方法公式、数据流图、复杂度与 collision matrix；
5. family-default 与 frozen confirmation；
6. 标签流与 HPO 过程；
7. 源码、测试、checkpoint/replay、失败 ledger；
8. root-relative size/SHA-256 compact index、Windows 独立复算、Git bundle。

最后将 `/usr/bin/shutdown` 作为严格最后一条远端命令；派发后不重连，且不把它表述为控制面板状态证明。
