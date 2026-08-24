# Night-16D 执行任务书

## 目标

在 Night-16C final commit `8537412a8c3f0f05000940cf1e5385561e2755e4` 基础上，实现并检验 CMBF-RL：一个统一的、三态跨模态边驱动的可训练残差表示模块。主要精力放在自身方法与分数，不做完整外部方法复现。

## 执行原则

1. 先完整读取 Night-16C report、decision、代码、family HPO、matched ablation、P0 记录和本目录的数学合约；核对 final commit/tag/bundle。
2. 保持模型公式统一，仅允许 RNA+protein 与 RNA+chromatin 两个家族各有一组冻结数值。不得按数据集名称写模型 routing。
3. 允许公开标签做候选训练完成后的透明 benchmark HPO；不得进入模型训练、loss、gradient、early stopping 或单次训练内 checkpoint 选择。
4. 开发阶段允许 Codex2根据真实诊断自由简化/改进公式、结构和训练策略。每个科学版本登记即可；冻结之后不再回看确认集改公式。
5. 失败实验和负 seed 保留。工程 bug 可正常修复并重跑，不设一次 correction 上限；不得把工程修复伪装成科学结果。
6. 主要证据必须比较同一强起点、同一 endpoint 下的 teacher、generic residual 与 full tri-state。单纯 head 或初始化刷新只进入 score frontier。
7. 不扩增无关 benchmark，不做大规模外部复现。若有余力，只做两个高价值数据入口的轻量 provenance/annotation 闭合：GSE213264 tonsil 与 MISAR E18.5；下载前记录官方来源、大小与白名单，不能延误主模型。
8. 历史 raw 只读；不 force push。

## 最低执行内容

### Stage 0：审计与冻结前修正

- 复核 Night-16C 两个失真的消融语义并修正/改名。
- 检索并审计 PRAGA、SpaMV、SpaBalance、ARISE、CoMo/PRESENT 等的论文和源码语义，形成 collision matrix；只使用许可证允许的思想或 clean-room 实现。
- 写出真实 tensor shape、sparse edge counts、teacher/start 来源和标签读入边界。

### Stage 1：真实 P0

- A1、P22 各至少 1 seed。
- preprocessing -> 两视图表示 -> teacher -> CMBF -> train -> checkpoint strict reload -> fresh-process Z/partition replay。
- optimizer step > 0；参数和 Z 都必须有限且改变；GPU/RAM/time 实测。

### Stage 2：快速结构筛选

至少比较：

- teacher only
- generic low-pass或signed residual
- support-only
- support+boundary
- support+conflict
- full tri-state
- full without trust/self-return
- shuffled edge-state negative control

允许先在 A1、tonsil s1、P22、MISAR 中做低预算筛选。训练 seed 和 endpoint seed 分开记录。标签辅助的 family-HPO 必须在锁定候选工件后由独立 evaluator 完成。

### Stage 3：家族冻结与迁移

- protein 配置：只由 A1 + tonsil s1 选择；冻结后 D1 + tonsil s2/s3。
- chromatin 配置：只由 P22 K9 选择；冻结后 MISAR K7。K18/K12 为 secondary。
- 同时输出逐 lane score frontier，但与方法证据严格分表。
- 若快速筛选完全无表示增益，可提前 stop，保留 negative，并把节省的算力用于定位机制而不是盲目加大网格。

### Stage 4：交付

主报告先写“我现在需要知道的三件事”，随后给：

- outcome class；
- 每数据集强起点、full、generic baseline 的绝对 ARI/NMI/AMI/FMI、差值、空间指标、best/median/mean/min、最小簇、运行时间；
- family-frozen 与逐 lane HPO 两张独立表；
- 最小贡献/碰撞/失败记录；
- 标签读取、candidate-lock、checkpoint/replay、资源与 raw 不变性审计；
- 5–8 句导师汇报版；
- compact index、Git bundle、commit/tag。

## 自主连续运行规则

本轮由 Worker2 直接监控，用户暂时不在。**Night-16D 完成后不要执行 `/usr/bin/shutdown`，不要关机，也不要等待用户审核。** 先把完整结果回复到 Codex2 任务中；Worker2 会读取结果并决定是否直接派发 Night-16E。只有 Worker2 后续明确写出“本轮为最后一轮，关机”时，才把 `/usr/bin/shutdown` 作为最后一条远端命令。

