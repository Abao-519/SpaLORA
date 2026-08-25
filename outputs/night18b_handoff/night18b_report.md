# Night-18B report

## 我现在需要知道的三件事

1. **本轮解决什么问题：** 原计划检验跨模态频率相干，但 SMGFM 已公开覆盖频带、语义角色和可靠性路由，因此该方向在运行前即被淘汰。实际检验的是更物理的“模态空间测量响应校准”：先估计 RNA 与第二模态各自的有效空间粗糙度，再用有界前向/逆向图响应对齐到共同组织带宽。
2. **实际改了哪一层：** 改的是进入共同残差 backbone 之前的表示输入层。公式、训练、候选数和 endpoint 在 A1 与 P22 完全相同；12 个真实训练工件均有参数更新、严格 checkpoint reload 和 fresh-process 分区逐字节重放。
3. **对论文意味着什么：** 终态是 **SCIENTIFIC_NEGATIVE**。A1 的 full 在 common KMeans 仅有 +0.001886/+0.001358，且 low-pass-only 几乎解释全部；P22 的 full 在 common KMeans 双降。P22 的 GMM 条件增益与 sharpen-only 分区逐字节相同，属于单模态锐化/endpoint 条件信号，不是统一响应校准贡献。没有进入多 seed，也没有刷新历史 frontier。

## 绝对指标主表

| 数据集 | 同口径 endpoint | Matched backbone | Full calibration | ΔARI / ΔNMI | 最强解释性 control | 历史 frontier |
|---|---|---|---|---|---|---|
| A1 K10 | COMMON_KMEANS | .232410 / .388158 | **.234296 / .389516** | +.001886 / +.001358 | low-pass-only .234001 / .389618 | .276003 / .421740 |
| P22 K9 | COMMON_KMEANS | **.419206 / .551188** | .417729 / .549751 | -.001477 / -.001437 | sharpen-only = full | .595552 / .717931 |
| P22 K9 | COMMON_GMM context | .408791 / .553338 | .451848 / .578726 | +.043057 / +.025388 | **sharpen-only byte-exact = full** | .595552 / .717931 |

完整 48 行的 ARI/NMI/AMI/FMI、空间指标、簇大小和资源见 `absolute_metrics_main_table.csv`；每个 endpoint 的同端点差值和对齐后 changed spots 见 `matched_contribution_table.csv`。

## 机制归因

- A1 原始粗糙度为 1.2012/0.9306；full 自动选择 `(beta,gamma)=(.35,.10)` 与 `(.15,.35)`，log-gap 从 .2552 降到 .0420。这证明校准算子确实按输入统计工作，但分数增量过小且未独立胜过单向 controls。
- P22 原始粗糙度已很接近（1.2061/1.1602）；full 只对第二模态选择轻锐化 `(beta,gamma)=(.10,.15)`，因此与 sharpen-only 表示和分区相同。这里不能声称“双模态共同带宽校准”。
- Swapped response 在 A1 的固定结构 endpoint 得到 .241818/.397338，但它是负对照且依赖另一 endpoint；它不能反向证明 full，反而说明当前粗糙度匹配目标与聚类质量并不一致。
- 所有结果仍远低于项目历史强 representation/head frontier；没有把 endpoint 变化伪装成 representation 胜利。

## 新颖性分诊

SMGFM 已覆盖多模态图频带语义和可靠性路由，GatorPrism 已覆盖 joint/private experts 与 prototype router；DRIFT、SpaGFT/DeepGFT 和 SpaDDM 分别覆盖低通/图频/方向扩散。Night-18B 的最小候选对象是“输入粗糙度驱动的有界逆/前向测量响应等化”，源码层面未发现完全同构，但本轮实证门失败，因此不进入论文贡献清单，只保留为可复算负结果。

## 失败与局限

- 只有 seed 0，因为预注册独立贡献门未通过；补 seed 无法把 control-identical 的 P22 结果变成新机制证据。
- 粗糙度是全局统计，可能无法描述局部组织带宽异质性；但继续加局部 router 会与最新先例高度碰撞，且本轮没有信号支持扩张。
- P22 的锐化条件信号可作为后续成熟 preprocessing 诊断，但不得作为 SpaLORA 新模块。
- 没有新数据下载，没有在 inode 已满的持久盘写文件。

## 导师汇报版

1. 我们先主动否决了已被 SMGFM 覆盖的“频率专家”方案，没有换名抢创新。
2. 随后实现了一个更有物理含义的统一模块：估计每种组学的空间测量带宽，再用有界有理图响应校准到共同尺度。
3. A1 与 P22 的真实训练、checkpoint、候选生成、独立评价和 fresh-process 重放全部闭合。
4. A1 只得到千分位提升，而且低通单臂几乎可解释。
5. P22 在共同 KMeans 上反而下降；GMM 上的较大提升与“只锐化一个模态”完全相同。
6. 因而这轮不能证明统一测量响应校准是论文方法贡献，分类为 SCIENTIFIC_NEGATIVE。
7. 我们保留了 P22 的单模态锐化/head 条件线索，但没有补 seed 或扩大网格美化结果。
8. 服务器保持开机，未派发 shutdown。

## 技术附录

- taskbook SHA-256: `96aba215df9c8d6ef18e386c515e2da43a587f1433fef6cd7b01e1bb867b483f`
- parent commit: `2c9d59098afd6ef927746ddf2774887fcf681abc`
- formal artifacts: 12; exact representation/partition replays: 12/12; targeted tests: 6/6.
- labels: producer 0 reads; evaluator 1 post-lock read per artifact; benchmark endpoint selection is transparently label-assisted.
- shutdown_dispatched=false; KEEP_ON_FOR_CONTINUOUS_RESEARCH.
