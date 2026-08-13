# SpaLORA Night-5A 独立规划审计

日期：2026-08-13  
审计方式：仅只读读取 D 盘 official compact handoff；未连接 AutoDL，未修改 Night-5A 结果。

## 1. 交付完整性

用户给出的五个 SHA-256 均由本机重新计算并逐字匹配：

| 文件 | SHA-256 | 结果 |
|---|---|---|
| `night5a_report.md` | `f7a1ae426ec6e019f49639e67d58f775e902cdbc7718cad866a40678c2a2282b` | PASS |
| `night5a_planner_handoff_20260813.tar.gz` | `7fd03857e38919603702835cb4eb7bc7af249d4533469b5c795dfc28f7c30673` | PASS |
| `SpaLORA_night4a_to_night5a_20260813.bundle` | `f5d43ad20e5ee0eefb4ee6435051e10839b40a4be6d1ca721d92c6f2482c4ef4` | PASS |
| `delivery_index.json` | `61bda9bd62fd861a9531f5dce682c4a0ca1fe0d6cafc0d9a49c6b9043bac89b8` | PASS |
| `shutdown_command_status.json` | `b44d4df01fc3e5c7b60290c562aa8ea9e27c62d46302761fdb4d6096377165fe` | PASS |

compact archive 可正常列出 79 个内部条目，delivery index 报告 79/79、0 failures。报告、逐 run 汇总、三阶段 decision、源码、测试、标签防火墙、原始结果 manifest 和增量 bundle 均在精简包内。

## 2. 执行完整性

- R1 34/34、R2 24/24、R3 16/16；
- 总计 74 个训练单元，0 个 scientific run failure；
- specialized tests 最终 39 passed、0 failed；
- 首次 26 passed/13 failed 的日志得到保留，原因是 audit assertion 把结构零误计为 support loss；修复只改 audit expression，没有改图公式或候选参数；
- Night-3B 1186/1186、Night-4A 76/76 历史保护通过；
- P22、D1、GSE198353、Night-4B 均未运行；
- candidate/seed/parameter/budget 未扩展；
- GitHub branch、commit、tag 已非 force 持久化。

审计未发现需要推翻 Night-5A 结果的完整性问题。

## 3. C04 的真实强度

C04 相对 C00 的五 seed 结果：

- macro delta ARI `+0.040030`；
- macro delta NMI `+0.025219`；
- macro delta Q `+0.032624`；
- A1 delta Q `+0.013413`；
- Placenta delta Q `+0.051835`；
- paired wins `8/10`；
- runtime ratio `0.843`；
- GPU ratio `0.968`；
- Night-5A 空间保护门 PASS。

绝对五 seed mean：

| 数据集 | C00 ARI/NMI | C04 ARI/NMI |
|---|---:|---:|
| A1 | 0.246859 / 0.372468 | 0.262235 / 0.383919 |
| Placenta | 0.604405 / 0.659636 | 0.669088 / 0.698623 |

C04 是目前最稳妥的跨开发集候选，但不是所有指标和 seed 都提升：A1 seed 3 与 Placenta seed 3 的 paired Q 分别为 `-0.005563`、`-0.008527`。因此应表述为“平均提升且 8/10 paired wins”，不能写成逐 seed 全胜。

## 4. 不应丢弃的 C09/C10

### C09

- macro delta Q `+0.056966`；
- A1 delta Q `+0.005087`；
- Placenta delta Q `+0.108845`；
- paired wins `9/10`；
- Placenta ARI/NMI `0.722441 / 0.759290`；
- runtime/GPU ratio `0.645 / 0.887`。

空间代价主要来自 Placenta：neighbor 从 `0.477256` 降到 `0.415644`，Moran 从 `0.469405` 降到 `0.368785`，Geary 从 `0.674023` 升到 `0.732992`。

### C10

- macro delta Q `+0.065371`；
- A1 delta Q `-0.002333`；
- Placenta delta Q `+0.133075`；
- paired wins `7/10`；
- Placenta ARI/NMI `0.748868 / 0.781323`；
- runtime/GPU ratio `0.692 / 0.889`。

Placenta neighbor/Moran/Geary 分别为 `0.412635 / 0.370617 / 0.731695`，同样是准确率大增而连续性下降。

这两个候选按原 Night-5A 规则停止是正确执行，但“停止”不是“科学机制被否定”。值得注意的是，历史 Placenta C1 的 neighbor/Moran 约为 `0.4308 / 0.3753`，与 C09/C10 很接近，而 C09/C10 的 ARI/NMI 更高。因此应将其保留为 accuracy frontier，并尝试轻量空间救援。

## 5. 一个需要继续审查的损失现象

`attention_loss_collapse_audit.json` 按预设阈值判定 C04 没有 collapse；数值本身显示：

- A1 的 RNA reconstruction contribution fraction 约 `98.74%–99.10%`；
- A1 Corr1 约 `0.17%–0.20%`；
- A1 modality-2 reconstruction 约 `0.72%–1.06%`；
- Placenta RNA reconstruction 约 `94.17%–94.51%`。

这不构成协议失败，因为预注册阈值没有触发；但从机制解释看，step-0 IGE 并没有保证训练后各目标持续平衡。后续论文不能简单宣称“所有损失始终均衡”。它也是未来动态校准研究的依据，但 Night-5B 不同时扩大到另一套大规模动态 loss grid，以免一次改变过多机制。

## 6. 第三方源码带来的真实信息

本次不是只读论文摘要，而是核对了官方源码：

- SMART：共享 embedding 上叠加各模态 reconstruction 与 triplet loss；源码中的 Laplacian 是可选项且默认权重为 0；
- COSMOS：两层 GCN+PReLU，训练中途基于 learned latent 一次性计算 spot-wise WNN 权重并冻结；这与 Night-5A 的 input-space frozen reliability 不同，值得独立测试；
- COSMOS：默认 accelerated spatial penalty 中第二组坐标索引与第一组相同，按源码会使 sampled spatial distance 失真，不能直接复制；
- ARISE：官方 `train.py` 和 `code/hln.py` 每个 epoch 都读取真实标签、聚类、计算 ARI，并保留 best-ARI embedding。其 headline 数值不能当成严格无标签 checkpoint-selection 上限；
- SpaMFG：公开实现是 dataset-specific 单脚本，含大量 hard-coded feature/cluster/spatial 参数，不能直接当通用库移植；
- SpatialCOC：坐标 INR 与 DCCA 思路有启发性，但 3000-wide SIREN/CCA 路线成本和改动面都较大，不作为下一轮首要候选。

## 7. 关机异常复盘

`shutdown_command_status.json` 证明：

- 尝试的命令确实是 `/usr/bin/shutdown`；
- 它是最后一次远端命令尝试；
- 但 AutoDL gateway 在 SSH banner 前关闭连接；
- 因此命令送达未确认，结果只能是 `UNKNOWN_NOT_CONFIRMED`。

根因不是 shutdown 命令拼写，而是流程把唯一关机动作放在所有工作结束后的全新 SSH 握手上。网关或实例在这一刻不可达，就失去关机保障。

永久修订：

1. 不再以末尾新 SSH 连接作为唯一关机手段；
2. GPU 实验开始时设置远端最长运行 fail-safe；
3. 交付完成后从 Windows 侧调用官方 `power_off` API；
4. 官方 status 连续确认停止三次才报告成功；
5. API 未确认时立即告警用户，不得把 readiness 文件写成 shutdown proof。

## 8. 规划结论

Night-5A 是一次成功的开发集研发轮：C04 给出跨 A1/Placenta 的稳定平均提升，C09/C10 给出更强的 accuracy-frontier 信号。最合理的下一步不是立刻消费 P22，而是先进行一次有限、预注册、无 P22 的 second-look + rescue：

- 补完被 family cap 截断的正向候选；
- 组合 C04 与 anchor/MNN/DGI；
- 测试 learned-latent reliability；
- 用单步 diffusion 或小权重梯度归一化 Laplacian 救援 C09/C10；
- 同时保留 balanced 与 accuracy 两条 Pareto 前沿；
- 最后才签发一次真正锁定的 P22 检验。
