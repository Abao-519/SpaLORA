# SpaLORA Night-16G 报告：多模态候选解选择器

## 我现在需要知道的三件事

1. **问题**：同一 RNA+ATAC 数据、同一个已登记 `K`，不同初始化、能量强度和图尺度会产生很多候选分区。本轮检验能否不用当前数据的逐点标签，从这些候选中稳定选出高质量解。
2. **实际动作**：我们在聚类终端新增了候选锁定、分子分离、稀疏空间拓扑、小簇风险与有序能量路径证据。这里“解盆地”通俗讲就是：参数稍微改变时仍反复出现的一群相似分区。候选先保存并哈希，随后公开标签才由独立 evaluator 用于诊断和跨运行 HPO。
3. **论文意义**：正式拟合把“路径持久性”权重压到 **0**，所以稳定盆地假设没有得到支持；留下的是多证据选择器。共同标签辅助 HPO 在 4 个协议中相对普通 medoid 提升 3 个，但严格留一研究在人海马上失败。因此终态是 **`LOCAL SIGNAL`**，不是新盆地方法、不是可部署统一 head、也不是 SOTA/confirmed milestone。

## 明确终态

- `classification`: **LOCAL_SIGNAL**
- `status`: **NIGHT16G_MULTI_EVIDENCE_SELECTOR_LOCAL_SIGNAL**
- 证据层级：公开 benchmark 的共同标签辅助 HPO；另做严格留一研究的参数转移。
- 路径持久性结论：**SCIENTIFIC NEGATIVE**（正式权重 `0.0`）。

## 绝对指标主表（共同公开 benchmark HPO 输出）

| 协议 | N / eval | K | 选择分区 | ARI | NMI | AMI | FMI | Moran宏平均 | Geary宏平均 | 最小簇 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| P22 | 9196 / 9196 | 9 | pairwise path L03 / authority start | **0.596390** | **0.718243** | 0.717756 | 0.656194 | 0.931968 | 0.070524 | 166 |
| MISAR E15.5 | 1949 / 1949 | 7 | authority direct | **0.535306** | **0.658265** | 0.656463 | 0.627985 | 0.922517 | 0.079430 | 125 |
| Human hippocampus | 2500 / 2500 | 7 | retained S4 / uniform-mass matched | **0.596178** | **0.585490** | 0.583641 | 0.687224 | 0.774107 | 0.217786 | 34 |
| Slide-tags melanoma tumour | 833 / 833 | 2 | retained S3 / direct | **0.914177** | **0.838439** | 0.838286 | 0.962071 | 0.934295 | 0.000019 | 276 |

这些是共同规则经四个公开 benchmark 标签做跨运行 HPO 后的输出，不是对四个研究的盲测。

## 与普通规则、严格转移和 oracle 的对照

| 协议 | 普通 partition medoid | 固定 authority direct | 全局多证据规则 | 严格 LOSO | label-assisted oracle |
|---|---|---|---|---|---|
| P22 K9 | .514452 / .644919 | .594837 / .715742 | **.596390 / .718243** | **.596390 / .718243** | .596390 / .718243 |
| MISAR K7 | .363107 / .550287 | **.535306 / .658265** | **.535306 / .658265** | **.535306 / .658265** | .541932 / .665867 |
| Human K7 | .565455 / .569491 | .167510 / .266756 | **.596178 / .585490** | **.231039 / .426182** | .634591 / .589995 |
| Melanoma K2 | **.918830 / .845173** | .293643 / .195757 | .914177 / .838439 | .914177 / .838439 | **.975766 / .942756** |

黑色素瘤的 minimum inertia 与 maximum Calinski–Harabasz（CH）都直接选中 oracle S3，说明该数据不需要新的“盆地”叙事。人海马的严格留一研究配置去掉了小簇风险权重，随后选出最小簇为 1 的脆弱分区；这是转移失败，原样保留。

## 哪一层真正有信号

冻结的全局权重是：分子分离 `0.25`、空间拓扑 `1.0`、小簇风险 `0.25`、持久性 `0.0`。消融显示：

- 去掉空间拓扑后，四条均明显恶化；它是主要证据轴。
- 去掉分子轴对 P22/MISAR影响小，但人海马从 `.596/.585` 降到 `.231/.426`。
- 去掉小簇风险对 P22/MISAR/黑色素瘤不变，却使人海马同样降到 `.231/.426`。
- 强制持久性为正的最佳共同配置虽然保持 P22、人海马和黑色素瘤，却把 MISAR 降到 `.361589/.548696`。
- basin-first 在 P22/MISAR没有形成可靠提升；因此不支持把 ordinary consensus 或路径持久性作为论文贡献。

这说明当前信号来自**候选库 + 分子/拓扑/小簇风险的多证据排序**，而不是有序路径本身。P22 选中的路径候选提供了一个很小的分数前沿，但选择它的正式打分没有使用路径持久性。

## 严格 LOSO 的进程边界

每个 leave-one-study-out（LOSO，留一研究）配置在一个只接收其余三项 evaluation 的进程中拟合；held-out evaluation 未打开。配置写入并哈希后，另一 producer 才对 held-out candidate bank 选择，最后独立 evaluator 打分。三条保持强结果，但人海马失败，所以没有形成跨研究 confirmed milestone。

## 真实路径与复现

四条路径均完成：真实 reduced views/稀疏图与 locked bank → 89×89 candidate similarity → evidence selector → partition save → fresh-process reload → 独立 evaluator。每条候选库为 89 个分区，其中 42 个来自六个 start 的七级有序能量路径。最终源码后两次独立运行 4/4 在 ordered IDs、candidate ID、config SHA、partition SHA 上完全一致；targeted tests 为 9/9。GPU 时间 0，dense observation×observation 矩阵 0。

| 协议 | retained | RNA view | ATAC view | 三个稀疏图 nnz | start bank |
|---|---|---|---|---|---|
| P22 K9 | 9196×64 float32 | 9196×30 float32 | 9196×50 float32 | 37,168 / 64,466 / 178,238 | 6×9196 |
| MISAR K7 | 1949×64 | 1949×30 | 1949×50 | 8,070 / 13,666 / 38,212 | 6×1949 |
| Human hippocampus K7 | 2500×64 | 2500×64 | 2500×64 | 10,200 / 20,212 / 48,770 | 6×2500 |
| Melanoma tumour K2 | 833×64 | 833×64 | 833×64 | 4,222 / 8,176 / 18,100 | 6×833 |

## 最重要的失败和限制

1. 正式 persistence 权重为 0，不能声称稳定盆地方法成功。
2. 严格 LOSO 人海马崩溃并产生 singleton，统一参数转移仍不可靠。
3. 全局规则使用四个公开 benchmark 标签做跨运行 HPO，不能当成新数据自动输出。
4. 候选起点含历史资产；P22/MISAR authority start 的历史开发 provenance 不会因本轮 selector 而消失。
5. 黑色素瘤普通 inertia/CH 已优于全局多证据输出。
6. STCC、SACCELERATOR、PHD-MS、SCALE、SMODEL 已覆盖 consensus、multiscale persistence 或 weighted ensemble 的关键先例；新颖性尚未成立。

## 5–8句导师汇报版

Night-16G 研究的是在多个聚类候选之间怎样自动选解，而不是继续改 RNA+ATAC 表示。我们锁定了四个真实研究共 89 个候选，并把分子可分性、稀疏空间拓扑、小簇风险和能量路径稳定性放进同一选择框架。共同公开 benchmark HPO 相对普通 medoid 在 P22、MISAR和人海马上提高，黑色素瘤略降。黑色素瘤的普通 inertia/CH 已经能选到最高分解，因此这里没有新方法贡献。更关键的是，正式拟合把路径持久性权重压到零，说明“解盆地”故事没有独立证据。严格留一研究在三项保持，但在人海马失败且出现 singleton，所以不能升级为跨研究里程碑。当前最准确的结论是多证据 head 的局部工程信号，下一步若继续应解决真正无标签的跨研究标定，而不是扩大候选库后做标签 HPO。

## 技术附录

- 分支：`revision/q2-night16g-multimodal-basin-selector-20260824`
- 父级：Night-16F commit `b138195a2cbfb673ef95962a06a460ab1f842b2d`
- 线程：OMP/MKL/OpenBLAS/threadpool 均固定为 1
- 普通 push、final commit/tag、bundle 与 compact SHA 见交付 verification；GitHub SSH 若仍缺钥匙会保留普通 push 失败事实。
- `shutdown_dispatched=false`；按连续夜间任务要求保持 AutoDL 在线。
