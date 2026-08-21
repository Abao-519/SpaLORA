# SpaLORA Night-10A REV2：真实组合覆盖、跨维度协调与安全恢复任务书

日期：2026-08-21  
运行：AutoDL 有卡模式  
目标：在不读取标签的前提下修复 P22 参考表示维度，证明全部真实组合可运行，尽量复用未受影响的 REV1 计算，再完成原定 R1/条件 R2。

## 1. 先说明白本轮在修什么

REV1 不是“候选分数低”，而是 P22 的真实输入维度没有在 P0 组合测试中出现。REV2 只允许三类变化：

1. 冻结的参考维度协调器；
2. 覆盖 30 个真实数据-seed 单元、7 个候选的运行时预检；
3. 对 REV1 未受影响结果进行无标签、整组复用审计。

QCRD 的质量特征、mask、loss 权重、候选、优化器、endpoint 和 frontier 规则都不得改动。原 Night-10A 与 REV1 的失败 commit/tag/report 永久保留。

## 2. 最高权威文件

开始前完整读取并独立复算：

1. `night10a_qcrd_rev2_dimension_and_reuse_contract.json`；
2. `Night10A_REV1_Failure_Root_Cause_and_REV2_Safety_Audit_2026-08-21.md`；
3. REV1 compact：`D:/文档/ChatGPT/博士第一篇科研论文项目/night10a_rev1_handoff_20260821/official_compact`，内部 118/118；
4. REV1 report、retrospective false-pass audit、resource audit、source 和 tests；
5. 原 REV1 semantic contract 与 taskbook。

以上文件的精确 SHA 由配套复制提示词给出。任一不匹配立即停止，不用相似文件替代。

## 3. Git 和目录

- 从 `c7cb3ecd1631db7a21606e62c69eaec05fe31ac1` 新建 `revision/q2-night10a-rev2-real-schema-harmonization-20260821`。
- 创建一次保护标签 `baseline/pre-night10a-rev2-real-schema-harmonization-20260821`，指向父提交。
- final tag：`night10a-rev2-final-20260821`，交付文件进入 final commit 后只创建一次。
- 普通 push；无 force；不移动任何旧 tag。
- 复用远端 `/root/autodl-fs/night10a_rev1_qcrd_20260821`，只读核验旧文件；REV2 新输出写入 `/root/autodl-fs/night10a_rev2_qcrd_20260821`。
- 不覆盖、移动或删除 REV1 raw。

## 4. P0-REV2A：静态定义和负向测试

严格实现 REV2 dimension contract。至少新增：

- 同维 identity 的 canonical array SHA 完全不变；
- 64→128 的真实/小型矩形 Procrustes shape、finite、秩、正交性和几何保持测试；
- 重复计算的 P SHA 一致性；
- `d_ref>d_view`、rank deficient、NaN、spot 顺序不一致均 fail closed；
- adapter 输入、最终三视图融合、boundary loss 均使用同一个 `zf_aligned`；
- 函数签名不得接收 dataset name；用 AST 和调用追踪检查无 dataset routing；
- Q06 额外坐标维度在混合维度输入下的真实 shape 断言；
- `prepare_one` 必须断言 N、全部 feature dims、dtype、finite、ordered spot SHA、graph shape 和最终 adapter 输入宽度，不能只检查 cardinality。

原 REV1 测试全部保留并通过。

## 5. P0-REV2B：30×7 真实运行矩阵

这是允许正式训练的必要条件，不能被合成测试替代。

### 5.1 真实单元

- A1：seeds 0-4；
- tonsil：seeds 0-4；
- D1：seeds 0-9；
- P22：seeds 0-9。

生成 `real_runtime_schema_matrix.csv/json`，每个单元记录 `N,d_z1,d_z2,d_zf_raw,d_zf_aligned,d_coords,K`、全部输入 SHA、ordered spot SHA、graph SHA、harmonizer mode/SHA、code commit 和 contract SHA。

### 5.2 210 个真实候选组合

每个单元依次运行 Q01-Q07：真实 forward、全部注册 loss、backward、gradient finite、一次临时 optimizer step、checkpoint state round-trip、eval forward、输出 shape/finite。临时状态不作为正式 checkpoint，不进入科学预算。

必须明确包含真实 P22 的 `128+128+64→aligned 128` 路径及 Q06 的 16 维坐标。不能只测一个 P22 seed，也不能只测一个候选。

每行输出唯一 `preflight_row_sha256`。210/210 全部通过后，普通 commit/push P0 证据；每个后续 formal config 必须引用完全匹配的 row SHA。若 209/210 或存在未覆盖项，立即终止并关机，不启动正式训练。

为降低费用，P0 阶段使用已有 frozen quality/partition 时先核验 SHA；不重复运行昂贵 transform。预检目标是证明真实训练计算图完整，而不是提前聚类或评价。

## 6. REV1 结果复用审计

P0 通过后、仍不读标签：

1. 复算 REV1 raw 声明文件 SHA；
2. 对 A1、tonsil、D1 分别执行 21/21 checkpoint load 和 corrected-view replay；
3. 验证 identity harmonizer、state、config、input、corrected arrays、已有 partition；
4. 以数据集为组给出 `REUSE_ALL_21` 或 `RETRAIN_ALL_21`，禁止部分复用；
5. A1+tonsil 的 42 个 transform 只有在全部证据吻合时复用；D1 的缺失 transform从锁定 corrected views 补齐；
6. P22 的 21 个单元新训练，不能使用 REV1 failure placeholders。

如果需要重训某个 RNA+protein 数据集整组，这属于 REV2 正式训练，不是 scientific retry；原因和授权必须在训练前写入锁定 manifest。不能在看到标签后改变复用决定。

## 7. R1、评价与条件 R2

其余顺序沿用 REV1：

1. 完成/复用 Q01-Q07 × 4 datasets × seeds 0/1/2 的全部 R1 输出；
2. checkpoint/embedding/affinity/partition/manifests 总锁并普通 push；
3. 只在总锁后打开各核心数据集唯一 R1 评价窗口，计算 ARI、NMI、Q、扩展标签指标、无监督几何、跨模态配对和空间指标；
4. 按原三条 frontier 规则冻结最多三个候选；
5. 只有满足原条件才进入 R2，R2 不得重新调整 REV2 映射、损失、阈值或候选。

任何 ordinary numerical failure 原样保留并继续独立单元；真实 schema、错误 source/config、标签泄漏或系统性实现错误才 fail closed。

## 8. 时间与资源控制

- 使用有卡模式；真实 adapter 训练必须 CUDA。
- P0-REV2 预计应远短于完整实验；若 P0 总墙钟超过 90 分钟仍未完成，停止并交付性能诊断，不进入正式训练。
- 单正式 transform 30 分钟 timeout；最多 4 个 CPU 并行，先做内存探针。
- 整轮墙钟 12 小时；到点停止派发新单元并交付，不无限等待。
- 只在 P0 完成、复用审计完成、R1 总锁、评价、R2 总锁、最终交付等边界汇报，避免高频轮询耗费 token。

## 9. 交付和关机

D 盘只交付 compact：报告、通俗总结、两份 REV2 权威文件、真实 schema matrix、210-row preflight、reuse audit、代码/tests、配置/输出索引、指标/资源/标签审计、增量 bundle 和根相对 SHA 索引。不要下载 raw runs、checkpoints、full embeddings、affinities 或全历史 bundle。

Windows 独立复算全部 SHA。最后保留同一 SSH 会话，将 `/usr/bin/shutdown` 作为最后一条远端命令；派发后不重连，只报告派发事实。

## 10. 最终回复必须让用户看懂

先用通俗中文回答：

1. 这次是否真正测全了所有真实输入组合；
2. 旧的 63 个训练中哪些被整组复用、哪些重跑；
3. P22 的维度问题是否彻底解决；
4. QCRD 是否让 A1、D1、P22 协同提高，tonsil 是否保住；
5. 若没有候选，明确说 QCRD 方向暂时失败，不包装。

随后再给技术表、失败单元、标签读取次数、Git、SHA 和关机派发记录。

