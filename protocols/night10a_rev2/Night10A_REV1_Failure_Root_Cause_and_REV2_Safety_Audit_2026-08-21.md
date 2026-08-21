# Night-10A REV1 失败根因与 REV2 安全审计

日期：2026-08-21  
性质：本地只读审计；没有连接或启动 AutoDL；没有读取任何标签。

## 一、结论

Night-10A REV1 的失败不是普通数值波动，而是正式训练前的真实输入组合覆盖不足。

- RNA+protein 的 `z1/z2/zf` 均为 128 维，所以 63 个训练单元能够完成。
- P22 的 G04 私有视图 `z1/z2` 为 128 维，F00/R02 参考表示 `zf` 为 64 维。
- 原适配器按 `3*128=384` 构造输入层，但真实拼接只有 `128+128+64=320`；Q06 加 16 维坐标后是 336，而非 400。
- 原 P0 分别验证了真实 P22 endpoint 和合成的同维 adapter，却没有运行“真实 P22 混合维度输入 + 真实候选 + 全损失”的组合测试。因此原 P0 的 PASS 是覆盖不足造成的 false pass。

权威执行计数以 compact 文件为准：63 个 CUDA 训练完成，42 个 transform 完成，21 个 P22 单元在首次 forward、优化器更新前失败；A1、tonsil、D1、P22 标签读取均为 0。用户转述中的“39 个 transform”不是最终权威计数。

## 二、责任边界

这次问题的主要责任在任务设计，而不是执行者临时操作：

1. 维度合约只写了“统一 adapter”，没有把 `d_private` 与 `d_reference` 分开注册。
2. P0 用两个局部测试替代了真实跨组件测试。
3. `prepare_one` 只检查了 spot 数量，没有断言全部 feature 维度及 adapter 实际输入宽度。
4. 正式训练前没有生成覆盖所有真实数据/seed/候选的 runtime schema matrix。

## 三、REV2 唯一修复方向

REV2 引入一个冻结、无标签、与数据集名称无关的参考维度协调器：

1. 强制 `d(z1)=d(z2)=d_view`。
2. 若 `d(zf)=d_view`，返回原数组，严格恒等，不做 SVD 或重新归一化。
3. 若 `0<d(zf)<d_view`，用全部配对 spots 的无标签表示计算矩形正交 Procrustes 映射：
   - `X = row_normalize(zf)`；
   - `T = row_normalize((z1+z2)/2)`；
   - `X^T T = U S V^T`；
   - `P = U V^T`；
   - `zf_aligned = row_normalize(X P)`。
4. `P` 在 CPU float64 中一次性计算、冻结、保存并哈希；训练中没有梯度。
5. 若 `d(zf)>d_view`、SVD 非有限或矩阵秩不足，预检失败；不得静默截断、补零或训练可学习投影。

矩形正交映射满足 `P P^T=I`，因此在数值容差内保留 64 维参考表示的样本间内积，同时把它的坐标轴对齐到 128 维私有视图空间。它不是按 P22 名称写死的分支；触发条件只有维度关系。

## 四、为什么不能只修 Linear 输入宽度

原代码还有第二处同维假设：

`zc = normalize((z1c + z2c + zf) / 3)`

即使把 adapter 的 Linear 改成接收 320 维，最终融合仍然无法把 128、128、64 维直接相加。因此 REV2 必须先得到统一的 `zf_aligned`，并在 adapter 输入、最终融合和 boundary loss 中一致使用它。

## 五、正式训练前的新硬门

下一轮 P0-REV2 不允许再用合成数据代替真实组合。必须生成 30 个历史/可能进入 R2 的真实数据-seed 单元：

- A1 seeds 0-4；
- tonsil seeds 0-4；
- D1 seeds 0-9；
- P22 seeds 0-9。

对每个单元记录真实 `N,d_z1,d_z2,d_zf,d_coords,K`、有序 spot SHA、视图/图/参考表示 SHA、dtype、finite 状态和协调器 SHA。随后对 Q01-Q07 共 210 个组合逐一运行：

- 真实输入加载；
- 真实冻结质量证据；
- 真实 `qcrd_forward`；
- 该候选全部注册 loss；
- backward 和一次临时 optimizer step；
- checkpoint round-trip；
- 输出 shape/finite 检查。

这些只是临时冒烟，不进入科学结果。210/210 全部通过、且每个正式 config 都能精确关联到相同 input/code/contract SHA 的预检行后，才能进入正式训练。

## 六、63 个既有训练单元的处理

不能因为 REV1 整体无效就盲目重跑，也不能未经审计直接当成有效结果。REV2 采用数据集整组复用门：

- A1、tonsil、D1 的协调器必须走 byte-exact identity 分支；
- 复核全部声明 artifact 的 SHA；
- 逐 checkpoint 重载，重放 corrected views，验证各数组和已存在 partition；
- 对同一数据集的 21 个候选-seed 单元实行“全部复用或全部重跑”，禁止按表现挑选；
- 42 个已完成 transform 只有在 partition/输入/config/hash 全部吻合时复用；D1 缺失 transform 从锁定 corrected views 继续；
- P22 21 个单元必须在 REV2 下新训练，因为 REV1 在首次 forward 前失败。

上述决策发生在标签仍完全关闭的情况下，不构成按结果挑选。

## 七、上下文与对话风险

当前规划对话已经发生过自动上下文压缩，说明历史长度很大。虽然关键状态已由文件和摘要保留，但继续把下一次付费运行建立在旧对话隐含上下文上，风险已经不划算。

建议在下一次 AutoDL 开机前：

1. 新建一条规划 Worker 对话，第一条消息读取本目录的当前状态交接文件；
2. 新建一条执行 Codex 对话，第一条消息使用本目录的 REV2 复制提示词；
3. 两条旧对话暂时保留，直到新对话明确回复已完成权威文件定位与 SHA 复核。

无法从应用界面得到精确“还剩多少上下文百分比”，但已经发生压缩是确定证据，足以支持现在换新对话。

