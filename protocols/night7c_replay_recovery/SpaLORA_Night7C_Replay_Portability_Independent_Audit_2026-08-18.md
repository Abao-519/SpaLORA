# SpaLORA Night-7C 跨 GPU replay 可移植性独立审计

日期：2026-08-18  
性质：只读失败原因审计与协议修订依据；不是科研结果

## 1. 审计结论

Night-7C 按原任务书终止为 `IMPLEMENTATION_SEMANTICS_INVALID` 是正确执行旧硬门的结果，必须永久保留，不能移动原 tag、改写原报告或把失败尝试包装成成功。

但当前证据不支持“R02 公式或 Night-7C 科学设计已经错误”。更准确的判断是：旧 P1 把跨执行上下文、跨 GPU 型号的 float32 首轮 loss 绝对误差 `<=1e-7` 当作语义必要条件，门槛超出了 PyTorch 官方承诺的可复现范围；同时，Night-7C replay 的操作顺序并没有完全复刻 Night-7B 的原始首轮训练顺序。

因此允许创建一个新的、独立留痕的 replay-portability recovery。它只修订 `m_initial` 的权威来源与 replay 审计方法，不修改任何 routing/weighted-MNN 候选、阈值、seed、预算、selection order、评价门或标签防火墙。

## 2. 本地交付独立核验

- Night-7C compact index SHA：`1797d4f1583d3dea0fedca93aa53ad1ade9417d1b08f0bc79fb03729c472e734`。
- 按 compact index 对 D 盘文件重新计算 size 与 SHA：`38/38` 通过。
- Compact root：`3d40e0a4445dade341c6ae32a5505996cca0a2d6789e457289bc476c321fc992`。
- Incremental bundle SHA：`a0eb375e1796b766498287ea0e2d4c0993d05909276dad00bd5f745aa2be966a`；`git bundle list-heads` 显示 branch head 为 `e34567db5ace4f0fcdd2526cfb94a84fc9148020`。
- P0 权威、输入、checkpoint、历史 GPU 训练证据和标签防火墙均通过；正式训练 0、正式 transform 0、标签访问 0。

## 3. 失败单元与数值规模

`u005` 是 human tonsil、seed 0、`K=4`、4326 spots；此前 `u000`–`u004` 是 A1 的五个 seed。

- Night-7B 锁定 loss curve 首行 MNN：`0.0974991172552108`
- Night-7C 重算：`0.09747149795293808`
- 恢复 RNG 后第二次重算：同为 `0.09747149795293808`
- 绝对误差：`2.7619302272713364e-05`
- 相对误差：约 `0.02833%`
- 旧硬门：`1e-7`

该历史值距注册的 `0.20/0.25/0.30` 冲突阈值分别约 `0.1025/0.1525/0.2025`，所以已观察偏差不会改变 `u005` 的 hard-router 分支。由于旧任务在首个失败处停止，尚不能假设其余 24 个单元也同样稳定，恢复轮必须完成 30/30 无标签诊断后才能继续。

## 4. replay 代码顺序并非历史原顺序

Night-7B 原训练的关键顺序是：

1. `configure_seed`；
2. 构建 graph summaries/reliability；
3. 初始化模型并移到 CUDA；
4. 建立 targets、relation/MNN/mask tensors、optimizer、scheduler；
5. 保存 pre-forward RNG；
6. 执行第一次 training forward 并记录 MNN。

Night-7C P1 的顺序则是：

1. 加载最终 checkpoint，初始化 final model 并执行 final forward；
2. 计算 fixed MNN；
3. 再次 `configure_seed`，初始化 initial model；
4. 恢复保存的 RNG；
5. 执行 initial forward。

恢复 RNG 能控制 dropout 随机流，但不能把 CUDA allocator、cuBLAS 工作区、已执行 kernel 和进程内部库状态恢复到原训练首次 forward 之前。因此旧 P1 的“fresh process”只代表每个 unit 新开进程，并不代表它完整复刻了历史首轮 forward 的执行上下文。A1 与 tonsil 的矩阵形状不同，`u005` 恰好是第一个新形状单元，这使执行顺序/算法选择差异成为必须排查的解释。

## 5. GPU 差异只能作为合理解释，不能冒充已证明原因

Night-7B 记录为 RTX 4080，Night-7C 为 RTX 4080 SUPER。PyTorch 官方 reproducibility 文档明确指出，不同平台、release 或设备之间不保证完全复现；numerical accuracy 文档也说明浮点加法/乘法不满足结合律，跨平台不保证 bitwise-identical。相关官方依据：

- https://docs.pytorch.org/docs/stable/notes/randomness
- https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
- https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html

旧代码使用 `torch.use_deterministic_algorithms(True, warn_only=True)`。该设置提高同环境重复性，但不能承诺跨不同 GPU 型号得到 `1e-7` 以内的聚合 loss；`warn_only=True` 也不是 fail-on-nondeterminism。

当前不能断言偏差一定由 GPU 造成，因为原始大文件仍在已关机服务器的持久盘，尚未运行“历史同顺序、首个 CUDA forward”的隔离探针。恢复轮必须把执行顺序效应与硬件效应分开记录。

## 6. 权威修订决定

### 6.1 `m_initial` 的权威来源

Night-7B 每个 R02 单元的 `loss_curve.csv` 是原训练当时真实计算并已由 manifest SHA 锁定的无标签观测。恢复轮把其首行 `MNN` 设为 routing 的唯一权威 `m_initial`。不得使用 Night-7C invalid partial probe 的重算值，也不得在看到评价指标后选择历史值或重算值。

### 6.2 replay 的新角色

跨 GPU replay 改为语义与数值可移植性诊断，而不是要求重造一个并未保存的历史 initial model state。30/30 单元必须在完全复刻历史顺序的独立进程中重复两次，并通过新 registry 的有界一致性门。任何正式训练和 transform 在该门前都为 0。

### 6.3 为什么这不是事后放水

- 旧 invalid 结果和 tag 保持不动。
- 还没有任何 Night-7C 正式科学结果、标签或候选胜负可供迎合。
- 候选、阈值、seed、预算、评价与保护门完全不变。
- 最终 routing 固定使用实验前已经存在并哈希锁定的 Night-7B 原始 loss，而不是选择更有利的新值。

## 7. 对项目的通俗解释

这次不是“尝试失败后把规则改松继续跑”，而是发现旧规则要求在另一张显卡上把一个 float32 首轮 loss 复刻到小数点后七位，这并不是 PyTorch 保证的能力。修复方案也不是忽略差异，而是使用原实验真正记录、已经锁死的无标签数值，同时增加更严格的执行顺序诊断。只有确认没有公式、输入、索引或初始化语义错误后，才恢复尚未开始的 Night-7C 科学实验。
