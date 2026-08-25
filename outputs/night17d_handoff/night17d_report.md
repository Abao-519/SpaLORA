# Night-17D 报告：训练表示证据选择器未通过独立贡献门

## 我现在需要知道的三件事

1. **问题**：Night-17C 的训练表示虽然能改善自己的 KMeans 表示端点，但能否作为新证据，从 Night-16H 已锁定的 89 个结构可行候选中选出更好的分区？
2. **实际动作 / 流水线层级**：本轮没有改候选、图或训练网络；只在“最终候选选择层”计算三个冻结训练 seed 的分子分离、centroid margin、relation alignment 和 seed 不确定度。relation alignment 对参与 UNBIASED bank 的候选做了 leave-one-candidate-out，避免候选给自己加分。
3. **论文意义**：严格整研究留出只有 P22 双升，MISAR 轻微下降，人海马持平；更关键的是 learned 与 ZERO 在 3/3 研究选择完全相同。因此训练表示没有形成独立 selector 贡献，结论是 **SCIENTIFIC_NEGATIVE**，应停止继续修补这一选择器家族。

## 绝对结果

| 数据 | Night-16H ARI/NMI | 固定全局 learned | 严格 LOSO learned | ΔARI/ΔNMI | learned=ZERO |
|---|---:|---:|---:|---:|---|
| P22 K9 | 0.587533/0.708974 | 0.482738/0.626282 | 0.596390/0.718243 | +0.008858/+0.009270 | 是 |
| MISAR K7 | 0.534624/0.656772 | 0.359643/0.542269 | 0.532956/0.655982 | -0.001667/-0.000790 | 是 |
| Human hippocampus K7 | 0.596178/0.585490 | 0.135535/0.237378 | 0.596178/0.585490 | +0.000000/+0.000000 | 是 |

三折均拟合到 `M0_T1_L0.5_U0`。P22 的数值是候选库内已有可达解；ZERO selector 也选中同一 partition，故不能归因于 Night-17C learned representation。

## 证据边界与限制

- 训练表示是基于 UNBIASED candidate ensemble 的 transductive distillation，不是独立外部教师；候选级 alignment 已扣除自身权重。
- 固定全局配置表现弱，尤其在人海马选择到低分候选；这是原样保留的预注册负结果。
- 公开 labels 只在 feature CSV 两次重放并锁 SHA 后进入 direct HPO / LOSO evaluator。直接 HPO 是开发上限，不是迁移证据。
- strict LOSO 拟合只打开两个训练 study 的 evaluation 与单行 authority；held-out selection 先锁定，再由独立 evaluator 打开 held-out 指标。
- 未达到 2/3 晋级门，因此按合约没有扩展 melanoma，也没有扩大 grid 或追加 selector 规则。

## 导师汇报版

1. 我们检验了 Night-17C 的训练表示能否帮助 Night-16H 从原 89 个候选中选得更好。
2. 新证据包含表示内分离、原型 margin、关系一致性和三 seed 稳定性，候选自身贡献已用 leave-one-out 扣除。
3. 严格整研究留出在 P22 提高到 0.5964/0.7182，但 MISAR 略降，人海马持平。
4. 最关键的是，learned 与 zero-residual 对照在三条数据上都选到完全相同的 partition。
5. 因此 P22 的数值来自原候选库和拓扑排序，不是训练表示的独立增益。
6. 本轮按预注册门判为 SCIENTIFIC_NEGATIVE，不再扩大 selector 网格或补规则。
7. 下一步应转向新的高质量 RNA+ATAC 外部数据和更强的原生 backbone，而不是继续修补候选选择器。

## 技术附录摘要

- 三条 lane 的 learned/zero/permuted checkpoint authority 均通过；最终 feature CSV 两次 fresh-process SHA 完全一致。
- targeted tests：6/6 PASS；dense N×N：0；新下载：0；新环境：0；melanoma label reads：0。
- AutoDL 按连续自主研发要求保持开机，`shutdown_dispatched=false`。
