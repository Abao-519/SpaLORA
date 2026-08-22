# Night-11B RNA–蛋白不一致可识别性 P0 报告

## 负责人现在需要知道的三件事

1. 本轮只判断 RNA–蛋白不一致能否被拆成 RNA 可解释的共享部分、稳定且有空间结构的蛋白特异残差、不可复现技术噪声三个可观测对象。
2. 实际工作停在 feature/residual/fusion 之前的证据层：从真实 RNA/ADT feature matrix 出发，做空间块交叉拟合、残差重采样稳定性、稀疏图空间超额和固定 controls；没有训练完整融合模型。
3. 冻结终态是 `NIGHT11B_EVIDENCE_AXES_NOT_IDENTIFIABLE`（`SCIENTIFIC NEGATIVE`）。这只回答 P0 可识别性，不能解释成聚类提高、SOTA 或论文已经成立。

## 可识别性主表

| 数据集 | 真实输入 shape | 29 对 | shared OOF R² 中位数 | bootstrap stability 中位数 | spatial excess 中位数 | control AUROC | 运行秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| a1 | 3484x18085 RNA; 3484x31 ADT | 29 | 0.0944 | 0.8438 | 0.1441 | 0.8750 | 14.0 |
| tonsil | 4326x18085 RNA; 4326x31 ADT | 29 | 0.1651 | 0.8837 | 0.2437 | 0.8793 | 18.0 |
| d1 | 3359x18085 RNA; 3359x31 ADT | 29 | 0.0749 | 0.8838 | 0.2506 | 0.8750 | 12.9 |

## 冻结判定

- pooled control AUROC = 0.8764，95% bootstrap CI [0.8549, 0.8980]。
- combined 相对 boot 单轴 AUROC 差 = 0.0244，95% CI [0.0129, 0.0359]。
- combined 相对 space 单轴 AUROC 差 = 0.0029，95% CI [-0.0000, 0.0072]。
- A1↔D1 同 target evidence-rank Spearman = 0.0000，预注册 permutation p = 1.0000。
- 真实端到端 smoke 与 fresh-process reload：3/3。formal correction cycle：1。scientific retry：0。

## 导师汇报版

Night-11B 没有训练一个新模型，而是先检查 RNA–蛋白不一致是否有可观测的三分证据。
三个真实组织单元都从 feature-level RNA/ADT 和登记的稀疏空间图出发，并只用了 29 个完全同名的 deposited feature pairs。
共享部分由空间块外推的 ridge prediction 定义，蛋白特异部分是其残差。
残差证据由重采样稳定性和相对空间 permutation null 的超额共同构成，shared predictability 单独报告。
known controls 与 A1–D1 现实复现门严格按 formal 前冻结的阈值判定。
最终分类见上文，任何 synthetic control 结果都没有被包装成真实生物学成功。
本轮未读取标签，未计算 ARI/NMI/Q，也没有证明 SOTA 或聚类性能提高。

## 技术附录

- wall time：53.1 秒；peak RSS：483.2 MiB；GPU attributed peak：0 MiB。
- fixed formula：`sqrt(p_boot * p_space)`；32 bootstraps；199 permutations。
- 标签与全部禁区计数见 `label_firewall_audit.json`，均为 0。
