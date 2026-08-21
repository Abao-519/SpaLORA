# SpaLORA Night-10A REV2 最终报告

日期：2026-08-21  
分支：`revision/q2-night10a-rev2-real-schema-harmonization-20260821`

## 结论

REV2 的工程目标已经完成，但 QCRD 没有通过最终 frontier。

- 30 个真实 data-seed 单元、Q01–Q07 共 210 行真实 runtime preflight 全部通过；每行都完成真实 forward、全部注册 loss、backward、有限梯度、一次临时 optimizer step、checkpoint round-trip 和 eval forward。
- P22 的真实 `128+128+64` 输入已经统一为冻结的 `64→128` 矩形正交 Procrustes：10 个 P22 seeds 都得到独立冻结并哈希的 `P[64,128]`，秩均为 64，正交性和几何保持门全部通过。adapter、三视图融合及 boundary loss 使用同一个 `zf_aligned`。
- A1、tonsil、D1 的真实维度均为 `64→64`，走原对象、原字节 identity，不做复制、转换、SVD 或归一化。
- REV1 的 A1、tonsil、D1 各自 21/21 整组通过无标签 checkpoint replay，合计复用 63 个训练单元；A1+tonsil 的 42 个 transforms 复用，D1 的 21 个 transforms 在 REV2 补齐。P22 的 R1 21 个单元全部在 REV2 新训练。
- R1 依据预注册规则仅冻结 Q06 进入条件 R2。R2 扩展 18 个新单元，训练与 transform 全部锁定。
- 最终 Q06 只有 D1 相对 Q00 提高；A1、tonsil、P22 均下降，protein family 与 ATAC family 都为负，故 `frontier entry=false`。本轮结果为 `QCRD_NO_FINAL_FRONTIER_ENTRY`，不能包装成协同改进。

## 权威输入与安全边界

四份 REV2 文件及 REV1 compact 均在执行前独立定位并复算：四份给定 SHA 全部一致；REV1 compact index SHA 为 `28140f95f532947d9ecb1f223d4793492daecfcec4f7d41d63a876080c920ef1`，内部 118/118 文件大小与 SHA 全部一致。随后完整读取 REV1 report、false-pass audit、resource audit、QCRD/runner source、全部 REV1 tests 及原 semantic contract。

本轮训练、复用决定和候选冻结前均不读取标签；不访问 MISAR Y、E18.5、新外部数据或第三方 benchmark。最终授权标签读取只有两个锁后窗口：R1 和 R2 各对 A1、tonsil、D1、P22 读取一次，即每个数据集 2 次、总计 8 次；训练标签读取为 0。

## P0-REV2

| 项目 | 结果 |
|---|---:|
| 真实 schema 单元 | 30/30 |
| 真实 runtime rows | 210/210 |
| 唯一 row SHA | 210/210 |
| P22 `128+128+64→128` rows | 70 |
| P22 Q06 坐标 rows | 10 |
| identity manifests | 20 |
| rectangular `64→128` manifests | 10 |
| synthetic substitution | 0 |
| 标签读取 | 0 |
| P0 墙钟 | 474.21 秒 |

P0 在 90 分钟门限内完成。所有 84 个 R1 configs 和 18 个条件 R2 configs 都绑定了匹配的 `preflight_row_sha256`、input manifest SHA、runner SHA、implementation SHA 与 contract SHA。

## 无标签整组复用

| 数据集 | checkpoint replay | 决定 | transform |
|---|---:|---|---|
| A1 | 21/21 | `REUSE_ALL_21` | 21/21 复用 |
| tonsil | 21/21 | `REUSE_ALL_21` | 21/21 复用 |
| D1 | 21/21 | `REUSE_ALL_21` | 21/21 REV2 补做 |
| P22 | REV1 在 optimizer 前失败 | `RETRAIN_ALL_21_REV2_REQUIRED` | 21/21 REV2 新做 |

复用审计在标签 0 时重放 63 个 CUDA checkpoints，并核验 state、config、input、identity reference、corrected arrays、partition 与冻结工件 SHA。旧 REV1 raw artifact manifest 的 473 项再次复算为 473/473；旧 raw 未覆盖、移动或删除。

## R1 与条件 R2

R1 的 84/84 trainable outputs 和 12 个 Q00 references 在标签窗口前总锁。R1 三个预注册角色（accuracy、balanced、spatial）都选中 Q06；其 protein family mean ΔQ 为正且空间保护门通过，因此按冻结规则进入 R2，尽管当时 macro ΔQ 略负。

条件 R2 仅扩展 Q06：A1 2、tonsil 2、D1 7、P22 7，共 18 个新单元。R2 标签窗口前，连同 R1 seeds 共锁定 30 个 Q06 outputs 与 30 个 Q00 references。

最终相对 Q00 的 Q 结果：

| 数据集 | mean ΔQ | 95% bootstrap CI | 单侧 exact sign-flip p |
|---|---:|---:|---:|
| A1 | -0.0007068 | [-0.0025650, 0.0006374] | 0.6875 |
| tonsil | -0.0019826 | [-0.0028095, -0.0011971] | 1.0 |
| D1 | +0.0017996 | [0.0006347, 0.0031477] | 0.0068359375 |
| P22 | -0.0087522 | [-0.0103860, -0.0071128] | 1.0 |

汇总：macro ΔQ `-0.0024105`；protein family mean ΔQ `-0.0002966`；ATAC family mean ΔQ `-0.0087522`；worst family/dataset ΔQ 均为 `-0.0087522`。空间退化保护门通过，但两个 family 都没有正向 Q，因此最终 `entry=false`。

这意味着 QCRD 没有让 A1、D1、P22 协同提高，tonsil 也没有在 Q 指标上保住；只能如实报告 D1 的独立正向结果，不能据此声称整体方法成功。

## 失败与异常保留

- P0 启动前两次实现校正分别是 package module invocation 与坐标 dtype 断言；两次均为 0 runtime rows、0 optimizer steps、0 标签。
- 复用审计先后修正了 identity 数据默认 128 维和 loss-curve 证据路由；成功复用行均在修正后从零重放，未使用结果或标签挑选。
- `/usr/bin/time` 未安装，实际内存探针改用 `/proc` 轮询；未安装工具的尝试没有启动 transform，并已保留。
- R1 的 P22 / seed 2 / Q02 冻结谱聚类产生 8 个非空簇而非 K=9。这是单独的 ordinary numerical failure；其文件、顺序、SHA 与 partition 均锁定，未重跑、改 seed 或补簇。其余 83 个 R1 单元及全部 R2 新单元簇数符合 K。

## 资源与 Git

正式训练使用 NVIDIA GeForce RTX 4080 SUPER。真实 P22 transform 内存探针峰值约 1.01 GiB；4 路投影约 4.03 GiB，随后按最多 4 个 CPU workers、每 worker 3 个 BLAS threads、单 transform 1800 秒 timeout 执行。整轮在 12 小时门限内完成。

分支从 `c7cb3ecd1631db7a21606e62c69eaec05fe31ac1` 创建；baseline tag 指向该父提交。所有 push 均为普通 push，无 force；旧 branch/tag/report/raw 未修改。最终 commit、final tag、远端 peel、compact index 和 Windows 独立 SHA 复算记录由 delivery envelope 给出。

## 保守表述

本轮证明了 REV1 false pass 的真实维度缺口已经被真实 210-row gate 修复，也证明 P22 可以在 REV2 下完整训练与评价；它没有证明 QCRD 在四个核心数据集上具有联合收益。当前最准确的结论是：**工程修复成功，QCRD 最终科学候选失败。**
