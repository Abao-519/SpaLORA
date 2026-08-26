# Night-19C 报告

## 我现在需要知道的三件事

1. **问题**：Night-17C 的零起步可训练表示核心能否从 P22/MISAR/人海马转移到独立 human placenta，而不是把后续 selector/direct-cut 的失败误算到它头上？
2. **实际动作所在层**：我们保持 `Z01_CONSERVATIVE` 网络、损失、40 steps 与 KMeans endpoint 完全不变；只把 Night-19B 在标签前锁定的 16 个非消融/非置换 placenta 分区按原 ID、等权构成 relation bank。
3. **论文含义**：旧三 lane 的 9 个 lane×seed artifact/checkpoint/replay 重新核验通过，Night-17C 旧的 `LOCAL_SIGNAL` 仍成立；但 placenta 的可训练 full 明显低于 deterministic relation smooth，因此本轮分类为 **SCIENTIFIC_NEGATIVE**，不授权多 seed 或结构 head 集成。

## Placenta seed0 绝对指标（同一 KMeans endpoint）

| arm | ARI | NMI | AMI | FMI | min cluster | changed vs zero |
|---|---:|---:|---:|---:|---:|---:|
| Frozen retained | 0.308612 | 0.484822 | 0.478280 | 0.417099 | 64 | 355 |
| Relation smooth | 0.377102 | 0.554557 | 0.548777 | 0.478716 | 64 | 0 |
| Zero residual | 0.377102 | 0.554557 | 0.548777 | 0.478716 | 64 | 0 |
| Permuted relation | 0.290602 | 0.477734 | 0.471054 | 0.402151 | 63 | 276 |
| **Z01 full** | **0.316241** | **0.505123** | 0.498808 | 0.424255 | 63 | 234 |

Full 相对 coordinate-wise strongest matched control 的差值为 **-0.060861 ARI / -0.049434 NMI**，远低于预注册的 `+0.005/+0.005` 门。没有 singleton/empty cluster，但结构健康不能补偿方法分数失败。

## 旧证据恢复与贡献边界

- Night-17C final compact 25/25 size+SHA 重新通过；当前 frozen core SHA 与 final compact 一致。
- P22、MISAR、人海马的 seeds 0–2 共 9 个 producer artifact、checkpoint、fresh replay 和 metrics authority 均验证通过，未重跑旧科学结果。
- 旧 Z01 common-KMeans mean：P22 `0.480888/0.609897`，MISAR `0.363704/0.541180`，human `0.199789/0.270691`。这是旧的局部表示信号，不是 Night-19C 新确认。
- Placenta relation smooth 相对 frozen retained 提高 **+0.068490/+0.069734**，但它是确定性 control/head signal，不是可训练核心贡献；且低于 Night-19B concatenated-feature control 的 `0.499999/0.631180`。
- Stage B seed0 失败后，严格没有运行 seeds 1–2，也没有进入 Stage C 或新增参数补救。

## 真实 P0 与技术边界

- Shape：N=1662，RNA=30，ATAC=30，retained=30，pair bank=11650，K=10。
- 训练确有非零 gradient/parameter change；zero-start 在 step0 对 smooth byte-exact，zero-gate self-return 由原 frozen core 保证。
- checkpoint strict load、fresh-process learned/permuted representation 与 partition 2/2 exact。
- Producer 只读取 numeric allow-list；标签在所有 partition 写出并哈希之后由独立 evaluator 读取。
- 首次启动因缺少 `PYTHONPATH` 在 import 前失败；未建 bank、未训练、未读标签，修正 launcher 后从头执行冻结路径。

## 导师汇报版

1. 我们先把 Night-17C 的旧证据重新核清，确认它的零起步训练信号没有被后来失败的 selector 或 direct-cut 自动推翻。
2. 新实验完全复用原 Z01，不改网络、损失、步数和阈值。
3. Placenta 的教师来自 Night-19B 标签前锁定的 16 个分区，保留原 ID 并严格等权，没有伪装成旧候选前缀。
4. 工程路径真实完成了训练、参数更新、checkpoint reload 和 fresh-process exact replay。
5. 但 Z01 full 只有 `0.316241/0.505123`，明显低于 relation smooth 的 `0.377102/0.554557`。
6. 因此 Night-17C 的局部信号没有迁移到 placenta，训练 residual 反而破坏了更好的 deterministic smooth carrier。
7. Relation smooth 本身相对 retained 有明显 control signal，但低于 Night-19B concat frontier，不能当作可训练方法成功。
8. 本轮按预注册门停止，不补 seed、不接结构 head，结论为 `SCIENTIFIC_NEGATIVE`。
