# Night-17E report: Learned Relation-Conditioned Cut

## 我现在需要知道的三件事

1. **问题**：我们检验了 Night-17C 学到的表示关系，放进 Night-16H 强表示和强起点的稀疏图割能量后，能否独立改善分区，而不再只当选择器特征。
2. **实际动作所在层**：改动发生在候选生成的 pairwise capacity（边的非负平滑容量）层；动态 prototype unary、强分子表示、起点和 alpha-expansion 求解器保持不变。冻结网格共 3 条 lane × 126 行，全部成功并两进程 exact 重放。
3. **论文含义与分类**：结论为 **SCIENTIFIC_NEGATIVE**。严格整研究留出 0/3 双指标提升、独立控制贡献 0/3；learned edge support 在 P22 与 ZERO 完全同分，在 MISAR 与 ZERO/disabled/uniform 都是同一分区，在人海马被 relation-disabled、uniform 和 permuted 支配。因此它不能成为论文核心机制，且不再扩展多 seed 或 melanoma。

## 绝对指标主表

| dataset | profile | ARI | NMI | ΔARI vs Night-16H | ΔNMI vs Night-16H | min cluster |
|---|---|---:|---:|---:|---:|---:|
| P22_K9 | NIGHT16H_INPUT_AUTHORITY | 0.587533 | 0.708974 | +0.000000 | +0.000000 | 164 |
| P22_K9 | FIXED_GLOBAL_L02 | 0.587619 | 0.709109 | +0.000086 | +0.000135 | 165 |
| P22_K9 | DIRECT_HPO_BALANCED | 0.587903 | 0.709586 | +0.000370 | +0.000612 | 165 |
| P22_K9 | STRICT_LOSO | 0.587409 | 0.708980 | -0.000124 | +0.000006 | 165 |
| MISAR_K7 | NIGHT16H_INPUT_AUTHORITY | 0.534624 | 0.656772 | +0.000000 | +0.000000 | 125 |
| MISAR_K7 | FIXED_GLOBAL_L02 | 0.534624 | 0.656772 | +0.000000 | +0.000000 | 125 |
| MISAR_K7 | DIRECT_HPO_BALANCED | 0.534624 | 0.656772 | +0.000000 | +0.000000 | 125 |
| MISAR_K7 | STRICT_LOSO | 0.534624 | 0.656772 | +0.000000 | +0.000000 | 125 |
| HUMAN_HIPPOCAMPUS_K7 | NIGHT16H_INPUT_AUTHORITY | 0.596178 | 0.585490 | +0.000000 | +0.000000 | 34 |
| HUMAN_HIPPOCAMPUS_K7 | FIXED_GLOBAL_L02 | 0.594310 | 0.584188 | -0.001869 | -0.001301 | 33 |
| HUMAN_HIPPOCAMPUS_K7 | DIRECT_HPO_BALANCED | 0.599975 | 0.591711 | +0.003797 | +0.006222 | 25 |
| HUMAN_HIPPOCAMPUS_K7 | STRICT_LOSO | 0.594073 | 0.583674 | -0.002105 | -0.001815 | 34 |

`DIRECT_HPO_BALANCED` 是公开标签后置评价的开发上限，不是自动输出。P22 的该行（0.587903/0.709586）与同配置 ZERO 分区 byte-exact；人海马的该行（0.599975/0.591711）也与 ZERO byte-exact。它们是可审计的分数变化，但不是 learned relation 的独立贡献。

## 强对照与晋级门

- Strict LOSO：P22 为 0.587409/0.708980（ARI 略降），MISAR 为 0.534624/0.656772（no-op），人海马为 0.594073/0.583674（双降）。
- 控制独立性：P22 learned 与 ZERO partition SHA 相同；MISAR learned 与 disabled/ZERO/uniform 相同；人海马 learned 的 ARI/NMI 同时低于 relation-disabled、uniform 和 permuted。
- Gate：需要至少 2/3 strict LOSO 双升和至少 2/3 不被强对照等价/支配；实际为 0/3 与 0/3。停止规则已触发。

## 工程与真实路径

- P22：N=9196、K=9；MISAR：N=1949、K=7；human hippocampus：N=2500、K=7。
- 三条真实路径均完成 frozen checkpoint strict CUDA replay、三尺度稀疏关系容量、alpha-expansion、artifact reload 和第二进程 126/126 partition/metric exact replay；dense N×N=0。
- ZERO/PERMUTED/UNIFORM 三个控制在每个尺度的 base-weighted mass 误差最大不超过 9.1e-13；learned 与 disabled 的容量确实不同，所以负结论不是“代码没有用到 learned signal”。
- Formal producer 没有读取标签。公开标签只在 partition 锁定后由 evaluator 读取；direct HPO 与两研究训练的 LOSO 配置选择均透明登记。

## 数据与新颖性审计

SpatialGlue、PRAGA 和 spaMGCN 已覆盖多模态图学习、动态/多尺度图和 prototype 机制；普通 learned graph weighting 或 graph cut 不能当原创。Night-17E 的窄组合对象又未通过强对照，故不形成新颖性主张。GSE213264 已纠正为 spatial-CITE-seq 人扁桃体而非 MISAR；目前只闭合到 GC/病理语义，不能制造全域 K-class ARI。

## 失败与限制

- 只有四个机制配置、一个确定性冻结 checkpoint bank；这是有意的可证伪小网格，不证明所有可能的 edge learner 都失败。
- 表示来自 Night-17C transductive public-benchmark development artifacts；本轮不是外部盲测。
- CUDA strict replay只为复现历史 checkpoint 数值；GPU 时间/峰值没有单独 profile，不能填 0。
- GitHub SSH 私钥仍缺失，普通 push 仅尝试一次并失败；bundle 可从 Night-17D parent 恢复唯一 final tag。

## 导师汇报版

Night-17E 把 Night-17C 的 learned 表示关系真正放进了 Night-16H 的图割候选生成能量，而不再只做末端打分。三条 RNA+ATAC 主数据、每条 126 个冻结候选都完整运行并精确重放。工程上 learned capacity 确实改变，三个质量匹配控制也闭合到约 1e-12。科学上严格跨研究留出没有任何一条同时提升 ARI 与 NMI。P22 和人海马的开发小幅提升都能由 ZERO 控制完全复现，MISAR则完全不动。人海马上不使用 learned relation 的 base 能量反而更好。因此本轮归类为科学负结果，停止这条 relation-conditioned cut 路线，不把局部分数包装成方法贡献。外部数据审计还纠正了 GSE213264 的来源错误，并确认其当前更适合 GC 区域生物学验证而不是全域 ARI。

## 技术附录

- Parent commit: `dff4cd44c123c7467134b40464315be6ab170195`
- Science commit: `5165f1f149a8c526063e3f98f67918bc9db3572c`
- Final delivery commit: resolve by peeling annotated tag `night17e-final-20260826`; the exact
  commit and tag-object SHA are recorded after the delivery commit in
  `technical_delivery_manifest.json`.  This avoids embedding a commit SHA in
  the commit that computes that same SHA.
- Git push: failed once with `Permission denied (publickey)`; no force push
- Incremental bundle SHA-256: recorded after the final delivery commit in
  `technical_delivery_manifest.json`.
- Targeted tests: 5 passed
- shutdown_dispatched: false; AutoDL kept on
