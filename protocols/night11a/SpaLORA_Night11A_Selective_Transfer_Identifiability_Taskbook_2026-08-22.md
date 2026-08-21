# SpaLORA Night-11A 选择性跨模态传递无标签可识别性任务书

日期：2026-08-22  
任务类型：两种真实模态家族、各一个 seed、无标签机制 go/no-go  
不是：正式聚类科学比较、候选大搜索、新数据或第三方 benchmark

## 0. 本轮结果边界

Night-11A 想回答的唯一问题是：**在每个空间点上，无标签留出证据能否预先判断“加入另一模态更好”还是“保持本模态更安全”。**

实现位置是 encoder 之后、融合之前。输入是与真实 observation 顺序绑定的 G00/G04 模态私有表示；输出是本模态预测、跨模态预测、门值和选择性更新表示，最后只做固定聚类端点的工程 round-trip。

即使全部通过，科学分类也只能是 `LOCAL SIGNAL`。它不证明 ARI/NMI 改善，不证明跨真实数据泛化，不是 `CONFIRMED MILESTONE`，更不是 `PAPER-READY EVIDENCE`。

## 1. 启动前确认

执行 Codex 必须先在 Windows 完整读取并复算同目录下：

1. 全局路线图；
2. `night11a_selective_transfer_identifiability_contract.json`；
3. 本任务书；
4. `README_COPY_TO_CODEX2_NIGHT11A_2026-08-22.md`；
5. planning delivery index 中的全部路径、size 和 SHA-256。

再独立复算 Night-10B compact：54/54、missing/size/SHA mismatch 均为 0，index SHA 必须为 `696b9d99720e2168bd8930cdf4c0b6d2071cd71263df8960d89c648bb2e89910`，bundle SHA 必须为 `c31888546aaad21d2be912ef9646ee3819212caad90d4605c5b387c1a3f0b15a`。

任何文件缺失、JSON 不能解析、hash 不匹配或 parent 不唯一，立即停在 `NIGHT11A_AUTHORITY_OR_ARTIFACT_BLOCKED`，不得连接 AutoDL。

本地全部通过后，只用一段简短回复说明 `planning N/N`、Night-10B `54/54` 和准备连接；不要逐文件复述。

## 2. 硬范围

正式单元只有：

| unit | 数据 | seed | 模态家族 | 作用 |
|---|---|---:|---|---|
| u000 | A1 | 0 | RNA+protein | 蛋白家族真实路径 |
| u020 | P22 | 0 | RNA+ATAC | ATAC 家族真实路径 |

固定扰动复本 seeds：`17, 29, 43`。

固定条件：

- `CLEAN_HOLDOUT`：只有特征留出，没有局部破坏；
- `LOCAL_TARGET_DAMAGE`：局部损伤被预测的本模态可见信息；
- `LOCAL_AUXILIARY_CONFLICT`：局部打乱另一模态的 spot 配对，同时保持其边际分布。

固定四条路径：

- `B0_SELF_ONLY`：只用本模态，不接受跨模态消息；
- `B1_ALWAYS_TRANSFER`：所有 spot 都完全接受跨模态消息；
- `B2_UNCERTAINTY_ONLY`：只按不稳定程度衰减，不比较传递前后的有符号效用；
- `B3_SELECTIVE_NULL`：本轮提出的路径，用交叉拟合效用减去不确定性，并允许精确回到零传递。

禁止新增近邻候选、候选编号、温度、alpha、fold、patch fraction 或 endpoint。禁止根据初步数值挑一个“看起来更好的”版本。

## 3. 标签与范围防火墙

全轮必须为 0：

- 训练、选择和评价标签读取；
- ARI、NMI、Q、AMI、FMI 及依赖 annotation 的空间指标；
- MISAR Y、E18.5、新外部数据、第三方 benchmark；
- QCRD 训练、恢复和评价；
- dataset-name routing；
- dense N×N 构造；
- scientific retry/fallback。

人工扰动区域和干净表示是本轮无标签机制真值，不是生物 annotation。报告中必须写清：这是“可识别性试验”，不能替代真实聚类评价。

## 4. P0-REMOTE：独立工作区和父边界

只有用户已手动以有卡模式开机后才连接。不要调用 AutoDL 电源 API。

1. 在远端定位含 parent commit `10f16eab93d9dbc4c33c3aaff2b84ee782694a80` 的权威仓库；
2. 验证 `night10b-final-20260821^{}` 精确 peel 到 parent；
3. 从 parent 建立 `/root/autodl-fs/SpaLORA-night11a`；
4. 创建 branch `revision/q2-night11a-selective-transfer-identifiability-20260822`；
5. protection tag `baseline/pre-night11a-selective-transfer-identifiability-20260822` 只能指向 parent；已正确存在则记录，不移动；错误存在则停止；
6. raw root 固定为 `/root/autodl-fs/night11a_selective_transfer_identifiability_20260822`；
7. 历史 raw、branch、tag、report 和 compact 全部只读；普通 push，禁止 force。

## 5. P0-SOURCE：先闭合真实语义

执行 Codex 必须亲自读当前 parent 中的真实实现，不得只读 report 或 README：

- Night-6C/6D：G00/G04 view 产生、G04 forward、H05 endpoint；
- Night-7B：六视图、R02 projector/adapter、C06/H01；
- Night-10A REV2：真实输入 manifest、dimension/reuse contract、u000/u020 observation order；
- Night-10B：family recipe、runtime、两条 full smoke 和 round-trip；
- 所有将被复用的 checkpoint/view/graph loader 和标签边界。

生成 `outputs/night11a_handoff/p0_source_and_shape_audit.json`，至少包括：

1. 每个真实函数、配置、checkpoint、view 和 graph 的 repo-relative 或 absolute path、size、SHA；
2. u000、u020 的 observation 数及顺序 SHA；
3. 每个 G00/G04 `modality1/modality2/fused` tensor 的实际 `N x D`、dtype、finite、语义；
4. 原始输入到 preprocessing、真实 forward、checkpoint reload、fusion、fixed endpoint 的调用链；
5. C00/F00 是复用哪个锁定 artifact，而不是重新构造；
6. pilot 只使用 modality-private view，不把已经融合过另一模态的表示伪装成本模态证据；
7. 固定 `H05_EQUAL3_AFFINITY_SPECTRAL` 是否保持 sparse，是否出现 dense N×N；
8. 标签模块是否从本轮 worker 进程不可达。

若 G00/G04 的真实 modality-private 表示、ordered IDs 或 endpoint 不能在两个家族闭合，不猜公式或维度，立即使用 `NIGHT11A_IMPLEMENTATION_SEMANTICS_INVALID`。不得用“真实 endpoint 单独通过 + 合成 adapter 单独通过”代替真实混合路径。

治理文件中“Night-10A 只有 REV1 实现失败”的旧句已被 REV2 结果更新；本轮以合约锁定解释为准：REV2 工程修复成功，但 QCRD 科学候选总体失败。

## 6. P1：最小实现

建议职责边界，可按现有仓库命名微调：

```text
SpaLORA/selective_transfer.py                # 交叉拟合 predictor、utility、uncertainty、null gate
SpaLORA/night11a_corruption.py               # 稀疏连通 patch、target damage、auxiliary derangement
scripts/night11a/run_identifiability.py      # 单一正式入口
configs/night11a/*.json                      # 冻结四路径与统一参数
tests/night11a/*                             # 最小语义与防火墙测试
```

### 6.1 交叉拟合

- spot 使用两个由 `unit_id + ordered observation ID` 哈希决定的折；
- 每个方向只以接收模态的干净 G04 modality-private 表示为 prediction target；feature folds 只在这个真实 target space 中定义，不假设 G00/G04 或不同家族维度相同；
- feature 使用五个由 `unit_id + direction + canonical feature index` 哈希决定的折；evaluation fold 为 `e` 时，evidence folds 固定为 `(e+1)%5`、`(e+2)%5`，剩余两个 target folds 才能作为 predictor input；
- 对某个 held-out spot 的标准化、ridge 拟合和温度估计只能来自另一 spot fold；
- self 输入为接收模态 G00 加两个可见 G04 target folds；两个 transfer predictors 保持完全相同的 self 输入、训练行和固定 `alpha=1.0`，分别只增加另一模态 G00 或 G04；二者预测等权平均形成唯一 transfer proposal，不按性能选择 graph view；
- gate utility 只来自两个 evidence folds，uncertainty 是两个 evidence folds × 两个 auxiliary graph views 共四个 utility 的 median absolute deviation；
- 不做 alpha、维度、阈值、fold 或 solver 搜索。

这里 ridge 是带固定 L2 正则的线性预测器，只是便宜的可识别性探针，不是论文最终 decoder，也不能被包装成原创模块。

### 6.2 门控

- `u = L_self_evidence - L_transfer_evidence`；
- `s` 是 G00/G04 和已注册 evidence folds 上效用的稳健离散程度；
- `T=max(training-fold median(abs(u)), 1e-6)`；
- `g=sigmoid((u-s)/T)`；
- `B3 = self + g*(transfer-self)`；
- 强制 `g=0` 时，B3 在 prediction 和重建表示边界必须与 B0 byte-exact。

`B2_UNCERTAINTY_ONLY` 使用合约固定公式，只是检验“有符号效用”是否比普通不确定性衰减多提供信息。

### 6.3 人工局部条件

- 先复制并哈希 clean held-out target；
- patch 使用注册稀疏空间图，从哈希 anchor 按 observation-ID 稳定 tie-break 做 BFS，取 `ceil(0.2*N)`；
- target damage 分别在接收模态 G00 和两个可见 G04 folds 内按各自 feature hash 替换 30% 坐标为 training-fold median；干净 held-out G04 target 与另一模态不变；
- auxiliary conflict 在 patch 内对另一模态 G00/G04 使用同一个无 fixed point row permutation，并拒绝仍为直接空间邻居的配对；
- patch、mask、permutation 在运行四个 arm 前一次冻结，四条路径必须共用完全相同的输入。

## 7. P1-TEST：只保留能挡住错误的测试

正式 smoke 前必须通过：

1. 两个真实 family 的 shape/observation contract；
2. spot/feature folds、patch 和 permutation 在 fresh process byte-exact；
3. evaluation feature 不进入自身 gate evidence；held-out spot 不进入自身 predictor 拟合；
4. self/transfer 训练行与标准化完全相同；
5. `g=0` 精确 identity，`g=1` 精确 always-transfer，gate 有界且 finite；
6. auxiliary permutation 无 fixed point、保持 cardinality/feature marginals、没有直接空间邻居配对；
7. source AST 和 runtime 无 dataset/tissue/species/stage/path/label/metric 路由；
8. worker 不 import annotation/evaluator；
9. sparse graph 守卫拒绝 dense N×N；
10. checkpoint 和 manifest 原子写，失败不会生成 PASS；
11. 两家族真实 preprocessing→forward→reload→pilot→fusion→endpoint 各 1 条；
12. 独立汇总脚本不复用执行循环中的内存对象。

不要求为每个 getter 或格式化函数写低价值测试；测试目标是语义、防泄漏、可复现和真实边界。

## 8. P2：两家族无标签 smoke 与正式冻结

先各运行一个固定最小 smoke：

- u000：replicate seed 17，`LOCAL_TARGET_DAMAGE`；
- u020：replicate seed 17，`LOCAL_AUXILIARY_CONFLICT`。

每条 smoke 必须从注册输入开始，经过真实视图 loader/forward、pilot、四 arm、选择性表示、checkpoint reload 和固定 `H05_EQUAL3_AFFINITY_SPECTRAL` endpoint。每个 arm 都把更新后的 modality-1、modality-2 和二者等权 fused 表示作为 H05 的三个输入；K 只读取已注册 input manifest 中的冻结值，不读标签、不重新推断。数值 round-trip 为 `atol=1e-6, rtol=1e-5`，partition canonical exact。

smoke 只检查链路，不能据其数值修改公式。两条通过后冻结并哈希：代码、config、实际 shape 表、ordered IDs、fold、patch、mask、permutation、四 arm 和 go/no-go 门。冻结后才进入正式执行。

## 9. P3：正式无标签 identifiability

执行矩阵固定为：2 units × 3 conditions × 3 replicate seeds × 2 transfer directions × 4 arms。可以共享 predictor 和同一输入上的中间量，但 manifest 必须能恢复每一行，不能少跑或用报告数字补行。

每行保存：

- unit、family、direction、condition、replicate；
- input/view/ordered-ID/fold/patch/mask/permutation/implementation/config SHA；
- B0–B3 的 held-out MSE；
- evaluation-only oracle utility；
- frozen gate score、gate、uncertainty；
- regret、有效/模糊 spot 数；
- runtime、CPU、RSS、GPU；
- checkpoint/endpoint fresh-process round-trip；
- label/metric/read counters。

独立汇总必须报告：

| family/unit | 实际 view shapes | condition | arm | held-out MSE | AUROC/AUPRC | Spearman | gate separation | regret | runtime |
|---|---|---|---|---:|---:|---:|---:|---:|---:|

Night-11A 不得生成 ARI/NMI 主表，因为标签始终关闭。

## 10. 唯一 go/no-go 门

严格使用 JSON 合约中的全部门，包括：

- pooled AUROC 至少 0.75、每家族至少 0.70；
- pooled Spearman 至少 0.35、每家族至少 0.25；
- 每家族正、负非模糊 spot-events 各至少 100；
- gate separation 每家族至少 0.25，且不塌缩；
- clean 条件相对 self-only 退化不超过 2%；
- target damage 中 B3 至少比 B0/B1 中更好的一个再好 1%；
- auxiliary conflict 中 B3 不比 B0 差 1%，并至少比 B1 好 10%；
- B3 pooled normalized regret 至少比 B2 低 10%；
- 2/2 checkpoint 和 2/2 endpoint round-trip；
- 总墙钟不超过 4 小时，peak GPU 不超过 8192 MiB。

所有门同时通过才允许：

`NIGHT11A_SELECTIVE_TRANSFER_IDENTIFIABILITY_PASS`

实现正确但任一识别/效用门失败：

`NIGHT11A_NO_IDENTIFIABLE_TRANSFER_UTILITY`

这属于当前新机制的 `SCIENTIFIC NEGATIVE`（机制负结果），必须保留，不得调 threshold、重挑 patch 或派生相邻候选。

## 11. 正式修正规则

正式 manifest 开始后最多一个全局 correction cycle，且只允许修复能在不看正式结果方向的情况下定位的全局实现错误。

修正时必须：

1. 保存并标记整个旧 attempt invalid；
2. 说明 root cause；
3. 重新冻结实现 SHA；
4. 从两个 family、三个条件、三个 replicate、两个方向、四 arm 全部重跑；
5. 禁止只补失败 family、condition、direction、spot 或 row。

普通数值失败原样锁定。资源或外部环境阻塞使用 `NIGHT11A_INFRASTRUCTURE_BLOCKED`，不能改成科学结论。

## 12. 报告、Git 与 compact

最终报告先写“负责人现在需要知道的三件事”，再给状态分类、主结果表和 5–8 句导师汇报版。hash、commit、tag 和清单放技术审计附录。

Git：

- 小而有意义的 commits，普通 push；
- final tag 只在最终报告、decision、tracked index 都进入 final commit 后创建；
- remote branch 和 tag peel 必须等于 final commit；
- 从 Night-10B parent 生成增量 bundle 并验证 prerequisite/tag。

Windows compact 到：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night11a_delivery_20260822/official_compact`

compact 只含代码、配置、测试、JSON/CSV/MD/log、审计、索引和增量 bundle；不下载 raw、checkpoint、embedding、affinity、`.npy/.npz/.pt/.pth/.ckpt/.h5/.h5ad` 等大工件。

Windows 独立复算 root-relative path、size、SHA-256；任何 missing/size/SHA mismatch 都不能报告完成。

## 13. 关机与汇报节制

Codex 2 只在四个节点向用户更新：P0 完成、两家族 smoke 完成、formal 终态、Windows delivery 完成；遇到失败立即报告。不要逐条命令直播，也不要重复抄写大段合约。

完成 final commit/tag/peel、bundle、compact 下载和 Windows 独立复算后，最后一条远端命令派发 `/usr/bin/shutdown`，随后不重连。只声称“关机命令已派发”。

最终回复必须给：终态、为什么是该类别、关键 identifiability 数字、2/2 round-trip、是否发生 correction、全部禁区计数、资源、final commit/tag、compact N/N、index SHA、bundle SHA、关机命令派发状态。不得把 Night-11A pass 写成聚类性能提高或论文已成立。
