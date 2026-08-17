# SpaLORA Night-6B 独立审计与 Night-6C 决策

日期：2026-08-17  
审计角色：规划 Worker，只读本地证据审计  
Night-6B 权威终态：`IMPLEMENTATION_SEMANTICS_INVALID`  
下一步权威决策：`PROCEED_TO_NIGHT6C_WITH_CLEAN_PAIRED_BASELINE_RETRAINING`

## 1. 审计结论

Night-6B 的停止是正确的。任务书要求从 Night-5 C04/B01 的历史 final checkpoint 做 A1 五个 seed 的只读 forward replay，并据此派生此前未保存的 modality-private views；但该 checkpoint 从未被 Night-5A runner 保存。SHA 摘要、fused embedding 或旧 clusters 都不是可加载模型状态，也不能逆推出权重。

这不是候选表现不佳，也不是 Night-6B 图/聚类假设被否定。本轮正式训练、head transform 和科研评价均为零，因此没有有效正结论或负结论。

## 2. 独立核验结果

本地交付根：

`D:\文档\ChatGPT\博士第一篇科研论文项目\night6b_handoff_20260817\official_compact`

- `handoff/delivery_index.json` SHA-256：`883824f88b26ef10aa53297178859e123f25ef529c82537f82050b68819793f7`
- 索引内 31/31 文件存在、size 匹配、SHA-256 匹配。
- 报告 SHA-256：`06de92eaf16aa7d2fc9fdfc1bd127451a6db0d22d03f5805ba1a9181fc30ed02`。
- checkpoint 审计 SHA-256：`72a5ed9db0f67fb0119ce78d8048c71555d1b7ba12c72bbdec79013344e2a276`。
- ontology contract SHA-256：`9b0473b22807a0408a04759c1b1ba664c1d15f77433069b1d8725300665a9e66`。
- P0 semantic contract SHA-256：`108cbeb0a493eb9c0e122daeb3b3e11e848db1ebb8ab0798bcd533b1671f0088`。
- Night-6B commit、远端 branch 与 final tag 均为 `301f49be2ddd15d2d36823c194df3549874729a4`；记录显示未 force、final tag 只创建一次。

### 2.1 checkpoint 缺失证据

- Night-5 C04/B01 seeds 0–4 共 30 个已声明 artifact 全部重新哈希且 0 mismatch。
- 每个 seed 只有 `embedding.npz`、`attention.npz`、`clusters.csv`、`observation_ids.csv`、`loss_trajectory.csv`、`coefficient_probe.json` 及 manifest。
- 五个 seed 均没有 `.pt`、`.pth`、`.ckpt` 或等价 model-state 文件。
- Night-5A runner 的 required-artifact 列表不含 model state，代码中也没有对应的 `torch.save`。
- D 盘 Night-5A loose files 和五个相关归档均未发现 C04/B01 checkpoint。
- Night-3AF checkpoint 语义不同；Night-6A 权重属于无效实验；二者均不得替代。

因此 `valid_checkpoint_available=false`、`forward_replay_executed=false`、`private_views_derived=false` 是证据支持的唯一结论。

### 2.2 可继承的有效前置证据

Night-6B 的下列证据在进入科研训练前形成，且不依赖缺失 checkpoint，可在逐字节复核后继承：

- tonsil section 1 的目标列在读取前已锁为 `final_annot`，真实 known K 为 4；四类计数为 731、183、834、2578。
- RNA/ADT 的 `final_annot` 逐 spot 完全一致，标签向量 SHA-256 为 `a587e8029c83d002f96b72e40bb2fe4ea46867f9a655d01c963d94dd18897c9a`。
- 新建 RNA/ADT label-free H5AD 均为零 `obs` 列，配对 barcode 完全一致。
- label-free RNA SHA-256：`a53bb5a1d5fcea356db65f8de9f8f6c158d1b96db8dfbef0b85d6d4d45131a0a`。
- label-free ADT SHA-256：`0146aabbd9845b90932c68711c985832895f93082a25a6adabdeccd9795f0ce5`。
- 防火墙负向测试 9/9 通过；trainer、transformer、evaluator 均未启动。
- D1、P22、GSE198353、Night-4B、Night-5D 指标内容和 Night-6A raw runs 均未访问。

Night-6C 必须只读复核这些文件和 SHA，不得在训练前再次打开原始 tonsil annotation；若持久盘副本 SHA 不一致，才按阻塞终态停下，而不是静默重建。

## 3. 根因与责任边界

根因是上一份规划任务书把“历史 private-view parity”设为硬门，却没有先证明历史 runner 保存过可加载权重。这是规划合约错误，不应归因于实验执行者，也不能通过伪造、替代或把 embedding 当 checkpoint 来绕过。

Night-6B 的 `IMPLEMENTATION_SEMANTICS_INVALID` 只描述该任务书无法合法执行；它不使已通过的 ontology、label-free 副本、访问日志和 Git 证据失效。

## 4. Night-6C 的唯一授权路线

Night-6C 明确授权一次全新的 clean baseline retraining：

1. 在 A1 与 tonsil 上分别训练 `G00/H00` seeds 0–4；全部计为新的科学训练，不称 replay，不称 retry。
2. C04/B01 encoder 的结构、loss、优化器、epoch、学习率、attention shrink 和 preprocessing 必须从 Night-5 权威代码、注册表与 manifests 重建并锁定；除 dataset、graph candidate 和 seed 外不得改变。
3. Night-6C 所有 graph/head 候选只与同一 Night-6C、同 dataset、同 seed 的 fresh `G00/H00` 比较。
4. Night-5 C04 指标只做历史漂移诊断，不作为 hard parity、候选选择或成败门。
5. 每个正式训练单元必须保存真实 final model state、完整配置、input/cache/code SHA，并在独立新进程中 reload-forward；未通过 round-trip 的单元无效。
6. 原 Night-6B 9 个 graph 和 12 个 head 的候选定义、选择阈值、seeds、标签防火墙与保护数据集保持不变。

由此，最大科学训练从 61 调整为 66：R1 36，R2 最多 30。实现/基础设施纠正最多 12，总训练尝试最多 78。head transforms 仍最多 552，transform corrections 仍最多 48。

## 5. 科学解释边界

- fresh G00 是 Night-6C 内部公平参照，不自动等于 Night-5 历史结果。
- 若 Night-6C G00 与 Night-5 数值存在差异，只能报告为环境/实现漂移诊断；只要预锁定语义、输入和 round-trip 均通过，不得据此挑 seed、调参或重跑。
- Night-6C 若锁定候选，仍只是 A1+tonsil 开发证据；D1/P22 只有下一轮才能做封存确认。
- 若没有候选通过原阈值，应如实终止 `NO_GRAPH_OR_CLUSTER_RESCUE_CANDIDATE`，不得临时放宽门槛。

## 6. 操作边界

本次规划审计未连接或查询 AutoDL，未修改任何历史交付，未创建替代 checkpoint。用户在粘贴 Night-6C 启动提示前手动确认实例关机/开机；规划 Worker 不自动唤起另一个 Codex 对话。
