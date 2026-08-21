# 复制到全新执行 Codex：Night-10B

你是 SpaLORA Night-10B 的全新执行 Codex。你的任务是完成**冻结模态家族策略整合与复现封口**，不是继续 QCRD、不是搜索新候选、不是重新评价科学分数。

先在 Windows 本地完整读取并独立复算以下三个文件：

1. `D:/文档/ChatGPT/博士第一篇科研论文项目/night10b_family_policy_integration_planning_20260821/SpaLORA_Post_Night10A_REV2_Research_Decision_2026-08-21.md`  
   SHA-256：`bae8a2e1a47b36a818d013df8b5782d6182db63bd657eb9f333231cc6f5305c2`
2. `D:/文档/ChatGPT/博士第一篇科研论文项目/night10b_family_policy_integration_planning_20260821/night10b_frozen_family_policy_integration_contract.json`  
   SHA-256：`9f0582032aa07e895343f8d32d2561faa913d2c047b9bb4b703b8f325e86d5e7`
3. `D:/文档/ChatGPT/博士第一篇科研论文项目/night10b_family_policy_integration_planning_20260821/SpaLORA_Night10B_Frozen_Family_Policy_Integration_Taskbook_2026-08-21.md`  
   SHA-256：`123ac0fd7c7d04e39912979295291b2602ea5b2fd7ccd4b06e26bb9313525ec8`

再读取同目录 planning delivery index，验证其中列出的全部文件、size 和 SHA。任一文件缺失、JSON 不能解析或 hash 不匹配，立即停止告知用户；不要连接 AutoDL。

随后亲自复核合约 `authority_files`，尤其是：

- Night-10A REV2 compact index 必须为 182/182，SHA 为 `d2b1a1c6baf469aede5b7042b2133b6d065b6ffe0f30919682a952b08decc804`；
- parent commit/tag 必须为 `8b83fe38f3fbd4d3cb24fef6512596be2d49408b` / `night10a-rev2-final-20260821`；
- 30 个 Q00 references 必须是 A1 5、tonsil 5、D1 10、P22 10，不能少、不能替换 seed；
- Night-10A 的终态是 QCRD 工程修复成功、科学候选失败，禁止重新解释。

完成本地核验后先明确回复：

> 已完成 Night-10B 规划文件定位、planning index 复核、REV2 182/182 复核和 30-row Q00 authority 审查；等待或确认 AutoDL 已由用户以有卡模式启动后再连接。

只有用户已手动以 GPU 有卡模式开机，才可连接远端。不要调用 AutoDL 电源 API，不要自动派发另一个 Codex 或子代理。

远端从 parent 新建独立工作区和分支：

- repo：`/root/autodl-fs/SpaLORA-night10b`
- branch：`revision/q2-night10b-frozen-family-policy-integration-20260821`
- protection tag：`baseline/pre-night10b-frozen-family-policy-integration-20260821`
- raw root：`/root/autodl-fs/night10b_family_policy_integration_20260821`
- final tag：`night10b-final-20260821`

不得覆盖、移动或修改任何历史 raw root、branch、tag、report。所有 push 必须普通 push，禁止 force。

严格按 Taskbook 顺序执行：

1. P0-REMOTE：父 commit/tag、独立工作区、保护 tag；
2. P0-SOURCE：亲自读取 G04/H05、full F00/R02/C06/H01、Night-8B identity-blind policy 和 Night-10A 30-row Q00 源代码/配置/工件；
3. P1：建立单一 family-policy API 和 CLI，路由只能由显式 `RNA+PROTEIN` 或 `RNA+ATAC` 决定；
4. P1-TEST：AST、运行时身份盲、config SHA、稀疏边界和 fail-closed 测试；
5. P2：对全部 30 个 Q00 references 做零标签正式精确重放；
6. P3：仅执行 A1 seed 0 的 C00 full smoke 与 P22 seed 0 的 full F00 smoke；
7. P4：独立审计、普通 push、final commit/tag、增量 bundle、compact、Windows SHA 复算；
8. `/usr/bin/shutdown` 作为最后一条远端命令，派发后不重连。

本轮硬边界：

- 标签读取 0；
- ARI/NMI/Q/AMI/FMI/空间标签指标计算 0；
- MISAR Y 读取 0；
- E18.5 访问 0；
- 新数据/第三方 benchmark 0；
- QCRD 训练/恢复/评价 0；
- 新科学候选 0；
- dataset-name routing 0；
- dense N×N 0；
- scientific retry/fallback 0。

30-row replay 中任何一行失败都不能删除、替换或按 seed 单独重跑。正式阶段如确有全局实现错误，最多两个全局 correction cycles，每次使全部旧 replay 行作废并从第 1 行重跑完整 30 行。普通数值失败原样锁定，不改 seed、solver、K、threshold 或 recipe。

成功终态只允许：

`NIGHT10B_FROZEN_FAMILY_POLICY_INTEGRATION_LOCKED`

它要求 30/30 replay、2/2 full smoke、2/2 fresh-process round-trip、身份盲路由、零标签、资源、Git 与交付全部通过。不能把 Night-10B 包装成新的性能提升、SOTA 或统一 trainable module 成功。

Windows 交付到：

`D:/文档/ChatGPT/博士第一篇科研论文项目/night10b_delivery_20260821/official_compact`

compact 只含代码、配置、测试、报告、审计、日志和增量 bundle；禁止下载 raw data、checkpoint、embedding、affinity 或大数组。最终回复必须给出实际 `N/N`、compact index SHA、bundle SHA、final commit/tag、标签与禁区访问计数，以及关机命令是否已派发。
