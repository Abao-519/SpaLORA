# Night-10A 通俗结论

Night-10A 没有产生新的分数，也没有训练任何 QCRD 候选。原因不是候选效果差，而是正式训练前的逐字段复核发现：现有实现漏掉了注册表明确要求的部分质量特征，同时权威文件没有给出几项训练损失和 masked denoising 的唯一数值定义。执行者如果自行补默认值，就会把预注册实验变成事后自创实验。

因此本轮严格停在 P0，终态为 `IMPLEMENTATION_SEMANTICS_INVALID`。Stage M、R1 和 R2 均未启动；没有候选、seed 或 checkpoint 被挑选，没有 fallback，也没有访问 MISAR Y、E18.5 或新外部数据。A1 标签只在隔离的 P0 指标校验中读取，用于确认 ARI/NMI 实现一致，未进入训练或选择。

后续若要继续，应由规划方签发一个机器可读 REV1：明确 global/spot quality 的完整公式、mask 比例与确定性规则、三项 mandatory regularizer 权重、Q07 MNN 权重，并同步更新真实构造测试。不能仅口头补一句“使用合理默认值”。
