# Night-8A evaluation-only recovery report

终态：`NIGHT8A_DEV_WINDOW1_RECOVERED_NO_ELIGIBLE_NEW_CANDIDATE`

原 Night-8A 仍永久保持 `IMPLEMENTATION_SEMANTICS_INVALID`。本恢复任务完成 0 training、0 transform、0 external benchmark、0 GPU。

## 科学结果

- 116 次真实 CUDA 训练与 12 个合法 alias 全部救回；109/109 新候选依赖隔离通过。
- comparator parity 最大误差：`4.441e-16`；独立复算最大误差：`1.055e-15`。
- 固定失败保持：B03/P22/seed1；B03 只有 11/12，未晋级。
- shortlist：空。

## R2 pilot 相对锁定 family comparator 的均值变化

|配置|HLN ΔQ|P22 ΔQ|tonsil ΔQ|priority macro ΔQ|空间门|完整|
|---|---:|---:|---:|---:|---|---|
|B00_FAMILY_REFERENCE|+0.000000|-0.000000|+0.000000|-0.000000|True|True|
|B01_SP_RR10|-0.002977|-0.000924|+0.005435|-0.001212|False|True|
|B02_SP_RR30|-0.010472|-0.002559|+0.005614|-0.005302|False|True|
|B03_RR10_PROTO|-0.003674|-0.004127|+0.003593|-0.003151|True|False|
|B04_SP_RR10_PROTO|-0.003018|-0.002342|+0.004307|-0.001981|False|True|
|B05_SP_RR10_PROTO_RNAANCHOR|-0.003018|-0.001518|+0.004307|-0.001610|False|True|
|B06_SP_RR10_PROTO_DGI|-0.003660|-0.001843|+0.004968|-0.001979|False|True|
|B07_SP_RR10_PROTO_TRIPLET|-0.004456|-0.001149|+0.005544|-0.001968|False|True|

## Shortlist slots

- unified_balanced: None
- human_lymph_frontier: None
- P22_frontier: None

MISAR 现在不能运行：本轮只恢复 3-seed DEV_WINDOW_1。必须先由规划方审查 pilot，并对锁定的最多三个候选运行预注册 R3 补 seed、冻结最终 family candidate，之后才允许一次性外部确认。
