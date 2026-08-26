# Night-19C 导师口述稿

1. 我们先把 Night-17C 的旧证据重新核清，确认它的零起步训练信号没有被后来失败的 selector 或 direct-cut 自动推翻。
2. 新实验完全复用原 Z01，不改网络、损失、步数和阈值。
3. Placenta 的教师来自 Night-19B 标签前锁定的 16 个分区，保留原 ID 并严格等权，没有伪装成旧候选前缀。
4. 工程路径真实完成了训练、参数更新、checkpoint reload 和 fresh-process exact replay。
5. 但 Z01 full 只有 `0.316241/0.505123`，明显低于 relation smooth 的 `0.377102/0.554557`。
6. 因此 Night-17C 的局部信号没有迁移到 placenta，训练 residual 反而破坏了更好的 deterministic smooth carrier。
7. Relation smooth 本身相对 retained 有明显 control signal，但低于 Night-19B concat frontier，不能当作可训练方法成功。
8. 本轮按预注册门停止，不补 seed、不接结构 head，结论为 `SCIENTIFIC_NEGATIVE`。
