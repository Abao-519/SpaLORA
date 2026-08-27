# Night-23C 方法与选择合同

- Night-23B 数值有效，唯一下游勘误是决策顺序；scale=2 固定。
- 每个 outer fold 只用另外两个 primary studies 的 teacher relations；inner leave-one-study-out 产生跨研究 score。
- MLP 为 primary，logistic 为解释性对照；calibration 是每研究 score midrank，阈值最大化最差 source selective utility。
- WITHIN 吸引，BOUNDARY 排斥，FULL 的 UNKNOWN 权重严格为零；另有 retained-support unknown 原子臂。
- carrier geometry 与 signed relation embedding 拼接，relation scale 固定 2，deterministic exact-K endpoint。
- 所有 primary bank/checkpoint 先锁定并 fresh replay，之后 evaluator 才打开 heldout teacher/reference。
- FULL 必须在至少 2/3 lane 严格双胜每个 matched control，macro 双增且安全门通过。实际 0/3，placenta 禁止。
