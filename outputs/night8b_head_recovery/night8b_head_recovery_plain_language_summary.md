# Night-8B uniform-head recovery: plain-language summary

这次不能判断 F00 是否优于 U00。20 个统一稳健聚类头都已成功得到恰好 12 簇，说明原 Night-8B 的 U00 seed6 聚类端点失败已被统一 head 工程上消除；但在 partitions 全部锁定并推送后，唯一一次打开 MISAR Y 时，真实标注的唯一类别数与注册表预先锁定的 K=12 不一致。评价因此在计算任何 ARI、NMI 或 Q 之前硬停止。

按照预注册规则，本轮不得重新读取 Y、尝试别的 K、改变标注粒度或重新聚类，所以没有合法的科学比较表，也不能据此支持或否定“RNA+ATAC 使用 R02”的 family policy。原 Night-8B 继续保持 INFRASTRUCTURE_BLOCKED；本恢复终态为 RECOVERY_BLOCKED_INPUT_INTEGRITY。

原 9 对 spectral 结果没有在本次标签窗口中评价，也没有影响终态。没有训练、adapter、affinity 重建、第三方 benchmark 或 SOTA 声明。
