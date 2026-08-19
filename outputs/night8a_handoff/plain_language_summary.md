# Night-8A 通俗总结

这轮把 MF-SPC 的家族路由、shared/private、同位点对齐、软原型、RNA 锚点、DGI 和 MNN triplet 都实现并完成了 R1/R2，但第一次开标签核验时发现基线语义错误，所以本轮科研结论作废，不能报“涨了多少”。

错误只定位在 RNA+蛋白的 C00 基线路径：程序重新构造了近似 affinity，而不是复用已经锁定的 G04/H05 affinity 和 partition。A1、D1、tonsil 因而都没有可报告的新方法增减；P22 的 R02 基线本身精确复现，但因为联合选择窗口整体失效，也没有候选被晋级。R3 没有启动，失败后新增训练和 transform 都是 0。

修复代码和回归测试已经准备好，但没有拿它回填本轮结果。外部 MISAR E15.5 S1 已完成无标签锁定（1949 spots，RNA+ATAC），本轮没有运行外部 benchmark。
