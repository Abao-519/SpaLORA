Night-8A 的 116 次训练没有白跑：checkpoint 和候选结果都通过了全量重哈希，错误只在旧 comparator 变换。
本次用真正锁定的 C00/R02 comparator 重算后，shortlist 为：空。
B03/P22/seed1 仍是原始数值失败，没有补跑。
当前还不能跑 MISAR，因为这里只是三 seed pilot 恢复；还需先审查 shortlist、完成 R3 固定补 seed 并冻结最终候选。
