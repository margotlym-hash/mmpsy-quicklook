# L2 · BERT 微调（回归：预测 PHQ-9 分数）

- 模型：hfl/chinese-macbert-base，4 epochs，5 折 out-of-fold
- **Pearson r = 0.551**（论文 BERT 多任务：0.706）
- MAE = 2.50 分（量表满分 27）