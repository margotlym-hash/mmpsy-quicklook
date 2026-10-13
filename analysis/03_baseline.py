# -*- coding: utf-8 -*-
"""
L0/L1 · 第 3 步：最简单的模型能走多远？

TF-IDF（把文本变成"哪些词出现、多重不重要"的数字向量）
+ 逻辑回归（最基础的分类器）——没有 BERT，没有神经网络。

任务：二分类 Normal vs 有症状（Mild/Moderate/Severe）
评估：5 折交叉验证（每人都被当一次"新学生"测过，不靠一次运气划分）

# Question: how far do the simplest word features go?
跑完看：AUC 离论文 BERT 的 0.877 差多少——那个差距就是深度模型的增量。

输出：results/03_baseline.md
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"

stu = pd.read_csv(DATA / "mmpsy-data-学生评分总表.csv", encoding="utf-8-sig")
X = stu["所有回答文本"].astype(str)
y = (stu["PHQ9_level"] != "Normal").astype(int)  # 1=有症状 0=Normal

pipe = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000),  # 单词+双词（"想死""难受"）
    LogisticRegression(max_iter=1000, class_weight="balanced"),  # balanced=给少数类更高权重
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
pred = (proba >= 0.5).astype(int)

acc = accuracy_score(y, pred)
auc_score = roc_auc_score(y, proba)
report = classification_report(y, pred, target_names=["Normal", "有症状"], digits=3)

L = []
add = L.append
add("# L0/L1 · 最简基线：TF-IDF + 逻辑回归\n")
add(f"- N={len(y)}，有症状 {int(y.sum())} 人（{y.mean()*100:.0f}%），Normal {int((1-y).sum())} 人")
add(f"- 模型：词频向量（1-2元词组）+ 逻辑回归——没有任何神经网络\n")
add("## 5 折交叉验证结果\n")
add("```")
add(report)
add("```")
add(f"- Accuracy：{acc:.3f}")
add(f"- **AUC：{auc_score:.3f}**　← 和论文 BERT 多任务系统的抑郁 AUC 0.877 放在一起看\n")
add("```")
add(f"基线参照：全部猜'有症状'的准确率 = {y.mean()*100:.0f}%")
add("```")
add("\n这两个数之间的差，就是深度模型在这个任务上的真实增量。")
add("（注意：论文测的是量表分数回归+三分类，这里是二分类，数字不能直接对齐，")
add("方向性对比即可；下一步 L2 换 BERT 时用同一套 5 折划分才能严格对比。）")

(RESULTS / "03_baseline.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print("\n✅ 已写入 results/03_baseline.md")
