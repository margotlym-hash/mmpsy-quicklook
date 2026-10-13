# -*- coding: utf-8 -*-
"""
L0 · 第 1 步：数据画像（只看数字，不看病例内容）

回答三个问题：
  1. 数据长什么样？多少人、什么级别分布？
  2. 类别有多不平衡？（论文说重症只有 ~15 人）
  3. 不同严重度的人，写的话有多长？

输出：results/01_overview.md + figures/ 两张图
跑法：python analysis/01_overview.py（在 mmpsy-quicklook 目录下）
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接把图存成文件
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # Windows 中文字体
plt.rcParams["axes.unicode_minus"] = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent  # mmpsy-quicklook/
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# ── 读数据（utf-8-sig 处理文件头的 BOM 字符）──────────────────
per = pd.read_csv(DATA / "mmpsy-data-逐条回答.csv", encoding="utf-8-sig")
stu = pd.read_csv(DATA / "mmpsy-data-学生评分总表.csv", encoding="utf-8-sig")

L = []
add = L.append

add("# L0 · 数据画像\n")
add(f"- 学生数：{stu.shape[0]}　逐条回答行数：{per.shape[0]}")
add(f"- 每人回答段数：平均 {stu['回答段数'].mean():.1f}，范围 {stu['回答段数'].min()}–{stu['回答段数'].max()}")
add("")

# ── 级别分布：类别不平衡的体感数字 ──────────────────────────────
add("## PHQ-9（抑郁）级别分布\n")
add("```")
add(stu["PHQ9_level"].value_counts().to_string())
add("```")
severe_n = int((stu["PHQ9_level"] == "Severe").sum())
add(f"\n重症：{severe_n} 人 = {severe_n / stu.shape[0] * 100:.1f}%　← 论文要靠 VAE 造数据的原因\n")

add("## GAD-7（焦虑）级别分布\n")
add("```")
add(stu["GAD7_level"].value_counts().to_string())
add("```")
add("")

# ── 按级别：人数 / 平均分 / 人均总字数 ──────────────────────────
# 级别顺序不写死，按 PHQ9 平均分排序（Normal→…→Severe 自然就位）
stu2 = stu.assign(总字数=stu["所有回答文本"].str.len())
order = stu2.groupby("PHQ9_level")["PHQ9_score"].mean().sort_values().index.tolist()
agg = stu2.groupby("PHQ9_level").agg(
    人数=("user_id", "count"),
    PHQ9均分=("PHQ9_score", "mean"),
    人均总字数=("总字数", "mean"),
).round(1).reindex(order)

add("## 各级别：人数 / 平均分 / 人均总字数\n")
add("```")
add(agg.to_string())
add("```")
add("\n（人均总字数随严重度的走势——越严重写得越多还是越少？看图 2）\n")

# ── 图 1：PHQ-9 分数直方图 ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(stu["PHQ9_score"], bins=27, edgecolor="white", color="#4C72B0")
ax.set_xlabel("PHQ-9 分数（0–27，越高越抑郁）")
ax.set_ylabel("人数")
ax.set_title(f"PHQ-9 分数分布（N={stu.shape[0]}，重症 {severe_n} 人）")
fig.tight_layout()
fig.savefig(FIGURES / "01_phq9_hist.png", dpi=150)
plt.close(fig)

# ── 图 2：人均总字数按级别 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(agg.index, agg["人均总字数"], color="#55A868", edgecolor="white")
ax.set_xlabel("PHQ-9 级别")
ax.set_ylabel("人均总字数")
ax.set_title("写的字数随严重度的变化")
for i, v in enumerate(agg["人均总字数"]):
    ax.text(i, v + 3, f"{v:.0f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "02_len_by_level.png", dpi=150)
plt.close(fig)

add("## 产出")
add("- figures/01_phq9_hist.png（分数分布）")
add("- figures/02_len_by_level.png（字数×严重度）")

(RESULTS / "01_overview.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print("\n✅ 已写入 results/01_overview.md")
