# -*- coding: utf-8 -*-
"""
L0+ · 第 5 步：转折连词密度 × 抑郁严重度（转折假设检验）

假设出处：学习窗口 2026-10-20/6 讨论种子（语气校准-01 卡的姐妹假设）——
抑郁表达可能更多"逻辑转折"：先说好的，然后"但是"转向不开心。

设计要点（与 02 的两点不同）：
  1. 直接数原始字符串（像语气词那样），**不走分词和停用词**——
     "但是"在 02 的 STOP 表里，走分词会被滤掉
  2. **双口径**：人均次数 + 每千字密度。L0 已发现字数随严重度涨 2.6 倍
     （275→719），只报次数会被"写得多的转折词自然也多"混淆，
     密度才能消掉长度

词表：但是 / 可是 / 不过 / 然而 / 却
（注："却"直接数会混入极少数非转折用法如"退却"，口语语料里可忽略）

输出：results/05_transition.md
"""
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"

WORDS = ["但是", "可是", "不过", "然而", "却"]

stu = pd.read_csv(DATA / "mmpsy-data-学生评分总表.csv", encoding="utf-8-sig")
text = stu["所有回答文本"].astype(str)
chars = text.str.len()
counts = pd.DataFrame({w: text.str.count(w) for w in WORDS})
stu = stu.assign(
    转折总数=counts.sum(axis=1),
    总字数=chars,
    密度=counts.sum(axis=1) / chars.replace(0, 1) * 1000,  # 每千字转折词数
    用过转折=(counts.sum(axis=1) > 0).astype(int),
)

order = stu.groupby("PHQ9_level")["PHQ9_score"].mean().sort_values().index.tolist()

L = []
add = L.append
add("# 转折连词密度 × 抑郁严重度\n")
add(f"词表：{' / '.join(WORDS)}（直接数原始字符串，不过分词）\n")

add("## 按 PHQ-9 级别（双口径）\n")
add("```")
add(f"{'级别':<10}{'人数':>5}{'人均次数':>9}{'每千字密度':>11}{'用过转折%':>10}")
for lv in order:
    sub = stu[stu["PHQ9_level"] == lv]
    add(f"{lv:<10}{len(sub):>5}{sub['转折总数'].mean():>9.2f}{sub['密度'].mean():>11.2f}{sub['用过转折'].mean()*100:>9.0f}%")
add("```")
add("（密度=每人每千字转折词数，先算每人的再平均——不被个别长文本带偏）\n")

add("⚠️ Severe 仅 15 人，样本小信号抖，看趋势别抠小数。\n")

# ── Normal vs Moderate+Severe 对比 ──
nor = stu[stu["PHQ9_level"] == "Normal"]
heavy = stu[stu["PHQ9_level"].isin(["Moderate", "Severe"])]
add("## Normal vs Moderate+Severe\n")
add("```")
add(f"  人均次数：Normal {nor['转折总数'].mean():.2f} → 有症状 {heavy['转折总数'].mean():.2f}（{heavy['转折总数'].mean()/max(nor['转折总数'].mean(),1e-9):.2f} 倍）")
add(f"  每千字密度：Normal {nor['密度'].mean():.2f} → 有症状 {heavy['密度'].mean():.2f}（{heavy['密度'].mean()/max(nor['密度'].mean(),1e-9):.2f} 倍）")
add(f"  用过转折：Normal {nor['用过转折'].mean()*100:.0f}% → 有症状 {heavy['用过转折'].mean()*100:.0f}%")
add("```\n")

# ── 分词细看：哪个词在承载信号 ──
add("## 分词密度（每千字）\n")
add("```")
header = f"{'级别':<10}" + "".join(f"{w:>7}" for w in WORDS)
add(header)
for lv in order:
    sub = stu[stu["PHQ9_level"] == lv]
    row = f"{lv:<10}"
    for w in WORDS:
        d = (text[sub.index].str.count(w) / chars[sub.index].replace(0, 1) * 1000).mean()
        row += f"{d:>7.2f}"
    add(row)
add("```")

# ── 单调性判定 ──
dens = [stu[stu["PHQ9_level"] == lv]["密度"].mean() for lv in order]
mono = all(dens[i] < dens[i + 1] for i in range(len(dens) - 1))
add("\n## 判定\n")
if mono:
    add(f"密度逐级单调上升（{' → '.join(f'{d:.2f}' for d in dens)}）——**转折假设初步支持，出自 [[语气校准-01-词频测不出不接纳]] 卡的姐妹假设**。")
    add("注：相关性≠因果；密度受写作主题/能力干扰；下一步可做的严格版=控制字数分箱后对比、或在模型里把转折密度作为特征看增量。")
else:
    add(f"密度非单调（{' → '.join(f'{d:.2f}' for d in dens)}）——转折假设在本数据上未见单调支持，看分组对比和分词表判断信号形态。")

out = RESULTS / "05_transition.md"
out.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print(f"\n✅ 已写入 {out}")
