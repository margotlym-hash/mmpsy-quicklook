# -*- coding: utf-8 -*-
"""
L0 · 第 2 步：词频对比（重症的人 vs 无症状的人，各自反复说什么）

这就是你 2026-04 在论文笔记里自己提的问题的第一半：
"能不能不用 BERT，用词频统计，看看能发现什么？"

两个对比：
  A. Severe vs Normal（纯重症，人少，信号可能抖）
  B. Moderate+Severe vs Normal（合并有症状组，更稳）

特别设计：口语里的语气填充词（嗯/呃/哎/唉…）单独统计——
犹豫和停顿本身可能就是信号，这是论文没做过的角度。

输出：results/02_wordfreq.md（只有词和数字，不展示任何病例原文）
"""
import re
import sys
from collections import Counter
from pathlib import Path

import jieba
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# ── 词表设置 ────────────────────────────────────────────────
# 只保留纯中文词（自动滤掉标点、数字、英文）
CJK = re.compile(r"^[一-鿿]+$")

# 极简停用词表：只去掉随处可见的虚词，故意保留所有可能带情感的词
#（包括单字——"累""哭""烦""死"这种单字恰恰是关键信号）
STOP = set("""的 了 是 我 你 他 她 它 我们 你们 他们 就 都 也 和 与 及 或 在 有 很
这个 那个 这些 那些 什么 怎么 为什么 因为 所以 但是 然后 还有 就是 不是 一个 一些
自己 觉得 可能 现在 时候 东西 没有 会 被 把 让 对 到 说 去 来 要 还 想 做 真的 很多
非常 特别 如果 对于 而且 关于 通过 应该 可以 的话 一样 一般 感觉 时候 而且 大家
知道 可能 一定 开始 每天 一些 以后 老师 学校 同学 爸妈 父母""".split())

# 语气填充词：从主统计里摘出来，单独看（犹豫/停顿的痕迹）
FILLERS = ["嗯", "呃", "哎", "哎呀", "唉", "哦"]

MIN_COUNT = 5  # 一个词至少出现 5 次才进差异榜，防止"只出现2次但全在重症组"的假信号


def tokenize(text: str) -> list[str]:
    words = jieba.lcut(str(text))
    return [w for w in words if CJK.match(w) and w not in STOP and w not in FILLERS]


def freq_per_10k(counter: Counter, total: int) -> dict[str, float]:
    return {w: c / total * 10000 for w, c in counter.items()} if total else {}


# ── 读数据，按人聚合文本，按 PHQ9 级别分组 ─────────────────────
per = pd.read_csv(DATA / "mmpsy-data-逐条回答.csv", encoding="utf-8-sig")
stu = pd.read_csv(DATA / "mmpsy-data-学生评分总表.csv", encoding="utf-8-sig")

group_tokens: dict[str, Counter] = {}
group_total: dict[str, int] = {}
filler_tokens: dict[str, Counter] = {}

for level, sub in per.groupby("PHQ9_level"):
    tokens, fills = Counter(), Counter()
    for t in sub["回答文本"]:
        tokens.update(tokenize(t))
        for f in FILLERS:  # 填充词按原文本直接数（它们常被分词切碎）
            fills[f] += str(t).count(f)
    group_tokens[level] = tokens
    group_total[level] = sum(tokens.values()) or 1
    filler_tokens[level] = fills

levels_by_n = stu["PHQ9_level"].value_counts()
L = []
add = L.append
add("# L0 · 词频对比\n")
add(f"分组人数：{levels_by_n.to_dict()}\n")


def compare(name: str, heavy_name: str, heavy_levels: list[str], light_level: str = "Normal"):
    """heavy 组（合并若干级别）vs light 组（Normal）的词频对比"""
    heavy_c = Counter()
    for lv in heavy_levels:
        if lv in group_tokens:
            heavy_c += group_tokens[lv]
    heavy_total = sum(group_total[lv] for lv in heavy_levels if lv in group_total) or 1
    light_c = group_tokens.get(light_level, Counter())
    light_total = group_total.get(light_level, 1)

    heavy_f = freq_per_10k(heavy_c, heavy_total)
    light_f = freq_per_10k(light_c, light_total)

    add(f"\n## 对比{name}：{heavy_name}（{heavy_total:,} 词）vs Normal（{light_total:,} 词）\n")

    # 各组自己的 top20（先感受"两组日常都在说什么"）
    for tag, c, total in [(heavy_name, heavy_c, heavy_total), ("Normal", light_c, light_total)]:
        top = c.most_common(20)
        rows = "　".join(f"{w}×{n}" for w, n in top)
        add(f"**{tag} 组 top20**：{rows}\n")

    # 差异榜：≥MIN_COUNT 次，按"每万字频率倍数"排
    diff = []
    for w, hc in heavy_c.items():
        nc = light_c.get(w, 0)
        if hc < MIN_COUNT and nc < MIN_COUNT:
            continue
        hf, lf = heavy_f.get(w, 0), light_f.get(w, 0)
        ratio = (hf / lf) if lf > 0 else float("inf")  # Normal 组完全没有 → ∞
        diff.append((w, hc, hf, nc, lf, ratio))
    diff.sort(key=lambda x: -x[5])

    add("**重症方向差异词 top25**（每万字频率 / Normal倍数）\n")
    add("```")
    add(f"{'词':<6}{'本组次':>6}{'/万字':>8}{'Normal次':>9}{'/万字':>8}{'倍数':>8}")
    for w, hc, hf, nc, lf, r in diff[:25]:
        rs = "∞" if r == float("inf") else f"{r:.1f}"
        add(f"{w:<6}{hc:>6}{hf:>8.1f}{nc:>9}{lf:>8.1f}{rs:>8}")
    add("```\n")

    # 反向：Normal 多、重症少的词（"保护性"语言？）
    diff_r = [d for d in diff if d[5] < 1]
    diff_r.sort(key=lambda x: x[5])
    add("**反向 top15**（Normal 组里更常见的词）\n")
    add("```")
    for w, hc, hf, nc, lf, r in diff_r[:15]:
        add(f"{w:<8} 本组×{hc}（{hf:.1f}/万字）　Normal×{nc}（{lf:.1f}/万字）　{r:.2f}倍")
    add("```")


compare("A", "Severe（纯重症）", ["Severe"])
compare("B", "Moderate+Severe（有症状合并）", ["Moderate", "Severe"])

# ── 语气填充词：双口径按级别（人均次数 + 每千字密度）─────────────
# 2026-10-07 改：加密度口径（同 05 转折）——L0 已知字数随严重度涨 2.6 倍，
# 人均次数会被"写得多所以填充词多"混淆，密度才能消掉长度
add("\n## 语气填充词（双口径：人均次数 / 每千字密度）\n")

# 先算每个学生的（次数、总字数），再按级别平均——不被个别长文本带偏
person_stats = {}  # user_id -> [level, Counter, chars]
for uid, level, t in zip(per["user_id"], per["PHQ9_level"], per["回答文本"]):
    s = person_stats.setdefault(uid, [level, Counter(), 0])
    s[2] += len(str(t))
    for f in FILLERS:
        s[1][f] += str(t).count(f)

by_level = {}
for uid, (lv, fills, chars) in person_stats.items():
    by_level.setdefault(lv, []).append((fills, max(chars, 1)))

order_sev = stu.groupby("PHQ9_level")["PHQ9_score"].mean().sort_values().index.tolist()
n_by_level = stu["PHQ9_level"].value_counts().to_dict()
dens = {f: [] for f in FILLERS}

add("```")
add(f"{'级别':<10}{'人数':>5}　人均次数（嗯/呃/哎/哎呀/唉/哦）　每千字密度（同序）")
for lv in order_sev:
    rows = by_level.get(lv, [])
    n = n_by_level.get(lv, len(rows))
    cnt = {f: sum(r[0].get(f, 0) for r in rows) / max(len(rows), 1) for f in FILLERS}
    d = {f: sum(r[0].get(f, 0) / r[1] * 1000 for r in rows) / max(len(rows), 1) for f in FILLERS}
    for f in FILLERS:
        dens[f].append(d[f])
    c_row = "/".join(f"{cnt[f]:.2f}" for f in FILLERS)
    d_row = "/".join(f"{d[f]:.2f}" for f in FILLERS)
    add(f"{lv:<10}{n:>5}　{c_row}　　{d_row}")
add("```\n")

# 单调性判定（密度口径，按严重度升序）
for f in ["嗯", "呃"]:
    seq = dens[f]
    mono_up = all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
    if mono_up:
        add(f"- **{f}：密度归一后仍单调上升**（{' → '.join(f'{v:.2f}' for v in seq)}）——结论变硬")
    else:
        add(f"- {f}：密度非单调（{' → '.join(f'{v:.2f}' for v in seq)}）——人均次数的上升可（至少部分）被篇幅解释")

(RESULTS / "02_wordfreq.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print("\n✅ 已写入 results/02_wordfreq.md")
