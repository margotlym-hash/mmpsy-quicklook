# -*- coding: utf-8 -*-
"""
07 - Same word, different contexts: an observational probe

Six positive words (kai xin / xing fu / gao xing / kuai le / xi huan / man yi):
for each, sample two context buckets from the transcripts - matched contexts
(no negative or modal markers) vs mismatched contexts (containing one of the
markers such as negation, "should", "originally", adversative adverbs). Then
compare pretrained-MacBERT cosine similarity for the same word across buckets
against the similarity between different words: how far apart are the two
faces of the same word?

Note: deliberately sampled examples make this an observational probe, not a
test - selection is non-random, outputs are illustrations rather than inference.
Outputs containing verbatim examples stay local (gitignored); only the
aggregate figure is published.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"

EMO = ["开心", "幸福", "高兴", "快乐", "喜欢", "满意"]
MARKERS = ["不", "没", "应该", "本来", "却", "但是", "可是", "为什么"]
PER_BUCKET = 8   # 例句桶：每词每桶最多取几条（交接单 5-10）
PER_BUCKET_VEC = 50  # 向量桶：每词每桶最多编码几条（求稳定质心）

rng = np.random.RandomState(42)
per = pd.read_csv(DATA / "mmpsy-data-逐条回答.csv", encoding="utf-8-sig")
txt = per["回答文本"].astype(str)

L = []
add = L.append
add("# 实验 A · 同词异义取样（语气校准种子的数据版）\n")
add("> ⚠️ **含病例原文 · 未脱敏 · 勿上传**——仅本地，已 .gitignore\n")
add("> ⚠️ 口径：**观察例证非检验**（刻意取样有选择偏差）——对外措辞用「我注意到」，禁用「我验证/检验」\n")

# ── 分桶 ────────────────────────────────────────────────
buckets = {}  # (word, tag) -> list[sentence]
for w in EMO:
    hit = per[txt.str.contains(w)].copy()
    hit["文本"] = hit["回答文本"].astype(str)
    mm = hit[hit["文本"].apply(lambda t: any(m in t for m in MARKERS))]
    pos = hit[~hit.index.isin(mm.index)]
    buckets[(w, "错配")] = list(mm["文本"])
    buckets[(w, "正向")] = list(pos["文本"])
    add(f"- {w}：命中 {len(hit)} 段；错配桶 {len(mm)}，正向桶 {len(pos)}")
add("（「错配」=句含标记词之一；是否真反转由人读例句判断，机器只做初筛）\n")

# ── 例句库 ──────────────────────────────────────────────
for w in EMO:
    add(f"\n## {w}\n")
    for tag in ["正向", "错配"]:
        sents = buckets.get((w, tag), [])
        if not sents:
            add(f"**{tag}**：无样本\n")
            continue
        take = list(rng.choice(len(sents), size=min(PER_BUCKET, len(sents)), replace=False))
        add(f"**{tag}**（{len(sents)} 条中抽 {len(take)}）\n")
        for j in take:
            s = sents[j].replace("\n", " ")
            add(f"- {s}")
        add("")

# ── 加分项：同词跨语境 vs 异词 的向量余弦 ──────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")
model = AutoModel.from_pretrained("hfl/chinese-macbert-base").to(device).eval()


def word_vec(sent: str, word: str):
    """返回句中 word 的 token 向量（首个出现位置，span 平均）"""
    enc = tok(sent, return_offsets_mapping=True, truncation=True, max_length=128,
              return_tensors="pt")
    spans = enc.pop("offset_mapping")[0].tolist()
    ids = {k: v.to(device) for k, v in enc.items()}
    ids["input_ids"] = ids["input_ids"][0:1]
    ids["attention_mask"] = ids["attention_mask"][0:1]
    ids["token_type_ids"] = ids["token_type_ids"][0:1] if "token_type_ids" in ids else None
    with torch.no_grad():
        out = model(**{k: v for k, v in ids.items() if v is not None}).last_hidden_state[0]
    for i in range(len(spans) - 1):
        if sent[spans[i][0]:spans[i][1]] == word[0] and sent[spans[i + 1][0]:spans[i + 1][1]] == word[1:]:
            if len(word) == 2:
                return (out[i] + out[i + 1]) / 2
            return out[i]
    return None


def centroid(word: str, tag: str):
    sents = buckets.get((word, tag), [])[:PER_BUCKET_VEC]
    vecs = [v for v in (word_vec(s, word) for s in sents) if v is not None]
    if not vecs:
        return None, 0
    return torch.stack(vecs).mean(0), len(vecs)


def cos(a, b):
    return (a @ b / (a.norm() * b.norm())).item()


cent = {}
for w in EMO:
    for tag in ["正向", "错配"]:
        c, n = centroid(w, tag)
        if c is not None:
            cent[(w, tag)] = (c, n)

add("\n## 向量对照（预训练 MacBERT，词向量=语境句中该词 span 平均后求质心）\n")
same, cross = [], []
for w in EMO:
    if (w, "正向") in cent and (w, "错配") in cent:
        v = cos(*[cent[(w, t)][0] for t in ["正向", "错配"]])
        same.append((w, v))
for i, w1 in enumerate(EMO):
    for w2 in EMO[i + 1:]:
        if (w1, "正向") in cent and (w2, "正向") in cent:
            cross.append(cos(cent[(w1, "正向")][0], cent[(w2, "正向")][0]))

add("```")
add(f"{'词':<6}{'同词跨语境 cos':>14}")
for w, v in same:
    add(f"{w:<6}{v:>14.3f}")
add(f"\n异词对照均值（正向桶内互比）: {np.mean(cross):.3f}")
add("```")
add("\n读法：**同词跨语境 cos 越低于异词均值 ⇒ 语境把同一个词的表示拉得比「换了个词」还远**。")
sizes = {f"{w}-{t}": n for (w, t), (_, n) in cent.items()}
add(f"每质心句数：{sizes}")

# ── 图（不含原文，可公开）───────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(7, 4))
ws = [w for w, _ in same]
vs = [v for _, v in same]
ax.bar(ws, vs, color="#4C72B0", edgecolor="white")
ax.axhline(np.mean(cross), color="#C44E52", ls="--", label=f"异词对照均值 {np.mean(cross):.3f}")
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("余弦相似度")
ax.set_title("同一个词的两副面孔：正向语境 vs 错配语境")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "07_same_word_two_faces.png", dpi=150)

add(f"\n图：figures/07_same_word_two_faces.png（无原文，可公开）")
(RESULTS / "07_same_word_contexts.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L[-12:]))
print(f"\n✅ 已写入 {RESULTS / '07_same_word_contexts.md'}（含原文，勿上传）")
