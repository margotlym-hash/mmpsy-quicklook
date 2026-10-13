# -*- coding: utf-8 -*-
"""
L2 · 第 4 步：BERT 微调——商业分析技能迁移的"升级模型"环节

与 03 基线严格同口径：同一个 5 折分层划分（同 DataFrame 顺序 + 同 seed=42），
所以 04 的 AUC 减 03 的 0.634 = 深度模型在这个任务上的干净增量。

任务：
  分类（默认）  Normal vs 有症状 —— 和 03 基线直接比
  回归（--task regression） 预测 PHQ-9 具体分数 —— 对齐论文 r=0.706

跑法：
  python analysis/04_bert.py                       # 分类任务，5 折全量
  python analysis/04_bert.py --quick               # 快速验证管线（1 折 1 epoch）
  python analysis/04_bert.py --task regression     # 回归任务

说明：
  - 未调参，小数据微调标准配方（lr 2e-5 / batch 16 / 4 epochs）——demo 目标是
    增量对比，不是刷分
  - 国内网络走 hf-mirror 镜像下载模型（首次 ~400MB，之后走缓存）
输出：results/04_bert_classification.md 或 04_bert_regression.md
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "hfl/chinese-macbert-base"  # 中文 BERT（哈工大讯飞 MacBERT）
SEED = 42
BATCH = 16
EPOCHS = 4
LR = 2e-5
MAX_LEN = 256  # 每人全部回答 300-700 字，截到 256 token 覆盖大部分

task = "regression" if "--task" in sys.argv and "regression" in sys.argv else "classification"
quick = "--quick" in sys.argv

torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
stu = pd.read_csv(DATA / "mmpsy-data-学生评分总表.csv", encoding="utf-8-sig")
X = stu["所有回答文本"].astype(str).tolist()

if task == "classification":
    y = (stu["PHQ9_level"] != "Normal").astype(int).values  # 与 03 完全一致
    num_labels = 2
else:
    y = stu["PHQ9_score"].astype(float).values  # 回归：预测具体分数
    num_labels = 1

print(f"任务：{task}　N={len(y)}　模型：{MODEL_NAME}")
print("下载/加载 tokenizer 与模型（首次约 400MB，请等待）…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TextDS(Dataset):
    def __init__(self, texts, ys):
        self.enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding="max_length")
        self.ys = ys

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(float(self.ys[i]))
        return item


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 与 03 同款划分器
splits = list(skf.split(X, stu["PHQ9_level"]))  # 分层依据=级别，两任务同折
if quick:
    splits = splits[:1]

oof_pred, oof_true, oof_uid = [], [], []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备：{device}")

for fold, (tr, te) in enumerate(splits):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n_epochs = 1 if quick else EPOCHS
    dl = DataLoader(TextDS([X[i] for i in tr], y[tr]), batch_size=BATCH, shuffle=True)
    total_steps = len(dl) * n_epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * 0.1), total_steps)

    model.train()
    for ep in range(n_epochs):
        losses = []
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            if task == "classification":
                batch["labels"] = batch["labels"].long()
            out = model(**batch)  # HF 默认损失：分类=交叉熵（未加权），回归=MSE
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(out.loss.item())
        print(f"  fold{fold} epoch{ep}  loss={np.mean(losses):.4f}")

    # ── 在 held-out 折上预测 ──
    model.eval()
    preds = []
    with torch.no_grad():
        dl_te = DataLoader(TextDS([X[i] for i in te], y[te]), batch_size=32)
        for batch in dl_te:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: batch[k] for k in ["input_ids", "attention_mask", "token_type_ids"]}).logits
            if task == "classification":
                preds += torch.softmax(logits, -1)[:, 1].tolist()  # P(有症状)
            else:
                preds += logits[:, 0].tolist()
    oof_pred += preds
    oof_true += y[te].tolist()
    oof_uid += stu["user_id"].iloc[te].tolist()
    print(f"  fold{fold} 完成（{len(te)} 人）")

oof_pred, oof_true = np.array(oof_pred), np.array(oof_true)

L = []
add = L.append
if task == "classification":
    pred_cls = (oof_pred >= 0.5).astype(int)
    add(f"# L2 · BERT 微调（分类：Normal vs 有症状）\n")
    add(f"- 模型：{MODEL_NAME}，{EPOCHS} epochs，未调参标准配方")
    add(f"- 同 5 折 out-of-fold 评估（与 03 基线同划分）\n")
    add("```")
    add(classification_report(oof_true, pred_cls, target_names=["Normal", "有症状"], digits=3))
    add("```")
    auc_s = roc_auc_score(oof_true, oof_pred)
    add(f"- Accuracy：{accuracy_score(oof_true, pred_cls):.3f}")
    add(f"- **AUC：{auc_s:.3f}**")
    if "--dump-preds" in sys.argv and not quick:
        pd.DataFrame(
            {"user_id": oof_uid, "y_true": oof_true.astype(int), "bert_proba": oof_pred.round(4)}
        ).to_csv(RESULTS / "oof_bert.csv", index=False, encoding="utf-8-sig")
        add("- OOF 预测已存 results/oof_bert.csv（--dump-preds）")
    add(f"- 对照：TF-IDF+逻辑回归基线 AUC = 0.634（03_baseline）")
    add(f"- **深度模型增量 = {auc_s - 0.634:+.3f}**")
    if quick:
        add("\n⚠️ 本次为 --quick 模式（仅 1 折 1 epoch），数字仅验证管线用，非最终结果。")
    out = RESULTS / "04_bert_classification.md"
else:
    r = np.corrcoef(oof_pred, oof_true)[0, 1]
    mae = np.abs(oof_pred - oof_true).mean()
    add(f"# L2 · BERT 微调（回归：预测 PHQ-9 分数）\n")
    add(f"- 模型：{MODEL_NAME}，{EPOCHS} epochs，5 折 out-of-fold")
    add(f"- **Pearson r = {r:.3f}**（论文 BERT 多任务：0.706）")
    add(f"- MAE = {mae:.2f} 分（量表满分 27）")
    if quick:
        add("\n⚠️ --quick 模式，仅验证管线。")
    out = RESULTS / "04_bert_regression.md"

out.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print(f"\n✅ 已写入 {out}")
