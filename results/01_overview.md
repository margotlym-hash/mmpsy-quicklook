# L0 · 数据画像

- 学生数：1275　逐条回答行数：9417
- 每人回答段数：平均 7.4，范围 4–10

## PHQ-9（抑郁）级别分布

```
PHQ9_level
Mild        581
Normal      436
Moderate    243
Severe       15
```

重症：15 人 = 1.2%　← 论文要靠 VAE 造数据的原因

## GAD-7（焦虑）级别分布

```
GAD7_level
Normal      883
Mild        256
Moderate    122
Severe       14
```

## 各级别：人数 / 平均分 / 人均总字数

```
             人数  PHQ9均分  人均总字数
PHQ9_level                    
Normal      436     3.1  275.4
Mild        581     6.7  332.0
Moderate    243    12.5  475.8
Severe       15    21.2  719.3
```

（人均总字数随严重度的走势——越严重写得越多还是越少？看图 2）

## 产出
- figures/01_phq9_hist.png（分数分布）
- figures/02_len_by_level.png（字数×严重度）