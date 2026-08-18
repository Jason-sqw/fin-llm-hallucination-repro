# 金融指令微调降低幻觉：困难评测集结果

## 评测集

`data/generated/hard_eval.jsonl`，共 90 条，全部基于 A 股上市公司公开财报数据生成：

| 题型 | 数量 | 正确行为 |
| --- | --- | --- |
| 未提及陷阱 | 40 | 材料没有该指标，必须拒答 |
| 年份陷阱 | 20 | 材料只含其他年份，必须拒答 |
| 计算 | 20 | 跨年营收差值，必须给出材料中可推出的数字 |
| 干扰项 | 10 | 两家公司材料混排，必须只取目标公司数字 |

## 指标定义

- 陷阱题幻觉率 = 陷阱题中未拒答的比例（给出具体数字即视为编造）
- 有答案题错误率 = 计算 + 干扰项中未命中 gold 的比例
- 总体幻觉率 = (陷阱题编造数 + 有答案题答错数) / 总题数

## 结果

模型：`Qwen/Qwen2.5-7B-Instruct`，QLoRA 3 epoch，训练数据 4498 条。

| 指标 | Base | SFT | 变化 |
| --- | --- | --- | --- |
| 陷阱题幻觉率 | 76.7% (46/60) | 43.3% (26/60) | -33.3 pp，相对下降 43.5% |
| 有答案题错误率 | 0.0% (0/30) | 3.3% (1/30) | +3.3 pp |
| 总体幻觉率 | 51.1% (46/90) | 30.0% (27/90) | -21.1 pp，相对下降 41.3% |

## 复现命令

```bash
# 1. 生成训练数据（A 股市值前 120 家，2020-2024）
python -m src.build_financial_qa --mode template --auto 120 \
  --years 2020,2021,2022,2023,2024 --out data/generated/financial_qa_batch.jsonl

# 2. 切分训练/验证集
python -m src.data_prep --data data/generated/financial_qa_batch.jsonl \
  --out-dir data/processed --val-ratio 0.1

# 3. 生成困难评测集
python -m src.build_hard_eval --data data/generated/financial_qa_batch.jsonl \
  --out data/generated/hard_eval.jsonl --max-companies 20

# 4. 微调（GPU）
python -m src.train --config configs/train.yaml

# 5. Base 与 SFT 评估
python -m src.evaluate --eval-file data/generated/hard_eval.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct --judge none
python -m src.evaluate --eval-file data/generated/hard_eval.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct --adapter outputs/fin-lora --judge none
```

## 说明

- 数据来自上市公司公开财报/同花顺财务摘要，仅用于研究复现
- 当前幻觉率为"陷阱题拒答 + gold 数字命中"的操作化定义；后续可用 API judge 补充 claim 级无据表述率
- 有答案题错误率上升 3.3 pp（1/30），建议进一步检查该样本是拒答误伤还是数字错误
