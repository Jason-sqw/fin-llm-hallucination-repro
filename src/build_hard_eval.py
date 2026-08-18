"""Build a harder hallucination eval set from the generated financial QA batch.

Question types:
  missing-metric: metric absent from the material; correct answer is refusal
  wrong-year:     material only covers one year; asking another year must refuse
  distractor:     two companies' materials are mixed; answer must stay grounded
  calculation:    cross-year revenue difference
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

MISSING_METRICS = [
    ("扣非净利润", "扣非净利润"),
    ("资产负债率", "资产负债率"),
    ("研发投入", "研发投入"),
    ("研发人员数量", "研发人员数量"),
]


def company_of(record):
    match = re.search(r"回答：(.+?)(?:20\d{2})年", record.get("instruction", ""))
    if match:
        return match.group(1).strip()
    match = re.match(r"(.+?)公开财务数据", record.get("input", ""))
    return match.group(1).strip() if match else ""


def year_of(record):
    parts = record["id"].split("-")
    if len(parts) >= 2 and re.fullmatch(r"20\d{2}", parts[1]):
        return parts[1]
    return None


def parse_yi(gold):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)亿元", gold)
    return float(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/generated/financial_qa_batch.jsonl")
    parser.add_argument("--out", default="data/generated/hard_eval.jsonl")
    parser.add_argument("--max-companies", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_stock = defaultdict(list)
    for record in records:
        by_stock[record["id"].split("-")[0]].append(record)

    rng = random.Random(args.seed)
    out = []
    stock_list = sorted(by_stock)[: args.max_companies]

    def add(item):
        item["id"] = f"{item['stock']}-hard-{len(out):04d}"
        out.append(item)

    for stock in stock_list:
        company_records = by_stock[stock]
        company = company_of(company_records[0])
        by_year = defaultdict(list)
        for record in company_records:
            year = year_of(record)
            if year:
                by_year[year].append(record)
        years = sorted(by_year)

        combos = rng.sample(
            [(year, metric) for year in years for metric in MISSING_METRICS],
            min(2, len(years) * len(MISSING_METRICS)),
        )
        for year, (metric, _) in combos:
            add(
                {
                    "stock": stock,
                    "instruction": (
                        f"根据材料回答：{company}{year}年{metric}是多少？"
                    ),
                    "input": by_year[year][0]["input"],
                    "output": (
                        f"材料未提及{metric}，我无法从现有材料中给出该数据。"
                    ),
                    "evidence": [],
                    "gold": [],
                    "category": "未提及陷阱",
                }
            )

        if len(years) >= 2:
            material_year = rng.choice(years)
            target_year = rng.choice([y for y in years if y != material_year])
            add(
                {
                    "stock": stock,
                    "instruction": (
                        f"根据材料回答：{company}{target_year}年营业收入是多少？"
                    ),
                    "input": by_year[material_year][0]["input"],
                    "output": (
                        f"材料未提及{target_year}年营业收入，"
                        "我无法从现有材料中给出该数据。"
                    ),
                    "evidence": [],
                    "gold": [],
                    "category": "年份陷阱",
                }
            )

        if len(years) >= 2:
            latest, previous = years[-1], years[-2]
            revenue = {}
            for year in (latest, previous):
                for record in by_year[year]:
                    if record["id"].endswith("-revenue"):
                        revenue[year] = record
            if len(revenue) == 2:
                gold_latest = parse_yi(revenue[latest]["gold"][0])
                gold_previous = parse_yi(revenue[previous]["gold"][0])
                if gold_latest is not None and gold_previous is not None:
                    delta = gold_latest - gold_previous
                    if delta >= 0:
                        delta_text = f"增加{delta:.2f}亿元"
                    else:
                        delta_text = f"减少{abs(delta):.2f}亿元"
                    add(
                        {
                            "stock": stock,
                            "instruction": (
                                f"根据材料计算：{company}{latest}年营业收入比"
                                f"{previous}年多多少亿元？"
                            ),
                            "input": (
                                revenue[latest]["input"]
                                + "\n"
                                + revenue[previous]["input"]
                            ),
                            "output": (
                                f"根据材料，{company}{latest}年营业收入为"
                                f"{revenue[latest]['gold'][0]}，{previous}年为"
                                f"{revenue[previous]['gold'][0]}，"
                                f"{delta_text}。"
                            ),
                            "evidence": [
                                revenue[latest]["input"],
                                revenue[previous]["input"],
                            ],
                            "gold": [f"{abs(delta):.2f}亿元"],
                            "category": "计算",
                        }
                    )

    for index in range(0, len(stock_list) - 1, 2):
        stock_a, stock_b = stock_list[index], stock_list[index + 1]
        record_a = next(
            (r for r in by_stock[stock_a] if r.get("gold")),
            None,
        )
        record_b = next(
            (r for r in by_stock[stock_b] if r.get("gold")),
            None,
        )
        if record_a is None or record_b is None:
            continue
        company_a = company_of(record_a)
        metric = re.search(
            r"回答：.+?年(.+?)是多少",
            record_a["instruction"],
        )
        metric_text = metric.group(1) if metric else "营业收入"
        year_a = year_of(record_a) or "2024"
        add(
            {
                "stock": stock_a,
                "instruction": (
                    f"根据材料回答：{company_a}{year_a}年{metric_text}是多少？"
                ),
                "input": record_a["input"] + "\n" + record_b["input"],
                "output": record_a["output"],
                "evidence": [record_a["input"]],
                "gold": record_a["gold"],
                "category": "干扰项",
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in out:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts = defaultdict(int)
    for item in out:
        counts[item["category"]] += 1
    print(f"wrote {out_path} ({len(out)} records)")
    print(dict(counts))


if __name__ == "__main__":
    main()
