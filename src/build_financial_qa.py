"""Build grounded financial QA from public reports or structured data.

Two modes:
  template: pull real financial indicators with AkShare and generate
            deterministic, evidence-grounded QA pairs (no API key needed)
  api:      chunk annual-report text and ask an OpenAI-compatible model
            (DashScope Qwen, OpenAI, etc.) to write QA pairs
"""

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path

API_SYSTEM_PROMPT = (
    "你是一名金融数据标注员。请根据【材料】生成不超过 4 个中文问答题。"
    "要求：1) 问题必须能被材料直接回答；2) 答案只使用材料中的信息，不得编造；"
    "3) 答案包含数字时，把关键数字提取到 gold 数组；"
    "4) 材料没有的信息不要提问。"
    '只输出 JSON 数组，格式：[{"instruction": "...", "output": "...", '
    '"gold": ["..."], "evidence": ["材料原句"]}]'
)


def parse_num(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if not math.isnan(value) else None
    text = str(value).strip().replace(",", "").replace("%", "").replace("--", "")
    if not text or text.lower() in ("none", "nan", "-"):
        return None
    match = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(亿|万元|万|元)?", text)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2) or ""
    if unit in ("万", "万元"):
        number = number / 10000
    return number


def to_yi(number):
    """Normalize absolute amounts to 亿元."""
    if number is None:
        return None
    if abs(number) >= 1e9:
        return number / 1e8
    if abs(number) >= 1e6:
        return number / 1e4
    return number


def to_percent(number):
    if number is None:
        return None
    return number * 100 if abs(number) < 1 else number


def fetch_financial_data(stock):
    import akshare as ak

    errors = []
    try:
        df = ak.stock_financial_abstract_ths(symbol=stock, indicator="按报告期")
        return df, "ths"
    except Exception as exc:
        errors.append(f"同花顺: {exc}")
    try:
        df = ak.stock_financial_analysis_indicator(symbol=stock)
        return df, "eastmoney"
    except Exception as exc:
        errors.append(f"东财: {exc}")
    raise SystemExit("AkShare 获取财务数据失败：" + "; ".join(errors))


def auto_pairs(count):
    import akshare as ak

    errors = []
    try:
        spot = ak.stock_zh_a_spot_em().sort_values("总市值", ascending=False)
        return list(zip(spot["代码"].head(count), spot["名称"].head(count)))
    except Exception as exc:
        errors.append(f"东财行情: {exc}")
    try:
        cons = ak.index_stock_cons_csindex(symbol="000300")
        return list(
            zip(
                cons["成分券代码"].head(count),
                cons["成分券名称"].head(count),
            )
        )
    except Exception as exc:
        errors.append(f"沪深300: {exc}")
    try:
        codes = ak.stock_info_a_code_name()
        return list(zip(codes["code"].head(count), codes["name"].head(count)))
    except Exception as exc:
        errors.append(f"交易所代码表: {exc}")
    raise SystemExit("获取股票列表失败：" + "; ".join(errors))


def pick_year(value):
    text = str(value)
    match = re.search(r"(20\d{2})", text)
    return match.group(1) if match else None


def build_ths_metrics(df, years):
    data = {}
    for _, row in df.iterrows():
        year = pick_year(row.get("报告期") or "")
        if year not in years:
            continue
        data[year] = {
            "revenue": parse_num(row.get("营业总收入")),
            "revenue_yoy": parse_num(row.get("营业总收入同比增长率")),
            "profit": parse_num(row.get("净利润")),
            "profit_yoy": parse_num(row.get("净利润同比增长率")),
            "gross_margin": parse_num(row.get("销售毛利率")),
            "net_margin": parse_num(row.get("销售净利率")),
            "roe": parse_num(row.get("净资产收益率")),
            "eps": parse_num(row.get("基本每股收益")),
        }
    return data


def build_eastmoney_metrics(df, years):
    data = {}
    for _, row in df.iterrows():
        year = pick_year(row.get("日期") or row.get("报告期") or "")
        if year not in years:
            continue
        data[year] = {
            "revenue": to_yi(parse_num(row.get("营业收入"))),
            "revenue_yoy": to_percent(parse_num(row.get("营业收入同比增长率"))),
            "profit": to_yi(parse_num(row.get("净利润"))),
            "profit_yoy": to_percent(parse_num(row.get("净利润同比增长率"))),
            "gross_margin": to_percent(parse_num(row.get("销售毛利率"))),
            "net_margin": to_percent(parse_num(row.get("销售净利率"))),
            "roe": to_percent(parse_num(row.get("净资产收益率"))),
            "eps": parse_num(row.get("每股收益")),
        }
    return data


def year_lines(company, data):
    lines = {}
    for year in sorted(data):
        m = data[year]
        parts = []
        if m["revenue"] is not None:
            parts.append(f"营业收入 {m['revenue']:.2f}亿元")
        if m["revenue_yoy"] is not None:
            parts.append(f"同比增长 {m['revenue_yoy']:.2f}%")
        if m["profit"] is not None:
            parts.append(f"净利润 {m['profit']:.2f}亿元")
        if m["profit_yoy"] is not None:
            parts.append(f"同比增长 {m['profit_yoy']:.2f}%")
        if m["gross_margin"] is not None:
            parts.append(f"毛利率 {m['gross_margin']:.2f}%")
        if m["net_margin"] is not None:
            parts.append(f"净利率 {m['net_margin']:.2f}%")
        if m["roe"] is not None:
            parts.append(f"净资产收益率 {m['roe']:.2f}%")
        if m["eps"] is not None:
            parts.append(f"每股收益 {m['eps']:.2f}元")
        lines[year] = (
            f"{company}公开财务数据：{year}年：" + "；".join(parts) + "。"
        )
    return lines


def material_text(company, data):
    header = f"{company}公开财务数据（单位：亿元，来源：公开年报/财务摘要）："
    return "\n".join([header] + list(year_lines(company, data).values()))


def fmt_yi(value):
    return None if value is None else f"{value:.2f}亿元"


def fmt_pct(value):
    return None if value is None else f"{value:.2f}%"


def make_template_records(company, data, lines=None):
    lines = lines or year_lines(company, data)
    records = []
    for year in sorted(data):
        m = data[year]
        evidence = lines.get(year, f"{company}公开财务数据：{year}年相关指标")
        if m["revenue"] is not None:
            records.append(
                {
                    "id": f"{year}-revenue",
                    "instruction": f"根据材料回答：{company}{year}年营业收入是多少？",
                    "output": f"根据材料，{company}{year}年营业收入为{fmt_yi(m['revenue'])}。",
                    "evidence": [evidence],
                    "gold": [fmt_yi(m["revenue"])],
                    "category": "营收",
                }
            )
        if m["revenue_yoy"] is not None:
            records.append(
                {
                    "id": f"{year}-revenue-yoy",
                    "instruction": f"根据材料回答：{company}{year}年营业收入同比增长多少？",
                    "output": (
                        f"根据材料，{company}{year}年营业收入同比增长"
                        f"{fmt_pct(m['revenue_yoy'])}。"
                    ),
                    "evidence": [evidence],
                    "gold": [fmt_pct(m["revenue_yoy"])],
                    "category": "增长",
                }
            )
        if m["profit_yoy"] is not None:
            records.append(
                {
                    "id": f"{year}-profit-yoy",
                    "instruction": f"根据材料回答：{company}{year}年净利润同比增长多少？",
                    "output": (
                        f"根据材料，{company}{year}年净利润同比增长"
                        f"{fmt_pct(m['profit_yoy'])}。"
                    ),
                    "evidence": [evidence],
                    "gold": [fmt_pct(m["profit_yoy"])],
                    "category": "增长",
                }
            )
        if m["profit"] is not None:
            records.append(
                {
                    "id": f"{year}-profit",
                    "instruction": f"根据材料回答：{company}{year}年净利润是多少？",
                    "output": f"根据材料，{company}{year}年净利润为{fmt_yi(m['profit'])}。",
                    "evidence": [evidence],
                    "gold": [fmt_yi(m["profit"])],
                    "category": "盈利",
                }
            )
        if m["gross_margin"] is not None:
            records.append(
                {
                    "id": f"{year}-margin",
                    "instruction": f"根据材料回答：{company}{year}年毛利率是多少？",
                    "output": (
                        f"根据材料，{company}{year}年毛利率为"
                        f"{fmt_pct(m['gross_margin'])}。"
                    ),
                    "evidence": [evidence],
                    "gold": [fmt_pct(m["gross_margin"])],
                    "category": "盈利",
                }
            )
        if m["net_margin"] is not None:
            records.append(
                {
                    "id": f"{year}-net-margin",
                    "instruction": f"根据材料回答：{company}{year}年净利率是多少？",
                    "output": (
                        f"根据材料，{company}{year}年净利率为"
                        f"{fmt_pct(m['net_margin'])}。"
                    ),
                    "evidence": [evidence],
                    "gold": [fmt_pct(m["net_margin"])],
                    "category": "盈利",
                }
            )
        elif m["profit"] is not None and m["revenue"]:
            margin = m["profit"] / m["revenue"] * 100
            records.append(
                {
                    "id": f"{year}-net-margin",
                    "instruction": f"根据材料计算：{company}{year}年净利率是多少？",
                    "output": (
                        f"根据材料计算，{company}{year}年净利率为"
                        f"{fmt_pct(margin)}。"
                    ),
                    "evidence": [evidence],
                    "gold": [fmt_pct(margin)],
                    "category": "计算",
                }
            )
        if m["roe"] is not None:
            records.append(
                {
                    "id": f"{year}-roe",
                    "instruction": f"根据材料回答：{company}{year}年净资产收益率是多少？",
                    "output": (
                        f"根据材料，{company}{year}年净资产收益率为"
                        f"{fmt_pct(m['roe'])}。"
                    ),
                    "evidence": [evidence],
                    "gold": [fmt_pct(m["roe"])],
                    "category": "盈利",
                }
            )
        if m["eps"] is not None:
            records.append(
                {
                    "id": f"{year}-eps",
                    "instruction": f"根据材料回答：{company}{year}年基本每股收益是多少？",
                    "output": (
                        f"根据材料，{company}{year}年基本每股收益为"
                        f"{m['eps']:.2f}元。"
                    ),
                    "evidence": [evidence],
                    "gold": [f"{m['eps']:.2f}元"],
                    "category": "盈利",
                }
            )

    years = sorted(data)
    if len(years) >= 2:
        latest, prev = years[-1], years[-2]
        if data[latest]["revenue"] is not None and data[prev]["revenue"] is not None:
            delta = data[latest]["revenue"] - data[prev]["revenue"]
            records.append(
                {
                    "id": "revenue-delta",
                    "instruction": (
                        f"根据材料回答：{company}{latest}年营业收入比{prev}年"
                        "多多少亿元？"
                    ),
                    "output": (
                        f"根据材料，{company}{latest}年营业收入"
                        f"{fmt_yi(data[latest]['revenue'])}，{prev}年为"
                        f"{fmt_yi(data[prev]['revenue'])}，增加{fmt_yi(delta)}。"
                    ),
                    "evidence": [lines.get(latest, evidence)],
                    "gold": [fmt_yi(delta)],
                    "category": "计算",
                }
            )

    records.append(
        {
            "id": "refusal-salary",
            "instruction": f"根据材料回答：{company}研发人员平均薪酬是多少？",
            "output": "材料未提及研发人员平均薪酬，我无法从现有材料中给出该数据。",
            "evidence": [],
            "gold": [],
            "category": "拒答",
        }
    )
    return records


def extract_text(path):
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text, size=1400, overlap=150):
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) > size and current:
            chunks.append(current)
            current = current[-overlap:]
        current += paragraph + "\n"
    if current:
        chunks.append(current)
    return chunks


def api_generate(material, model, api_base, api_key):
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": API_SYSTEM_PROMPT + "\n\n【材料】\n" + material}
        ],
        "temperature": 0.3,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    match = re.search(r"\[.*\]", content, re.S)
    if not match:
        raise ValueError(f"模型未返回 JSON 数组：{content[:200]}")
    return json.loads(match.group(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["template", "api"], default="template")
    parser.add_argument("--stock", default="600519")
    parser.add_argument("--company", default="贵州茅台")
    parser.add_argument(
        "--stocks",
        default=None,
        help="批量模式：600519=贵州茅台,000858=五粮液",
    )
    parser.add_argument(
        "--auto",
        type=int,
        default=0,
        help="自动取 A 股市值最大的 N 家公司批量生成",
    )
    parser.add_argument("--years", default="2020,2021,2022,2023,2024")
    parser.add_argument("--source-dir", help="API 模式：财报 txt/pdf 文件夹")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--api-base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", default="data/generated/financial_qa.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    if args.mode == "template":
        years = set(args.years.split(","))
        if args.auto:
            pairs = auto_pairs(args.auto)
            print(f"自动选取 A 股市值最大的 {len(pairs)} 家公司")
        elif args.stocks:
            pairs = [tuple(pair.split("=")) for pair in args.stocks.split(",")]
        else:
            pairs = [(args.stock, args.company)]
        failures = []
        for stock, company in pairs:
            try:
                df, source = fetch_financial_data(stock)
                data = (
                    build_ths_metrics(df, years)
                    if source == "ths"
                    else build_eastmoney_metrics(df, years)
                )
                if not data:
                    raise ValueError(f"没有取到年份 {args.years} 的财务数据")
                lines = year_lines(company, data)
                material = material_text(company, data)
                batch = make_template_records(company, data, lines)
                for record in batch:
                    if record["id"] == "revenue-delta":
                        years_sorted = sorted(data)
                        record["input"] = (
                            lines[years_sorted[-1]] + "\n" + lines[years_sorted[-2]]
                        )
                    elif record["id"] == "refusal-salary":
                        record["input"] = lines[sorted(data)[-1]]
                    else:
                        record["input"] = lines[record["id"][:4]]
                    record["id"] = f"{stock}-{record['id']}"
                records.extend(batch)
                print(
                    f"{company}({stock}): {len(batch)} 条, "
                    f"来源 {source}, 年份 {sorted(data)}"
                )
            except Exception as exc:
                failures.append(f"{stock}({company}): {exc}")
                print(f"跳过 {stock}({company}): {exc}")
        if failures:
            print("失败:", *failures, sep="\n")
    else:
        source_dir = Path(args.source_dir)
        if not source_dir.exists():
            raise SystemExit(f"找不到材料目录：{source_dir}")
        files = sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in (".txt", ".md", ".pdf"))
        for path in files:
            for chunk in chunk_text(extract_text(path)):
                items = api_generate(chunk, args.model, args.api_base, args.api_key)
                for item in items:
                    records.append(
                        {
                            "id": f"{path.stem}-{len(records)}",
                            "instruction": item.get("instruction", ""),
                            "input": chunk,
                            "output": item.get("output", ""),
                            "evidence": item.get("evidence", [chunk]),
                            "gold": item.get("gold", []),
                            "category": item.get("category", "api"),
                        }
                    )
        if not files:
            raise SystemExit(f"{source_dir} 里没有 txt/md/pdf 文件")

    if args.limit > 0:
        records = records[: args.limit]
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({len(records)} records)")


if __name__ == "__main__":
    main()
