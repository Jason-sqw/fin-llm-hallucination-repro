"""Generate grounded answers and measure hallucination rate.

Judge modes:
  local: reuse the loaded HF model as the fact checker
  api:   call any OpenAI-compatible /chat/completions endpoint
  none:  only report numeric gold hits and refusal behavior
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat_utils import ensure_chat_template

SYSTEM_PROMPT = (
    "你是一位严谨的金融分析师。回答必须严格基于给定的材料；"
    "材料没有提到的事情要明确说明'材料未提及'，不要编造数据或结论。"
)
JUDGE_SYSTEM = (
    "你是一个严格的金融事实核验员。请逐条核对【回答】中的事实性表述"
    "是否被【材料】支持。数值、主体、年份必须与材料一致；"
    "材料未提及但回答给出具体数值或结论的，判定为 unsupported；"
    "回答明确说明'材料未提及'且材料确实未提及的，不算幻觉。"
    "只输出 JSON，格式："
    '{"claims": [{"claim": "...", "verdict": "supported|unsupported|no_info"}]}'
)

NUM_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|亿元|万元|亿|万|元|倍)?")
REFUSAL_PATTERNS = [
    "材料未提及",
    "年报未提及",
    "未披露",
    "没有提到",
    "无法从",
    "无法回答",
    "不知道",
]


def build_user_text(record):
    text = record["instruction"].strip()
    material = (record.get("input") or "").strip()
    if material:
        text += "\n\n【材料】\n" + material
    return text


def load_model(base_model, adapter, load_in_4bit):
    torch_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if torch.cuda.is_available() else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        adapter or base_model, trust_remote_code=True
    )
    ensure_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, record, max_new_tokens, temperature):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_text(record)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def extract_numbers(text):
    return [re.sub(r"\s+", "", m.group(0)) for m in NUM_RE.finditer(text)]


def gold_numeric_hit(answer, gold):
    if not gold:
        return None
    answer_nums = extract_numbers(answer)
    return all(g in answer_nums for g in gold)


def is_refusal(answer):
    return any(pattern in answer for pattern in REFUSAL_PATTERNS)


def build_judge_prompt(record, answer):
    return (
        "【材料】\n"
        + (record.get("input") or "").strip()
        + "\n\n【回答】\n"
        + answer
    )


def parse_judge_response(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def local_judge(model, tokenizer, record, answer, max_new_tokens):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_judge_prompt(record, answer)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return parse_judge_response(text)


def api_judge(record, answer, judge_model, api_base, api_key):
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_judge_prompt(record, answer)},
        ],
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    return parse_judge_response(body["choices"][0]["message"]["content"])


def summarize(results, judged):
    total = len(results)
    hallucinated = sum(1 for r in judged if r["unsupported_claims"] > 0)
    unsupported_total = sum(r["unsupported_claims"] for r in judged)
    claims_total = sum(
        r["supported_claims"] + r["unsupported_claims"] + r["no_info_claims"]
        for r in judged
    )
    gold_hits = sum(1 for r in results if r["gold_hit"] is True)
    gold_total = sum(1 for r in results if r["gold_hit"] is not None)
    refusals = sum(1 for r in results if r["refusal"])
    by_category = {}
    for result in results:
        category = result.get("category") or "other"
        entry = by_category.setdefault(
            category,
            {"samples": 0, "gold_hits": 0, "gold_total": 0, "refusals": 0},
        )
        entry["samples"] += 1
        if result["gold_hit"] is True:
            entry["gold_hits"] += 1
        if result["gold_hit"] is not None:
            entry["gold_total"] += 1
        if result["refusal"]:
            entry["refusals"] += 1
    category_summary = {}
    for category, entry in by_category.items():
        category_summary[category] = {
            "samples": entry["samples"],
            "gold_hit_rate": (
                round(entry["gold_hits"] / entry["gold_total"], 4)
                if entry["gold_total"]
                else None
            ),
            "refusal_rate": round(entry["refusals"] / entry["samples"], 4),
        }

    summary = {
        "samples": total,
        "judged_samples": len(judged),
        "hallucination_rate": (
            round(hallucinated / len(judged), 4) if judged else None
        ),
        "unsupported_claim_rate": (
            round(unsupported_total / claims_total, 4) if claims_total else None
        ),
        "gold_hit_rate": round(gold_hits / gold_total, 4) if gold_total else None,
        "refusal_rate": round(refusals / total, 4) if total else None,
        "by_category": category_summary,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/sample_qa.jsonl")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--judge", choices=["local", "api", "none"], default="local")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--judge-api-base", default="https://api.openai.com/v1")
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/eval")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.eval_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        records = records[: args.limit]
    model, tokenizer = load_model(
        args.base_model, args.adapter, load_in_4bit=True
    )

    results = []
    judged = []
    for record in records:
        answer = generate(
            model, tokenizer, record, args.max_new_tokens, args.temperature
        )
        result = {
            "id": record.get("id", ""),
            "category": record.get("category", ""),
            "question": record["instruction"],
            "gold": record.get("gold", []),
            "answer": answer,
            "refusal": is_refusal(answer),
            "gold_hit": gold_numeric_hit(answer, record.get("gold", [])),
        }

        verdict = None
        if args.judge == "local":
            verdict = local_judge(model, tokenizer, record, answer, 512)
        elif args.judge == "api":
            verdict = api_judge(
                record, answer, args.judge_model, args.judge_api_base,
                args.judge_api_key,
            )

        if verdict and "claims" in verdict:
            claims = verdict["claims"]
            result["claims"] = claims
            result["supported_claims"] = sum(
                1 for c in claims if c.get("verdict") == "supported"
            )
            result["unsupported_claims"] = sum(
                1 for c in claims if c.get("verdict") == "unsupported"
            )
            result["no_info_claims"] = sum(
                1 for c in claims if c.get("verdict") == "no_info"
            )
            judged.append(result)
        results.append(result)
        print(f"[{result['id']}] gold_hit={result['gold_hit']} "
              f"refusal={result['refusal']} answer={answer[:80]!r}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results),
        encoding="utf-8",
    )
    summary = summarize(results, judged)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
