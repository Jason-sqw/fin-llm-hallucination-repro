"""Convert downloaded Chinese datasets into the lab's QA JSONL schema."""

import argparse
import json
import re
from pathlib import Path


def letter_to_index(answer):
    if answer is None:
        return None
    text = str(answer).strip().upper()
    if re.fullmatch(r"[A-D]", text):
        return ord(text) - ord("A")
    try:
        index = int(text)
        return index if 0 <= index < 26 else None
    except ValueError:
        return None


def convert_fineval(row, index):
    question = str(row.get("question") or "").strip()
    choices = row.get("choices") or []
    answer = row.get("answer")
    explanation = str(row.get("explanation") or "").strip()
    if not question or not choices or answer is None:
        return None

    choice_lines = "\n".join(
        f"{chr(65 + i)}. {str(choice).strip()}"
        for i, choice in enumerate(choices)
    )
    answer_index = letter_to_index(answer)
    if answer_index is None or answer_index >= len(choices):
        output = str(answer).strip()
    else:
        output = f"{chr(65 + answer_index)}. {str(choices[answer_index]).strip()}"
    if explanation:
        output += f"\n解析：{explanation}"

    return {
        "id": f"fineval-{index:05d}",
        "instruction": (
            "请根据金融专业知识，从下列选项中选择正确答案。\n"
            f"问题：{question}\n{choice_lines}"
        ),
        "input": "",
        "output": output,
        "evidence": [],
        "gold": [str(answer).strip()],
        "category": row.get("category") or row.get("source") or "fineval",
    }


def convert_alpaca(row, index):
    instruction = str(row.get("instruction") or "").strip()
    input_text = str(row.get("input") or "").strip()
    output = str(row.get("output") or "").strip()
    if not instruction or not output:
        return None
    return {
        "id": row.get("id") or f"alpaca-{index:05d}",
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "evidence": row.get("evidence") or [],
        "gold": row.get("gold") or [],
        "category": row.get("category") or "alpaca",
    }


CONVERTERS = {
    "fineval": convert_fineval,
    "alpaca": convert_alpaca,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--kind", choices=list(CONVERTERS), default="fineval")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    converter = CONVERTERS[args.kind]
    converted = []
    for index, record in enumerate(records):
        item = converter(record, index)
        if item is not None:
            converted.append(item)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in converted:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({len(converted)} converted)")


if __name__ == "__main__":
    main()
