"""Convert raw grounded QA JSONL into chat-format train/val splits."""

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "你是一位严谨的金融分析师。回答必须严格基于给定的材料；"
    "材料没有提到的事情要明确说明'材料未提及'，不要编造数据或结论。"
)


def to_messages(record):
    user_text = record["instruction"].strip()
    material = (record.get("input") or "").strip()
    if material:
        user_text += "\n\n【材料】\n" + material
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": record["output"].strip()},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_qa.jsonl")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-examples", type=int, default=0, help="只用前 N 条（0=全部）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_examples > 0:
        records = records[: args.max_examples]
    random.Random(args.seed).shuffle(records)

    val_n = max(1, int(len(records) * args.val_ratio))
    splits = {
        "train": records[val_n:],
        "val": records[:val_n],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        converted = [
            {"id": item.get("id", f"{name}-{i}"), "messages": to_messages(item)}
            for i, item in enumerate(items)
        ]
        path = out_dir / f"{name}.json"
        path.write_text(
            json.dumps(converted, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {path} ({len(converted)} examples)")


if __name__ == "__main__":
    main()
