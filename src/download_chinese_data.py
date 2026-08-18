"""Download open Chinese financial datasets into data/raw."""

import argparse
import json
import itertools
from pathlib import Path

KNOWN_DATASETS = {
    "fineval": {
        "id": "SUFE-AIFLM-Lab/FinEval",
        "kind": "multiple_choice",
        "archive_file": "FinEval.zip",
    },
    "fingpt-corpus": {
        "id": "AI4Finance-Foundation/FinGPT-FinCorpus",
        "kind": "corpus",
    },
    "bbt-corpus": {
        "id": "BAAI/BBT-FinCorpus",
        "kind": "corpus",
    },
}


def normalize_row(row, kind):
    if kind == "corpus":
        text = (
            row.get("text")
            or row.get("content")
            or row.get("document")
            or ""
        )
        source = (
            row.get("source")
            or row.get("source_file")
            or row.get("source_name")
            or ""
        )
        return {"text": text, "source": source}
    return dict(row)


def load_all(dataset_id, config, split):
    from datasets import load_dataset

    if config:
        return load_dataset(dataset_id, config, split=split)
    try:
        return load_dataset(dataset_id, split=split)
    except Exception:
        from datasets import get_dataset_config_names

        configs = get_dataset_config_names(dataset_id)
        rows = []
        for name in configs:
            rows.extend(list(load_dataset(dataset_id, name, split=split)))
        return rows


def iter_dataset(dataset_id, kind, config, split, limit):
    from datasets import load_dataset

    if kind == "corpus":
        ds = load_dataset(dataset_id, split=split, streaming=True)
    else:
        ds = load_all(dataset_id, config, split)
    if limit and limit > 0:
        return itertools.islice(ds, limit)
    return ds


def save_archive_as_jsonl(dataset_id, archive_file, out_path):
    """Download a repo zip and merge every JSON/JSONL file into one JSONL."""
    import tempfile
    import zipfile

    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        dataset_id, archive_file, repo_type="dataset"
    )
    count = 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_dir)
        with out_path.open("w", encoding="utf-8") as handle:
            for path in sorted(Path(tmp_dir).rglob("*.json*")):
                if path.suffix not in (".json", ".jsonl"):
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    handle.write(
                        json.dumps(dict(row), ensure_ascii=False) + "\n"
                    )
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(KNOWN_DATASETS) + ["custom"],
        default="fineval",
    )
    parser.add_argument("--hf-dataset", default=None, help="自定义 HF 数据集 ID")
    parser.add_argument("--config", default=None, help="数据集子配置名")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="最多保存多少行，0 表示全部")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    if args.dataset == "custom" and not args.hf_dataset:
        parser.error("--dataset custom 时需要提供 --hf-dataset")
    spec = KNOWN_DATASETS.get(args.dataset)
    dataset_id = spec["id"] if spec else args.hf_dataset
    kind = spec["kind"] if spec else "corpus"

    if kind == "corpus" and args.limit == 0:
        print("警告：语料数据集非常大，建议使用 --limit（例如 --limit 50000）")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.jsonl"

    try:
        if spec and spec.get("archive_file"):
            count = save_archive_as_jsonl(dataset_id, spec["archive_file"], out_path)
        else:
            rows = iter_dataset(dataset_id, kind, args.config, args.split, args.limit)
            count = 0
            with out_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(normalize_row(row, kind), ensure_ascii=False)
                        + "\n"
                    )
                    count += 1
    except ImportError as exc:
        raise SystemExit(
            "缺少 huggingface_hub/datasets 库，请先运行："
            "pip install -r requirements-cpu.txt"
        ) from exc
    except Exception as exc:
        raise SystemExit(
            f"下载 {dataset_id} 失败（{exc}），请确认网络；"
            "国内可使用镜像：$env:HF_ENDPOINT = \"https://hf-mirror.com\""
        ) from exc
    print(f"wrote {out_path} ({count} rows)")


if __name__ == "__main__":
    main()
