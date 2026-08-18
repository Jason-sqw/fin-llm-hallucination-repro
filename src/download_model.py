"""Download a model from ModelScope into a local directory."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="AI-ModelScope/TinyLlama-1.1B-Chat-v1.0",
        help="ModelScope 模型 ID",
    )
    parser.add_argument("--output-dir", default="models/tinyllama-1.1b-chat")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} 已存在文件，跳过下载（加 --overwrite 可重下）")

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise SystemExit("缺少 modelscope，请先运行：pip install modelscope") from exc

    path = snapshot_download(args.model, local_dir=str(out_dir))
    print(f"model saved to {path}")


if __name__ == "__main__":
    main()
