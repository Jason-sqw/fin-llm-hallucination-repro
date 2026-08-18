# 金融大模型微调降低幻觉复现实验

这是一个"金融领域指令微调 + 幻觉率评估"的可复现参考工程。它复现的不是某一个固定论文仓库，而是这类项目共同的核心链路：

1. 构造"问题 + 证据材料 + 有据答案"的金融 QA 数据（含"材料未提及"的拒答样本）
2. 用 QLoRA/LoRA 对开源基座模型做指令微调
3. 用逐条事实核验（LLM judge 或 NLI）计算幻觉率
4. 做 Base vs SFT vs SFT+RAG 的对照实验

如果你想复现某篇具体论文（例如 FinGPT、FinTral、XuanYuan、FinanceBench 或某篇 arXiv），把论文标题或仓库链接发过来，我可以把数据源和超参数对齐到那篇论文。

## 幻觉在这个项目里怎么定义

评估时把"幻觉"定义成可操作的几条：

- 数值幻觉：答案中的金额、增速、比例等数字与材料不一致
- 主体/时间错配：把 A 公司的数据说成 B 公司，或张冠李戴到错误年份
- 编造信息：材料完全没有提及，却给出具体数据或结论
- 过度自信：材料不足以回答时没有说"材料未提及"，而是强行给出答案

## 目录结构

```text
fin-llm-hallucination-lab/
├── README.md              # 本指南
├── requirements.txt
├── configs/
│   └── train.yaml         # QLoRA 微调配置
├── data/
│   ├── sample_qa.jsonl    # 演示数据：问题+材料+答案+证据+gold
│   └── processed/         # 数据预处理输出（自动生成）
├── src/
│   ├── data_prep.py       # JSONL -> 对话格式 -> train/val
│   ├── train.py           # LoRA/QLoRA 指令微调
│   └── evaluate.py        # 生成答案 + 幻觉率评估
└── outputs/               # 模型与评估结果（自动生成）
```

## 环境要求

本机是 Windows + Python 3.13，没有 NVIDIA GPU，无法直接跑 7B 微调。推荐下面任意一种环境：

- Linux 或 WSL2 + NVIDIA 显卡（12GB 显存可跑 QLoRA 7B，24GB 更宽裕）
- AutoDL、RunPod、Lambda 等云 GPU 实例
- Google Colab（选 T4/A100 运行时）

Python 建议 3.10/3.11。`bitsandbytes` 在原生 Windows 上支持较差，训练请优先 Linux/WSL。

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 数据准备

工程自带的 `data/sample_qa.jsonl` 只用于跑通流程，字段如下：

```json
{
  "id": "sample-001",
  "instruction": "根据材料回答：示例科技2024年营业收入是多少？",
  "input": "【示例科技2024年年报】2024年公司实现营业收入128.6亿元……",
  "output": "根据年报，示例科技2024年营业收入为128.6亿元。",
  "evidence": ["2024年营业收入128.6亿元"],
  "gold": ["128.6亿元"]
}
```

真正复现时替换成金融数据，推荐来源：

- FinGPT 系列数据集：`AI4Finance-Foundation` 下的 FinGPT 指令数据
- FinTral 论文附带数据（金融多任务 + 幻觉评估集）
- 自己构造：从财报、公告、研报抽取句子生成 QA，自动生成"问题 + 材料 + 答案"
- 拒答样本：把"材料里没有的信息"做成问题，答案固定为"材料未提及"

论文级实验建议至少 5k-20k 条高质量指令，且保证每条答案有可回溯的证据文本；数据里最好混入 5%-10% 的拒答样本。

```bash
python -m src.data_prep --data data/sample_qa.jsonl --out-dir data/processed --val-ratio 0.1
```

### 中文数据：下载与转换

工程内置三个开源中文数据集的下载入口，数据落到 `data/raw/`：

| 数据 | HF 仓库 | 用途 |
| --- | --- | --- |
| FinEval | `SUFE-AIFLM-Lab/FinEval` | 中文金融知识题，4661 题，可转训练或评测 |
| FinGPT-FinCorpus | `AI4Finance-Foundation/FinGPT-FinCorpus` | 中英金融语料，原始文本 |
| BBT-FinCorpus | `BAAI/BBT-FinCorpus` | 中文金融语料，原始文本 |

```bash
# 1. 下载 FinEval 并转成工程 QA 格式
python -m src.download_chinese_data --dataset fineval --output-dir data/raw
python -m src.convert_chinese --input data/raw/fineval.jsonl --kind fineval --out data/fineval_qa.jsonl

# 2. 下载中文语料（原始文本，先取一部分）
python -m src.download_chinese_data --dataset bbt-corpus --output-dir data/raw --limit 50000
python -m src.download_chinese_data --dataset fingpt-corpus --output-dir data/raw --limit 50000
```

语料不是现成的问答对，下一步需要用 LLM 从材料生成"问题 + 有据答案"；
FinEval 转换后是知识题（不带证据材料），适合先验证训练流程和评测准确率。
另有 FinBen（`SUFE-AIFLM/FinBen`，中文金融综合基准，含幻觉评测子集）和
DISC-FinLLM（`FudanDISC/DISC-FinLLM`，中文金融指令数据）可以直接从各自仓库获取。
注意 FinGPT 系列多为 CC BY-NC-SA 许可，非商用实验没问题，商用前要确认。

## 微调

默认配置是 `Qwen/Qwen2.5-7B-Instruct` + QLoRA（4-bit + LoRA r=16），可改 `configs/train.yaml` 换成其他基座和超参：

```bash
python -m src.train --config configs/train.yaml
```

训练完成后，`outputs/fin-lora` 里是 LoRA adapter，合并回完整权重时用：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(model, "outputs/fin-lora")
merged = model.merge_and_unload()
merged.save_pretrained("outputs/fin-lora-merged")
```

### 没有 GPU：TinyLlama CPU 路线

工程自带 `configs/train-cpu-tinyllama.yaml`，用
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` + LoRA 在纯 CPU 上跑通整个流程：

```bash
python -m src.data_prep --data data/sample_qa.jsonl --out-dir data/processed --val-ratio 0.2
python -m src.train --config configs/train-cpu-tinyllama.yaml
```

说明：

- 该配置关闭 4-bit 和 fp16/bf16，自动用 float32；`max_seq_len` 降到 1024，batch size 为 1
- 快速验证用 `configs/train-cpu-smoke.yaml`（1 epoch、几十条以内、seq 512），
  几十秒到几分钟即可跑完
- 1B 模型 CPU 训练可行但慢，适合验证数据、脚本和"微调后幻觉率相对下降"的结论
- 若 tokenizer 没有内置 chat template，脚本会自动套用一个 Zephyr 风格模板，TinyLlama 正常加载即可
- 评估 judge 建议用 API 模式（1B 模型自评不可靠），或者人工抽检
- 想跑正式效果时，把同一份数据放到 AutoDL/Colab 上用 `configs/train.yaml` 训 7B

国内如果 Hugging Face 直连或镜像不稳定，模型改用 ModelScope 下载：

```bash
pip install modelscope
python -m src.download_model
```

下载完成后，把 `configs/train-cpu-tinyllama.yaml` 里的
`base_model` 改成 `models/tinyllama-1.1b-chat`（本地路径），
之后训练和评估都不再走网络。数据可以先直接用工程自带的
`data/sample_qa.jsonl` 跑通全流程，再替换成自己的中文金融数据。

## 幻觉评估

评估脚本会：

1. 对测试集逐条生成答案（温度 0，保证可复现）
2. 检查答案是否命中 gold 数字
3. 用 judge 逐条核验"回答中的事实表述是否被材料支持"
4. 输出每个样本的判定和整体幻觉率

judge 有两种模式：

```bash
# 本地模式：用同一个模型当核验员（不依赖网络/API key）
python -m src.evaluate \
  --eval-file data/sample_qa.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter outputs/fin-lora \
  --judge local

# API 模式：任意 OpenAI 兼容接口
python -m src.evaluate \
  --eval-file data/sample_qa.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter outputs/fin-lora \
  --judge api \
  --judge-model gpt-4o-mini \
  --judge-api-base https://api.openai.com/v1 \
  --judge-api-key $OPENAI_API_KEY
```

结果写入 `outputs/eval/results.jsonl` 和 `outputs/eval/summary.json`，核心指标：

- `hallucination_rate`：存在无据/矛盾表述的答案占比
- `unsupported_claim_rate`：无据事实表述占所有表述的比例
- `gold_hit_rate`：答案命中 gold 数字的比例
- `refusal_rate`：正确拒答的比例

## 实验结果（困难评测集）

在 90 条困难财报问答（60 条"材料未提及"陷阱 + 30 条有答案题）上，
Base 与 SFT（Qwen2.5-7B-Instruct，QLoRA 3 epoch）的对比：

| 指标 | Base | SFT | 变化 |
| --- | --- | --- | --- |
| 陷阱题幻觉率 | 76.7% | 43.3% | -33.3 pp（相对 -43.5%） |
| 有答案题错误率 | 0.0% | 3.3% | +3.3 pp |
| 总体幻觉率 | 51.1% | 30.0% | -21.1 pp（相对 -41.3%） |

详细指标定义与复现命令见 [results/hallucination_results.md](results/hallucination_results.md)。

## 对照实验怎么设计

论文里最有说服力的是一张对照表：

| 方法 | 幻觉率 | gold 命中率 | 拒答正确率 |
| --- | --- | --- | --- |
| Base（不微调） | ... | ... | ... |
| Base + RAG 提示 | ... | ... | ... |
| SFT（本项目） | ... | ... | ... |
| SFT + RAG 提示 | ... | ... | ... |

评估集要固定，随机种子固定，同一个 judge 配置跑全部实验；最后人工抽检 100-200 条确认 judge 结论。

## 进阶方向

- DPO：把"有据正确回答"和"幻觉回答"配成偏好对，在 SFT 之后做一轮 DPO，通常能再降一截幻觉
- RAG：微调时把证据放进上下文，推理时检索财报/公告/新闻再回答
- 数字约束：对金额、比例做格式化输出或校验
- 金融基准：FinBen、FinanceBench、FinQA 等测试集作为外部验证

## 常见问题

- `bitsandbytes` 装不上：先确认是 Linux/WSL；Windows 上优先用云端 GPU
- 显存不够：换 `Qwen/Qwen2.5-3B-Instruct`，或降低 `max_seq_len`/`per_device_train_batch_size`
- 训练完幻觉还高：先查数据质量，重点看答案是否和 evidence 一致、拒答样本是否足够
- judge 判定不稳定：把 judge 的判定 prompt 写严格，多条不一致时取多数或人工复核
