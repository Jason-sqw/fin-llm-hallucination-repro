"""LoRA/QLoRA instruction fine-tuning for grounded financial QA."""

import argparse
import inspect
import json
import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.chat_utils import ensure_chat_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_tokenizer_fn(tokenizer, max_seq_len):
    def tokenize(examples):
        prompts = [
            tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            for messages in examples["messages"]
        ]
        fulls = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in examples["messages"]
        ]
        enc_prompts = tokenizer(
            prompts, add_special_tokens=True, truncation=True, max_length=max_seq_len
        )
        enc_full = tokenizer(
            fulls, add_special_tokens=True, truncation=True, max_length=max_seq_len
        )

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(prompts)):
            prompt_ids = enc_prompts["input_ids"][i]
            full_ids = enc_full["input_ids"][i]
            if len(full_ids) <= len(prompt_ids):
                logger.warning(
                    "prompt fills the whole window for example %s; "
                    "falling back to full-sequence supervision",
                    i,
                )
                batch["input_ids"].append(full_ids)
                batch["attention_mask"].append(enc_full["attention_mask"][i])
                batch["labels"].append(list(full_ids))
                continue
            if full_ids[: len(prompt_ids)] != prompt_ids:
                logger.warning(
                    "chat template prefix mismatch for example %s; "
                    "falling back to full-sequence supervision",
                    i,
                )
                batch["input_ids"].append(full_ids)
                batch["attention_mask"].append(enc_full["attention_mask"][i])
                batch["labels"].append(list(full_ids))
                continue
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            batch["input_ids"].append(full_ids)
            batch["attention_mask"].append(enc_full["attention_mask"][i])
            batch["labels"].append(labels)
        return batch

    return tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(cfg.get("seed", 42))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True
    )
    ensure_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = (
        torch.bfloat16 if use_bf16 else torch.float16
        if torch.cuda.is_available() else torch.float32
    )
    quantization_config = None
    if cfg.get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["train_file"],
            "validation": cfg["val_file"],
        },
    )
    tokenize = make_tokenizer_fn(tokenizer, cfg["max_seq_len"])
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["messages", "id"])
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100
    )

    output_dir = cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    num_update_steps = max(
        1,
        len(tokenized["train"])
        // (
            cfg["per_device_train_batch_size"]
            * cfg["gradient_accumulation_steps"]
        ),
    )
    warmup_ratio = float(cfg.get("warmup_ratio", 0.03))
    warmup_steps = int(warmup_ratio * num_update_steps * cfg["num_train_epochs"])
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 2),
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        optim="paged_adamw_8bit" if quantization_config is not None else "adamw_torch",
        seed=cfg.get("seed", 42),
        report_to="none",
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": data_collator,
    }
    if "tokenizer" in inspect.signature(Trainer.__init__).parameters:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)
    else:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()
    (Path(output_dir) / "train_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("adapter saved to %s", output_dir)


if __name__ == "__main__":
    main()
