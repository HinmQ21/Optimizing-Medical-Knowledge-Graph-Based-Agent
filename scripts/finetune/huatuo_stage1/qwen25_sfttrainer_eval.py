#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import LoraConfig, prepare_model_for_kbit_training
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: peft. Install with `pip install peft accelerate` before training."
    ) from exc

try:
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: trl. Install with `pip install trl` before using SFTTrainer."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HuatuoGPT-o1 style stage-1 SFT using TRL SFTTrainer with eval and best-checkpoint selection."
    )
    parser.add_argument("--model-path", required=True, help="Local model path or HF model id.")
    parser.add_argument(
        "--data-path",
        default="FreedomIntelligence/medical-o1-reasoning-SFT",
        help="HF dataset id, local load_from_disk path, or local json/jsonl file.",
    )
    parser.add_argument(
        "--dataset-config",
        default="en",
        help="HF dataset config name. Use en/zh for the official medical-o1-reasoning-SFT dataset.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default=None)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.02,
        help="Used only when --eval-split is not provided.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--dataset-num-proc", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/huatuo_stage1_qwen25_3b_sfttrainer_eval")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none", choices=["none", "wandb"])
    parser.add_argument("--run-name", default="huatuo-stage1-qwen25-3b-sfttrainer-eval")
    parser.add_argument("--wandb-project", default="my-awesome-project")
    parser.add_argument(
        "--wandb-entity",
        default="qminhlb-vietnam-national-university-hanoi",
    )
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--modules-to-save", nargs="*", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return parser.parse_args()


def load_dataset_maybe_dict(data_path: str, dataset_config: str | None):
    local_path = Path(data_path)
    if local_path.exists():
        if local_path.is_file() and local_path.suffix in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(local_path))
        return load_from_disk(str(local_path))

    if dataset_config:
        return load_dataset(data_path, dataset_config)
    return load_dataset(data_path)


def get_split_or_first(loaded, split: str) -> Dataset:
    if isinstance(loaded, DatasetDict):
        if split in loaded:
            return loaded[split]
        first_split = next(iter(loaded.keys()))
        return loaded[first_split]
    return loaded


def maybe_select(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def build_train_eval_splits(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    loaded = load_dataset_maybe_dict(args.data_path, args.dataset_config)
    train_base = get_split_or_first(loaded, args.train_split)

    if args.eval_split:
        eval_base = get_split_or_first(loaded, args.eval_split)
    else:
        split_dataset = train_base.train_test_split(
            test_size=args.validation_ratio,
            seed=args.seed,
        )
        train_base = split_dataset["train"]
        eval_base = split_dataset["test"]

    train_base = maybe_select(train_base, args.max_train_samples)
    eval_base = maybe_select(eval_base, args.max_eval_samples)
    return train_base, eval_base


def resolve_field(example: dict[str, Any], *candidates: str) -> str:
    for name in candidates:
        if name in example and example[name] is not None:
            return str(example[name])
    raise KeyError(f"Missing expected field. Tried: {', '.join(candidates)}")


def format_completion(example: dict[str, Any]) -> str:
    cot = resolve_field(example, "Complex_CoT", "complex_cot", "complex_cot_en")
    response = resolve_field(example, "Response", "response", "final_response")
    return f"## Thinking\n\n{cot}\n\n## Final Response\n\n{response}"


def convert_to_conversational_prompt_completion(dataset: Dataset, dataset_num_proc: int | None) -> Dataset:
    remove_columns = list(dataset.column_names)

    def map_row(example: dict[str, Any]) -> dict[str, Any]:
        question = resolve_field(example, "Question", "question", "prompt")
        return {
            "prompt": [{"role": "user", "content": question}],
            "completion": [{"role": "assistant", "content": format_completion(example)}],
        }

    return dataset.map(
        map_row,
        remove_columns=remove_columns,
        num_proc=dataset_num_proc,
        desc="Formatting dataset for SFTTrainer",
    )


def build_model(args: argparse.Namespace):
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}

    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--load-in-4bit requires a CUDA-capable GPU.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model


def build_peft_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        modules_to_save=args.modules_to_save,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_run_config(
    args: argparse.Namespace,
    train_base: Dataset,
    eval_base: Dataset,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        **vars(args),
        "output_dir": str(output_dir),
        "train_num_rows": len(train_base),
        "eval_num_rows": len(eval_base),
        "cuda_available": torch.cuda.is_available(),
        "bf16_enabled": torch.cuda.is_available(),
        "trainer_type": "trl.SFTTrainer",
        "dataset_format": "conversational_prompt_completion",
        "loss_behavior": "completion_only",
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def configure_wandb(args: argparse.Namespace, output_dir: Path, run_config: dict[str, Any]):
    if args.report_to != "wandb":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: wandb. Install with `pip install wandb` or use `--report-to none`."
        ) from exc

    wandb_dir = args.wandb_dir or str(output_dir / "wandb")
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_NAME"] = args.run_name
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_DIR"] = wandb_dir
    os.environ.setdefault("WANDB_WATCH", "false")
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_tags:
        os.environ["WANDB_TAGS"] = ",".join(args.wandb_tags)

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        dir=wandb_dir,
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config=run_config,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_base, eval_base = build_train_eval_splits(args)
    run_config = build_run_config(args, train_base, eval_base, output_dir)
    save_json(run_config, output_dir / "run_config.json")
    wandb_run = configure_wandb(args, output_dir, run_config)

    train_dataset = convert_to_conversational_prompt_completion(
        train_base,
        dataset_num_proc=args.dataset_num_proc,
    )
    eval_dataset = convert_to_conversational_prompt_completion(
        eval_base,
        dataset_num_proc=args.dataset_num_proc,
    )

    model = build_model(args)
    peft_config = build_peft_config(args)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
        dataloader_num_workers=2,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        dataset_num_proc=args.dataset_num_proc,
        max_length=args.max_seq_len,
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        pad_token=tokenizer.pad_token,
        eos_token=tokenizer.eos_token,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    final_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    best_info = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "final_eval_metrics": final_metrics,
    }
    save_json(best_info, output_dir / "best_checkpoint.json")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
