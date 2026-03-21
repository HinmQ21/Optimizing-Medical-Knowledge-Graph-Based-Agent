#!/usr/bin/env python3
"""Unified zero-shot baseline benchmark for medical QA datasets.

Supported datasets (--dataset):
  medqa       – MedQA 4-option MCQ (A-D), split=test
  medmcqa     – MedMCQA 4-option MCQ (A-D, 4options or raw schema), split=validation
  pubmedqa    – PubMedQA yes/no/maybe, split=train
  medxpertqa  – MedXpertQA 10-option MCQ (A-J), split=test

To add a new dataset, define a DatasetAdapter and register it in ADAPTERS.
"""
import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
import transformers.utils.import_utils as hf_import_utils

hf_import_utils._torchvision_available = False

from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# DatasetAdapter – the only thing that differs between datasets
# ---------------------------------------------------------------------------

@dataclass
class DatasetAdapter:
    """Encapsulates all dataset-specific behaviour for a single benchmark."""

    # CLI / path defaults
    default_path: str
    default_split: str
    default_num_samples: int
    default_output_dir: str

    # Row processing: raw HF row -> normalized dict {question, gold, ...extras}
    normalize: Callable[[dict], dict]
    # Prompt builder: normalized dict -> prompt string
    build_prompt: Callable[[dict], str]
    # Answer parser: decoded model output -> canonical prediction or None
    parse: Callable[[str], str | None]


# ---------------------------------------------------------------------------
# MCQ adapter (MedQA / MedMCQA)
# ---------------------------------------------------------------------------

_MCQ_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_COP_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", "a": "A", "b": "B", "c": "C", "d": "D"}


def _mcq_normalize(row: dict) -> dict:
    """Handle MedQA/MedMCQA_4options (options+answer_idx) and raw MedMCQA (opa-opd+cop)."""
    if "options" in row and "answer_idx" in row:
        return {
            "question": row["question"],
            "options": dict(row["options"]),
            "gold": str(row["answer_idx"]).upper(),
        }
    options = {"A": row["opa"], "B": row["opb"], "C": row["opc"], "D": row["opd"]}
    cop = row.get("cop")
    gold = _COP_TO_LETTER.get(cop, "") if cop is not None else ""
    return {"question": row["question"], "options": options, "gold": gold}


def _mcq_build_prompt(norm: dict) -> str:
    ordered = [f"{k}. {norm['options'][k]}" for k in ["A", "B", "C", "D"] if k in norm["options"]]
    options_text = "\n".join(ordered)
    return (
        "Read the medical multiple-choice question and select the single best answer.\n"
        "Return only one letter: A, B, C, or D.\n\n"
        f"Question:\n{norm['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        "Answer:"
    )


def _mcq_parse(text: str) -> str | None:
    text = text.strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    match = _MCQ_LETTER_RE.search(text)
    return match.group(1).upper() if match else None


# ---------------------------------------------------------------------------
# MedXpertQA adapter  (10-option MCQ, A-J)
# ---------------------------------------------------------------------------

_MEDXPERT_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)
_MEDXPERT_VALID = set("ABCDEFGHIJ")

# Question text already contains embedded "Answer Choices: (A)...(J)..."
# so the prompt only needs the question — no need to list options again.


def _medxpertqa_normalize(row: dict) -> dict:
    return {
        "question": row["question"],
        "options": dict(row["options"]),          # kept for reference in records
        "gold": str(row["label"]).strip().upper(),
        # extra metadata preserved in prediction records
        "id": row.get("id"),
        "medical_task": row.get("medical_task"),
        "body_system": row.get("body_system"),
        "question_type": row.get("question_type"),
    }


def _medxpertqa_build_prompt(norm: dict) -> str:
    # The question already has embedded answer choices — avoid duplicating them.
    return (
        "Read the medical multiple-choice question and select the single best answer.\n"
        "Return only one letter: A, B, C, D, E, F, G, H, I, or J.\n\n"
        f"{norm['question']}\n\n"
        "Answer:"
    )


def _medxpertqa_parse(text: str) -> str | None:
    text = text.strip().upper()
    if text in _MEDXPERT_VALID:
        return text
    match = _MEDXPERT_LETTER_RE.search(text)
    return match.group(1).upper() if match else None


# ---------------------------------------------------------------------------
# PubMedQA adapter
# ---------------------------------------------------------------------------

_PUBMEDQA_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
_PUBMEDQA_VALID = {"yes", "no", "maybe"}


def _pubmedqa_normalize(row: dict) -> dict:
    paragraphs = row.get("context", {}).get("contexts", [])
    context_text = "\n\n".join(p.strip() for p in paragraphs if p and p.strip())
    return {
        "question": row["question"],
        "context_text": context_text,
        "gold": str(row["final_decision"]).strip().lower(),
        # extra field kept in prediction records
        "pubid": row.get("pubid"),
    }


def _pubmedqa_build_prompt(norm: dict) -> str:
    return (
        "Read the following biomedical research abstract and answer the question.\n"
        "Return only one word: yes, no, or maybe.\n\n"
        f"Abstract:\n{norm['context_text']}\n\n"
        f"Question:\n{norm['question']}\n\n"
        "Answer:"
    )


def _pubmedqa_parse(text: str) -> str | None:
    text = text.strip().lower()
    if text in _PUBMEDQA_VALID:
        return text
    match = _PUBMEDQA_RE.search(text)
    return match.group(1).lower() if match else None


# ---------------------------------------------------------------------------
# Registry – add new datasets here
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, DatasetAdapter] = {
    "medqa": DatasetAdapter(
        default_path="dataset/MedQA",
        default_split="test",
        default_num_samples=20,
        default_output_dir="results/medqa_baseline",
        normalize=_mcq_normalize,
        build_prompt=_mcq_build_prompt,
        parse=_mcq_parse,
    ),
    "medmcqa": DatasetAdapter(
        default_path="dataset/MedMCQA_4options",
        default_split="validation",
        default_num_samples=20,
        default_output_dir="results/medmcqa_baseline",
        normalize=_mcq_normalize,
        build_prompt=_mcq_build_prompt,
        parse=_mcq_parse,
    ),
    "medxpertqa": DatasetAdapter(
        default_path="dataset/MedXpertQA_Text",
        default_split="test",
        default_num_samples=2450,
        default_output_dir="results/medxpertqa_baseline",
        normalize=_medxpertqa_normalize,
        build_prompt=_medxpertqa_build_prompt,
        parse=_medxpertqa_parse,
    ),
    "pubmedqa": DatasetAdapter(
        default_path="dataset/PubMedQA",
        default_split="train",
        default_num_samples=1000,
        default_output_dir="results/pubmedqa_baseline",
        normalize=_pubmedqa_normalize,
        build_prompt=_pubmedqa_build_prompt,
        parse=_pubmedqa_parse,
    ),
}


# ---------------------------------------------------------------------------
# Core evaluation loop (dataset-agnostic)
# ---------------------------------------------------------------------------

def run_model(
    model_path: Path,
    dataset,
    adapter: DatasetAdapter,
    num_samples: int,
    batch_size: int,
    seed: int,
    output_dir: Path,
    device_preference: str,
    dataset_name: str,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wants_cuda = device_preference == "cuda" or (
        device_preference == "auto" and torch.cuda.is_available()
    )
    load_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if wants_cuda:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    model_input_device = next(model.parameters()).device

    sampled = dataset.shuffle(seed=seed).select(range(min(num_samples, len(dataset))))
    records = []
    correct = 0
    invalid = 0
    started = time.time()

    total_batches = (len(sampled) + batch_size - 1) // batch_size
    progress = tqdm(
        range(0, len(sampled), batch_size),
        total=total_batches,
        desc=model_path.name,
        unit="batch",
        leave=False,
    )
    for i in progress:
        batch = sampled.select(range(i, min(i + batch_size, len(sampled))))
        normalized = [adapter.normalize(row) for row in batch]
        prompts = [adapter.build_prompt(n) for n in normalized]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model_input_device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for idx, norm in enumerate(normalized):
            decoded = tokenizer.decode(outputs[idx][prompt_len:], skip_special_tokens=True).strip()
            pred = adapter.parse(decoded)
            gold = norm["gold"]
            ok = pred == gold
            if ok:
                correct += 1
            if pred is None:
                invalid += 1
            # base record; merge any extra fields from normalized row
            record = {"question": norm["question"], "pred": pred, "gold": gold, "raw_output": decoded, "correct": ok}
            record.update({k: v for k, v in norm.items() if k not in ("question", "gold")})
            records.append(record)

        processed = min(i + len(batch), len(sampled))
        if processed > 0:
            progress.set_postfix({"acc": f"{correct / processed:.3f}", "invalid": invalid}, refresh=False)

    elapsed = time.time() - started
    total = len(records)
    acc = correct / total if total else 0.0
    result = {
        "model": str(model_path),
        "dataset": dataset_name,
        "num_samples": total,
        "correct": correct,
        "accuracy": acc,
        "invalid_predictions": invalid,
        "elapsed_seconds": elapsed,
        "samples_per_second": (total / elapsed) if elapsed > 0 else 0.0,
        "device": str(model_input_device),
    }

    model_out = output_dir / model_path.name
    model_out.mkdir(parents=True, exist_ok=True)
    with (model_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (model_out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    del model
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified zero-shot baseline for medical QA datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available datasets: {', '.join(ADAPTERS)}",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(ADAPTERS),
        help="Dataset to benchmark.",
    )
    parser.add_argument("--dataset-path", default="", help="Override dataset path.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--model-path",
        action="append",
        default=None,
        help="Model directory to evaluate. Repeat to compare multiple models.",
    )
    parser.add_argument("--split", default="", help="Dataset split (uses dataset default if empty).")
    parser.add_argument("--num-samples", type=int, default=0, help="Samples to evaluate (uses dataset default if 0).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="", help="Override output directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = ADAPTERS[args.dataset]

    dataset_path = args.dataset_path or adapter.default_path
    split = args.split or adapter.default_split
    num_samples = args.num_samples or adapter.default_num_samples
    output_dir = Path(args.output_dir or adapter.default_output_dir)

    dataset = load_from_disk(dataset_path)[split]
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_path:
        model_paths = [Path(p) for p in args.model_path]
    else:
        models_dir = Path(args.models_dir)
        model_paths = sorted(p for p in models_dir.iterdir() if p.is_dir())

    if not model_paths:
        raise RuntimeError("No model directories found for evaluation.")

    all_results = []
    for model_path in tqdm(model_paths, desc="Models", unit="model"):
        print(f"\n=== {model_path.name} | {args.dataset.upper()} ({split}, n={num_samples}) ===")
        result = run_model(
            model_path=model_path,
            dataset=dataset,
            adapter=adapter,
            num_samples=num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            output_dir=output_dir,
            device_preference=args.device,
            dataset_name=args.dataset,
        )
        all_results.append(result)
        print(
            f"Accuracy: {result['accuracy']:.4f} "
            f"({result['correct']}/{result['num_samples']}) | "
            f"invalid={result['invalid_predictions']} | "
            f"time={result['elapsed_seconds']:.1f}s"
        )

    with (output_dir / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for r in all_results:
        print(
            f"{Path(r['model']).name}: "
            f"acc={r['accuracy']:.4f} ({r['correct']}/{r['num_samples']}), "
            f"invalid={r['invalid_predictions']}, "
            f"samples/s={r['samples_per_second']:.3f}"
        )


if __name__ == "__main__":
    main()
