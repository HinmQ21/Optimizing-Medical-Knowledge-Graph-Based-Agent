#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
import transformers.utils.import_utils as hf_import_utils

# Force-disable torchvision backend to avoid optional vision import errors
# in text-only evaluation environments.
hf_import_utils._torchvision_available = False

from transformers import AutoModelForCausalLM, AutoTokenizer


ANSWER_RE = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)
VALID_ANSWERS = {"yes", "no", "maybe"}


def build_context_text(context: dict) -> str:
    """Concatenate all paragraph strings from the context dict."""
    paragraphs = context.get("contexts", [])
    return "\n\n".join(p.strip() for p in paragraphs if p and p.strip())


def make_prompt(question: str, context: dict) -> str:
    context_text = build_context_text(context)
    return (
        "Read the following biomedical research abstract and answer the question.\n"
        "Return only one word: yes, no, or maybe.\n\n"
        f"Abstract:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def parse_answer(text: str) -> str | None:
    text = text.strip().lower()
    if text in VALID_ANSWERS:
        return text
    match = ANSWER_RE.search(text)
    return match.group(1).lower() if match else None


def run_model(
    model_path: Path,
    dataset,
    num_samples: int,
    batch_size: int,
    seed: int,
    output_dir: Path,
    device_preference: str,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wants_cuda = device_preference == "cuda" or (
        device_preference == "auto" and torch.cuda.is_available()
    )
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
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
    batch_starts = range(0, len(sampled), batch_size)
    progress = tqdm(
        batch_starts,
        total=total_batches,
        desc=f"{model_path.name}",
        unit="batch",
        leave=False,
    )
    for i in progress:
        batch = sampled.select(range(i, min(i + batch_size, len(sampled))))
        prompts = [make_prompt(row["question"], row["context"]) for row in batch]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
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
        for idx, row in enumerate(batch):
            gen_ids = outputs[idx][prompt_len:]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred = parse_answer(decoded)
            gold = str(row["final_decision"]).strip().lower()
            ok = pred == gold
            if ok:
                correct += 1
            if pred is None:
                invalid += 1
            records.append(
                {
                    "pubid": row["pubid"],
                    "question": row["question"],
                    "pred": pred,
                    "gold": gold,
                    "raw_output": decoded,
                    "correct": ok,
                }
            )
        processed = min(i + len(batch), len(sampled))
        if processed > 0:
            progress.set_postfix(
                {
                    "acc": f"{(correct / processed):.3f}",
                    "invalid": invalid,
                },
                refresh=False,
            )

    elapsed = time.time() - started
    total = len(records)
    acc = correct / total if total else 0.0
    result = {
        "model": str(model_path),
        "num_samples": total,
        "correct": correct,
        "accuracy": acc,
        "invalid_predictions": invalid,
        "elapsed_seconds": elapsed,
        "samples_per_second": (total / elapsed) if elapsed > 0 else 0.0,
        "device": str(model_input_device),
    }

    model_name = model_path.name
    model_out = output_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    with (model_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (model_out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    del model
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PubMedQA zero-shot baseline for local models.")
    parser.add_argument("--dataset-path", default="dataset/PubMedQA")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--model-path",
        action="append",
        default=None,
        help="Specific model directory to evaluate. Repeat to compare multiple models.",
    )
    parser.add_argument("--split", default="train", help="Dataset split (PubMedQA only has 'train' with 1000 samples).")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/pubmedqa_baseline")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)[args.split]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_path:
        model_paths = [Path(p) for p in args.model_path]
    else:
        models_dir = Path(args.models_dir)
        model_paths = sorted([p for p in models_dir.iterdir() if p.is_dir()])

    if not model_paths:
        raise RuntimeError("No model directories were provided for evaluation.")

    all_results = []
    for model_path in tqdm(model_paths, desc="Models", unit="model"):
        print(f"\n=== Running {model_path.name} on PubMedQA ({args.split}, n={args.num_samples}) ===")
        result = run_model(
            model_path=model_path,
            dataset=dataset,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            output_dir=output_dir,
            device_preference=args.device,
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
