#!/usr/bin/env python3
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
import transformers.utils.import_utils as hf_import_utils

# Force-disable torchvision backend to avoid optional vision import errors
# in text-only evaluation environments.
hf_import_utils._torchvision_available = False

from transformers import AutoModelForCausalLM, AutoTokenizer


LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
EXPLICIT_ANSWER_PATTERNS = [
    re.compile(r"final\s+answer\s*[:\-]\s*([ABCD])\b", re.IGNORECASE),
    re.compile(r"the\s+answer\s+is\s+([ABCD])\b", re.IGNORECASE),
    re.compile(r"the\s+correct\s+answer\s+is\s+([ABCD])\b", re.IGNORECASE),
    re.compile(r"the\s+correct\s+option\s+is\s+([ABCD])\b", re.IGNORECASE),
    re.compile(r"option\s+([ABCD])\b", re.IGNORECASE),
    re.compile(r"answer\s*[:\-]\s*([ABCD])\b", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MedQA/MedMCQA benchmark for models trained with qwen25_medreason_full_trainer_v2."
    )
    parser.add_argument(
        "--dataset",
        default="medqa",
        choices=["medqa", "medmcqa"],
        help="Which benchmark dataset to evaluate on.",
    )
    parser.add_argument("--dataset-path", default="", help="Override dataset path (auto-selected if empty).")
    parser.add_argument("--model-path", default="models/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--split",
        default="",
        help="Dataset split. Defaults to 'test' for MedQA and 'validation' for MedMCQA.",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--prompt-max-length", type=int, default=3072)
    parser.add_argument("--output-dir", default="", help="Override output dir (auto-selected if empty).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        action="store_true",
        help="Enable KV cache during generation.",
    )
    parser.add_argument(
        "--no-use-cache",
        dest="use_cache",
        action="store_false",
        help="Disable KV cache during generation.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional run tag appended in output metadata.",
    )
    # Match the section titles used during MedReason training so the model
    # sees the same format it was trained on.
    parser.add_argument(
        "--thinking-section-title",
        default="Thinking",
        help="Section title used before the reasoning chain (must match training).",
    )
    parser.add_argument(
        "--final-section-title",
        default="Final Response",
        help="Section title used before the final answer (must match training).",
    )
    parser.set_defaults(use_cache=True)
    return parser.parse_args()


_COP_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", "a": "A", "b": "B", "c": "C", "d": "D"}


def normalize_row(row: dict) -> dict:
    """Return a unified dict with question / options / answer_idx regardless of source schema.

    Supports:
    - MedQA / MedMCQA_4options: already has ``options`` dict and ``answer_idx``.
    - Raw MedMCQA: has ``opa``/``opb``/``opc``/``opd`` and ``cop`` (ClassLabel int or letter).
    """
    if "options" in row and "answer_idx" in row:
        return {
            "question": row["question"],
            "options": dict(row["options"]),
            "answer_idx": str(row["answer_idx"]).upper(),
        }
    options = {"A": row["opa"], "B": row["opb"], "C": row["opc"], "D": row["opd"]}
    cop = row.get("cop")
    answer_idx = _COP_TO_LETTER.get(cop, "") if cop is not None else ""
    return {"question": row["question"], "options": options, "answer_idx": answer_idx}


def make_final_section_re(final_section_title: str) -> re.Pattern:
    escaped = re.escape(final_section_title)
    return re.compile(
        rf"##\s*{escaped}\s*(.*)$",
        re.IGNORECASE | re.DOTALL,
    )


def make_prompt(
    question: str,
    options: dict[str, str],
    thinking_section_title: str,
    final_section_title: str,
) -> str:
    ordered = []
    for key in ["A", "B", "C", "D"]:
        if key in options:
            ordered.append(f"{key}. {options[key]}")
    options_text = "\n".join(ordered)
    return (
        "You are answering a medical multiple-choice question.\n"
        "Respond in exactly this format:\n"
        f"## {thinking_section_title}\n"
        "<step-by-step reasoning>\n\n"
        f"## {final_section_title}\n"
        "The answer is <A/B/C/D>. <brief justification>\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n"
    )


def render_chat(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"### User:\n{prompt}\n\n### Assistant:\n"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_final_section(text: str, final_section_re: re.Pattern) -> str | None:
    match = final_section_re.search(text)
    if match:
        return match.group(1).strip()
    return None


def parse_by_explicit_pattern(text: str) -> str | None:
    if text.strip().upper() in {"A", "B", "C", "D"}:
        return text.strip().upper()

    for pattern in EXPLICIT_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def parse_by_option_text(text: str, options: dict[str, str]) -> str | None:
    normalized_text = normalize_text(text)
    found = []
    for label, option_text in options.items():
        normalized_option = normalize_text(option_text)
        if normalized_option and normalized_option in normalized_text:
            found.append((label, normalized_text.rfind(normalized_option)))

    if len(found) == 1:
        return found[0][0]
    if len(found) > 1:
        found.sort(key=lambda item: item[1], reverse=True)
        return found[0][0]
    return None


def parse_answer(
    text: str,
    options: dict[str, str],
    final_section_re: re.Pattern,
) -> tuple[str | None, str]:
    cleaned = text.replace("<think>", " ").replace("</think>", " ").strip()
    sections = []

    final_section = extract_final_section(cleaned, final_section_re)
    if final_section:
        sections.append(("final_section", final_section))
    sections.append(("full_text", cleaned))

    for source_name, section in sections:
        pred = parse_by_explicit_pattern(section)
        if pred is not None:
            return pred, f"{source_name}:explicit"

    if final_section:
        pred = parse_by_option_text(final_section, options)
        if pred is not None:
            return pred, "final_section:option_text"

    pred = parse_by_option_text(cleaned, options)
    if pred is not None:
        return pred, "full_text:option_text"

    if final_section:
        matches = LETTER_RE.findall(final_section.upper())
        if len(matches) == 1:
            return matches[0].upper(), "final_section:single_letter"

    return None, "unparsed"


def force_extract_final_answer(
    *,
    tokenizer,
    model,
    model_input_device,
    question: str,
    options: dict[str, str],
    draft_response: str,
    use_cache: bool,
    prompt_max_length: int,
    final_section_re: re.Pattern,
) -> tuple[str | None, str]:
    ordered = []
    for key in ["A", "B", "C", "D"]:
        if key in options:
            ordered.append(f"{key}. {options[key]}")
    options_text = "\n".join(ordered)

    prompt = (
        "You are extracting the final option from a draft answer to a medical multiple-choice question.\n"
        "Return exactly one line in this format and nothing else:\n"
        "Final Answer: <A/B/C/D>\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Draft response:\n{draft_response}"
    )
    text = render_chat(tokenizer, prompt)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=prompt_max_length,
    ).to(model_input_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    pred, _ = parse_answer(decoded, options, final_section_re)
    return pred, decoded


def run_benchmark(
    model_path: Path,
    dataset,
    num_samples: int,
    batch_size: int,
    seed: int,
    max_new_tokens: int,
    prompt_max_length: int,
    use_cache: bool,
    run_tag: str,
    output_dir: Path,
    device_preference: str,
    thinking_section_title: str,
    final_section_title: str,
    dataset_name: str = "medqa",
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

    final_section_re = make_final_section_re(final_section_title)

    sampled = dataset.shuffle(seed=seed).select(range(min(num_samples, len(dataset))))
    records = []
    correct = 0
    invalid = 0
    fallback_attempts = 0
    fallback_recovered = 0
    started = time.time()

    total_batches = (len(sampled) + batch_size - 1) // batch_size
    batch_starts = range(0, len(sampled), batch_size)
    progress = tqdm(
        batch_starts,
        total=total_batches,
        desc=model_path.name,
        unit="batch",
        leave=False,
    )

    for i in progress:
        batch = sampled.select(range(i, min(i + batch_size, len(sampled))))
        rows = [normalize_row(row) for row in batch]
        prompts = [
            make_prompt(
                r["question"],
                r["options"],
                thinking_section_title,
                final_section_title,
            )
            for r in rows
        ]
        texts = [render_chat(tokenizer, prompt) for prompt in prompts]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_max_length,
        ).to(model_input_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for idx, row in enumerate(rows):
            gen_ids = outputs[idx][prompt_len:]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred, parse_source = parse_answer(decoded, row["options"], final_section_re)
            fallback_used = False
            fallback_output = None

            if pred is None:
                fallback_attempts += 1
                fallback_used = True
                pred, fallback_output = force_extract_final_answer(
                    tokenizer=tokenizer,
                    model=model,
                    model_input_device=model_input_device,
                    question=row["question"],
                    options=row["options"],
                    draft_response=decoded,
                    use_cache=use_cache,
                    prompt_max_length=prompt_max_length,
                    final_section_re=final_section_re,
                )
                if pred is not None:
                    fallback_recovered += 1
                    parse_source = "fallback_extract"

            gold = row["answer_idx"]
            ok = pred == gold
            if ok:
                correct += 1
            if pred is None:
                invalid += 1
            records.append(
                {
                    "question": row["question"],
                    "pred": pred,
                    "gold": gold,
                    "raw_output": decoded,
                    "correct": ok,
                    "run_tag": run_tag,
                    "parse_source": parse_source,
                    "fallback_used": fallback_used,
                    "fallback_output": fallback_output,
                }
            )

        processed = min(i + len(batch), len(sampled))
        if processed > 0:
            progress.set_postfix(
                {
                    "acc": f"{(correct / processed):.3f}",
                    "invalid": invalid,
                    "fb_ok": fallback_recovered,
                },
                refresh=False,
            )

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
        "max_new_tokens": max_new_tokens,
        "prompt_max_length": prompt_max_length,
        "use_cache": use_cache,
        "run_tag": run_tag,
        "fallback_attempts": fallback_attempts,
        "fallback_recovered": fallback_recovered,
        "thinking_section_title": thinking_section_title,
        "final_section_title": final_section_title,
        "prompt_style": "medreason_sft_reasoning_headers",
        "expected_output_format": f"## {thinking_section_title} / ## {final_section_title} / The answer is <A-D>",
    }

    model_out = output_dir / model_path.name
    model_out.mkdir(parents=True, exist_ok=True)
    with (model_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (model_out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    del model
    return result


_DATASET_DEFAULTS = {
    "medqa":   {"path": "dataset/MedQA",           "split": "test",       "output_dir": "results/medqa_qwen25_medreason_pipeline"},
    "medmcqa": {"path": "dataset/MedMCQA_4options", "split": "validation", "output_dir": "results/medmcqa_qwen25_medreason_pipeline"},
}


def main() -> None:
    args = parse_args()

    defaults = _DATASET_DEFAULTS[args.dataset]
    dataset_path = args.dataset_path or defaults["path"]
    split = args.split or defaults["split"]
    output_dir = Path(args.output_dir or defaults["output_dir"])

    dataset = load_from_disk(dataset_path)[split]
    model_path = Path(args.model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_tag = f"{args.dataset}_qwen25_medreason"
    if args.tag:
        run_tag = f"{run_tag}_{args.tag}"

    print(f"\n=== Running {model_path.name} on {args.dataset.upper()} ({split}) with MedReason prompt ===")
    result = run_benchmark(
        model_path=model_path,
        dataset=dataset,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        prompt_max_length=args.prompt_max_length,
        use_cache=args.use_cache,
        run_tag=run_tag,
        output_dir=output_dir,
        device_preference=args.device,
        thinking_section_title=args.thinking_section_title,
        final_section_title=args.final_section_title,
        dataset_name=args.dataset,
    )
    print(
        f"Accuracy: {result['accuracy']:.4f} "
        f"({result['correct']}/{result['num_samples']}) | "
        f"invalid={result['invalid_predictions']} | "
        f"fallback={result['fallback_recovered']}/{result['fallback_attempts']} | "
        f"time={result['elapsed_seconds']:.1f}s | "
        f"samples/s={result['samples_per_second']:.3f}"
    )


if __name__ == "__main__":
    main()
