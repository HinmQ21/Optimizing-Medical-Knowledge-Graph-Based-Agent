#!/usr/bin/env python3
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from datasets import load_from_disk
from tqdm.auto import tqdm


LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
FINAL_SECTION_RE = re.compile(
    r"##\s*Final Response\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
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
        description="Run MedQA benchmark for Qwen2.5 reasoning-SFT models through an SGLang completions endpoint."
    )
    parser.add_argument("--dataset-path", default="dataset/MedQA")
    parser.add_argument("--split", default="test")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="default", help="Served model name exposed by SGLang.")
    parser.add_argument("--run-name", default="qwen25_reasoning_sglang")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", default="results/medqa_qwen25_reasoning_sglang")
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def make_prompt(question: str, options: dict[str, str]) -> str:
    ordered = []
    for key in ["A", "B", "C", "D"]:
        if key in options:
            ordered.append(f"{key}. {options[key]}")
    options_text = "\n".join(ordered)
    return (
        "You are answering a medical multiple-choice question.\n"
        "Respond in exactly this format:\n"
        "## Thinking\n"
        "<step-by-step reasoning>\n\n"
        "## Final Response\n"
        "The answer is <A/B/C/D>. <brief justification>\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n"
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_final_section(text: str) -> str | None:
    match = FINAL_SECTION_RE.search(text)
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


def parse_answer(text: str, options: dict[str, str]) -> tuple[str | None, str]:
    cleaned = text.replace("<think>", " ").replace("</think>", " ").strip()
    sections = []

    final_section = extract_final_section(cleaned)
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


def batched_completion_request(
    *,
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> list[str]:
    payload = {
        "model": model,
        "prompt": prompts,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    req = Request(
        url=base_url.rstrip("/") + "/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SGLang server returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to SGLang server at {base_url}: {exc}") from exc

    choices = body.get("choices", [])
    if len(choices) != len(prompts):
        raise RuntimeError(
            f"Expected {len(prompts)} completions, got {len(choices)}. Raw response keys={list(body.keys())}"
        )

    choices = sorted(choices, key=lambda item: item.get("index", 0))
    return [choice.get("text", "") for choice in choices]


def force_extract_final_answer(
    *,
    base_url: str,
    model: str,
    question: str,
    options: dict[str, str],
    draft_response: str,
    temperature: float,
    top_p: float,
    timeout: float,
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
    decoded = batched_completion_request(
        base_url=base_url,
        model=model,
        prompts=[prompt],
        max_tokens=24,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
    )[0].strip()
    pred, _ = parse_answer(decoded, options)
    return pred, decoded


def run_benchmark(
    *,
    dataset,
    args: argparse.Namespace,
    run_name: str,
    output_dir: Path,
) -> dict:
    sampled = dataset.shuffle(seed=args.seed).select(range(min(args.num_samples, len(dataset))))

    records = []
    correct = 0
    invalid = 0
    fallback_attempts = 0
    fallback_recovered = 0
    started = time.time()

    total_batches = (len(sampled) + args.batch_size - 1) // args.batch_size
    batch_starts = range(0, len(sampled), args.batch_size)
    progress = tqdm(
        batch_starts,
        total=total_batches,
        desc=run_name,
        unit="batch",
        leave=False,
    )

    for i in progress:
        batch = sampled.select(range(i, min(i + args.batch_size, len(sampled))))
        prompts = [make_prompt(row["question"], row["options"]) for row in batch]
        outputs = batched_completion_request(
            base_url=args.base_url,
            model=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
        )

        for row, decoded in zip(batch, outputs):
            decoded = decoded.strip()
            pred, parse_source = parse_answer(decoded, row["options"])
            fallback_used = False
            fallback_output = None

            if pred is None:
                fallback_attempts += 1
                fallback_used = True
                pred, fallback_output = force_extract_final_answer(
                    base_url=args.base_url,
                    model=args.model,
                    question=row["question"],
                    options=row["options"],
                    draft_response=decoded,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    timeout=args.timeout,
                )
                if pred is not None:
                    fallback_recovered += 1
                    parse_source = "fallback_extract"

            gold = str(row["answer_idx"]).upper()
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
                    "run_name": run_name,
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
        "base_url": args.base_url,
        "model": args.model,
        "num_samples": total,
        "correct": correct,
        "accuracy": acc,
        "invalid_predictions": invalid,
        "elapsed_seconds": elapsed,
        "samples_per_second": (total / elapsed) if elapsed > 0 else 0.0,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "run_name": run_name,
        "fallback_attempts": fallback_attempts,
        "fallback_recovered": fallback_recovered,
        "prompt_style": "qwen25_sft_reasoning_headers",
        "expected_output_format": "## Thinking / ## Final Response / The answer is <A-D>",
    }

    run_out = output_dir / run_name
    run_out.mkdir(parents=True, exist_ok=True)
    with (run_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (run_out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return result


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.dataset_path)[args.split]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name
    if args.tag:
        run_name = f"{run_name}_{args.tag}"

    print(f"=== Running {args.model} on MedQA through SGLang reasoning pipeline ===")
    result = run_benchmark(
        dataset=dataset,
        args=args,
        run_name=run_name,
        output_dir=output_dir,
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
