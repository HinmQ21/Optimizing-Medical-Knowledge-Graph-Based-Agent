"""Post-GRPO evaluation script.

Extends Stage 1.5 eval_sft.py with:
  - Multi-benchmark support (MedQA, MedMCQA, ...)
  - Forced no-tool ablation mode  (--no-tool)
  - Query quality scoring         (copy-paste vs reformulation)
  - Think-block depth analysis
  - GRPO-specific comparison metrics

Output JSON:
  {
    "model_path": "...",
    "mode": "with_tool" | "no_tool",
    "benchmarks": {
      "MedQA": { "metrics": {...}, "per_sample": [...] },
      ...
    }
  }

Usage:
    cd /home/vcsai/minhlbq/baseline

    # Standard eval (with tool):
    ./training_venv312/bin/python -m scripts.eval.grpo_eval \\
        --model-path outputs/grpo_v4_merged \\
        --benchmarks dataset/MedQA/test dataset/MedMCQA_4options_fixed/test \\
        --n-samples 200 --score-retrieval \\
        --output eval_results/grpo_v4_with_tool.json

    # No-tool ablation:
    ./training_venv312/bin/python -m scripts.eval.grpo_eval \\
        --model-path outputs/grpo_v4_merged \\
        --no-tool \\
        --benchmarks dataset/MedQA/test \\
        --n-samples 200 \\
        --output eval_results/grpo_v4_no_tool.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.serve.retrieval_tool import MedicalKnowledgeTool, search_medical_knowledge
from scripts.train_rl.data_prep import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

NO_TOOL_SYSTEM_PROMPT = (
    "You are a medical reasoning assistant.\n\n"
    "Structure your response:\n"
    "1. <think>Your reasoning here</think>\n"
    "2. <answer>Your final answer</answer>\n\n"
    "IMPORTANT: In <answer> tags, write ONLY the option letter (e.g. A) "
    "or a short answer, NOT an explanation."
)


# ---------------------------------------------------------------------------
# Regex / tokenization helpers
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_LETTER_RE = re.compile(r"^\s*([A-Ea-e])[.\):\s]")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "not", "with", "that",
    "this", "which", "what", "how", "does", "do", "from",
})


def _tokenize(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _copy_paste_ratio(query: str, question: str) -> float:
    """Fraction of query tokens present in the question.

    1.0 = pure copy-paste; low = reformulated / focused query.
    """
    q_tokens = _tokenize(query)
    src_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    return len(q_tokens & src_tokens) / len(q_tokens)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    return calls


def extract_answer_letter(content: str) -> str | None:
    """Extract MCQ letter from <answer>X</answer>."""
    match = _ANSWER_RE.search(content)
    if not match:
        return None
    ans = match.group(1).strip()
    letter_match = _LETTER_RE.match(ans)
    if letter_match:
        return letter_match.group(1).upper()
    if len(ans) == 1 and ans.upper() in "ABCDE":
        return ans.upper()
    for c in ans:
        if c.upper() in "ABCDE":
            return c.upper()
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_with_tools(
    model, tokenizer, question: str, options: dict, system_prompt: str,
    max_tool_iterations: int = 3, max_new_tokens: int = 1024,
    temperature: float = 0.3,
) -> dict:
    """Run one sample through the model with (optional) tool loop."""
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{question}\n\nOptions:\n{opt_text}"},
    ]
    tool_calls_made: list[dict] = []
    tool_responses: list[str] = []
    query_texts: list[str] = []

    for iteration in range(max_tool_iterations + 1):
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=False,
        ).replace("<|im_end|>", "").strip()

        tool_calls = extract_tool_calls(generated)
        if tool_calls and iteration < max_tool_iterations:
            messages.append({"role": "assistant", "content": generated})
            tool_calls_made.extend(tool_calls)
            for tc in tool_calls:
                try:
                    args = tc.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    query = args.get("query", "")
                    if query:
                        query_texts.append(query)
                    result = search_medical_knowledge(query)
                    tool_responses.append(result)
                    messages.append({
                        "role": "user",
                        "content": f"<tool_response>\n{result}\n</tool_response>",
                    })
                except Exception as e:
                    tool_responses.append(f"ERROR: {e}")
                    messages.append({
                        "role": "user",
                        "content": f"<tool_response>\nERROR: {e}\n</tool_response>",
                    })
        else:
            messages.append({"role": "assistant", "content": generated})
            break

    return {
        "messages": messages,
        "tool_calls": tool_calls_made,
        "tool_responses": tool_responses,
        "query_texts": query_texts,
        "n_tool_calls": len(tool_calls_made),
    }


def generate_no_tool(
    model, tokenizer, question: str, options: dict,
    max_new_tokens: int = 1024, temperature: float = 0.3,
) -> dict:
    """Single-turn generation without tool loop."""
    opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    messages = [
        {"role": "system", "content": NO_TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n\nOptions:\n{opt_text}"},
    ]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        out[0][input_ids.shape[1]:], skip_special_tokens=False,
    ).replace("<|im_end|>", "").strip()

    return {
        "messages": messages + [{"role": "assistant", "content": generated}],
        "tool_calls": [],
        "tool_responses": [],
        "query_texts": [],
        "n_tool_calls": 0,
    }


# ---------------------------------------------------------------------------
# Retrieval quality
# ---------------------------------------------------------------------------

def score_retrieval_relevance(
    encoder, question: str, answer_text: str, retrieved_texts: list[str],
) -> float:
    """Max cosine similarity between retrieved facts and (question + answer) anchor."""
    if not retrieved_texts:
        return 0.0
    anchor = f"{question} The answer is {answer_text}"
    embeddings = encoder.encode(
        [anchor] + retrieved_texts, normalize_embeddings=True,
    )
    sims = embeddings[1:] @ embeddings[0]
    return float(sims.max())


# ---------------------------------------------------------------------------
# Dataset loading helper
# ---------------------------------------------------------------------------

def load_benchmark(path: str, n_samples: int, seed: int):
    """Load benchmark dataset from path (handles both Dataset and DatasetDict)."""
    ds = load_from_disk(path)
    # DatasetDict: pick the split named in path or default to 'test'
    if hasattr(ds, "column_names") and isinstance(ds.column_names, dict):
        # It's a DatasetDict
        split_name = Path(path).name  # e.g. "test"
        if split_name in ds:
            ds = ds[split_name]
        elif "test" in ds:
            ds = ds["test"]
        else:
            ds = ds[list(ds.keys())[0]]
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return ds


# ---------------------------------------------------------------------------
# Per-benchmark evaluation
# ---------------------------------------------------------------------------

def eval_benchmark(
    model, tokenizer, ds, benchmark_name: str,
    no_tool: bool, max_tool_iterations: int,
    temperature: float, score_retrieval: bool,
    encoder=None,
) -> dict:
    """Run inference on one benchmark split and return metrics + per_sample list."""
    system_prompt = NO_TOOL_SYSTEM_PROMPT if no_tool else SYSTEM_PROMPT
    results = []
    n = len(ds)

    print(f"\n  Benchmark: {benchmark_name}  ({n} samples, {'no-tool' if no_tool else 'with-tool'})")

    for i, ex in enumerate(ds):
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{n}] ...", flush=True)

        if no_tool:
            res = generate_no_tool(model, tokenizer, ex["question"], ex["options"], temperature=temperature)
        else:
            res = generate_with_tools(
                model, tokenizer, ex["question"], ex["options"], system_prompt,
                max_tool_iterations=max_tool_iterations, temperature=temperature,
            )

        # Final assistant content
        final_content = ""
        for m in reversed(res["messages"]):
            if m["role"] == "assistant":
                final_content = m.get("content", "") or ""
                break

        pred_letter = extract_answer_letter(final_content)
        is_correct = pred_letter == ex["answer_idx"]

        # Think block analysis
        think_matches = _THINK_RE.findall(final_content)
        think_text = " ".join(think_matches)
        think_words = len(think_text.split()) if think_text else 0
        has_think = bool(think_matches)
        has_answer = bool(_ANSWER_RE.search(final_content))

        # Query quality: copy-paste ratio per query
        copy_paste_ratios = [
            _copy_paste_ratio(q, ex["question"]) for q in res["query_texts"]
        ]
        avg_copy_paste = float(np.mean(copy_paste_ratios)) if copy_paste_ratios else None

        # Retrieval score
        retrieval_score = None
        if score_retrieval and encoder and res["tool_responses"]:
            retrieval_score = score_retrieval_relevance(
                encoder, ex["question"], ex["answer"], res["tool_responses"],
            )

        results.append({
            "idx": i,
            "pred": pred_letter,
            "correct_idx": ex["answer_idx"],
            "is_correct": is_correct,
            "n_tool_calls": res["n_tool_calls"],
            "has_think": has_think,
            "has_answer": has_answer,
            "think_words": think_words,
            "query_texts": res["query_texts"],
            "avg_query_copy_paste": avg_copy_paste,
            "retrieval_score": retrieval_score,
        })

    # ── Aggregate ──
    n_correct = sum(1 for r in results if r["is_correct"])
    n_with_tool = sum(1 for r in results if r["n_tool_calls"] > 0)
    n_without_tool = n - n_with_tool
    n_has_think = sum(1 for r in results if r["has_think"])
    n_has_answer = sum(1 for r in results if r["has_answer"])
    n_extracted = sum(1 for r in results if r["pred"] is not None)

    correct_with = sum(1 for r in results if r["is_correct"] and r["n_tool_calls"] > 0)
    correct_without = sum(1 for r in results if r["is_correct"] and r["n_tool_calls"] == 0)

    acc_overall = n_correct / n
    acc_with = correct_with / n_with_tool if n_with_tool > 0 else None
    acc_without = correct_without / n_without_tool if n_without_tool > 0 else None

    think_words_list = [r["think_words"] for r in results if r["has_think"]]
    copy_paste_list = [r["avg_query_copy_paste"] for r in results if r["avg_query_copy_paste"] is not None]
    retrieval_scores = [r["retrieval_score"] for r in results if r["retrieval_score"] is not None]

    metrics = {
        "n_samples": n,
        "accuracy_overall": round(acc_overall, 4),
        "accuracy_with_tool": round(acc_with, 4) if acc_with is not None else None,
        "accuracy_without_tool": round(acc_without, 4) if acc_without is not None else None,
        "accuracy_diff_pts": round((acc_with - acc_without) * 100, 1) if (acc_with is not None and acc_without is not None) else None,
        "tool_call_frequency": round(n_with_tool / n, 4),
        "avg_tool_calls": round(sum(r["n_tool_calls"] for r in results) / n, 3),
        "multi_turn_rate": round(sum(1 for r in results if r["n_tool_calls"] >= 2) / n, 4),
        "has_think_rate": round(n_has_think / n, 4),
        "has_answer_rate": round(n_has_answer / n, 4),
        "pred_extracted_rate": round(n_extracted / n, 4),
        # Think depth
        "avg_think_words": round(float(np.mean(think_words_list)), 1) if think_words_list else 0,
        "median_think_words": int(np.median(think_words_list)) if think_words_list else 0,
        # Query quality
        "avg_query_copy_paste": round(float(np.mean(copy_paste_list)), 3) if copy_paste_list else None,
        "high_copy_paste_rate": round(
            sum(1 for x in copy_paste_list if x > 0.85) / len(copy_paste_list), 3
        ) if copy_paste_list else None,
    }

    if retrieval_scores:
        correct_ret = [r["retrieval_score"] for r in results if r["is_correct"] and r["retrieval_score"] is not None]
        wrong_ret = [r["retrieval_score"] for r in results if not r["is_correct"] and r["retrieval_score"] is not None]
        metrics["retrieval_cosine_mean"] = round(float(np.mean(retrieval_scores)), 3)
        metrics["retrieval_cosine_p50"] = round(float(np.median(retrieval_scores)), 3)
        metrics["retrieval_when_correct"] = round(float(np.mean(correct_ret)), 3) if correct_ret else None
        metrics["retrieval_when_wrong"] = round(float(np.mean(wrong_ret)), 3) if wrong_ret else None

    # ── Print summary ──
    print(f"\n  ── {benchmark_name} Results ──")
    print(f"  Accuracy:   {acc_overall:.1%}  ({n_correct}/{n})")
    if acc_with is not None and acc_without is not None:
        print(f"    with tool:    {acc_with:.1%}  ({correct_with}/{n_with_tool})")
        print(f"    without tool: {acc_without:.1%}  ({correct_without}/{n_without_tool})")
        diff = acc_with - acc_without
        print(f"    delta: {diff*100:+.1f} pts")
    print(f"  Tool freq:  {n_with_tool/n:.1%}  avg_calls={metrics['avg_tool_calls']:.2f}")
    print(f"  Format:     <think>={n_has_think/n:.1%}  <answer>={n_has_answer/n:.1%}")
    print(f"  Think depth: avg={metrics['avg_think_words']:.0f} words  median={metrics['median_think_words']}")
    if copy_paste_list:
        print(f"  Query copy-paste: avg={metrics['avg_query_copy_paste']:.2f}  high_rate={metrics['high_copy_paste_rate']:.2f}")
    if retrieval_scores:
        print(f"  Retrieval:  mean={metrics['retrieval_cosine_mean']:.3f}  correct={metrics['retrieval_when_correct']:.3f}  wrong={metrics['retrieval_when_wrong']:.3f}")

    return {"metrics": metrics, "per_sample": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-GRPO evaluation script")
    p.add_argument("--model-path", required=True, help="Path to merged model directory.")
    p.add_argument(
        "--benchmarks", nargs="+",
        default=["dataset/MedQA/test"],
        help="One or more load_from_disk paths. Split can be a sub-path (e.g. dataset/MedMCQA/test).",
    )
    p.add_argument("--data-dir", default="data/", help="KG data directory for retrieval tool.")
    p.add_argument("--no-tool", action="store_true", help="Forced no-tool ablation.")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--max-tool-iterations", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Lower than GRPO training temp (0.8) for deterministic eval.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-retrieval", action="store_true",
                   help="Score retrieval with MedEmbed (requires --no-tool=False).")
    p.add_argument("--output", default=None, help="Path to save JSON report.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mode = "no_tool" if args.no_tool else "with_tool"
    print(f"\n{'='*65}")
    print(f"Post-GRPO Evaluation  [{mode}]")
    print(f"  model:      {args.model_path}")
    print(f"  benchmarks: {args.benchmarks}")
    print(f"  n_samples:  {args.n_samples}  temp={args.temperature}  seed={args.seed}")
    print(f"{'='*65}")

    # Load retrieval tool (always load for with-tool mode; skip heavy load for no-tool)
    encoder = None
    if not args.no_tool:
        print("\nLoading KG retrieval tool ...")
        kg = MedicalKnowledgeTool.load(data_dir=args.data_dir)
        if args.score_retrieval:
            encoder = kg.encoder

    # Load model
    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto",
    ).eval()

    # Evaluate each benchmark
    benchmark_results = {}
    for bench_path in args.benchmarks:
        bench_name = Path(bench_path).name  # "test" or dataset name
        # Use parent dir name as readable name if split is "test"/"validation"
        if bench_name in ("test", "train", "validation"):
            bench_name = f"{Path(bench_path).parent.name}/{bench_name}"

        print(f"\nLoading {bench_path} ...")
        ds = load_benchmark(bench_path, args.n_samples, args.seed)

        result = eval_benchmark(
            model, tokenizer, ds, bench_name,
            no_tool=args.no_tool,
            max_tool_iterations=args.max_tool_iterations,
            temperature=args.temperature,
            score_retrieval=args.score_retrieval,
            encoder=encoder,
        )
        benchmark_results[bench_name] = result

    # ── Final summary ──
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"{'Benchmark':<30} {'Acc':>6} {'W/tool':>8} {'Wo/tool':>8} {'Delta':>7} {'Think':>6}")
    print("-" * 65)
    for bench, data in benchmark_results.items():
        m = data["metrics"]
        acc = f"{m['accuracy_overall']:.1%}"
        wt = f"{m['accuracy_with_tool']:.1%}" if m["accuracy_with_tool"] is not None else "  N/A "
        wot = f"{m['accuracy_without_tool']:.1%}" if m["accuracy_without_tool"] is not None else "  N/A "
        delta = f"{m['accuracy_diff_pts']:+.1f}" if m["accuracy_diff_pts"] is not None else " N/A "
        think = f"{m['avg_think_words']:.0f}w"
        print(f"{bench:<30} {acc:>6} {wt:>8} {wot:>8} {delta:>7} {think:>6}")

    # ── Save ──
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model_path": args.model_path,
            "mode": mode,
            "temperature": args.temperature,
            "n_samples_per_benchmark": args.n_samples,
            "seed": args.seed,
            "benchmarks": benchmark_results,
        }
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
