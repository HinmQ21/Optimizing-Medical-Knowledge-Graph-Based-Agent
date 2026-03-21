#!/usr/bin/env python3
import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/huggingface/datasets")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

DEFAULT_QWEN_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n\n' + message['content'] | trim + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n\n' }}{% endif %}"
)

DEFAULT_THRESHOLDS = [512, 1024, 2048, 4096, 8192]
DEFAULT_MEDREASON_PATTERNS = [
    "~/.cache/huggingface/hub/datasets--UCSC-VLAA--MedReason/snapshots/*/ours_quality_33000.jsonl",
]
DEFAULT_MEDICAL_O1_PATTERNS = [
    "~/.cache/huggingface/hub/datasets--FreedomIntelligence--medical-o1-reasoning-SFT/snapshots/*/medical_o1_sft.json",
]


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    display_name: str
    source: str
    config: str | None
    split: str
    prompt_definition: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze token-count statistics and distributions for MedReason and "
            "medical-o1-reasoning-SFT with the Qwen2.5-3B-Instruct tokenizer."
        )
    )
    parser.add_argument(
        "--tokenizer-path",
        default="models/Qwen2.5-3B-Instruct",
        help="Local tokenizer path or HF model id.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/dataset_token_stats/qwen25_3b_instruct",
        help="Directory for CSV, JSON summaries, and plots.",
    )
    parser.add_argument(
        "--medreason-source",
        default=None,
        help="MedReason source. Can be a local file/path or HF dataset id.",
    )
    parser.add_argument(
        "--medical-o1-source",
        default=None,
        help="medical-o1-reasoning-SFT source. Can be a local file/path or HF dataset id.",
    )
    parser.add_argument(
        "--medical-o1-config",
        default="en",
        help="Dataset config when loading medical-o1-reasoning-SFT from Hugging Face.",
    )
    parser.add_argument("--medreason-split", default="train")
    parser.add_argument("--medical-o1-split", default="train")
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for tokenizer calls.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Histogram bin count.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Plot resolution.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=int,
        default=DEFAULT_THRESHOLDS,
        help="Thresholds used for full-sequence length coverage summaries.",
    )
    return parser.parse_args()


def find_first_existing(patterns: list[str]) -> str | None:
    for pattern in patterns:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if matches:
            return matches[-1]
    return None


def resolve_medreason_source(user_source: str | None) -> str:
    if user_source:
        return user_source
    cached_source = find_first_existing(DEFAULT_MEDREASON_PATTERNS)
    return cached_source or "UCSC-VLAA/MedReason"


def resolve_medical_o1_source(user_source: str | None) -> str:
    if user_source:
        return user_source
    cached_source = find_first_existing(DEFAULT_MEDICAL_O1_PATTERNS)
    return cached_source or "FreedomIntelligence/medical-o1-reasoning-SFT"


def load_dataset_maybe_dict(source: str, dataset_config: str | None):
    source_path = Path(source).expanduser()

    if source_path.exists():
        if source_path.is_file() and source_path.suffix.lower() in {".json", ".jsonl"}:
            return load_dataset("json", data_files=str(source_path))
        return load_from_disk(str(source_path))

    if dataset_config:
        return load_dataset(source, dataset_config)
    return load_dataset(source)


def get_split_or_first(loaded: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(loaded, DatasetDict):
        if split in loaded:
            return loaded[split]
        return loaded[next(iter(loaded.keys()))]
    return loaded


def maybe_select(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def resolve_field(example: dict[str, Any], *candidates: str) -> str:
    for name in candidates:
        if name in example and example[name] is not None:
            return str(example[name])
    raise KeyError(f"Missing expected field. Tried: {', '.join(candidates)}")


def normalize_special_tokens(tokenizer) -> None:
    vocab = tokenizer.get_vocab()

    if tokenizer.eos_token in {None, "", "<EOS_TOKEN>"}:
        recovered = None
        if tokenizer.eos_token_id is not None:
            try:
                recovered = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
            except Exception:
                recovered = None

        if recovered and recovered not in {"", "<unk>", None, "<EOS_TOKEN>"}:
            tokenizer.eos_token = recovered
        elif "<|im_end|>" in vocab:
            tokenizer.eos_token = "<|im_end|>"
        else:
            raise ValueError(
                f"Could not resolve a valid eos_token. eos_token={tokenizer.eos_token!r}, "
                f"eos_token_id={tokenizer.eos_token_id!r}"
            )

    if tokenizer.pad_token in {None, "", "<PAD_TOKEN>", "<EOS_TOKEN>"}:
        tokenizer.pad_token = tokenizer.eos_token

    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = DEFAULT_QWEN_TEMPLATE

    tokenizer.padding_side = "right"


def render_chat(prompt: str, completion: str, tokenizer) -> tuple[str, str]:
    prompt_messages = [{"role": "user", "content": prompt}]
    full_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]

    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        prompt_text = f"<|im_start|>user\n\n{prompt}<|im_end|>\n<|im_start|>assistant\n\n"
        full_text = f"{prompt_text}{completion}<|im_end|>\n"

    return prompt_text, full_text


def format_medreason_record(example: dict[str, Any]) -> dict[str, str]:
    question = resolve_field(example, "question", "Question", "prompt")
    reasoning = resolve_field(
        example,
        "reasoning",
        "Reasoning",
        "Complex_CoT",
        "complex_cot",
    )
    response = resolve_field(
        example,
        "answer",
        "Answer",
        "Response",
        "response",
        "final_response",
    )
    completion = f"## Thinking\n\n{reasoning}\n\n## Final Response\n\n{response}"
    return {
        "question": question,
        "reasoning": reasoning,
        "response": response,
        "completion": completion,
    }


def format_medical_o1_record(example: dict[str, Any]) -> dict[str, str]:
    question = resolve_field(example, "Question", "question", "prompt")
    reasoning = resolve_field(
        example,
        "Complex_CoT",
        "complex_cot",
        "complex_cot_en",
        "reasoning",
        "Reasoning",
    )
    response = resolve_field(
        example,
        "Response",
        "response",
        "final_response",
        "answer",
        "Answer",
    )
    completion = f"## Thinking\n\n{reasoning}\n\n## Final Response\n\n{response}"
    return {
        "question": question,
        "reasoning": reasoning,
        "response": response,
        "completion": completion,
    }


def count_tokens(texts: list[str], tokenizer) -> list[int]:
    encodings = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        padding=False,
    )
    return [len(token_ids) for token_ids in encodings["input_ids"]]


def build_rows_from_batch(
    spec: DatasetSpec,
    batch: dict[str, list[Any]],
    tokenizer,
    offset: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    size = len(next(iter(batch.values())))
    formatter = format_medreason_record if spec.label == "medreason" else format_medical_o1_record

    formatted = [formatter({key: values[i] for key, values in batch.items()}) for i in range(size)]
    question_texts = [item["question"] for item in formatted]
    reasoning_texts = [item["reasoning"] for item in formatted]
    response_texts = [item["response"] for item in formatted]
    completion_texts = [item["completion"] for item in formatted]
    prompt_texts: list[str] = []
    full_texts: list[str] = []

    for question, completion in zip(question_texts, completion_texts, strict=True):
        prompt_text, full_text = render_chat(question, completion, tokenizer)
        prompt_texts.append(prompt_text)
        full_texts.append(full_text)

    question_tokens = count_tokens(question_texts, tokenizer)
    reasoning_tokens = count_tokens(reasoning_texts, tokenizer)
    response_tokens = count_tokens(response_texts, tokenizer)
    completion_tokens = count_tokens(completion_texts, tokenizer)
    prompt_tokens = count_tokens(prompt_texts, tokenizer)
    full_tokens = count_tokens(full_texts, tokenizer)

    source_ids = batch.get("id_in_dataset") or batch.get("id") or [None] * size

    for i in range(size):
        rows.append(
            {
                "dataset": spec.display_name,
                "dataset_label": spec.label,
                "row_idx": offset + i,
                "source_id": source_ids[i],
                "question_tokens": question_tokens[i],
                "reasoning_tokens": reasoning_tokens[i],
                "response_tokens": response_tokens[i],
                "completion_tokens": completion_tokens[i],
                "prompt_tokens": prompt_tokens[i],
                "full_tokens": full_tokens[i],
            }
        )

    return rows


def analyze_dataset(
    spec: DatasetSpec,
    tokenizer,
    max_samples: int | None,
    batch_size: int,
) -> pd.DataFrame:
    loaded = load_dataset_maybe_dict(spec.source, spec.config)
    dataset = maybe_select(get_split_or_first(loaded, spec.split), max_samples)
    rows: list[dict[str, Any]] = []

    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        batch = dataset[start:stop]
        rows.extend(build_rows_from_batch(spec, batch, tokenizer, start))
        if stop % max(batch_size * 10, 1) == 0 or stop == len(dataset):
            print(f"[{spec.label}] processed {stop}/{len(dataset)} samples")

    return pd.DataFrame(rows)


def summarize_series(values: pd.Series, thresholds: list[int] | None = None) -> dict[str, Any]:
    array = values.to_numpy(dtype=np.int32)
    quantiles = np.percentile(array, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    summary: dict[str, Any] = {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": int(array.min()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p10": float(quantiles[2]),
        "p25": float(quantiles[3]),
        "median": float(quantiles[4]),
        "p75": float(quantiles[5]),
        "p90": float(quantiles[6]),
        "p95": float(quantiles[7]),
        "p99": float(quantiles[8]),
        "max": int(array.max()),
        "iqr": float(quantiles[5] - quantiles[3]),
        "spread_p95_p05": float(quantiles[7] - quantiles[1]),
    }

    if thresholds:
        summary["share_lte_thresholds"] = {
            str(threshold): float((array <= threshold).mean()) for threshold in thresholds
        }
        summary["share_gt_thresholds"] = {
            str(threshold): float((array > threshold).mean()) for threshold in thresholds
        }

    return summary


def build_summary(
    df: pd.DataFrame,
    specs: list[DatasetSpec],
    thresholds: list[int],
    tokenizer,
) -> dict[str, Any]:
    metric_names = [
        "question_tokens",
        "reasoning_tokens",
        "response_tokens",
        "completion_tokens",
        "prompt_tokens",
        "full_tokens",
    ]
    dataset_summaries: dict[str, Any] = {}

    for spec in specs:
        dataset_df = df[df["dataset_label"] == spec.label]
        dataset_summaries[spec.label] = {
            "display_name": spec.display_name,
            "source": spec.source,
            "config": spec.config,
            "split": spec.split,
            "num_samples": int(len(dataset_df)),
            "prompt_definition": spec.prompt_definition,
            "metrics": {
                metric: summarize_series(
                    dataset_df[metric],
                    thresholds if metric == "full_tokens" else None,
                )
                for metric in metric_names
            },
        }

    medreason = dataset_summaries["medreason"]["metrics"]["full_tokens"]
    medical_o1 = dataset_summaries["medical_o1"]["metrics"]["full_tokens"]

    return {
        "tokenizer": {
            "path": tokenizer.name_or_path,
            "class": tokenizer.__class__.__name__,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "analysis_scope": {
            "full_sequence_definition": (
                "Chat-formatted user question plus assistant completion. "
                "Assistant completion is rendered as '## Thinking' + reasoning + "
                "'## Final Response' + answer, matching current fine-tuning scripts."
            ),
            "medreason_note": (
                "MedReason options are not appended to the prompt because the current "
                "training pipeline only uses the question text."
            ),
            "thresholds": thresholds,
        },
        "datasets": dataset_summaries,
        "comparison": {
            "sample_count_ratio_medreason_to_medical_o1": float(
                dataset_summaries["medreason"]["num_samples"]
                / dataset_summaries["medical_o1"]["num_samples"]
            ),
            "full_tokens_mean_delta_medreason_minus_medical_o1": float(
                medreason["mean"] - medical_o1["mean"]
            ),
            "full_tokens_median_delta_medreason_minus_medical_o1": float(
                medreason["median"] - medical_o1["median"]
            ),
            "full_tokens_p95_delta_medreason_minus_medical_o1": float(
                medreason["p95"] - medical_o1["p95"]
            ),
        },
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    medreason = summary["datasets"]["medreason"]
    medical_o1 = summary["datasets"]["medical_o1"]

    lines = [
        "# Dataset Token Statistics",
        "",
        f"- Tokenizer: `{summary['tokenizer']['path']}`",
        f"- EOS token: `{summary['tokenizer']['eos_token']}` ({summary['tokenizer']['eos_token_id']})",
        f"- PAD token: `{summary['tokenizer']['pad_token']}` ({summary['tokenizer']['pad_token_id']})",
        "",
        "## Scope",
        "",
        f"- {summary['analysis_scope']['full_sequence_definition']}",
        f"- {summary['analysis_scope']['medreason_note']}",
        "",
        "## Dataset Summary",
        "",
        (
            f"- {medreason['display_name']}: {medreason['num_samples']} samples, "
            f"full tokens mean={medreason['metrics']['full_tokens']['mean']:.2f}, "
            f"median={medreason['metrics']['full_tokens']['median']:.2f}, "
            f"p95={medreason['metrics']['full_tokens']['p95']:.2f}"
        ),
        (
            f"- {medical_o1['display_name']}: {medical_o1['num_samples']} samples, "
            f"full tokens mean={medical_o1['metrics']['full_tokens']['mean']:.2f}, "
            f"median={medical_o1['metrics']['full_tokens']['median']:.2f}, "
            f"p95={medical_o1['metrics']['full_tokens']['p95']:.2f}"
        ),
        "",
        "## Comparison",
        "",
        (
            "- MedReason minus medical-o1 full-token deltas: "
            f"mean={summary['comparison']['full_tokens_mean_delta_medreason_minus_medical_o1']:.2f}, "
            f"median={summary['comparison']['full_tokens_median_delta_medreason_minus_medical_o1']:.2f}, "
            f"p95={summary['comparison']['full_tokens_p95_delta_medreason_minus_medical_o1']:.2f}"
        ),
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_ecdf(ax, values: np.ndarray, label: str) -> None:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, label=label, linewidth=2)


def plot_overview(
    df: pd.DataFrame,
    thresholds: list[int],
    output_path: Path,
    bins: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    datasets = list(df["dataset"].unique())
    full_p99 = float(np.percentile(df["full_tokens"], 99))

    for dataset_name in datasets:
        subset = df[df["dataset"] == dataset_name]
        axes[0, 0].hist(
            subset["full_tokens"],
            bins=bins,
            alpha=0.45,
            label=dataset_name,
            density=True,
        )
        plot_ecdf(axes[0, 1], subset["full_tokens"].to_numpy(), dataset_name)

    axes[0, 0].set_title("Full Sequence Token Histogram")
    axes[0, 0].set_xlabel("Tokens")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_xlim(0, full_p99)
    axes[0, 0].legend()

    axes[0, 1].set_title("Full Sequence Token ECDF")
    axes[0, 1].set_xlabel("Tokens")
    axes[0, 1].set_ylabel("Cumulative share")
    axes[0, 1].legend()

    boxplot_values = [df[df["dataset"] == dataset_name]["full_tokens"] for dataset_name in datasets]
    axes[1, 0].boxplot(boxplot_values, tick_labels=datasets, showfliers=False)
    axes[1, 0].set_title("Full Sequence Token Boxplot")
    axes[1, 0].set_ylabel("Tokens")
    axes[1, 0].set_yscale("log")

    x = np.arange(len(thresholds))
    width = 0.35
    for idx, dataset_name in enumerate(datasets):
        subset = df[df["dataset"] == dataset_name]["full_tokens"].to_numpy()
        values = [(subset <= threshold).mean() for threshold in thresholds]
        axes[1, 1].bar(
            x + (idx - (len(datasets) - 1) / 2) * width,
            values,
            width=width,
            label=dataset_name,
        )
    axes[1, 1].set_title("Share Within Token Budget")
    axes[1, 1].set_xlabel("Full token threshold")
    axes[1, 1].set_ylabel("Share <= threshold")
    axes[1, 1].set_xticks(x, [str(t) for t in thresholds])
    axes[1, 1].set_ylim(0, 1.02)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_component_boxplots(df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    metrics = [
        "question_tokens",
        "reasoning_tokens",
        "response_tokens",
        "completion_tokens",
        "prompt_tokens",
        "full_tokens",
    ]
    titles = {
        "question_tokens": "Question",
        "reasoning_tokens": "Reasoning",
        "response_tokens": "Response",
        "completion_tokens": "Assistant Completion",
        "prompt_tokens": "Prompt Chat Prefix",
        "full_tokens": "Full Chat Sequence",
    }
    datasets = list(df["dataset"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, metric in zip(axes.flat, metrics, strict=True):
        values = [df[df["dataset"] == dataset_name][metric] for dataset_name in datasets]
        ax.boxplot(values, tick_labels=datasets, showfliers=False)
        ax.set_title(titles[metric])
        ax.set_ylabel("Tokens")
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    normalize_special_tokens(tokenizer)

    specs = [
        DatasetSpec(
            label="medreason",
            display_name="UCSC-VLAA/MedReason",
            source=resolve_medreason_source(args.medreason_source),
            config=None,
            split=args.medreason_split,
            prompt_definition="question only, aligned with current MedReason fine-tuning script",
        ),
        DatasetSpec(
            label="medical_o1",
            display_name="FreedomIntelligence/medical-o1-reasoning-SFT",
            source=resolve_medical_o1_source(args.medical_o1_source),
            config=args.medical_o1_config,
            split=args.medical_o1_split,
            prompt_definition="Question field rendered into the current Huatuo stage-1 chat format",
        ),
    ]

    frames = [
        analyze_dataset(
            spec=spec,
            tokenizer=tokenizer,
            max_samples=args.max_samples_per_dataset,
            batch_size=args.batch_size,
        )
        for spec in specs
    ]
    df = pd.concat(frames, ignore_index=True)
    summary = build_summary(df, specs, args.thresholds, tokenizer)

    df.to_csv(output_dir / "per_sample_token_counts.csv", index=False)
    save_json(summary, output_dir / "summary.json")
    write_markdown_summary(summary, output_dir / "summary.md")
    plot_overview(
        df=df,
        thresholds=args.thresholds,
        output_path=output_dir / "token_distribution_overview.png",
        bins=args.bins,
        dpi=args.dpi,
    )
    plot_component_boxplots(
        df=df,
        output_path=output_dir / "token_component_boxplots.png",
        dpi=args.dpi,
    )

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
