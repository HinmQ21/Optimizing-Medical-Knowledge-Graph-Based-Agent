# Baseline Benchmark

Zero-shot evaluation scripts — no fine-tuning required.
Load a raw/instruction-tuned model and directly prompt it on each dataset.

## Files

| File | Description |
|---|---|
| `medical_baseline.py` | **Unified** — supports all datasets via `--dataset` flag. **Recommended.** |
| `medqa_baseline.py` | MedQA-only (legacy, kept for reference) |
| `medmcqa_baseline.py` | MedMCQA-only (legacy, kept for reference) |
| `pubmedqa_baseline.py` | PubMedQA-only (legacy, kept for reference) |

## Supported Datasets

| `--dataset` | Path | Split | Samples | Task |
|---|---|---|---|---|
| `medqa` | `dataset/MedQA` | `test` | 1273 | 4-choice MCQ (A/B/C/D) |
| `medmcqa` | `dataset/MedMCQA_4options` | `validation` | 4183 | 4-choice MCQ (A/B/C/D) |
| `medxpertqa` | `dataset/MedXpertQA_Text` | `test` | 2450 | 10-choice MCQ (A/B/C/D/E/F/G/H/I/J) |
| `pubmedqa` | `dataset/PubMedQA` | `train` | 1000 | Yes / No / Maybe |

## Usage

```bash
# Evaluate one model
python baseline/medical_baseline.py \
    --dataset medqa \
    --model-path models/Qwen2.5-3B-Instruct

# Compare multiple models
python baseline/medical_baseline.py \
    --dataset medmcqa \
    --model-path models/Qwen2.5-3B-Instruct \
    --model-path outputs/my_checkpoint \
    --num-samples 500

# Full PubMedQA (1000 samples)
python baseline/medical_baseline.py \
    --dataset pubmedqa \
    --model-path models/Qwen2.5-3B-Instruct

# All models in a directory
python baseline/medical_baseline.py \
    --dataset medqa \
    --models-dir models/
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | `medqa`, `medmcqa`, or `pubmedqa` |
| `--model-path` | — | Model dir. Repeat for multiple models. |
| `--models-dir` | `models/` | Scan all subdirs if `--model-path` not set. |
| `--dataset-path` | *(auto)* | Override dataset path |
| `--split` | *(auto)* | Override dataset split |
| `--num-samples` | *(auto)* | Number of samples to evaluate |
| `--batch-size` | `1` | Inference batch size |
| `--output-dir` | *(auto)* | Override output directory |

## Output

Results are saved to `results/<dataset>_baseline/<model_name>/`:
- `summary.json` — accuracy, timing, metadata
- `predictions.jsonl` — per-sample predictions

An `all_results.json` comparing all models is written to the output dir root.

## Adding a New Dataset

In `medical_baseline.py`, add three functions and one registry entry:

```python
def _mydata_normalize(row): ...     # raw HF row -> {question, gold, ...}
def _mydata_build_prompt(norm): ... # normalized -> prompt string
def _mydata_parse(text): ...        # decoded output -> prediction or None

ADAPTERS["mydata"] = DatasetAdapter(
    default_path="dataset/MyData",
    default_split="test",
    default_num_samples=500,
    default_output_dir="results/mydata_baseline",
    normalize=_mydata_normalize,
    build_prompt=_mydata_build_prompt,
    parse=_mydata_parse,
)
```
