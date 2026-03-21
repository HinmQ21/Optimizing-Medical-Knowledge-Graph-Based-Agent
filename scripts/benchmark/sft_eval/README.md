# SFT Eval — Post-Training Evaluation Pipelines

Evaluation pipelines for models **after supervised fine-tuning (SFT)**.
Each pipeline uses the same reasoning prompt format the model was trained on,
then parses the structured output (`## Thinking / ## Final Response`).

## Files

| File | Description |
|---|---|
| `medical_sft_eval.py` | **Unified pipeline** — all datasets, all training styles |
| `medqa_qwen25_reasoning_huatuo_pipeline.py` | Legacy: HuaTuo, MedQA+MedMCQA only |
| `medqa_qwen25_reasoning_medreason_pipeline.py` | Legacy: MedReason, MedQA+MedMCQA only |
| `medqa_qwen25_reasoning_sglang.py` | SGLang HTTP server backend |

> **Prefer `medical_sft_eval.py`** for new evaluations. The legacy files are kept for reference.

---

## `medical_sft_eval.py` — Unified Pipeline

### Supported datasets

| `--dataset` | Default path | Split | Samples | Task |
|---|---|---|---|---|
| `medqa` | `dataset/MedQA` | `test` | 20 | 4-choice MCQ (A–D) |
| `medmcqa` | `dataset/MedMCQA_4options` | `validation` | 20 | 4-choice MCQ (A–D) |
| `medxpertqa` | `dataset/MedXpertQA_Text` | `test` | 2450 | 10-choice MCQ (A–J) |
| `pubmedqa` | `dataset/PubMedQA` | `train` | 1000 | yes / no / maybe |

### Supported training styles

| `--training` | Default section titles | Training pipeline |
|---|---|---|
| `huatuo` | `"Thinking"` / `"Final Response"` | `huatuo_stage1/qwen25_full_trainer_eval.py` |
| `medreason` | `"Thinking"` / `"Final Response"` | `medreason/qwen25_medreason_full_trainer_v2.py` |
| `custom` | `"Thinking"` / `"Final Response"` | Any — supply `--thinking-section-title` / `--final-section-title` |

### Prompt format

The model is expected to respond with:

```
## <thinking-section-title>
<step-by-step reasoning>

## <final-section-title>
The answer is <label>. <brief justification>
```

Where `<label>` is:
- `A/B/C/D` for MedQA / MedMCQA
- `A/B/C/D/E/F/G/H/I/J` for MedXpertQA
- `yes/no/maybe` for PubMedQA

### Usage

```bash
# HuaTuo model on MedQA
python sft_eval/medical_sft_eval.py \
    --training huatuo \
    --dataset medqa \
    --model-path outputs/huatuo_checkpoint \
    --num-samples 500

# MedReason model on MedMCQA
python sft_eval/medical_sft_eval.py \
    --training medreason \
    --dataset medmcqa \
    --model-path outputs/medreason_checkpoint

# MedReason model on MedXpertQA (10-option)
python sft_eval/medical_sft_eval.py \
    --training medreason \
    --dataset medxpertqa \
    --model-path outputs/medreason_checkpoint

# MedReason model on PubMedQA
python sft_eval/medical_sft_eval.py \
    --training medreason \
    --dataset pubmedqa \
    --model-path outputs/medreason_checkpoint

# Custom section titles (match training config exactly)
python sft_eval/medical_sft_eval.py \
    --training custom \
    --dataset medqa \
    --model-path outputs/my_checkpoint \
    --thinking-section-title "Reasoning" \
    --final-section-title "Answer"
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--training` | *(required)* | Training style: `huatuo`, `medreason`, `custom` |
| `--dataset` | *(required)* | Dataset: `medqa`, `medmcqa`, `medxpertqa`, `pubmedqa` |
| `--model-path` | *(required)* | Fine-tuned checkpoint directory |
| `--num-samples` | adapter default | Samples to evaluate (0 = use adapter default) |
| `--batch-size` | `4` | Inference batch size |
| `--max-new-tokens` | `384` | Max tokens for reasoning chain |
| `--prompt-max-length` | `3072` | Input truncation length |
| `--thinking-section-title` | style default | Override thinking header |
| `--final-section-title` | style default | Override final answer header |
| `--tag` | — | Extra label appended to `run_tag` in output metadata |

### Output

Results saved to `results/<dataset>_sft_eval/<model_name>/`:
- `summary.json` — accuracy, timing, fallback stats, section titles, training style
- `predictions.jsonl` — per-sample: question, pred, gold, raw_output, parse_source, fallback info

### Parse sources (in `predictions.jsonl`)

| `parse_source` | Meaning |
|---|---|
| `final_section:explicit` | Matched explicit pattern inside `## <final-title>` |
| `full_text:explicit` | Matched explicit pattern in full output |
| `final_section:option_text` | Matched option text inside `## <final-title>` (MCQ only) |
| `full_text:option_text` | Matched option text in full output (MCQ only) |
| `final_section:single_letter` | Only one valid letter found in `## <final-title>` (MCQ) |
| `final_section:single_word` | Only one valid word found in `## <final-title>` (PubMedQA) |
| `fallback_extract` | Primary parse failed; model was re-prompted to extract answer |
| `unparsed` | Could not extract answer |

### Extending to a new dataset

1. Implement five functions: `_X_normalize`, `_X_build_prompt`, `_X_parse`, `_X_build_fallback_prompt`
   (optionally `_X_parse_fallback`).
2. Register a `DatasetAdapter(...)` entry in `ADAPTERS`.
3. Add the dataset to this README table.

### Extending to a new training style

Add a `"style_name": {"thinking_title": "...", "final_title": "..."}` entry to `TRAINING_STYLES`.

---

## Legacy pipelines

The old per-training-style pipelines are kept for reference but support only MedQA and MedMCQA.
Migrate to `medical_sft_eval.py` for new experiments.

```bash
# Legacy HuaTuo
python sft_eval/medqa_qwen25_reasoning_huatuo_pipeline.py \
    --model-path outputs/huatuo_checkpoint --dataset medqa

# Legacy MedReason
python sft_eval/medqa_qwen25_reasoning_medreason_pipeline.py \
    --model-path outputs/medreason_checkpoint --dataset medmcqa \
    --thinking-section-title "Reasoning" --final-section-title "Answer"
```

---

## SGLang pipeline

Requires a running SGLang server (MedQA only):

```bash
python -m sglang.launch_server --model-path outputs/my_checkpoint --port 30000
python sft_eval/medqa_qwen25_reasoning_sglang.py \
    --model-path outputs/my_checkpoint --num-samples 500
```
