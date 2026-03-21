# Scripts layout

Folder `scripts/` is now organized by purpose instead of keeping all entrypoints in one level.

## Current structure

- `scripts/analysis/`: dataset profiling and token statistics utilities
- `scripts/finetune/huatuo_stage1/`: Huatuo stage-1 fine-tuning variants for Qwen2.5
- `scripts/finetune/medreason/`: MedReason SFT entrypoints adapted for single-node Unsloth training
- `scripts/benchmark/`: benchmark and evaluation scripts for MedQA and MedMCQA
- `scripts/serve/`: local serving helpers (for example Docker-based SGLang launch/stop wrappers)
- `scripts/setup/`: environment/bootstrap scripts

## Entry points

Canonical paths:

- `python scripts/analysis/dataset_token_stats_qwen25.py`
- `python scripts/finetune/huatuo_stage1/qwen25_lora.py`
- `python scripts/finetune/huatuo_stage1/qwen25_lora_eval.py`
- `python scripts/finetune/huatuo_stage1/qwen25_lora_logging.py`
- `python scripts/finetune/huatuo_stage1/qwen25_sfttrainer_eval.py`
- `python scripts/finetune/huatuo_stage1/qwen25_unsloth_sfttrainer_eval.py`
- `python scripts/finetune/merge_peft_adapter.py`
- `python scripts/finetune/medreason/qwen25_medreason_unsloth.py`
- `python scripts/finetune/medreason/qwen25_medreason_unsloth_sft.py`
- `python scripts/benchmark/medqa_baseline.py`
- `python scripts/benchmark/medmcqa_qwen35_pipeline.py`
- `python scripts/benchmark/medqa_qwen25_reasoning_pipeline.py`
- `python scripts/benchmark/medqa_qwen25_reasoning_sglang.py`
- `bash scripts/serve/launch_sglang_server.sh --model-path <path> --detach`
- `bash scripts/setup/unsloth_venv_dgx_spark.sh .venv_unsloth`

## Notes

- `scripts/__pycache__/` is generated runtime artifact, not part of the intended hand-maintained layout.
