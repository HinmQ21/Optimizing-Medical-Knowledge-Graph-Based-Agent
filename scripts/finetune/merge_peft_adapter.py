#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: peft. Install it in the merge environment before running this script."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT/LoRA adapter into its base CausalLM and save a dense model."
    )
    parser.add_argument("--adapter-path", required=True, help="Directory containing adapter_config.json.")
    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Optional override for the base model path. Defaults to adapter_config.json -> base_model_name_or_path.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save the merged dense model.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Dense dtype used while loading and saving the merged model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Where to perform the merge. Use cuda when merging large models.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--safe-serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save weights as safetensors when enabled.",
    )
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "auto":
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def infer_base_model_path(adapter_path: Path) -> str:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_model_path = config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError(f"Could not infer base model from {config_path}")
    return str(base_model_path)


def load_tokenizer(adapter_path: Path, base_model_path: str, trust_remote_code: bool):
    last_error = None

    # Prefer the base tokenizer config because adapter checkpoints saved by PEFT/Unsloth
    # can contain a stripped tokenizer_config.json that breaks some runtimes such as SGLang.
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
    except Exception as exc:  # pragma: no cover - best-effort fallback
        last_error = exc
    else:
        chat_template_path = adapter_path / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
        return tokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=trust_remote_code)
    except Exception as exc:  # pragma: no cover - best-effort fallback
        last_error = exc
    else:
        return tokenizer

    raise RuntimeError("Failed to load tokenizer from base model path or adapter path.") from last_error


def build_model_kwargs(device: str, dtype: torch.dtype, trust_remote_code: bool) -> dict:
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if device == "auto":
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        kwargs["device_map"] = "auto"
    return kwargs


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    base_model_path = args.base_model_path or infer_base_model_path(adapter_path)
    dtype = resolve_dtype(args.dtype)
    model_kwargs = build_model_kwargs(args.device, dtype, args.trust_remote_code)

    print(f"Loading base model from: {base_model_path}")
    print(f"Loading adapter from: {adapter_path}")
    print(f"Merge dtype: {dtype}")

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    merged_model = peft_model.merge_and_unload()

    tokenizer = load_tokenizer(adapter_path, base_model_path, args.trust_remote_code)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(str(output_dir))

    print(f"Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()
