"""Experimental generic scorer for the pinned native-context Gemma 4 12B runtime.

Input rows use the OpenJEV ``id``/``state``/``question``/``options`` interface.
The scorer emits native final-position option logits without truncation or a KV
cache.  It is an experimental module rather than a production CLI entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from experiments.long_context import pilot
from openjev_phase1.core import _pinned_source, load_causal_model, softmax, validate_row
from openjev_phase1.direct import PROMPT_VERSION, _forward, encode_prompt


SCHEMA = "openjev-experimental-long-context-score-v1"
READOUT = "native full-vocabulary last-position logits restricted to declared answer slots"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_rows(path: Path) -> tuple[list[dict], str]:
    """Read and validate generic OpenJEV JSONL without requiring labels."""
    raw = path.read_bytes()
    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on input line {line_number}: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Input line {line_number} must be a JSON object")
        validate_row(row)
        rows.append(row)
    if not rows:
        raise ValueError("--input must contain at least one JSONL row")
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"Duplicate input row ID: {row['id']}")
        seen.add(row["id"])
    return rows, _sha256(raw)


def build_prediction_record(
    row: dict,
    logits: list[float],
    *,
    input_tokens: int,
    forward_seconds: float,
    total_seconds: float,
    peak_memory_bytes: int,
    prompt_sha256: str,
    source_sha256: str,
    model_metadata: dict,
    runtime_backend: dict,
) -> dict:
    """Map numeric readout data back to caller-declared option IDs."""
    option_ids = [option["id"] for option in row["options"]]
    if len(logits) != len(option_ids) or any(not math.isfinite(value) for value in logits):
        raise ValueError("Option logits must be finite and match the declared options")
    probabilities = softmax(logits)
    prediction = option_ids[max(range(len(probabilities)), key=probabilities.__getitem__)]
    return {
        "schema": SCHEMA,
        "status": "ok",
        "id": row["id"],
        "option_ids": option_ids,
        "option_logits": logits,
        "probabilities": probabilities,
        "prediction": prediction,
        "actual_tokens": input_tokens,
        "input_tokens": input_tokens,
        "peak_memory_bytes": peak_memory_bytes,
        "forward_seconds": forward_seconds,
        "total_seconds": total_seconds,
        "timing_scope": {
            "forward_seconds": (
                "synchronized model forward and option-logit selection; excludes input tensor "
                "preparation, tokenization, mask construction, and model loading"
            ),
            "total_seconds": (
                "per-row input tensor preparation, synchronized forward, option-logit transfer, "
                "and finite-value validation; excludes record construction, tokenization, mask "
                "construction, preflight, and model loading"
            ),
        },
        "prompt_sha256": prompt_sha256,
        "prompt_version": PROMPT_VERSION,
        "source_sha256": source_sha256,
        "model": model_metadata,
        "base_model": {"source": pilot.MODEL, "revision": pilot.REVISION},
        "adapter_sha256": model_metadata.get("adapter_sha256"),
        "runtime_backend": runtime_backend,
        "flex_kernel_options": pilot.FLEX_KERNEL_OPTIONS,
        "readout": READOUT,
        "probability_status": "conditional option scores; uncalibrated as decision confidence",
    }


def score_row(
    model,
    row: dict,
    *,
    device,
    masks: dict,
    encoding: tuple[list[int], list[int], str],
    source_sha256: str,
    model_metadata: dict,
    runtime_backend: dict,
) -> dict:
    """Score one pre-encoded row with native analytical BlockMasks."""
    import torch

    started = time.perf_counter()
    ids, slots, prompt_sha256 = encoding
    if not masks:
        raise RuntimeError("Refusing scoring without native analytical BlockMasks")
    if device.type != "cuda":
        raise RuntimeError("Long-context scoring requires exactly one CUDA GPU")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    position_ids = torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    forward_started = time.perf_counter()
    with torch.inference_mode():
        vocabulary = _forward(model, {
            "input_ids": input_ids,
            "attention_mask": masks,
            "position_ids": position_ids,
            "kernel_options": pilot.FLEX_KERNEL_OPTIONS,
        })[0].float()
        selected = vocabulary[torch.tensor(slots, dtype=torch.long, device=device)]
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    if not bool(torch.isfinite(selected).all().item()):
        raise RuntimeError(f"Row {row['id']} produced nonfinite option logits")
    return build_prediction_record(
        row,
        selected.cpu().tolist(),
        input_tokens=len(ids),
        forward_seconds=forward_seconds,
        total_seconds=time.perf_counter() - started,
        peak_memory_bytes=torch.cuda.max_memory_allocated(device),
        prompt_sha256=prompt_sha256,
        source_sha256=source_sha256,
        model_metadata=model_metadata,
        runtime_backend=runtime_backend,
    )


def _validate_args(args) -> None:
    if args.output.exists():
        raise FileExistsError("--output must be a new create-only path")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--input and --output must be different paths")
    if isinstance(args.max_tokens, bool) or not 1 <= args.max_tokens <= pilot.MAX_NATIVE_CONTEXT:
        raise ValueError(f"--max-tokens must be between 1 and {pilot.MAX_NATIVE_CONTEXT}")
    if bool(args.adapter) != bool(args.adapter_revision):
        raise ValueError("--adapter and --adapter-revision must be provided together")
    _pinned_source(pilot.MODEL, pilot.REVISION, "models")
    if args.adapter:
        _pinned_source(args.adapter, args.adapter_revision, "adapters")


def run(args) -> list[dict]:
    """Validate, load the pinned runtime, and create one JSONL output."""
    _validate_args(args)
    rows, source_sha256 = read_rows(args.input)

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError("Compiler fallback must be disabled")
    device = torch.device("cuda:0")
    model, tokenizer, metadata = load_causal_model(
        pilot.MODEL,
        pilot.REVISION,
        adapter=args.adapter,
        adapter_revision=args.adapter_revision,
        quantization="nf4",
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    backend = pilot._select_and_verify_flex(model)
    model.config.use_cache = False
    model.eval()
    if bool(getattr(model.config, "use_cache", True)):
        raise RuntimeError("Model cache must remain disabled")
    config = pilot._model_config(model)
    native_limit = int(getattr(config, "max_position_embeddings", pilot.MAX_NATIVE_CONTEXT))
    if args.max_tokens > native_limit:
        raise RuntimeError("--max-tokens exceeds the loaded model's native context")

    # Complete tokenizer/boundary/length preflight before creating the output.
    encodings = [encode_prompt(tokenizer, row, args.max_tokens) for row in rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with args.output.open("x", encoding="utf-8") as output:
        for row, encoding in zip(rows, encodings):
            masks, mask_inventory = pilot._prepare_masks(model, len(encoding[0]), device)
            row_backend = {**backend, "analytical_block_masks": mask_inventory}
            record = score_row(
                model,
                row,
                device=device,
                masks=masks,
                encoding=encoding,
                source_sha256=source_sha256,
                model_metadata=metadata,
                runtime_backend=row_backend,
            )
            records.append(record)
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            del masks
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-revision")
    args = parser.parse_args(argv)
    records = run(args)
    print(json.dumps({
        "schema": SCHEMA,
        "status": "complete",
        "rows": len(records),
        "output": str(args.output),
        "output_sha256": _sha256(args.output.read_bytes()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
