"""Evaluate project-authored distant-evidence JSONL with pinned Gemma 4 12B.

This is an exploratory authored-data check, not a public benchmark.  It uses
native final-position option logits without truncation or cache fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from experiments.long_context import pilot
from openjev_phase1.core import load_causal_model, softmax, validate_row
from openjev_phase1.direct import _forward, encode_prompt


SCHEMA = "openjev-authored-long-context-evaluation-v1"
LABELS = ("supported", "contradicted", "insufficient")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_authored_rows(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError("--input must contain at least one JSONL row")
    seen = set()
    for row in rows:
        validate_row(row)
        if row["id"] in seen:
            raise ValueError(f"Duplicate input row ID: {row['id']}")
        seen.add(row["id"])
        option_ids = [option["id"] for option in row["options"]]
        if set(option_ids) != set(LABELS):
            raise ValueError(f"Row {row['id']} does not declare the authored semantic labels")
        if row.get("gold_option_id") not in option_ids:
            raise ValueError(f"Row {row['id']} has no matching semantic gold option")
        if row.get("task") != "evidence" or row.get("authored") is not True:
            raise ValueError(f"Row {row['id']} is not authored evidence data")
        provenance = row.get("provenance", {})
        if provenance.get("kind") != "project-authored":
            raise ValueError(f"Row {row['id']} lacks project-authored provenance")
        if row.get("evidence_position") not in {"early", "middle", "late"}:
            raise ValueError(f"Row {row['id']} lacks a supported evidence position")
    return rows, _sha256(raw)


def _macro_f1(records: list[dict], labels: list[str]) -> float:
    values = []
    for label in labels:
        true_positive = sum(record["gold_option_id"] == label == record["prediction"]
                            for record in records)
        false_positive = sum(record["gold_option_id"] != label and record["prediction"] == label
                             for record in records)
        false_negative = sum(record["gold_option_id"] == label and record["prediction"] != label
                             for record in records)
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(values) / len(values)


def _metrics(records: list[dict], labels: list[str]) -> dict:
    if not records:
        return {"n": 0, "accuracy": None, "macro_f1": None}
    return {
        "n": len(records),
        "accuracy": sum(record["prediction"] == record["gold_option_id"] for record in records) / len(records),
        "macro_f1": _macro_f1(records, labels),
    }


def summarize(records: list[dict], selected_rows: list[dict], all_rows: list[dict]) -> dict:
    labels = list(LABELS)
    by_gold = defaultdict(list)
    by_position = defaultdict(list)
    for record in records:
        by_gold[record["gold_option_id"]].append(record)
        by_position[record["evidence_position"]].append(record)

    by_label = {}
    for label in labels:
        subset = by_gold[label]
        class_tp = sum(record["prediction"] == label for record in subset)
        predicted = sum(record["prediction"] == label for record in records)
        denominator = len(subset) + predicted
        by_label[label] = {
            **_metrics(subset, labels),
            "class_f1": 0.0 if denominator == 0 else 2 * class_tp / denominator,
        }
    return {
        "overall": _metrics(records, labels),
        "by_label": by_label,
        "by_evidence_position": {
            position: _metrics(by_position[position], labels)
            for position in ("early", "middle", "late")
        },
        "coverage": {
            "input_rows": len(all_rows),
            "selected_rows": len(selected_rows),
            "omitted_rows": len(all_rows) - len(selected_rows),
            "selection": "deterministic first rows in input order",
            "selected_labels": dict(sorted(Counter(row["gold_option_id"] for row in selected_rows).items())),
            "selected_evidence_positions": dict(sorted(Counter(
                row["evidence_position"] for row in selected_rows
            ).items())),
            "full_labels": dict(sorted(Counter(row["gold_option_id"] for row in all_rows).items())),
            "full_evidence_positions": dict(sorted(Counter(
                row["evidence_position"] for row in all_rows
            ).items())),
        },
    }


def build_prediction_record(row: dict, logits: list[float], *, actual_tokens: int,
                            elapsed_seconds: float, peak_vram_bytes: int,
                            prompt_sha256: str, source_sha256: str,
                            model_metadata: dict) -> dict:
    """Map semantic row IDs and numeric readout data into a non-prompt output."""
    option_ids = [option["id"] for option in row["options"]]
    if len(logits) != len(option_ids):
        raise ValueError("Option logits do not match the declared options")
    probabilities = softmax(logits)
    prediction = option_ids[max(range(len(probabilities)), key=probabilities.__getitem__)]
    return {
        "schema": SCHEMA,
        "status": "ok",
        "id": row["id"],
        "gold_option_id": row["gold_option_id"],
        "prediction": prediction,
        "option_ids": option_ids,
        "option_logits": logits,
        "probabilities": probabilities,
        "probability_status": "conditional option scores; uncalibrated",
        "actual_tokens": actual_tokens,
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "prompt_sha256": prompt_sha256,
        "source_sha256": source_sha256,
        "model_revision": pilot.REVISION,
        "adapter_sha256": model_metadata.get("adapter_sha256"),
        "evidence_position": row["evidence_position"],
    }


def score_row(model, tokenizer, row: dict, *, max_tokens: int, device, masks: dict,
              source_sha256: str, model_metadata: dict,
              encoding: tuple[list[int], list[int], str] | None = None) -> dict:
    """Score one row with native BlockMasks prepared for its exact token length."""
    import torch

    started = time.perf_counter()
    ids, slots, prompt_sha256 = encoding or encode_prompt(tokenizer, row, max_tokens)
    if not masks:
        raise RuntimeError("Refusing evaluation without native analytical BlockMasks")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    position_ids = torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0)
    if device.type != "cuda":
        raise RuntimeError("Authored long-context evaluation requires exactly one CUDA GPU")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        vocabulary = _forward(model, {
            "input_ids": input_ids,
            "attention_mask": masks,
            "position_ids": position_ids,
            "kernel_options": pilot.FLEX_KERNEL_OPTIONS,
        })[0].float()
        selected = vocabulary[torch.tensor(slots, dtype=torch.long, device=device)]
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(selected).all().item()):
        raise RuntimeError(f"Row {row['id']} produced nonfinite option logits")
    logits = selected.cpu().tolist()
    return build_prediction_record(
        row, logits, actual_tokens=len(ids), elapsed_seconds=time.perf_counter() - started,
        peak_vram_bytes=torch.cuda.max_memory_allocated(device), prompt_sha256=prompt_sha256,
        source_sha256=source_sha256, model_metadata=model_metadata,
    )


def _write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def run(args) -> dict:
    import torch

    if args.output.exists() or args.summary.exists():
        raise FileExistsError("--output and --summary must both be new create-only paths")
    if args.output.resolve() == args.summary.resolve():
        raise ValueError("--output and --summary must be different paths")
    if isinstance(args.max_tokens, bool) or not 1 <= args.max_tokens <= pilot.MAX_NATIVE_CONTEXT:
        raise ValueError(f"--max-tokens must be between 1 and {pilot.MAX_NATIVE_CONTEXT}")
    if args.limit is not None and (isinstance(args.limit, bool) or args.limit < 1):
        raise ValueError("--limit must be positive")
    if bool(args.adapter) != bool(args.adapter_revision):
        raise ValueError("--adapter and --adapter-revision must be provided together")
    rows, source_sha256 = read_authored_rows(args.input)
    selected_rows = rows if args.limit is None else rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.open("x", encoding="utf-8")
    args.summary.open("x", encoding="utf-8").close()
    started = time.time()
    records = []
    base_summary = {
        "schema": SCHEMA,
        "status": "running",
        "exploratory": True,
        "public_benchmark": False,
        "limitations": [
            "Project-authored fictional records are an exploratory check, not a public benchmark.",
            "Probabilities are conditional on the supplied options and are not calibrated confidence.",
            "--limit selects the deterministic input prefix and can omit labels or evidence positions.",
        ],
        "source_sha256": source_sha256,
        "model": {"source": pilot.MODEL, "revision": pilot.REVISION},
        "evaluator_sha256": _sha256(Path(__file__).read_bytes()),
        "adapter": args.adapter,
        "adapter_revision": args.adapter_revision,
        "requested_max_tokens": args.max_tokens,
        "started_unix": started,
    }
    _write_summary(args.summary, base_summary)
    try:
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
        if args.max_tokens > int(getattr(config, "max_position_embeddings", pilot.MAX_NATIVE_CONTEXT)):
            raise RuntimeError("--max-tokens exceeds the loaded model's native context")

        inventories = {}
        for row in selected_rows:
            try:
                encoding = encode_prompt(tokenizer, row, args.max_tokens)
                ids, _slots, _prompt = encoding
                length = len(ids)
                masks, inventories[str(length)] = pilot._prepare_masks(model, length, device)
                record = score_row(
                    model, tokenizer, row, max_tokens=args.max_tokens, device=device,
                    masks=masks, source_sha256=source_sha256, model_metadata=metadata,
                    encoding=encoding,
                )
                del masks
            except Exception as error:
                failure = {
                    "schema": SCHEMA, "status": "failed", "id": row["id"],
                    "gold_option_id": row["gold_option_id"],
                    "source_sha256": source_sha256, "model_revision": pilot.REVISION,
                    "adapter_sha256": metadata.get("adapter_sha256"),
                    "error_type": type(error).__name__, "error": str(error),
                }
                output.write(json.dumps(failure, sort_keys=True) + "\n")
                output.flush()
                raise
            records.append(record)
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
        output.close()
        summary = {
            **base_summary,
            "status": "complete",
            "completed_unix": time.time(),
            "model_metadata": metadata,
            "backend": backend,
            "flex_kernel_options": pilot.FLEX_KERNEL_OPTIONS,
            "mask_inventory_by_actual_tokens": inventories,
            "output_sha256": _sha256(args.output.read_bytes()),
            "metrics": summarize(records, selected_rows, rows),
        }
        _write_summary(args.summary, summary)
        return summary
    except Exception as error:
        if not output.closed:
            output.flush()
            output.close()
        failure_summary = {
            **base_summary,
            "status": "failed",
            "failed_unix": time.time(),
            "completed_rows": len(records),
            "output_sha256": _sha256(args.output.read_bytes()),
            "error_type": type(error).__name__,
            "error": str(error),
            "metrics": summarize(records, selected_rows, rows),
        }
        _write_summary(args.summary, failure_summary)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-revision")
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
