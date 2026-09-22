"""Fixed nine-step authored long-context LoRA training pilot.

This is a bounded experiment on fictional project-authored records.  It is not
a public benchmark and cannot establish generalized long-context competence.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from experiments.long_context import authored_data, evaluate_authored, pilot
from openjev_phase1.core import load_causal_model
from openjev_phase1.direct import _forward, encode_prompt
from openjev_phase1.training import _configure_lora


SCHEMA = "openjev-authored-long-context-training-pilot-v1"
SEED = 42
LEARNING_RATE = 2e-4
GRAD_CLIP_NORM = 1.0
TRAIN_STEPS = 9
POSITIONS = ("early", "middle", "late")
LABELS = evaluate_authored.LABELS


class TrainingTerminated(RuntimeError):
    """Raised when an external termination request reaches a step boundary."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_id(row: dict) -> str:
    source = row.get("source")
    value = source.get("source_id") if isinstance(source, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"Row {row.get('id')} lacks a source.source_id")
    return value


def select_balanced_nine(rows: list[dict], *, split: str) -> list[dict]:
    """Select a fixed 3x3 label/position grid across exactly three source groups.

    Selection uses only the supplied split.  The cyclic group assignment gives
    each group three rows and makes all nine semantic cells explicit.
    """
    if split not in {"train", "validation"}:
        raise ValueError("Balanced selection supports train or validation only")
    raw_groups = [row.get("group_id") for row in rows]
    if any(not isinstance(group, str) or not group for group in raw_groups):
        raise ValueError(f"{split} rows require nonempty source groups")
    groups = sorted(set(raw_groups))
    if len(groups) != 3:
        raise ValueError(f"{split} must contain exactly three nonempty source groups")
    by_cell: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        if row.get("split") != split:
            raise ValueError(f"Row {row.get('id')} is not assigned to {split}")
        key = (row["group_id"], row.get("evidence_position"), row.get("gold_option_id"))
        by_cell.setdefault(key, []).append(row)

    selected = []
    for position_index, position in enumerate(POSITIONS):
        for label_index, label in enumerate(LABELS):
            group = groups[(position_index + label_index) % len(groups)]
            candidates = by_cell.get((group, position, label), [])
            if len(candidates) != 1:
                raise ValueError(
                    f"{split} requires exactly one row for group={group}, "
                    f"position={position}, label={label}"
                )
            selected.append(candidates[0])
    if len(selected) != TRAIN_STEPS or len({row["id"] for row in selected}) != TRAIN_STEPS:
        raise RuntimeError("Balanced selection did not produce nine unique rows")
    return selected


def reorder_and_target(row: dict, *, seed: int = SEED) -> tuple[dict, int]:
    """Reorder one row deterministically and derive its target from semantic IDs."""
    reordered = authored_data.reorder_options(row, seed)
    option_ids = [option["id"] for option in reordered["options"]]
    try:
        target = option_ids.index(row["gold_option_id"])
    except ValueError as error:
        raise ValueError(f"Row {row['id']} gold semantic ID is absent after reordering") from error
    if option_ids[target] != row["gold_option_id"]:
        raise RuntimeError("Semantic target derivation failed")
    return reordered, target


def preflight(args) -> dict:
    """Validate all CPU-visible invariants before creating output paths."""
    if args.output.exists():
        raise FileExistsError("--output must be a new create-only directory")
    if args.train.resolve() == args.validation.resolve():
        raise ValueError("--train and --validation must be different files")
    if isinstance(args.max_tokens, bool) or not 1 <= args.max_tokens <= pilot.MAX_NATIVE_CONTEXT:
        raise ValueError(f"--max-tokens must be between 1 and {pilot.MAX_NATIVE_CONTEXT}")
    if not args.activation_offload:
        raise ValueError("The frozen authored pilot plan requires --activation-offload")

    plan_raw = args.plan.read_bytes()
    plan = json.loads(plan_raw)
    expected_plan = {
        "schema": "worthify-authored-long-context-plan-v1",
        "base_model": pilot.MODEL,
        "base_revision": pilot.REVISION,
        "training_examples": TRAIN_STEPS,
        "epochs": 1,
        "optimizer_steps": TRAIN_STEPS,
        "micro_batch": 1,
        "effective_batch": 1,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "max_tokens": args.max_tokens,
        "activation_offload": True,
        "quantization": "NF4",
        "compute_dtype": "BF16",
        "lora_rank": 16,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "checkpoint_selection": "None: use the fixed ninth-step checkpoint, report even if worse.",
    }
    mismatched = {key: (plan.get(key), value) for key, value in expected_plan.items()
                  if plan.get(key) != value}
    if mismatched:
        raise ValueError(f"--plan does not match the fixed authored pilot: {mismatched}")
    if "All 27 held-out test rows" not in plan.get("evaluation", "") or \
            "all 27 validation rows" not in plan.get("evaluation", ""):
        raise ValueError("--plan must predeclare all 27 long validation rows and held-out tests")

    train_rows, train_sha256 = evaluate_authored.read_authored_rows(args.train)
    validation_rows, validation_sha256 = evaluate_authored.read_authored_rows(args.validation)
    if any(row.get("split") != "train" for row in train_rows):
        raise ValueError("--train may contain only rows assigned to train")
    if any(row.get("split") != "validation" for row in validation_rows):
        raise ValueError("--validation may contain only rows assigned to validation")

    train_ids = {row["id"] for row in train_rows}
    validation_ids = {row["id"] for row in validation_rows}
    train_groups = {row["group_id"] for row in train_rows}
    validation_groups = {row["group_id"] for row in validation_rows}
    train_sources = {_source_id(row) for row in train_rows}
    validation_sources = {_source_id(row) for row in validation_rows}
    for label, left, right in (
        ("row ID", train_ids, validation_ids),
        ("source group", train_groups, validation_groups),
        ("source ID", train_sources, validation_sources),
    ):
        overlap = sorted(left & right)
        if overlap:
            raise ValueError(f"Training/validation leakage through {label}: {overlap[:3]}")

    selected_train = select_balanced_nine(train_rows, split="train")
    if len(validation_rows) != 27:
        raise ValueError("The frozen authored pilot requires all 27 validation rows")
    return {
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "selected_train": selected_train,
        "selected_validation": validation_rows,
        "train_sha256": train_sha256,
        "validation_sha256": validation_sha256,
        "plan": plan,
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
    }


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _save_step_checkpoint(path: Path, model, *, step: int, specification: dict) -> None:
    """Atomically persist an adapter-only step export for interruption recovery."""
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=False, exist_ok=False)
    model.save_pretrained(temporary, safe_serialization=True)
    state = {
        "schema": SCHEMA,
        "completed_step": step,
        "specification": specification,
    }
    (temporary / "checkpoint-receipt.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = sorted(item.name for item in temporary.iterdir() if item.is_file())
    (temporary / "SHA256SUMS").write_text(
        "".join(f"{_sha256_file(temporary / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _train_step(model, tokenizer, row: dict, target: int, *, max_tokens: int,
                device, optimizer, activation_offload: bool,
                encoding: tuple[list[int], list[int], str] | None = None) -> dict:
    import torch

    encoded, slots, prompt_sha256 = encoding or encode_prompt(tokenizer, row, max_tokens)
    length = len(encoded)
    masks, mask_inventory = pilot._prepare_masks(model, length, device)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    position_ids = torch.arange(length, dtype=torch.long, device=device).unsqueeze(0)
    slot_tensor = torch.tensor(slots, dtype=torch.long, device=device)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    context = nullcontext(None)
    if activation_offload:
        from experiments.long_context.activation_offload import checkpoint_activation_offload
        context = checkpoint_activation_offload(model)
    with context as offload_stats:
        vocabulary = _forward(model, {
            "input_ids": input_ids,
            "attention_mask": masks,
            "position_ids": position_ids,
            "kernel_options": pilot.FLEX_KERNEL_OPTIONS,
        })[0].float()
        logits = vocabulary.index_select(0, slot_tensor)
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.tensor([target], dtype=torch.long, device=device)
        )
        if not bool(torch.isfinite(loss).item()) or not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError(f"Row {row['id']} produced nonfinite loss or option logits")
        loss.backward()

    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters")
    nonzero = 0
    for name, parameter in trainable:
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError(f"Missing or nonfinite gradient for {name}")
        nonzero += int(torch.count_nonzero(gradient).item() > 0)
    if nonzero == 0:
        raise RuntimeError("All LoRA gradients are zero")
    grad_norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], GRAD_CLIP_NORM)
    if not bool(torch.isfinite(grad_norm).item()):
        raise RuntimeError("Gradient norm is nonfinite")
    optimizer.step()
    torch.cuda.synchronize(device)
    if any(not bool(torch.isfinite(parameter).all().item()) for _, parameter in trainable):
        raise RuntimeError("Optimizer step produced nonfinite LoRA parameters")
    if activation_offload and not offload_stats["offloaded_bytes"]:
        raise RuntimeError("Activation offload copied no checkpoint inputs")
    elapsed = time.perf_counter() - started
    loss_value = float(loss.detach().item())
    del masks, input_ids, position_ids, vocabulary, logits, loss
    return {
        "row_id": row["id"],
        "source_id": _source_id(row),
        "group_id": row["group_id"],
        "evidence_position": row["evidence_position"],
        "gold_option_id": row["gold_option_id"],
        "target_index_after_reorder": target,
        "prompt_sha256": prompt_sha256,
        "actual_tokens": length,
        "loss": loss_value,
        "loss_finite": True,
        "gradients_finite": True,
        "optimizer_parameters_finite": True,
        "gradient_norm_before_clip": float(grad_norm.detach().item()),
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "nonzero_gradient_tensor_count": nonzero,
        "gradient_tensor_count": len(trainable),
        "elapsed_seconds": elapsed,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "host_rss_bytes": pilot._rss_bytes(),
        "activation_offload": offload_stats,
        "masks": mask_inventory,
    }


def _validate(model, tokenizer, rows: list[dict], *, max_tokens: int, device,
              source_sha256: str, output_path: Path, encodings: dict[str, tuple],
              adapter_sha256: str) -> tuple[list[dict], dict]:
    records = []
    model.eval()
    for row in rows:
        encoded = encodings[row["id"]]
        masks, _inventory = pilot._prepare_masks(model, len(encoded[0]), device)
        record = evaluate_authored.score_row(
            model, tokenizer, row, max_tokens=max_tokens, device=device, masks=masks,
            source_sha256=source_sha256,
            model_metadata={"adapter_sha256": adapter_sha256}, encoding=encoded,
        )
        records.append(record)
        _append_jsonl(output_path, record)
        del masks
    summary = evaluate_authored.summarize(records, rows, rows)
    return records, summary


def _export_final_adapter(path: Path, *, receipt: dict, validation_summary: dict,
                          validation_predictions: Path, temporary: Path,
                          adapter_sha256: str) -> dict:
    local_revision = f"sha256:{adapter_sha256}"
    training_manifest = {
        "schema": SCHEMA,
        "model": {"source": pilot.MODEL, "revision": pilot.REVISION},
        "seed": SEED,
        "adapter_sha256": adapter_sha256,
        "local_adapter_revision": local_revision,
        "train_source_sha256": receipt["inputs"]["train_sha256"],
        "validation_source_sha256": receipt["inputs"]["validation_sha256"],
        "fixed_optimizer_steps": TRAIN_STEPS,
        "validation_predictions_sha256": _sha256_file(validation_predictions),
        "validation": validation_summary,
        "limitations": receipt["limitations"],
    }
    (temporary / "training-manifest.json").write_text(
        json.dumps(training_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    adapter_receipt = {
        **training_manifest,
        "selected_training_rows": receipt["selected_training_rows"],
        "training_code_sha256": receipt["runtime"]["training_code_sha256"],
        "reload": {
            "adapter": str(path.resolve()),
            "adapter_revision": local_revision,
            "evaluator": "experiments.long_context.evaluate_authored",
        },
    }
    (temporary / "adapter-receipt.json").write_text(
        json.dumps(adapter_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names = sorted(item.name for item in temporary.iterdir() if item.is_file())
    (temporary / "SHA256SUMS").write_text(
        "".join(f"{_sha256_file(temporary / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return {"adapter_sha256": adapter_sha256, "local_adapter_revision": local_revision}


def run(args) -> dict:
    prepared = preflight(args)
    import torch

    selected_train = prepared["selected_train"]
    selected_validation = prepared["selected_validation"]
    specification = {
        "model": pilot.MODEL,
        "revision": pilot.REVISION,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "gradient_clip_norm": GRAD_CLIP_NORM,
        "microbatch_size": 1,
        "effective_batch_size": 1,
        "epochs": 1,
        "max_tokens": args.max_tokens,
        "activation_offload": args.activation_offload,
        "plan_sha256": prepared["plan_sha256"],
        "train_sha256": prepared["train_sha256"],
        "validation_sha256": prepared["validation_sha256"],
        "selected_training_ids": [row["id"] for row in selected_train],
    }
    receipt = {
        "schema": SCHEMA,
        "status": "running",
        "exploratory": True,
        "public_benchmark": False,
        "generalized_long_context_claim": False,
        "limitations": [
            "Nine fictional templated examples cannot establish generalized long-context competence.",
            "This fixed pilot uses microbatch one and effective batch one, unlike the main recipes.",
            "Validation is an exploratory authored-data check, not a public benchmark.",
        ],
        "specification": specification,
        "inputs": {
            "plan": str(args.plan.resolve()),
            "plan_sha256": prepared["plan_sha256"],
            "train": str(args.train.resolve()),
            "train_sha256": prepared["train_sha256"],
            "validation": str(args.validation.resolve()),
            "validation_sha256": prepared["validation_sha256"],
        },
        "selected_training_rows": [{
            "id": row["id"], "source_id": _source_id(row), "group_id": row["group_id"],
            "evidence_position": row["evidence_position"], "gold_option_id": row["gold_option_id"],
        } for row in selected_train],
        "validation_rows": len(selected_validation),
        "runtime": {
            **pilot._runtime_identity(Path(__file__).resolve().parents[2]),
            "training_code_sha256": _sha256_file(Path(__file__)),
            "authored_data_code_sha256": _sha256_file(Path(authored_data.__file__)),
            "evaluator_code_sha256": _sha256_file(Path(evaluate_authored.__file__)),
        },
        "started_unix": time.time(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir()
    steps_path = args.output / "steps.jsonl"
    validation_path = args.output / "validation.predictions.jsonl"
    steps_path.open("x").close()
    validation_path.open("x").close()
    _atomic_json(args.output / "run-receipt.json", receipt)

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
        if torch._dynamo.config.suppress_errors:
            raise RuntimeError("Compiler fallback must be disabled")
        torch.manual_seed(SEED)
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        receipt["hardware"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": pilot._nvidia_driver_version(),
            "visible_device_count": torch.cuda.device_count(),
            "host_cgroup_memory_limit_bytes": pilot._host_memory_limit_bytes(),
        }
        model, tokenizer, metadata = load_causal_model(
            pilot.MODEL, pilot.REVISION, quantization="nf4",
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        backend = pilot._select_and_verify_flex(model)
        model, peft_version = _configure_lora(model, "nf4")
        model.train()
        trainable = [(name, parameter) for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
        if not trainable or any(".lora_" not in name for name, _ in trainable):
            raise RuntimeError("Trainable parameters must be restricted to LoRA adapters")
        config = pilot._model_config(model)
        if args.max_tokens > int(getattr(config, "max_position_embeddings", pilot.MAX_NATIVE_CONTEXT)):
            raise RuntimeError("--max-tokens exceeds the loaded model's native context")
        optimizer = torch.optim.AdamW((parameter for _, parameter in trainable), lr=LEARNING_RATE)
        training_examples = [reorder_and_target(row) for row in selected_train]
        train_encodings = {
            row["id"]: encode_prompt(tokenizer, row, args.max_tokens)
            for row, _target in training_examples
        }
        validation_encodings = {
            row["id"]: encode_prompt(tokenizer, row, args.max_tokens)
            for row in selected_validation
        }
        receipt["model"] = {
            **metadata,
            "peft_version": peft_version,
            "backend": backend,
            "flex_kernel_options": pilot.FLEX_KERNEL_OPTIONS,
            "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        }
        receipt["token_preflight"] = {
            "training": {row_id: len(encoding[0]) for row_id, encoding in train_encodings.items()},
            "validation": {row_id: len(encoding[0]) for row_id, encoding in validation_encodings.items()},
            "no_truncation": True,
        }
        _atomic_json(args.output / "run-receipt.json", receipt)

        for index, (row, target) in enumerate(training_examples):
            step = _train_step(
                model, tokenizer, row, target, max_tokens=args.max_tokens, device=device,
                optimizer=optimizer, activation_offload=args.activation_offload,
                encoding=train_encodings[row["id"]],
            )
            step["step"] = index + 1
            step["option_ids_after_reorder"] = [option["id"] for option in row["options"]]
            checkpoint = checkpoints / f"step-{index + 1:02d}"
            _save_step_checkpoint(checkpoint, model, step=index + 1, specification=specification)
            step["checkpoint"] = str(checkpoint.relative_to(args.output))
            _append_jsonl(steps_path, step)
            receipt["completed_steps"] = index + 1
            receipt["last_checkpoint"] = step["checkpoint"]
            _atomic_json(args.output / "run-receipt.json", receipt)
            print(json.dumps({"status": "trained", **step}, sort_keys=True), flush=True)

        pending_adapter = args.output / ".final-adapter.tmp"
        pending_adapter.mkdir(exist_ok=False)
        model.save_pretrained(pending_adapter, safe_serialization=True)
        adapter_sha256 = _sha256_file(pending_adapter / "adapter_model.safetensors")
        records, validation_summary = _validate(
            model, tokenizer, selected_validation, max_tokens=args.max_tokens, device=device,
            source_sha256=prepared["validation_sha256"], output_path=validation_path,
            encodings=validation_encodings, adapter_sha256=adapter_sha256,
        )
        validation_summary_path = args.output / "validation.summary.json"
        _atomic_json(validation_summary_path, validation_summary)
        export = _export_final_adapter(
            args.output / "final-adapter", receipt=receipt,
            validation_summary=validation_summary, validation_predictions=validation_path,
            temporary=pending_adapter, adapter_sha256=adapter_sha256,
        )
        receipt.update({
            "status": "completed",
            "completed_steps": TRAIN_STEPS,
            "validation": validation_summary,
            "validation_predictions_sha256": _sha256_file(validation_path),
            "final_adapter": {"path": "final-adapter", **export},
            "finished_unix": time.time(),
        })
        _atomic_json(args.output / "run-receipt.json", receipt)
        return receipt
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)[:1000]}
        receipt["finished_unix"] = time.time()
        _atomic_json(args.output / "run-receipt.json", receipt)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True,
                        help="frozen predeclared authored pilot plan")
    parser.add_argument("--output", type=Path, required=True,
                        help="new create-only training run directory")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-tokens", type=int, default=pilot.MAX_NATIVE_CONTEXT)
    parser.add_argument("--activation-offload", action="store_true")
    return parser


def main() -> None:
    signal.signal(signal.SIGTERM, lambda _signum, _frame: (_ for _ in ()).throw(
        TrainingTerminated("External timeout terminated authored training")
    ))
    args = _parser().parse_args()
    try:
        run(args)
    except BaseException as error:
        print(f"authored training failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
