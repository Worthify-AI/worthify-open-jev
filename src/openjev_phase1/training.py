"""Supervised LoRA training for the native OpenJev next-letter decision readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Iterable

from .core import load_causal_model, validate_row
from .direct import _forward, encode_prompt


APPROVED_DEFAULTS = {
    "quantization": "nf4", "max_tokens": 2048, "effective_batch_size": 16,
    "epochs": 2, "learning_rate": 2e-4, "lora_rank": 16,
    "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def load_recipe(path: Path) -> dict:
    """Read the deliberately small, auditable task recipe format."""
    raw = path.read_bytes()
    try:
        recipe = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Recipe {path} is not valid JSON") from error
    required = {"version", "task", "dataset", "seeds", "training"}
    if not isinstance(recipe, dict) or not required <= recipe.keys():
        raise ValueError(f"Recipe requires fields: {sorted(required)}")
    if recipe["version"] != "openjev-training-recipe-v1":
        raise ValueError("Unsupported training recipe version")
    if recipe["task"] not in {"classification", "evidence"} or recipe["dataset"] != {"classification": "CLINC150", "evidence": "WANLI"}[recipe["task"]]:
        raise ValueError("Recipe task or dataset is invalid")
    if recipe["seeds"] != [42, 43] or not isinstance(recipe["training"], dict):
        raise ValueError("Recipe must declare the approved seed pair and training settings")
    if {key: recipe["training"].get(key) for key in APPROVED_DEFAULTS} != APPROVED_DEFAULTS:
        raise ValueError("Recipe training settings differ from approved defaults")
    return {**recipe, "digest": hashlib.sha256(raw).hexdigest(), "path": str(path)}


def apply_recipe_defaults(args, recipe: dict | None) -> dict | None:
    """Fill omitted CLI values from a recipe or approved defaults, never overwrite one."""
    defaults = (recipe or {}).get("training", APPROVED_DEFAULTS)
    for name in ("quantization", "max_tokens", "effective_batch_size", "epochs"):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])
    if args.seed is None:
        args.seed = (recipe or {"seeds": [42, 43]})["seeds"][0]
    if args.seed not in (recipe or {"seeds": [42, 43]})["seeds"]:
        raise ValueError("Selected seed is not permitted by this recipe")
    return recipe


def _validate_recipe_rows(rows: list[dict], recipe: dict | None) -> None:
    if recipe is None:
        return
    for row in rows:
        if row.get("task") != recipe["task"]:
            raise ValueError(f"Row {row['id']} does not match recipe task {recipe['task']}")
        dataset = row.get("source", {}).get("dataset")
        if dataset != recipe["dataset"]:
            raise ValueError(f"Row {row['id']} does not match recipe dataset {recipe['dataset']}")


def validate_training_row(row: dict) -> None:
    validate_row(row)
    gold = row.get("gold_option_id")
    if not isinstance(gold, str) or gold not in {item["id"] for item in row["options"]}:
        raise ValueError("Training rows require gold_option_id matching a declared option")


def randomized_row(row: dict, *, seed: int, epoch: int) -> tuple[dict, int]:
    """Return a deterministic option permutation and the correct resulting index."""
    validate_training_row(row)
    permutation = list(range(len(row["options"])))
    # Include row id to avoid one shared permutation for an entire epoch.
    random.Random(f"{seed}:{epoch}:{row['id']}").shuffle(permutation)
    options = [row["options"][index] for index in permutation]
    shuffled = {key: value for key, value in row.items() if key != "options"}
    shuffled["options"] = options
    gold_index = next(index for index, option in enumerate(options) if option["id"] == row["gold_option_id"])
    return shuffled, gold_index


def batch_loss(model, tokenizer, examples: list[tuple[dict, int]], max_tokens: int):
    """One forward pass and CE restricted to each row's declared A-P option slots."""
    import torch

    if not examples:
        raise ValueError("A training batch must contain at least one example")
    encoded = [encode_prompt(tokenizer, row, max_tokens) for row, _ in examples]
    device = next(model.parameters()).device
    width = max(len(ids) for ids, _, _ in encoded)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer needs a pad_token_id or eos_token_id")
    input_ids = torch.full((len(encoded), width), pad, dtype=torch.long, device=device)
    attention = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
    for index, (ids, _, _) in enumerate(encoded):
        # Left padding keeps each final real prompt token at the one retained logit.
        input_ids[index, width - len(ids):] = torch.tensor(ids, device=device)
        attention[index, width - len(ids):] = 1
    positions = (attention.cumsum(dim=1) - 1).clamp_min(0)
    logits = _forward(model, {"input_ids": input_ids, "attention_mask": attention,
                              "position_ids": positions}).float()
    # Rows can have different numbers of options.  Invalid padded classes are -inf,
    # so they cannot affect cross entropy or predicted labels.
    counts = [len(slots) for _, slots, _ in encoded]
    if any(not 0 <= label < count for (_, label), count in zip(examples, counts)):
        raise ValueError("Training target is outside the declared option slots")
    selected = torch.stack([
        torch.nn.functional.pad(logits[index, slots], (0, max(counts) - len(slots)), value=float("-inf"))
        for index, (_, slots, _) in enumerate(encoded)
    ])
    labels = torch.tensor([label for _, label in examples], dtype=torch.long, device=device)
    loss = torch.nn.functional.cross_entropy(selected, labels)
    return loss, selected.detach(), labels.detach()


def macro_f1(predictions: Iterable, labels: Iterable, classes: Iterable | int) -> float:
    predictions, labels = list(predictions), list(labels)
    if isinstance(classes, int):
        classes = range(classes)
    classes = list(classes)
    if not classes or len(predictions) != len(labels):
        raise ValueError("Macro F1 requires classes and equally sized predictions and labels")
    scores = []
    for cls in classes:
        tp = sum(pred == cls and label == cls for pred, label in zip(predictions, labels))
        fp = sum(pred == cls and label != cls for pred, label in zip(predictions, labels))
        fn = sum(pred != cls and label == cls for pred, label in zip(predictions, labels))
        scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(classes)


def evaluate(model, tokenizer, rows: list[dict], max_tokens: int) -> float:
    import torch

    model.eval()
    predictions: list[str] = []
    labels: list[str] = []
    with torch.no_grad():
        for row in rows:
            validate_training_row(row)
            loss, logits, target = batch_loss(model, tokenizer, [(row, next(i for i, x in enumerate(row["options"]) if x["id"] == row["gold_option_id"]))], max_tokens)
            del loss
            predictions.append(row["options"][int(logits.argmax(dim=1).item())]["id"])
            labels.append(row["gold_option_id"])
    return macro_f1(predictions, labels, sorted(set(predictions) | set(labels)))


def _read_jsonl(path: Path, payload: bytes | None = None) -> list[dict]:
    rows = [json.loads(line) for line in (path.read_bytes() if payload is None else payload).splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty")
    for row in rows:
        validate_training_row(row)
    return rows


def _validate_splits(train_rows: list[dict], validation_rows: list[dict]) -> None:
    """Reject reused rows, source groups, or normalized evidence across splits."""
    from .datasets import normalize_text

    def keys(rows, split):
        ids, groups, evidence = set(), set(), set()
        for row in rows:
            if row["id"] in ids:
                raise ValueError(f"Duplicate {split} row ID: {row['id']}")
            if row.get("split", split) != split:
                raise ValueError(f"Row {row['id']} is not assigned to {split}")
            ids.add(row["id"])
            if row.get("group_id"):
                groups.add(row["group_id"])
            state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], sort_keys=True)
            evidence.add(normalize_text(state))
        return ids, groups, evidence

    for label, train_keys, validation_keys in zip(
        ("row ID", "source group", "normalized evidence"),
        keys(train_rows, "train"), keys(validation_rows, "validation"),
    ):
        if train_keys & validation_keys:
            raise ValueError(f"Training/validation leakage through {label}")


def _configure_lora(model, quantization: str):
    import peft

    if quantization == "nf4":
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = False
    configured = peft.get_peft_model(model, peft.LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    # Non-reentrant checkpointing retains gradients when all base embeddings are
    # frozen, including the non-quantized path. Reentrant mode can silently drop them.
    configured.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [name for name, parameter in configured.named_parameters() if parameter.requires_grad]
    if not trainable or any(".lora_" not in name or not any(
        f".{target}." in name for target in ("q_proj", "k_proj", "v_proj", "o_proj")
    ) for name in trainable):
        raise RuntimeError("Trainable weights must be limited to q/k/v/o LoRA adapters")
    return configured, peft.__version__


def _save_checkpoint(path: Path, model, optimizer, epoch: int, best_f1: float,
                     *, best_epoch: int, best_checkpoint: Path, training_spec: dict) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path / "adapter")
    state = {"optimizer": optimizer.state_dict(), "epoch": epoch, "best_f1": best_f1,
             "best_epoch": best_epoch, "best_checkpoint": str(best_checkpoint.resolve()),
             "training_spec": training_spec,
             "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(state, path / "state.pt")


def _resume_checkpoint(path: Path, model, training_spec: dict) -> dict:
    """Restore adapter, optimizer, and both RNG streams from a saved checkpoint."""
    import torch

    if not (path / "adapter").is_dir() or not (path / "state.pt").is_file():
        raise ValueError("resume path must contain adapter/ and state.pt")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    if state.get("training_spec") != training_spec:
        raise ValueError("Resume checkpoint model, data, seed, and training settings must match")
    if not (Path(state["best_checkpoint"]) / "adapter").is_dir():
        raise ValueError("Resume checkpoint's validation-selected adapter is missing")
    model.delete_adapter("default")
    model.load_adapter(str(path / "adapter"), adapter_name="default", is_trainable=True)
    model.set_adapter("default")
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return state


def _adapter_reload_error(model, tokenizer, rows: list[dict], max_tokens: int, adapter: Path) -> float:
    """Prove the exported adapter produces the same fixed validation logits."""
    import torch

    subset = rows[: min(8, len(rows))]
    examples = [(row, next(i for i, option in enumerate(row["options"])
                           if option["id"] == row["gold_option_id"])) for row in subset]
    model.eval()
    with torch.no_grad():
        _, before, _ = batch_loss(model, tokenizer, examples, max_tokens)
    model.delete_adapter("default")
    model.load_adapter(str(adapter), adapter_name="default", is_trainable=False)
    model.set_adapter("default")
    model.eval()
    with torch.no_grad():
        _, after, _ = batch_loss(model, tokenizer, examples, max_tokens)
    if before.shape != after.shape:
        raise RuntimeError("Adapter reload changed the option logit shape")
    declared = torch.arange(before.shape[1], device=before.device)[None, :] < torch.tensor(
        [len(row["options"]) for row in subset], device=before.device)[:, None]
    if not torch.isfinite(before[declared]).all() or not torch.isfinite(after[declared]).all():
        raise RuntimeError("Adapter reload verification found nonfinite declared option logits")
    return float((before[declared] - after[declared]).abs().max().item())


def train(args) -> dict:
    import torch

    if args.output.exists():
        raise ValueError("output must be a new create-only path")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
    recipe_path = getattr(args, "recipe", None)
    recipe = load_recipe(recipe_path) if recipe_path else None
    apply_recipe_defaults(args, recipe)
    if args.max_tokens < 1 or args.effective_batch_size < 1:
        raise ValueError("max tokens and effective batch size must be positive")
    train_bytes, validation_bytes = args.train.read_bytes(), args.validation.read_bytes()
    train_rows, validation_rows = _read_jsonl(args.train, train_bytes), _read_jsonl(args.validation, validation_bytes)
    _validate_recipe_rows(train_rows, recipe)
    _validate_recipe_rows(validation_rows, recipe)
    _validate_splits(train_rows, validation_rows)
    training_spec = {"model": args.model, "revision": args.revision,
                     "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
                     "validation_sha256": hashlib.sha256(validation_bytes).hexdigest(),
                     "seed": args.seed, "quantization": args.quantization,
                     "max_tokens": args.max_tokens, "effective_batch_size": args.effective_batch_size,
                     "learning_rate": 2e-4,
                     "recipe": None if recipe is None else {key: recipe[key] for key in ("digest", "task", "dataset", "version")}}
    started = time.monotonic()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer, metadata = load_causal_model(args.model, args.revision, quantization=args.quantization, cache_dir=str(args.cache_dir) if args.cache_dir else None)
    model, peft_version = _configure_lora(model, args.quantization)
    metadata["peft_version"] = peft_version
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    args.output.mkdir(parents=True)
    best_f1, best_epoch, start_epoch, best_checkpoint = -1.0, -1, 0, None
    optimizer_state = None
    if args.resume:
        state = _resume_checkpoint(args.resume, model, training_spec)
        start_epoch, best_f1, optimizer_state = int(state["epoch"]) + 1, float(state["best_f1"]), state["optimizer"]
        best_epoch = int(state["best_epoch"])
        best_checkpoint = Path(state["best_checkpoint"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        ordered = list(train_rows)
        random.Random(f"{args.seed}:{epoch}").shuffle(ordered)
        optimizer.zero_grad(set_to_none=True)
        optimizer_step = 0
        step_started = time.monotonic()
        step_examples = 0
        for index, row in enumerate(ordered):
            shuffled, target = randomized_row(row, seed=args.seed, epoch=epoch)
            loss, _, _ = batch_loss(model, tokenizer, [(shuffled, target)], args.max_tokens)
            group_size = min(args.effective_batch_size, len(ordered) - (index // args.effective_batch_size) * args.effective_batch_size)
            (loss / group_size).backward()
            step_examples += 1
            if (index + 1) % args.effective_batch_size == 0 or index + 1 == len(ordered):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                # Materializing the loss synchronizes the GPU work before timing
                # this completed step, rather than reporting kernel enqueue speed.
                completed_loss = float(loss.detach().item())
                elapsed = time.monotonic() - step_started
                print(json.dumps({"epoch": epoch, "optimizer_step": optimizer_step,
                                  "loss": completed_loss, "examples_per_second": step_examples / elapsed if elapsed else 0.0,
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
                step_started, step_examples = time.monotonic(), 0
        score = evaluate(model, tokenizer, validation_rows, args.max_tokens)
        print(json.dumps({"epoch": epoch, "validation_macro_f1": score}), flush=True)
        checkpoint = args.output / f"checkpoint-epoch-{epoch:02d}"
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            best_checkpoint = checkpoint
        _save_checkpoint(checkpoint, model, optimizer, epoch, best_f1,
                         best_epoch=best_epoch, best_checkpoint=best_checkpoint, training_spec=training_spec)
    if best_checkpoint is None:
        raise RuntimeError("No checkpoint was available for final adapter export")
    # The final adapter is the validation-selected checkpoint, never merely the last epoch.
    model.delete_adapter("default")
    model.load_adapter(str(best_checkpoint / "adapter"), adapter_name="default", is_trainable=False)
    model.set_adapter("default")
    model.eval()
    final = args.output / "final-adapter"
    final.mkdir(exist_ok=False)
    model.save_pretrained(final)
    reload_error = _adapter_reload_error(model, tokenizer, validation_rows, args.max_tokens, final)
    reload_tolerance = 1e-4
    if reload_error > reload_tolerance:
        raise RuntimeError(f"Adapter reload changed fixed validation logits by {reload_error}")
    dataset_hash = hashlib.sha256(train_bytes + validation_bytes).hexdigest()
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    ).stdout.strip() or "unavailable"
    manifest = {"model": metadata, "training_spec": training_spec, "seed": args.seed, "epochs": args.epochs, "max_tokens": args.max_tokens,
                "effective_batch_size": args.effective_batch_size, "learning_rate": 2e-4,
                "best_validation_macro_f1": best_f1, "best_epoch": best_epoch,
                "validation_checkpoint": str(best_checkpoint), "dataset_sha256": dataset_hash,
                "code_revision": code_revision, "training_seconds": time.monotonic() - started,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                "trainable_parameter_count": trainable_parameter_count,
                "adapter_reload_verification": {"examples": min(8, len(validation_rows)),
                                                  "max_abs_logit_error": reload_error,
                                                  "tolerance": reload_tolerance},
                "adapter": {"rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train",))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43))
    parser.add_argument("--quantization", choices=("none", "nf4"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--epochs", type=int, choices=(1, 2))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
