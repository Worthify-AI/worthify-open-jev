#!/usr/bin/env python3
"""Verify an exported adapter against training logits in a fresh model process."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    import torch
    from openjev_phase1.core import load_causal_model
    from openjev_phase1.training import batch_loss, validate_training_row

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be a new create-only path")
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    adapter = args.run_dir / "final-adapter"
    reference_bytes = (adapter / "reload-reference.json").read_bytes()
    reference_hash = hashlib.sha256(reference_bytes).hexdigest()
    if manifest.get("fresh_reload_reference") != {
        "path": "final-adapter/reload-reference.json", "sha256": reference_hash
    }:
        raise ValueError("Reload reference differs from the training manifest")
    reference = json.loads(reference_bytes)
    if reference.get("schema") != "openjev-phase1-adapter-reload-reference-v1":
        raise ValueError("Unsupported reload reference schema")
    validation_bytes = args.validation.read_bytes()
    validation_hash = hashlib.sha256(validation_bytes).hexdigest()
    if validation_hash != manifest["training_spec"]["validation_sha256"]:
        raise ValueError("Validation data differ from the training run")
    rows = [json.loads(line) for line in validation_bytes.splitlines() if line.strip()][:8]
    if len(rows) != len(reference["rows"]):
        raise ValueError("Reload reference population differs from validation")
    for row, expected in zip(rows, reference["rows"]):
        validate_training_row(row)
        if row["id"] != expected["id"] or [x["id"] for x in row["options"]] != expected["option_ids"]:
            raise ValueError("Reload reference IDs or option order differ from validation")
        if len(expected["logits"]) != len(row["options"]) or not all(math.isfinite(x) for x in expected["logits"]):
            raise ValueError("Reload reference logits are invalid")
    max_tokens = manifest["max_tokens"]
    if reference["max_tokens"] != max_tokens or reference["tolerance"] != 1e-4:
        raise ValueError("Reload reference settings differ from the verification contract")
    model, tokenizer, metadata = load_causal_model(
        manifest["model"]["source"], manifest["model"]["revision"],
        adapter=str(adapter), adapter_revision="local-final-adapter",
        quantization=manifest["model"]["quantization"], cache_dir=str(args.cache_dir),
    )
    examples = [(row, next(i for i, option in enumerate(row["options"])
                          if option["id"] == row["gold_option_id"])) for row in rows]
    with torch.inference_mode():
        _, actual, _ = batch_loss(model, tokenizer, examples, max_tokens)
    if tuple(actual.shape) != (len(rows), max(len(row["options"]) for row in rows)):
        raise RuntimeError("Fresh adapter load changed the option logit shape")
    actual = actual.cpu().tolist()
    errors = [abs(value - target) for values, expected in zip(actual, reference["rows"])
              for value, target in zip(values[:len(expected["logits"])], expected["logits"])]
    maximum = max(errors)
    if not all(math.isfinite(error) for error in errors) or maximum > reference["tolerance"]:
        raise RuntimeError(f"Fresh adapter load changed validation logits: max error {maximum}")
    receipt = {
        "schema": "openjev-phase1-fresh-adapter-reload-v1", "passed": True,
        "examples": len(rows), "max_abs_logit_error": maximum,
        "tolerance": reference["tolerance"], "validation_sha256": validation_hash,
        "reference_sha256": reference_hash,
        "adapter_sha256": metadata["adapter_sha256"], "seed": manifest["seed"],
        "base_model": {"source": metadata["source"], "revision": metadata["revision"]},
    }
    with args.output.open("x") as destination:
        destination.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, allow_nan=False))


if __name__ == "__main__":
    main()
