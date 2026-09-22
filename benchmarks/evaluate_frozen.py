"""Evaluate a pinned frozen base model on a complete held-out prediction set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from openjev_phase1.evaluation import _parse_jsonl, evaluate_predictions


PROMPT_VERSION = "direct-options-v1"
_ABSENT_ADAPTER_FIELDS = ("adapter", "adapter_sha256", "adapter_revision", "training_seed")


def _verify_frozen_predictions(predictions: list[dict], model: str, revision: str,
                               quantization: str) -> dict:
    if not predictions:
        raise ValueError("Frozen evaluation requires predictions")
    for prediction in predictions:
        row_id = prediction.get("id")
        metadata = prediction.get("model")
        if not isinstance(metadata, dict):
            raise ValueError(f"Prediction {row_id} lacks model metadata")
        if metadata.get("source") != model or metadata.get("revision") != revision:
            raise ValueError(f"Prediction model provenance disagrees for {row_id}")
        if metadata.get("quantization") != quantization:
            raise ValueError(f"Prediction quantization disagrees for {row_id}")
        if any(metadata.get(field) is not None for field in _ABSENT_ADAPTER_FIELDS):
            raise ValueError(f"Frozen prediction {row_id} declares an adapter or training seed")
        if prediction.get("adapter_sha256") is not None or prediction.get("training_seed") is not None:
            raise ValueError(f"Frozen prediction {row_id} declares an adapter or training seed")
        if prediction.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"Prediction prompt provenance disagrees for {row_id}")
        if prediction.get("warm") is not True:
            raise ValueError(f"Prediction {row_id} is not marked as warm")
    return {
        "model_source_exact": True,
        "model_revision_exact": True,
        "revision_is_full_commit_sha": True,
        "adapter_absent": True,
        "training_seed_absent": True,
        "quantization": quantization,
        "prompt_version": PROMPT_VERSION,
        "all_predictions_warm": True,
        "prediction_rows_checked": len(predictions),
    }


def evaluate_frozen(gold_bytes: bytes, prediction_bytes: bytes, *, model: str,
                    revision: str, seed: int = 217, bootstrap_samples: int = 1000,
                    quantization: str = "nf4", gold_name: str = "gold",
                    predictions_name: str = "predictions") -> dict:
    """Verify frozen provenance, then apply the shared evaluation contract."""
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a nonempty identifier")
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("revision must be a pinned lowercase 40-character commit SHA")
    if quantization != "nf4":
        raise ValueError("Frozen baseline quantization is fixed to nf4")

    gold = _parse_jsonl(gold_bytes, gold_name)
    predictions = _parse_jsonl(prediction_bytes, predictions_name)
    if {row.get("split") for row in gold} != {"test"}:
        raise ValueError("Frozen held-out evaluation requires only the test split")
    checks = _verify_frozen_predictions(predictions, model, revision, quantization)
    inputs = {
        "gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
        "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
    }
    report = evaluate_predictions(
        gold,
        predictions,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        base_model={"id": model, "revision": revision},
        input_hashes=inputs,
    )
    # The shared evaluator only verifies tuned provenance when an adapter digest
    # exists. Frozen provenance has been checked directly above, without inventing
    # an adapter hash.
    report.update({
        "model_kind": "frozen",
        "provenance_verified": True,
        "frozen_provenance": {"verified": True, "checks": checks},
    })
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, default=217)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--quantization", choices=("nf4",), default="nf4")
    args = parser.parse_args(argv)

    gold_bytes = args.gold.read_bytes()
    prediction_bytes = args.predictions.read_bytes()
    report = evaluate_frozen(
        gold_bytes,
        prediction_bytes,
        model=args.model,
        revision=args.revision,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        quantization=args.quantization,
        gold_name=str(args.gold),
        predictions_name=str(args.predictions),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        destination.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
