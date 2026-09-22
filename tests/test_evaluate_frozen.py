import hashlib
import json

import pytest

from benchmarks.evaluate_frozen import evaluate_frozen


MODEL = "google/gemma-4-12B-it"
REVISION = "7" * 40


def _jsonl(rows):
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def _gold():
    return [
        {"id": "a", "task": "classification", "group_id": "g1", "split": "test",
         "state": "one", "question": "choose", "options": [{"id": "yes", "description": "Yes"},
                                                                  {"id": "no", "description": "No"}],
         "gold_option_id": "yes"},
        {"id": "b", "task": "classification", "group_id": "g2", "split": "test",
         "state": "two", "question": "choose", "options": [{"id": "yes", "description": "Yes"},
                                                                  {"id": "no", "description": "No"}],
         "gold_option_id": "no"},
    ]


def _predictions():
    metadata = {"source": MODEL, "revision": REVISION, "quantization": "nf4",
                "dtype": "bfloat16"}
    return [
        {"id": "a", "option_ids": ["yes", "no"], "probabilities": [0.9, 0.1],
         "total_seconds": 0.2, "warm": True, "prompt_version": "direct-options-v1",
         "model": metadata},
        {"id": "b", "option_ids": ["yes", "no"], "probabilities": [0.8, 0.2],
         "total_seconds": 0.4, "warm": True, "prompt_version": "direct-options-v1",
         "model": metadata},
    ]


def test_valid_frozen_report_uses_shared_math_and_actual_input_hashes():
    gold_bytes, prediction_bytes = _jsonl(_gold()), _jsonl(_predictions())
    report = evaluate_frozen(gold_bytes, prediction_bytes, model=MODEL, revision=REVISION,
                             bootstrap_samples=40)

    assert report["schema"] == "openjev-phase1-evaluation-v1"
    assert report["model_kind"] == "frozen"
    assert report["base_model"] == {"id": MODEL, "revision": REVISION}
    assert report["adapter_sha256"] is None
    assert report["provenance_verified"] is True
    assert report["frozen_provenance"]["checks"]["prediction_rows_checked"] == 2
    assert report["inputs"] == {
        "gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
        "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
    }
    assert report["accuracy"] == 0.5
    assert report["task_results"]["classification"]["macro_f1"] == pytest.approx(1 / 3)
    assert report["latency"]["warm_n"] == 2


@pytest.mark.parametrize("change, message", [
    ("model", "model provenance"),
    ("adapter", "adapter or training seed"),
])
def test_mismatched_model_or_adapter_is_rejected(change, message):
    predictions = _predictions()
    if change == "model":
        predictions[1]["model"] = {**predictions[1]["model"], "source": "wrong/model"}
    else:
        predictions[1]["model"] = {**predictions[1]["model"], "adapter_sha256": "a" * 64}

    with pytest.raises(ValueError, match=message):
        evaluate_frozen(_jsonl(_gold()), _jsonl(predictions), model=MODEL, revision=REVISION,
                        bootstrap_samples=40)
