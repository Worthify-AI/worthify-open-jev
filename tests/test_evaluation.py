import pytest

from openjev_phase1.evaluation import (
    align_predictions,
    assert_no_split_leakage,
    evaluate_predictions,
    select_candidate,
)


def _gold(row_id, task, gold, group):
    return {"id": row_id, "task": task, "state": f"state {row_id}", "question": "choose",
            "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
            "gold_option_id": gold, "group_id": group, "split": "validation"}


def test_evaluation_joins_by_id_and_semantic_option_ids():
    gold = [_gold("a", "classification", "yes", "g1"), _gold("b", "evidence", "no", "g2")]
    predictions = [
        {"id": "b", "option_ids": ["no", "yes"], "probabilities": [0.8, 0.2],
         "total_seconds": 0.3, "peak_memory_bytes": 20},
        {"id": "a", "option_ids": ["no", "yes"], "probabilities": [0.1, 0.9],
         "total_seconds": 0.1, "peak_memory_bytes": 10},
    ]
    for prediction in predictions:
        prediction["warm"] = True
    report = evaluate_predictions(gold, predictions, bootstrap_samples=40)
    assert report["accuracy"] == 1
    assert report["mean_task_macro_f1"] == 1
    assert report["latency"]["warm_median_per_row_seconds"] == pytest.approx(0.2)
    assert report["peak_memory_bytes"] == 20
    assert report["task_results"]["classification"]["brier"] == pytest.approx(0.02)
    assert "uncalibrated" in report["task_results"]["evidence"]["probability_status"]


def test_missing_duplicate_and_option_mismatch_fail_closed():
    gold = [_gold("a", "classification", "yes", "g1")]
    with pytest.raises(ValueError, match="differ"):
        align_predictions(gold, [])
    duplicate = [{"id": "a", "probabilities": [0.5, 0.5]}] * 2
    with pytest.raises(ValueError, match="Duplicate"):
        align_predictions(gold, duplicate)
    with pytest.raises(ValueError, match="options"):
        align_predictions(gold, [{"id": "a", "option_ids": ["yes", "other"],
                                  "probabilities": [0.5, 0.5]}])


def test_macro_f1_includes_predicted_classes_absent_from_gold():
    gold = [_gold("a", "classification", "yes", "g1"), _gold("b", "classification", "yes", "g2")]
    predictions = [{"id": row["id"], "option_ids": ["yes", "no"], "probabilities": [0.1, 0.9]}
                   for row in gold]
    report = evaluate_predictions(gold, predictions, bootstrap_samples=40)
    assert report["task_results"]["classification"]["macro_f1"] == 0


def test_verified_provenance_must_match_every_prediction():
    gold = [_gold("a", "classification", "yes", "g1")]
    model = {"id": "org/model", "revision": "a" * 40}
    adapter = "b" * 64
    prediction = {"id": "a", "option_ids": ["yes", "no"], "probabilities": [0.9, 0.1],
                  "base_model": model, "adapter_sha256": adapter, "seed": 7}
    report = evaluate_predictions(gold, [prediction], seed=7, bootstrap_samples=40,
                                  base_model=model, adapter_sha256=adapter,
                                  input_hashes={"gold_sha256": "c" * 64, "predictions_sha256": "d" * 64},
                                  require_provenance=True)
    assert report["schema"] == "openjev-phase1-evaluation-v1"
    assert report["provenance_verified"] is True
    with pytest.raises(ValueError, match="provenance"):
        evaluate_predictions(gold, [{**prediction, "seed": 8}], seed=7, bootstrap_samples=40,
                             base_model=model, adapter_sha256=adapter)


def test_split_leakage_checks_groups_and_normalized_text():
    left = _gold("a", "classification", "yes", "shared")
    right = {**_gold("b", "classification", "yes", "shared"), "split": "test"}
    with pytest.raises(ValueError, match="leakage"):
        assert_no_split_leakage([left, right])
    right = {**right, "group_id": "other", "state": "STATE A!"}
    with pytest.raises(ValueError, match="normalized"):
        assert_no_split_leakage([left, right])


def test_selection_uses_both_validation_tasks_and_warm_latency():
    candidate = _candidate
    selected = select_candidate([candidate("best-slow", 0.90, 2.0),
                                 candidate("near-fast", 0.895, 0.4),
                                 candidate("too-low", 0.88, 0.1)])
    assert selected["candidate"] == "near-fast"
    assert selected["test_metrics_used"] is False
    invalid = candidate("invalid", 1.0, 1.0)
    invalid["test"] = {"macro_f1": 1.0}
    with pytest.raises(ValueError, match="validation"):
        select_candidate([invalid])


def _candidate(name, score, latency):
    return {"candidate": name, "split": "validation", "n": 10, "input_sha256": "a" * 64,
            "comparison_protocol": {"quantization": "nf4", "max_tokens": 2048, "prompt_version": "direct-options-v1",
                                    "dtype": "bfloat16", "batch_size": 1, "hardware_name": "A100"},
            "task_results": {task: {"macro_f1": score, "n": 5} for task in ("classification", "evidence")},
            "latency": {"warm_median_per_row_seconds": latency, "warm_n": 10}}


@pytest.mark.parametrize("change", ["nan", "negative_latency", "input", "hardware", "counts", "cold", "duplicate"])
def test_selection_rejects_incomparable_or_invalid_candidates(change):
    first, second = _candidate("one", .9, 1), _candidate("two", .895, .5)
    if change == "nan":
        second["task_results"]["evidence"]["macro_f1"] = float("nan")
    elif change == "negative_latency":
        second["latency"]["warm_median_per_row_seconds"] = -1
    elif change == "input":
        second["input_sha256"] = "b" * 64
    elif change == "hardware":
        second["comparison_protocol"]["hardware_name"] = "3090"
    elif change == "counts":
        second["task_results"]["evidence"]["n"] = 4
    elif change == "cold":
        second["latency"]["warm_n"] = 0
    else:
        second["candidate"] = "one"
    with pytest.raises(ValueError):
        select_candidate([first, second])


def test_unlabelled_timing_is_not_claimed_warm():
    report = evaluate_predictions([_gold("a", "classification", "yes", "g1")],
                                  [{"id": "a", "option_ids": ["yes", "no"], "probabilities": [.9, .1], "total_seconds": .1}], bootstrap_samples=40)
    assert report["latency"]["warm_n"] == 0
    assert report["latency"]["warm_median_per_row_seconds"] is None


def test_probability_lists_cannot_implicitly_assume_option_order():
    with pytest.raises(ValueError, match="explicit prediction option_ids"):
        align_predictions([_gold("a", "classification", "yes", "g1")],
                          [{"id": "a", "probabilities": [.9, .1]}])
