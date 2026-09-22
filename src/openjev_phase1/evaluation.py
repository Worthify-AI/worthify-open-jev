"""Strict evaluation and validation-only candidate selection for OpenJev rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

from .datasets import normalize_text


def read_jsonl(path: Path) -> list[dict]:
    return _parse_jsonl(path.read_bytes(), str(path))


def _parse_jsonl(raw: bytes, name: str) -> list[dict]:
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    _index(rows, name)
    return rows


def _index(rows: list[dict], name: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"Missing ID in {name}")
        if row_id in result:
            raise ValueError(f"Duplicate ID in {name}: {row_id}")
        result[row_id] = row
    return result


def assert_no_split_leakage(rows: list[dict]) -> None:
    """Reject source groups or normalized evidence appearing across splits."""
    _index(rows, "dataset")
    groups, texts = defaultdict(set), defaultdict(set)
    for row in rows:
        if row.get("split") not in {"train", "validation", "test", "robustness"}:
            raise ValueError(f"Unknown split for {row['id']}: {row.get('split')}")
        if not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise ValueError(f"Missing group_id for {row['id']}")
        groups[row["group_id"]].add(row["split"])
        texts[normalize_text(str(row.get("state", "")))].add(row["split"])
    crossed_groups = sorted(group for group, splits in groups.items() if len(splits) > 1)
    crossed_texts = sorted(text for text, splits in texts.items() if len(splits) > 1)
    if crossed_groups or crossed_texts:
        raise ValueError(
            f"Split leakage: {len(crossed_groups)} groups and {len(crossed_texts)} normalized texts cross splits"
        )


def _probabilities(prediction: dict, option_ids: list[str]) -> list[float]:
    values = prediction.get("probabilities")
    if isinstance(values, dict):
        if set(values) != set(option_ids):
            raise ValueError("Probability keys do not match options")
        values = [values[option_id] for option_id in option_ids]
    elif isinstance(values, list):
        predicted_ids = prediction.get("option_ids")
        if predicted_ids is None:
            raise ValueError("List probabilities require explicit prediction option_ids")
        if not isinstance(predicted_ids, list) or len(set(predicted_ids)) != len(predicted_ids):
            raise ValueError("Invalid prediction option_ids")
        if set(predicted_ids) != set(option_ids) or len(values) != len(predicted_ids):
            raise ValueError("Prediction options do not match gold options")
        by_id = dict(zip(predicted_ids, values))
        values = [by_id[option_id] for option_id in option_ids]
    else:
        raise ValueError("A full probability distribution is required")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Probabilities must be finite numbers in [0, 1]")
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-5):
        raise ValueError("Probabilities do not sum to one")
    return [float(value) for value in values]


def align_predictions(gold: list[dict], predictions: list[dict]) -> list[dict]:
    """Join strictly by ID and validate the full declared option distribution."""
    truth, outputs = _index(gold, "gold"), _index(predictions, "predictions")
    missing, unknown = truth.keys() - outputs.keys(), outputs.keys() - truth.keys()
    if missing or unknown:
        raise ValueError(f"Prediction IDs differ: missing={sorted(missing)[:5]}, unknown={sorted(unknown)[:5]}")
    aligned = []
    for source in gold:
        option_ids = [option["id"] for option in source.get("options", [])]
        if len(option_ids) < 2 or len(option_ids) != len(set(option_ids)):
            raise ValueError(f"Invalid options for {source['id']}")
        gold_id = source.get("gold_option_id")
        if gold_id not in option_ids:
            raise ValueError(f"Missing gold option for {source['id']}")
        prediction = outputs[source["id"]]
        values = _probabilities(prediction, option_ids)
        chosen = option_ids[max(range(len(values)), key=values.__getitem__)]
        target = option_ids.index(gold_id)
        aligned.append({
            "id": source["id"], "task": source.get("task", "unknown"),
            "group_id": source.get("group_id", source["id"]), "split": source.get("split"),
            "gold_id": gold_id, "predicted_id": chosen, "correct": chosen == gold_id,
            "confidence": max(values), "gold_probability": values[target],
            "brier": sum((probability - (index == target)) ** 2
                         for index, probability in enumerate(values)),
            "latency_seconds": prediction.get("total_seconds", prediction.get("latency_seconds")),
            "warm": prediction.get("warm", False),
            "peak_memory_bytes": prediction.get("peak_memory_bytes", prediction.get("peak_cuda_bytes")),
        })
    return aligned


def _macro_f1(rows: list[dict]) -> float:
    labels = sorted({row["gold_id"] for row in rows} | {row["predicted_id"] for row in rows})
    if not labels:
        raise ValueError("Cannot score an empty population")
    gold_counts = Counter(row["gold_id"] for row in rows)
    prediction_counts = Counter(row["predicted_id"] for row in rows)
    correct_counts = Counter(row["gold_id"] for row in rows if row["gold_id"] == row["predicted_id"])
    values = []
    for label in labels:
        denominator = gold_counts[label] + prediction_counts[label]
        values.append(0.0 if denominator == 0 else 2 * correct_counts[label] / denominator)
    return sum(values) / len(values)


def _ece(rows: list[dict], bins: int = 10) -> tuple[float, list[dict]]:
    if bins < 2:
        raise ValueError("ECE requires at least two bins")
    details, total = [], len(rows)
    weighted = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        part = [row for row in rows if lower <= row["confidence"] < upper or
                (index == bins - 1 and row["confidence"] == 1)]
        if not part:
            continue
        confidence = statistics.fmean(row["confidence"] for row in part)
        accuracy = statistics.fmean(row["correct"] for row in part)
        weighted += len(part) / total * abs(accuracy - confidence)
        details.append({"lower": lower, "upper": upper, "n": len(part),
                        "mean_confidence": confidence, "accuracy": accuracy})
    return weighted, details


def _metrics(rows: list[dict], *, ece_bins: int) -> dict:
    ece, bins = _ece(rows, ece_bins)
    return {
        "n": len(rows), "accuracy": statistics.fmean(row["correct"] for row in rows),
        "macro_f1": _macro_f1(rows), "brier": statistics.fmean(row["brier"] for row in rows),
        "ece": ece, "reliability_bins": bins,
        "probability_status": "uncalibrated conditional option scores",
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _stratified_bootstrap(rows: list[dict], samples: int, seed: int) -> dict:
    if samples < 40:
        raise ValueError("At least 40 bootstrap samples are required")
    strata = defaultdict(lambda: defaultdict(list))
    for row in rows:
        strata[row["task"]][row["group_id"]].append(row)
    rng, accuracy, macro = random.Random(seed), [], []
    for _ in range(samples):
        draw = []
        for groups in strata.values():
            units = list(groups.values())
            draw.extend(item for _ in units for item in units[rng.randrange(len(units))])
        accuracy.append(statistics.fmean(row["correct"] for row in draw))
        task_rows = defaultdict(list)
        for row in draw:
            task_rows[row["task"]].append(row)
        macro.append(statistics.fmean(_macro_f1(part) for part in task_rows.values()))
    accuracy.sort()
    macro.sort()
    interval = lambda values: [values[int(0.025 * samples)], values[min(samples - 1, int(0.975 * samples))]]
    return {"samples": samples, "seed": seed, "unit": "source group stratified by task",
            "accuracy_95": interval(accuracy), "mean_task_macro_f1_95": interval(macro)}


def evaluate_predictions(gold: list[dict], predictions: list[dict], *, seed: int = 217,
                         bootstrap_samples: int = 1000, ece_bins: int = 10,
                         base_model: dict | None = None, adapter_sha256: str | None = None,
                         input_hashes: dict | None = None, require_provenance: bool = False) -> dict:
    if base_model is not None and (set(base_model) != {"id", "revision"}
                                   or not all(isinstance(value, str) and value for value in base_model.values())):
        raise ValueError("base_model requires nonempty id and revision")
    if adapter_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", adapter_sha256):
        raise ValueError("adapter_sha256 must be a lowercase SHA-256 digest")
    provenance_verified = False
    if base_model is not None and adapter_sha256 is not None:
        for prediction in predictions:
            declared_model = prediction.get("base_model") or {
                "id": prediction.get("model", {}).get("source"),
                "revision": prediction.get("model", {}).get("revision"),
            }
            declared_adapter = prediction.get("adapter_sha256", prediction.get("model", {}).get("adapter_sha256"))
            declared_seed = prediction.get("seed", prediction.get("model", {}).get("training_seed"))
            if declared_model != base_model or declared_adapter != adapter_sha256 or declared_seed != seed:
                raise ValueError(f"Prediction provenance disagrees for {prediction.get('id')}")
        provenance_verified = bool(predictions)
    if input_hashes is not None and (set(input_hashes) != {"gold_sha256", "predictions_sha256"}
                                     or not all(re.fullmatch(r"[0-9a-f]{64}", value or "")
                                                for value in input_hashes.values())):
        raise ValueError("input_hashes requires gold_sha256 and predictions_sha256")
    if require_provenance and (not provenance_verified or input_hashes is None):
        raise ValueError("Verified prediction provenance and input hashes are required")
    assert_no_split_leakage(gold)
    declared_splits = {row.get("split") for row in gold}
    if len(declared_splits) != 1:
        raise ValueError("One evaluation report must contain exactly one declared split")
    aligned = align_predictions(gold, predictions)
    if not aligned:
        raise ValueError("Cannot evaluate an empty population")
    tasks = defaultdict(list)
    for row in aligned:
        tasks[row["task"]].append(row)
    latencies = [float(row["latency_seconds"]) for row in aligned
                 if isinstance(row["latency_seconds"], (int, float)) and math.isfinite(row["latency_seconds"])]
    warm_latencies = [float(row["latency_seconds"]) for row in aligned if row["warm"] and
                      isinstance(row["latency_seconds"], (int, float)) and math.isfinite(row["latency_seconds"])]
    memories = [int(row["peak_memory_bytes"]) for row in aligned
                if isinstance(row["peak_memory_bytes"], (int, float)) and row["peak_memory_bytes"] >= 0]
    task_results = {task: _metrics(rows, ece_bins=ece_bins) for task, rows in sorted(tasks.items())}
    return {
        "schema": "openjev-phase1-evaluation-v1", "provenance_verified": provenance_verified,
        "inputs": input_hashes, "seed": seed, "base_model": base_model, "adapter_sha256": adapter_sha256,
        "split": next(iter(declared_splits)),
        "n": len(aligned), "accuracy": statistics.fmean(row["correct"] for row in aligned),
        "mean_task_macro_f1": statistics.fmean(result["macro_f1"] for result in task_results.values()),
        "task_results": task_results,
        "bootstrap_95": _stratified_bootstrap(aligned, bootstrap_samples, seed),
        "latency": {"n": len(latencies), "p50_seconds": _percentile(latencies, 0.50),
                    "p95_seconds": _percentile(latencies, 0.95),
                    "warm_n": len(warm_latencies),
                    "warm_median_per_row_seconds": _percentile(warm_latencies, 0.50)},
        "peak_memory_bytes": max(memories) if memories else None,
        "metric_contract": "All IDs must be present exactly once; probability scores are uncalibrated.",
    }


def evaluate_seeds(gold: list[dict], predictions_by_seed: dict[int, list[dict]], **kwargs) -> dict:
    if not predictions_by_seed:
        raise ValueError("At least one seed is required")
    reports = {str(seed): evaluate_predictions(gold, predictions, seed=seed, **kwargs)
               for seed, predictions in sorted(predictions_by_seed.items())}
    return {"per_seed": reports,
            "mean_macro_f1": statistics.fmean(report["mean_task_macro_f1"] for report in reports.values()),
            "mean_accuracy": statistics.fmean(report["accuracy"] for report in reports.values())}


def select_candidate(reports: list[dict], *, tolerance: float = 0.01,
                     required_tasks: tuple[str, ...] = ("classification", "evidence")) -> dict:
    """Choose the fastest candidate within one point of best validation quality."""
    if not reports or tolerance != 0.01:
        raise ValueError("Selection requires candidates and a frozen 0.01 tolerance")
    assessed = []
    population = None
    protocol = None
    seen = set()
    for report in reports:
        if report.get("split") != "validation" or "test" in report or "test_results" in report:
            raise ValueError("Candidate selection may use validation reports only")
        tasks = report.get("task_results", {})
        if any(task not in tasks for task in required_tasks):
            raise ValueError(f"Candidate {report.get('candidate')} lacks both required tasks")
        name = report.get("candidate")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Candidates must have unique nonempty identities")
        seen.add(name)
        quality = [tasks[task].get("macro_f1") for task in required_tasks]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in quality):
            raise ValueError("Candidate validation macro F1 must be finite and in [0, 1]")
        mean = statistics.fmean(quality)
        digest = report.get("input_sha256")
        counts = tuple(tasks[task].get("n") for task in required_tasks)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or any(type(count) is not int or count < 1 for count in counts):
            raise ValueError("Candidates require a validation input hash and measured task counts")
        current_population = (digest, counts)
        if population is not None and current_population != population:
            raise ValueError("Candidates must use the same frozen validation population")
        population = current_population
        current_protocol = report.get("comparison_protocol")
        protocol_fields = {"quantization", "max_tokens", "prompt_version", "dtype", "batch_size", "hardware_name"}
        if not isinstance(current_protocol, dict) or not protocol_fields <= current_protocol.keys() or any(
            not current_protocol[key] for key in protocol_fields
        ):
            raise ValueError("Candidates require the explicit comparison protocol")
        if protocol is not None and current_protocol != protocol:
            raise ValueError("Candidates must share the same comparison protocol")
        protocol = current_protocol
        latency = report.get("latency", {}).get("warm_median_per_row_seconds")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"Candidate {report.get('candidate')} lacks finite validation latency")
        if report.get("n") != sum(counts) or report.get("latency", {}).get("warm_n") != report["n"]:
            raise ValueError("Candidate latency must cover the entire warm validation population")
        assessed.append((name, mean, float(latency)))
    best = max(mean for _, mean, _ in assessed)
    eligible = [item for item in assessed if item[1] >= best - tolerance]
    chosen = min(eligible, key=lambda item: (item[2], -item[1], item[0]))
    return {"candidate": chosen[0], "validation_mean_macro_f1": chosen[1],
            "validation_warm_median_per_row_seconds": chosen[2], "best_validation_mean_macro_f1": best,
            "tolerance_absolute": tolerance, "required_tasks": list(required_tasks),
            "input_sha256": population[0], "comparison_protocol": protocol,
            "selection_rule": "fastest validation candidate within one percentage point of best mean macro F1",
            "test_metrics_used": False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=217)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument("--adapter-sha256", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be new")
    gold_bytes, prediction_bytes = args.gold.read_bytes(), args.predictions.read_bytes()
    report = evaluate_predictions(_parse_jsonl(gold_bytes, str(args.gold)), _parse_jsonl(prediction_bytes, str(args.predictions)), seed=args.seed,
                                  bootstrap_samples=args.bootstrap_samples,
                                  base_model={"id": args.base_model_id, "revision": args.base_model_revision},
                                  adapter_sha256=args.adapter_sha256,
                                  input_hashes={"gold_sha256": hashlib.sha256(gold_bytes).hexdigest(),
                                                "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest()},
                                  require_provenance=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
