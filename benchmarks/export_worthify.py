"""Export compact, text-free row evidence for the pinned Worthify 12B benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

try:
    from benchmarks.evaluate_frozen import evaluate_frozen
except ModuleNotFoundError:  # Direct execution: python benchmarks/export_worthify.py
    from evaluate_frozen import evaluate_frozen
from openjev_phase1.evaluation import _metrics, _parse_jsonl, _probabilities, align_predictions, evaluate_predictions


ROOT = Path(__file__).parents[1]
RECIPES = {"classification": "clinc150", "evidence": "wanli"}
SEEDS = (42, 43)


def _sha(raw_or_path: bytes | Path) -> str:
    raw = raw_or_path.read_bytes() if isinstance(raw_or_path, Path) else raw_or_path
    return hashlib.sha256(raw).hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _compact(gold: list[dict], predictions: list[dict]) -> list[dict]:
    outputs = {row["id"]: row for row in predictions}
    if len(outputs) != len(predictions) or set(outputs) != {row["id"] for row in gold}:
        raise ValueError("Prediction IDs do not exactly match gold IDs")
    rows = []
    for item in gold:
        prediction = outputs[item["id"]]
        option_ids = [option["id"] for option in item["options"]]
        rows.append({
            "id": item["id"], "task": item["task"], "group_id": item["group_id"],
            "split": item["split"], "evaluation_slice": item["evaluation_slice"],
            "gold_option_id": item["gold_option_id"],
            "option_ids": option_ids, "probabilities": _probabilities(prediction, option_ids),
            "latency_seconds": prediction.get("total_seconds", prediction.get("latency_seconds")),
            "peak_memory_bytes": prediction.get("peak_memory_bytes", prediction.get("peak_cuda_bytes")),
            "input_tokens": prediction.get("input_tokens"), "warm": prediction.get("warm", False),
        })
    return rows


def _throughput(rows: list[dict]) -> dict:
    latencies = [row["latency_seconds"] for row in rows]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in latencies):
        raise ValueError("Every compact row requires a finite positive measured latency")
    measured = sum(latencies)
    return {"rows": len(rows), "measured_seconds": measured, "rows_per_second": len(rows) / measured,
            "scope": "Serial scorer throughput; excludes file I/O and network."}


def _slice_metrics(rows: list[dict]) -> dict:
    slices = defaultdict(list)
    for row in rows:
        if not isinstance(row["evaluation_slice"], str) or not row["evaluation_slice"]:
            raise ValueError("Every compact row requires a nonempty evaluation_slice")
        slices[row["evaluation_slice"]].append(row)
    results = {}
    for name, part in sorted(slices.items()):
        gold = [{"id": row["id"], "task": row["task"], "group_id": row["group_id"],
                 "split": row["split"], "options": [{"id": item} for item in row["option_ids"]],
                 "gold_option_id": row["gold_option_id"]} for row in part]
        predictions = [{"id": row["id"], "option_ids": row["option_ids"],
                        "probabilities": row["probabilities"]} for row in part]
        results[name] = _metrics(align_predictions(gold, predictions), ece_bins=10)
    return results


def _summary(entries: list[dict], validation: dict[str, dict[str, float]], base: dict) -> dict:
    by_recipe = {}
    for recipe in RECIPES:
        records = {entry["variant"]: entry for entry in entries if entry["recipe"] == recipe}
        reports = {variant: entry["report"] for variant, entry in records.items()}
        baseline = reports["frozen"]
        seeds = {}
        for seed in SEEDS:
            report = reports[f"seed{seed}"]
            seeds[str(seed)] = {
                "accuracy": report["accuracy"], "mean_task_macro_f1": report["mean_task_macro_f1"],
                "throughput": records[f"seed{seed}"]["throughput"],
                "slice_metrics": records[f"seed{seed}"]["slice_metrics"],
                "accuracy_delta_vs_frozen": report["accuracy"] - baseline["accuracy"],
                "macro_f1_delta_vs_frozen": report["mean_task_macro_f1"] - baseline["mean_task_macro_f1"],
            }
        selected = min(SEEDS, key=lambda seed: (-validation[recipe][str(seed)], seed))
        by_recipe[recipe] = {
            "frozen": {"accuracy": baseline["accuracy"], "mean_task_macro_f1": baseline["mean_task_macro_f1"],
                       "throughput": records["frozen"]["throughput"],
                       "slice_metrics": records["frozen"]["slice_metrics"]},
            "seeds": seeds, "selected_seed": selected,
            "selection_basis": "highest best_validation_macro_f1; test metrics excluded",
            "validation_macro_f1": validation[recipe],
        }
    return {
        "schema": "worthify-jev-benchmark-summary-v1", "base_model": base,
        "scope": "Pinned 12B scope choice; not a claim that 12B won a candidate comparison.",
        "comparative_winner_claim": False, "test_metrics_used_for_seed_selection": False,
        "recipes": by_recipe,
    }


def export(run_dir: Path, data_dir: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("Output must be a new path")
    selection = _load(ROOT / "manifests/base-selection-v1.json")
    data_manifest = _load(ROOT / "manifests/data-v1.json")
    launch = _load(run_dir / "launch.json")
    base = {"id": selection["selected_model"], "revision": selection["revision"]}
    if (selection.get("recipes"), selection.get("seeds"), selection.get("quantization")) != (
            list(RECIPES), list(SEEDS), "nf4"):
        raise ValueError("Pinned base-selection manifest does not declare the expected benchmark matrix")
    if launch.get("base_selection") != selection or not re.fullmatch(r"[0-9a-f]{40}", launch.get("code_revision", "")):
        raise ValueError("Run launch provenance differs from the current pinned selection or lacks a code revision")

    entries, compact_files, validation = [], {}, {}
    for recipe, dataset in RECIPES.items():
        gold_path = data_dir / f"{dataset}-test.jsonl"
        gold_raw = gold_path.read_bytes()
        gold = _parse_jsonl(gold_raw, gold_path.name)
        gold_spec = data_manifest.get("files", {}).get(gold_path.name)
        if gold_spec != {"rows": len(gold), "sha256": _sha(gold_raw)}:
            raise ValueError(f"{gold_path.name} differs from the committed data-v1 manifest")
        validation[recipe] = {}
        variants = [("frozen", None)] + [(f"seed{seed}", seed) for seed in SEEDS]
        for variant, seed in variants:
            stem = f"{recipe}-frozen" if seed is None else f"{recipe}-seed{seed}"
            prediction_path = run_dir / (f"{recipe}-frozen-predictions.jsonl" if seed is None
                                         else f"{recipe}-predictions-seed{seed}.jsonl")
            report_path = run_dir / (f"{recipe}-frozen-metrics.json" if seed is None
                                     else f"{recipe}-metrics-seed{seed}.json")
            prediction_raw, report = prediction_path.read_bytes(), _load(report_path)
            predictions = _parse_jsonl(prediction_raw, prediction_path.name)
            source_hashes = {"gold_sha256": _sha(gold_raw),
                             "predictions_sha256": _sha(prediction_raw),
                             "report_sha256": _sha(report_path)}
            if report.get("inputs") != {key: source_hashes[key] for key in ("gold_sha256", "predictions_sha256")}:
                raise ValueError(f"{stem} report does not hash its actual inputs")
            if seed is None:
                computed = evaluate_frozen(gold_raw, prediction_raw, model=base["id"],
                                           revision=base["revision"], bootstrap_samples=1000)
            else:
                training_path = run_dir / "runs" / recipe / f"seed{seed}" / "manifest.json"
                training = _load(training_path)
                adapter_path = training_path.parent / "final-adapter" / "adapter_model.safetensors"
                adapter_sha = _sha(adapter_path)
                if training.get("seed") != seed or {"id": training.get("model", {}).get("source"),
                                                     "revision": training.get("model", {}).get("revision")} != base:
                    raise ValueError(f"{stem} training manifest disagrees with pinned base or seed")
                validation[recipe][str(seed)] = training.get("best_validation_macro_f1")
                source_hashes.update(training_manifest_sha256=_sha(training_path), adapter_sha256=adapter_sha)
                computed = evaluate_predictions(gold, predictions, seed=seed, bootstrap_samples=1000,
                    base_model=base, adapter_sha256=adapter_sha,
                    input_hashes={key: source_hashes[key] for key in ("gold_sha256", "predictions_sha256")},
                    require_provenance=True)
            if computed != report:
                raise ValueError(f"{stem} stored report differs from recomputation")
            compact = _compact(gold, predictions)
            compact_files[f"{stem}.jsonl"] = compact
            entry = {"recipe": recipe, "variant": variant, "seed": seed,
                     "path": f"{stem}.jsonl", "source_hashes": source_hashes, "report": report,
                     "throughput": _throughput(compact), "slice_metrics": _slice_metrics(compact)}
            if seed is not None:
                entry["validation_macro_f1"] = validation[recipe][str(seed)]
                entry["selection_field"] = "training manifest best_validation_macro_f1"
            entries.append(entry)

    if any(not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1
           for scores in validation.values() for score in scores.values()):
        raise ValueError("Training manifests require numeric validation macro F1 for seed selection")
    output.mkdir(parents=True)
    for name, rows in compact_files.items():
        (output / name).write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    for entry in entries:
        entry["sha256"] = _sha(output / entry["path"])
    attribution = {"schema": "worthify-jev-attribution-v1",
        "scope": "Compact rows replay aggregate metrics. Original gold/prediction hashes were checked by the exporter against the source files; those files are not included for independent rehashing.",
        "sources": [
        {"name": "CLINC150", "authors": "Larson et al.", "source_url": "https://github.com/clinc/oos-eval",
         "license": "CC-BY-3.0", "license_url": "https://creativecommons.org/licenses/by/3.0/",
         "revision": "828f8093932c8fe6ca7936c3d2e52903b1c523de",
         "modifications": "Converted to bounded-option rows; prompt text is excluded from this evidence export."},
        {"name": "WANLI", "authors": "Liu et al.", "source_url": "https://huggingface.co/datasets/alisawuffles/WANLI",
         "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
         "revision": "61c95318fd71c55b6ba355d76253254615f387ec",
         "modifications": "Converted to bounded-option rows; prompt text is excluded from this evidence export."},
    ]}
    summary = _summary(entries, validation, base)
    for name, value in (("attribution.json", attribution), ("summary.json", summary)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest = {"schema": "worthify-jev-benchmark-evidence-v1", "base_model": base,
        "launch_code_revision": launch["code_revision"],
        "base_selection_sha256": _sha(ROOT / "manifests/base-selection-v1.json"),
        "data_manifest_sha256": _sha(ROOT / "manifests/data-v1.json"),
        "quantization": "nf4", "prompt_version": "direct-options-v1", "bootstrap_samples": 1000,
        "entries": entries,
        "attribution": {"path": "attribution.json", "sha256": _sha(output / "attribution.json")},
        "summary": {"path": "summary.json", "sha256": _sha(output / "summary.json")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.run_dir, args.data_dir, args.output)


if __name__ == "__main__":
    main()
