"""Verify compact Worthify benchmark evidence and reproduce its public claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

try:
    from benchmarks.evaluate_frozen import evaluate_frozen
    from benchmarks.export_worthify import RECIPES, ROOT, SEEDS, _load, _sha, _slice_metrics, _summary, _throughput
except ModuleNotFoundError:  # Direct execution: python benchmarks/verify_worthify.py
    from evaluate_frozen import evaluate_frozen
    from export_worthify import RECIPES, ROOT, SEEDS, _load, _sha, _slice_metrics, _summary, _throughput
from openjev_phase1.evaluation import evaluate_predictions


ROW_KEYS = {"id", "task", "group_id", "split", "evaluation_slice", "gold_option_id", "option_ids",
            "probabilities", "latency_seconds", "peak_memory_bytes", "input_tokens", "warm"}


def _rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(set(row) != ROW_KEYS for row in rows):
        raise ValueError(f"Compact rows have unexpected fields: {path.name}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate compact row ID: {path.name}")
    return rows


def _inputs(rows: list[dict], *, base: dict, seed: int | None, adapter_sha: str | None) -> tuple[list[dict], list[dict]]:
    gold, predictions = [], []
    for row in rows:
        gold.append({"id": row["id"], "task": row["task"], "group_id": row["group_id"],
            "split": row["split"], "state": row["id"], "question": "option",
            "options": [{"id": item, "description": item} for item in row["option_ids"]],
            "gold_option_id": row["gold_option_id"]})
        prediction = {"id": row["id"], "option_ids": row["option_ids"],
            "probabilities": row["probabilities"], "total_seconds": row["latency_seconds"],
            "peak_memory_bytes": row["peak_memory_bytes"], "input_tokens": row["input_tokens"],
            "warm": row["warm"]}
        if seed is None:
            prediction.update(prompt_version="direct-options-v1", model={"source": base["id"],
                "revision": base["revision"], "quantization": "nf4", "dtype": "bfloat16"})
        else:
            prediction.update(base_model=base, adapter_sha256=adapter_sha, seed=seed)
        predictions.append(prediction)
    return gold, predictions


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def _assert_report(actual: dict, expected: dict, *, frozen: bool) -> None:
    if frozen:
        actual = {**actual, "inputs": expected.get("inputs")}
    if actual != expected:
        raise ValueError("Stored aggregate metric report differs from compact-row recomputation")


def verify(artifact_dir: Path) -> dict:
    manifest = _load(artifact_dir / "manifest.json")
    if manifest.get("schema") != "worthify-jev-benchmark-evidence-v1" or manifest.get("bootstrap_samples") != 1000:
        raise ValueError("Unsupported benchmark evidence manifest")
    base = manifest.get("base_model")
    selection = _load(ROOT / "manifests/base-selection-v1.json")
    data_manifest = _load(ROOT / "manifests/data-v1.json")
    if base != {"id": selection.get("selected_model"), "revision": selection.get("revision")}:
        raise ValueError("Evidence does not use the currently pinned 12B base model")
    if (manifest.get("base_selection_sha256") != _sha(ROOT / "manifests/base-selection-v1.json")
            or manifest.get("data_manifest_sha256") != _sha(ROOT / "manifests/data-v1.json")
            or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("code_revision", ""))):
        raise ValueError("Shared run provenance is incomplete or stale")
    if manifest.get("quantization") != "nf4" or manifest.get("prompt_version") != "direct-options-v1":
        raise ValueError("Benchmark protocol differs from the frozen protocol")
    entries = manifest.get("entries")
    expected_matrix = {(recipe, variant) for recipe in RECIPES
                       for variant in ("frozen", "seed42", "seed43")}
    if (not isinstance(entries, list) or len(entries) != 6
            or {(item.get("recipe"), item.get("variant")) for item in entries} != expected_matrix):
        raise ValueError("Manifest must contain the complete two-recipe baseline/two-seed matrix")

    for expected_path, link in (("attribution.json", manifest.get("attribution")),
                                ("summary.json", manifest.get("summary"))):
        if (not isinstance(link, dict) or link.get("path") != expected_path
                or _sha(artifact_dir / expected_path) != link.get("sha256")):
            raise ValueError("Linked artifact hash mismatch")
    validation: dict[str, dict[str, float]] = {recipe: {} for recipe in RECIPES}
    computed_entries = []
    for entry in entries:
        expected_seed = None if entry["variant"] == "frozen" else int(entry["variant"].removeprefix("seed"))
        if entry.get("seed") != expected_seed:
            raise ValueError("Variant and seed disagree")
        if (not isinstance(entry.get("path"), str) or Path(entry["path"]).name != entry["path"]
                or entry["path"] != f"{entry['recipe']}-{entry['variant']}.jsonl"):
            raise ValueError("Compact evidence paths must be known flat artifact names")
        path = artifact_dir / entry["path"]
        if _sha(path) != entry.get("sha256"):
            raise ValueError(f"Compact evidence hash mismatch: {entry['path']}")
        report, sources = entry.get("report"), entry.get("source_hashes")
        if not isinstance(report, dict) or not isinstance(sources, dict):
            raise ValueError("Entry lacks report or source hashes")
        required_hashes = {"gold_sha256", "predictions_sha256", "report_sha256"}
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in sources.values()):
            raise ValueError("Original source hashes must be lowercase SHA-256 digests")
        if report.get("inputs") != {key: sources.get(key) for key in ("gold_sha256", "predictions_sha256")}:
            raise ValueError("Stored report input hashes differ from original source hashes")
        seed = entry.get("seed")
        frozen = entry["variant"] == "frozen"
        if not frozen:
            required_hashes |= {"training_manifest_sha256", "adapter_sha256"}
        if set(sources) != required_hashes:
            raise ValueError("Entry has missing or unexpected original source hashes")
        adapter_sha = None if frozen else sources.get("adapter_sha256")
        rows = _rows(path)
        if entry.get("throughput") != _throughput(rows):
            raise ValueError("Stored throughput differs from compact measured latencies")
        if entry.get("slice_metrics") != _slice_metrics(rows):
            raise ValueError("Stored slice metrics differ from compact-row recomputation")
        dataset = RECIPES[entry["recipe"]]
        gold_spec = data_manifest.get("files", {}).get(f"{dataset}-test.jsonl", {})
        if sources["gold_sha256"] != gold_spec.get("sha256") or len(rows) != gold_spec.get("rows"):
            raise ValueError("Compact population differs from the committed held-out benchmark")
        gold, predictions = _inputs(rows, base=base, seed=seed, adapter_sha=adapter_sha)
        if frozen:
            actual = evaluate_frozen(_jsonl(gold), _jsonl(predictions), model=base["id"],
                                     revision=base["revision"], bootstrap_samples=1000)
        else:
            actual = evaluate_predictions(gold, predictions, seed=seed, bootstrap_samples=1000,
                base_model=base, adapter_sha256=adapter_sha,
                input_hashes={key: sources[key] for key in ("gold_sha256", "predictions_sha256")},
                require_provenance=True)
            score = entry.get("validation_macro_f1")
            if (entry.get("selection_field") != "training manifest best_validation_macro_f1"
                    or isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1):
                raise ValueError("Tuned entry lacks training-validation seed selection evidence")
            validation[entry["recipe"]][str(seed)] = score
        _assert_report(actual, report, frozen=frozen)
        computed_entries.append({**entry, "report": actual if not frozen else report})

    stored_summary = _load(artifact_dir / manifest["summary"]["path"])
    actual_summary = _summary(computed_entries, validation, base)
    if actual_summary != stored_summary:
        raise ValueError("Stored final summary differs from verified reports")
    return actual_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.artifact_dir), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
