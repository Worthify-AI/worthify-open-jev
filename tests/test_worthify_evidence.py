import hashlib
import json
from pathlib import Path

import pytest

import benchmarks.export_worthify as exporter
import benchmarks.verify_worthify as verifier
from benchmarks.evaluate_frozen import evaluate_frozen
from benchmarks.export_worthify import RECIPES, export
from benchmarks.verify_worthify import verify
from openjev_phase1.evaluation import evaluate_predictions


BASE = {"id": "google/gemma-4-12B-it", "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"}


def _jsonl(rows):
    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def _fixture(tmp_path):
    run, data, root = tmp_path / "run", tmp_path / "data", tmp_path / "root"
    run.mkdir()
    data.mkdir()
    (root / "manifests").mkdir(parents=True)
    selection = json.loads((Path(__file__).parents[1] / "manifests/base-selection-v1.json").read_text())
    (root / "manifests/base-selection-v1.json").write_text(json.dumps(selection))
    (run / "launch.json").write_text(json.dumps({"code_revision": "6" * 40, "base_selection": selection}))
    files = {}
    for recipe, dataset in RECIPES.items():
        gold = [{"id": f"{recipe}-1", "task": recipe, "group_id": f"{recipe}-group",
                 "split": "test", "evaluation_slice": "unseen_label_test" if recipe == "classification" else "internal_group_holdout",
                 "state": f"private {recipe} prompt", "question": "private question",
                 "options": [{"id": "yes", "description": "private yes"},
                             {"id": "no", "description": "private no"}], "gold_option_id": "yes"}]
        gold_raw = _jsonl(gold)
        (data / f"{dataset}-test.jsonl").write_bytes(gold_raw)
        files[f"{dataset}-test.jsonl"] = {"rows": 1, "sha256": hashlib.sha256(gold_raw).hexdigest()}
        model = {"source": BASE["id"], "revision": BASE["revision"], "quantization": "nf4", "dtype": "bfloat16"}
        frozen = [{"id": gold[0]["id"], "option_ids": ["yes", "no"], "probabilities": [.25, .75],
                   "total_seconds": .3, "peak_memory_bytes": 30, "input_tokens": 9, "warm": True,
                   "prompt_version": "direct-options-v1", "model": model}]
        frozen_raw = _jsonl(frozen)
        (run / f"{recipe}-frozen-predictions.jsonl").write_bytes(frozen_raw)
        frozen_report = evaluate_frozen(gold_raw, frozen_raw, model=BASE["id"], revision=BASE["revision"])
        (run / f"{recipe}-frozen-metrics.json").write_text(json.dumps(frozen_report))
        for seed, probability in ((42, .8), (43, .6)):
            adapter_dir = run / "runs" / recipe / f"seed{seed}" / "final-adapter"
            adapter_dir.mkdir(parents=True)
            adapter_path = adapter_dir / "adapter_model.safetensors"
            adapter_path.write_bytes(f"{recipe}-{seed}".encode())
            adapter_sha = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
            training = {"seed": seed, "model": model, "best_validation_macro_f1": probability}
            (adapter_dir.parent / "manifest.json").write_text(json.dumps(training))
            option_ids = ["no", "yes"] if seed == 43 else ["yes", "no"]
            probabilities = [1 - probability, probability] if seed == 43 else [probability, 1 - probability]
            predictions = [{"id": gold[0]["id"], "option_ids": option_ids,
                "probabilities": probabilities, "total_seconds": seed / 100,
                "peak_memory_bytes": seed, "input_tokens": 10, "warm": True,
                "base_model": BASE, "adapter_sha256": adapter_sha, "seed": seed}]
            prediction_raw = _jsonl(predictions)
            (run / f"{recipe}-predictions-seed{seed}.jsonl").write_bytes(prediction_raw)
            report = evaluate_predictions(gold, predictions, seed=seed, bootstrap_samples=1000,
                base_model=BASE, adapter_sha256=adapter_sha,
                input_hashes={"gold_sha256": hashlib.sha256(gold_raw).hexdigest(),
                              "predictions_sha256": hashlib.sha256(prediction_raw).hexdigest()},
                require_provenance=True)
            (run / f"{recipe}-metrics-seed{seed}.json").write_text(json.dumps(report))
    (root / "manifests/data-v1.json").write_text(json.dumps({"files": files}))
    return run, data, root


def test_compact_export_recomputes_with_shared_metric_function(tmp_path, monkeypatch):
    run, data, root = _fixture(tmp_path)
    monkeypatch.setattr(exporter, "ROOT", root)
    monkeypatch.setattr(verifier, "ROOT", root)
    artifact = tmp_path / "artifact"
    export(run, data, artifact)
    summary = verify(artifact)
    assert summary["recipes"]["classification"]["selected_seed"] == 42
    assert summary["recipes"]["classification"]["seeds"]["42"]["accuracy"] == 1
    assert summary["recipes"]["classification"]["seeds"]["42"]["throughput"]["rows_per_second"] == pytest.approx(1 / .42)
    assert summary["recipes"]["classification"]["seeds"]["42"]["slice_metrics"]["unseen_label_test"]["accuracy"] == 1
    permuted = json.loads((artifact / "classification-seed43.jsonl").read_text())
    assert permuted["option_ids"] == ["yes", "no"] and permuted["probabilities"] == [.6, .4]
    compact = (artifact / "classification-seed42.jsonl").read_text()
    assert "private" not in compact and "state" not in compact and "question" not in compact


def test_verify_rejects_tampered_stored_metric(tmp_path, monkeypatch):
    run, data, root = _fixture(tmp_path)
    monkeypatch.setattr(exporter, "ROOT", root)
    monkeypatch.setattr(verifier, "ROOT", root)
    artifact = tmp_path / "artifact"
    export(run, data, artifact)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["report"]["accuracy"] = .123
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metric report"):
        verify(artifact)


def test_verify_rejects_tampered_original_source_hash(tmp_path, monkeypatch):
    run, data, root = _fixture(tmp_path)
    monkeypatch.setattr(exporter, "ROOT", root)
    monkeypatch.setattr(verifier, "ROOT", root)
    artifact = tmp_path / "artifact"
    export(run, data, artifact)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["entries"][0]["source_hashes"]["gold_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source hashes"):
        verify(artifact)
