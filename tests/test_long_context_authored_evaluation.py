import json
from types import SimpleNamespace

import pytest

from experiments.long_context import authored_data, evaluate_authored


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_reader_preserves_authored_semantic_ids_without_prompt_data_in_outputs(tmp_path):
    source = tmp_path / "input.jsonl"
    rows = authored_data.build_dataset(seed=3, source_count=3, distractor_count=3)[:4]
    _write_rows(source, rows)
    loaded, digest = evaluate_authored.read_authored_rows(source)
    assert loaded == rows
    assert len(digest) == 64

    records = []
    for row in loaded:
        records.append({
            "id": row["id"], "gold_option_id": row["gold_option_id"],
            "prediction": row["gold_option_id"], "evidence_position": row["evidence_position"],
        })
    summary = evaluate_authored.summarize(records, loaded[:2], loaded)
    assert summary["overall"]["accuracy"] == 1.0
    assert summary["coverage"]["selection"] == "deterministic first rows in input order"
    assert summary["coverage"]["selected_rows"] == 2
    assert summary["coverage"]["omitted_rows"] == 2
    assert "state" not in records[0] and "question" not in records[0]


def test_prediction_record_maps_semantic_ids_and_metadata_without_prompt_text():
    row = authored_data.build_dataset(seed=4, source_count=3, distractor_count=3)[0]
    record = evaluate_authored.build_prediction_record(
        row, [3.0, 1.0, -2.0], actual_tokens=8192, elapsed_seconds=1.25,
        peak_vram_bytes=1234, prompt_sha256="a" * 64, source_sha256="b" * 64,
        model_metadata={"adapter_sha256": "c" * 64},
    )
    assert record["id"] == row["id"]
    assert record["gold_option_id"] == row["gold_option_id"]
    assert record["option_ids"] == [option["id"] for option in row["options"]]
    assert record["prediction"] == row["options"][0]["id"]
    assert record["actual_tokens"] == 8192
    assert record["source_sha256"] == "b" * 64
    assert record["adapter_sha256"] == "c" * 64
    assert record["model_revision"] == evaluate_authored.pilot.REVISION
    assert record["probability_status"].endswith("uncalibrated")
    assert "state" not in record and "question" not in record and "provenance" not in record


def test_summary_reports_accuracy_macro_f1_by_label_and_position():
    rows = authored_data.build_dataset(seed=8, source_count=3, distractor_count=3)
    records = [{
        "id": row["id"],
        "gold_option_id": row["gold_option_id"],
        "prediction": row["gold_option_id"] if index else "insufficient",
        "evidence_position": row["evidence_position"],
    } for index, row in enumerate(rows)]
    summary = evaluate_authored.summarize(records, rows, rows)
    assert 0 < summary["overall"]["accuracy"] < 1
    assert 0 < summary["overall"]["macro_f1"] < 1
    assert set(summary["by_label"]) == {"supported", "contradicted", "insufficient"}
    assert set(summary["by_evidence_position"]) == {"early", "middle", "late"}
    assert all("accuracy" in value and "macro_f1" in value
               for value in summary["by_label"].values())
    assert all("accuracy" in value and "macro_f1" in value
               for value in summary["by_evidence_position"].values())


def test_reader_rejects_non_authored_rows(tmp_path):
    path = tmp_path / "input.jsonl"
    row = authored_data.build_dataset(source_count=3, distractor_count=3)[0]
    row["authored"] = False
    _write_rows(path, [row])
    with pytest.raises(ValueError, match="not authored"):
        evaluate_authored.read_authored_rows(path)


def test_create_only_preflight_happens_without_loading_gpu(tmp_path, monkeypatch):
    path = tmp_path / "input.jsonl"
    _write_rows(path, authored_data.build_dataset(source_count=3, distractor_count=3)[:1])
    output = tmp_path / "predictions.jsonl"
    output.write_text("keep")
    args = SimpleNamespace(
        input=path, output=output, summary=tmp_path / "summary.json", cache_dir=None,
        max_tokens=1024, limit=None, adapter=None, adapter_revision=None,
    )
    monkeypatch.setattr(evaluate_authored, "load_causal_model",
                        lambda *_args, **_kwargs: pytest.fail("model must not load"))
    with pytest.raises(FileExistsError, match="create-only"):
        evaluate_authored.run(args)
    assert output.read_text() == "keep"


def test_failure_summary_retains_status_and_prior_output(tmp_path, monkeypatch):
    path = tmp_path / "input.jsonl"
    _write_rows(path, authored_data.build_dataset(source_count=3, distractor_count=3)[:1])
    args = SimpleNamespace(
        input=path, output=tmp_path / "predictions.jsonl", summary=tmp_path / "summary.json",
        cache_dir=None, max_tokens=1024, limit=1, adapter=None, adapter_revision=None,
    )

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    import torch
    monkeypatch.setattr(torch, "cuda", FakeCuda())
    with pytest.raises(RuntimeError, match="exactly one CUDA"):
        evaluate_authored.run(args)
    summary = json.loads(args.summary.read_text())
    assert summary["status"] == "failed"
    assert summary["completed_rows"] == 0
    assert summary["output_sha256"]
    assert args.output.exists()
