import json
from types import SimpleNamespace

import pytest

from experiments.long_context import score


def _row(row_id="generic-1"):
    return {
        "id": row_id,
        "state": {"events": ["a", "b"], "accepted": True},
        "question": "Which outcome follows?",
        "options": [
            {"id": "accept", "description": "Accept the request."},
            {"id": "reject", "description": "Reject the request."},
            {"id": "review", "description": "Request review."},
        ],
    }


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_reader_accepts_generic_rows_without_gold_or_provenance(tmp_path):
    path = tmp_path / "input.jsonl"
    rows = [_row("one"), _row("two")]
    _write(path, rows)
    loaded, source_sha256 = score.read_rows(path)
    assert loaded == rows
    assert len(source_sha256) == 64
    assert all("gold_option_id" not in row and "provenance" not in row for row in loaded)


def test_prediction_maps_logits_to_declared_option_ids_without_prompt_text():
    row = _row()
    record = score.build_prediction_record(
        row,
        [-2.0, 4.0, 1.0],
        input_tokens=260_123,
        forward_seconds=2.5,
        total_seconds=2.75,
        peak_memory_bytes=1234,
        prompt_sha256="a" * 64,
        source_sha256="b" * 64,
        model_metadata={"source": score.pilot.MODEL, "revision": score.pilot.REVISION,
                        "adapter_sha256": "c" * 64},
        runtime_backend={"configured_backend": "flex_attention"},
    )
    assert record["option_ids"] == ["accept", "reject", "review"]
    assert record["prediction"] == "reject"
    assert record["actual_tokens"] == 260_123
    assert record["adapter_sha256"] == "c" * 64
    assert record["prompt_sha256"] == "a" * 64
    assert record["runtime_backend"]["configured_backend"] == "flex_attention"
    assert "model loading" in record["timing_scope"]["forward_seconds"]
    assert "mask construction" in record["timing_scope"]["total_seconds"]
    assert sum(record["probabilities"]) == pytest.approx(1.0)
    assert record["probability_status"].startswith("conditional option scores")
    assert "state" not in record and "question" not in record
    assert "gold_option_id" not in record and "provenance" not in record


@pytest.mark.parametrize(
    "rows, match",
    [
        ([_row("same"), _row("same")], "Duplicate input row ID"),
        ([{**_row(), "options": [_row()["options"][0]]}], "2-16"),
        ([{**_row(), "question": ""}], "nonempty"),
    ],
)
def test_reader_rejects_invalid_input(rows, match, tmp_path):
    path = tmp_path / "input.jsonl"
    _write(path, rows)
    with pytest.raises(ValueError, match=match):
        score.read_rows(path)


def test_static_preflight_rejects_existing_output_context_overflow_and_bare_adapter(tmp_path):
    source = tmp_path / "input.jsonl"
    _write(source, [_row()])
    output = tmp_path / "output.jsonl"
    output.write_text("keep")
    base = dict(input=source, output=output, cache_dir=None, max_tokens=1024,
                adapter=None, adapter_revision=None)
    with pytest.raises(FileExistsError, match="create-only"):
        score.run(SimpleNamespace(**base))
    assert output.read_text() == "keep"

    output.unlink()
    with pytest.raises(ValueError, match="between 1 and"):
        score.run(SimpleNamespace(**{**base, "output": output, "max_tokens": 262_145}))
    assert not output.exists()

    with pytest.raises(ValueError, match="provided together"):
        score.run(SimpleNamespace(**{**base, "output": output, "adapter_revision": "a" * 40}))
    assert not output.exists()


def test_token_overflow_preflight_does_not_create_output(tmp_path, monkeypatch):
    source = tmp_path / "input.jsonl"
    output = tmp_path / "output.jsonl"
    _write(source, [_row()])
    args = SimpleNamespace(input=source, output=output, cache_dir=None, max_tokens=16,
                           adapter=None, adapter_revision=None)

    import torch

    import torch._dynamo
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)
    model = SimpleNamespace(config=SimpleNamespace(use_cache=False), eval=lambda: None)
    monkeypatch.setattr(score, "load_causal_model", lambda *_a, **_kw: (model, object(), {}))
    monkeypatch.setattr(score.pilot, "_select_and_verify_flex", lambda _model: {})
    monkeypatch.setattr(score.pilot, "_model_config",
                        lambda _model: SimpleNamespace(max_position_embeddings=262_144))
    monkeypatch.setattr(score, "encode_prompt",
                        lambda *_a, **_kw: (_ for _ in ()).throw(
                            ValueError("17 input tokens exceed limit 16; no truncation allowed")))

    with pytest.raises(ValueError, match="no truncation"):
        score.run(args)
    assert not output.exists()
