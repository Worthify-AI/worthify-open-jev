import json
from collections import Counter
from types import SimpleNamespace

import pytest

from experiments.long_context import authored_data, pilot, train_authored


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _inputs(tmp_path):
    rows = authored_data.build_dataset(seed=291_607, source_count=9, distractor_count=3)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(train, [row for row in rows if row["split"] == "train"])
    _write_jsonl(validation, [row for row in rows if row["split"] == "validation"])
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "schema": "worthify-authored-long-context-plan-v1",
        "base_model": pilot.MODEL,
        "base_revision": pilot.REVISION,
        "training_examples": 9,
        "epochs": 1,
        "optimizer_steps": 9,
        "micro_batch": 1,
        "effective_batch": 1,
        "learning_rate": 2e-4,
        "seed": 42,
        "max_tokens": pilot.MAX_NATIVE_CONTEXT,
        "activation_offload": True,
        "quantization": "NF4",
        "compute_dtype": "BF16",
        "lora_rank": 16,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "checkpoint_selection": "None: use the fixed ninth-step checkpoint, report even if worse.",
        "evaluation": (
            "All 27 held-out test rows at matched short and long lengths; "
            "all 27 validation rows at long length."
        ),
    }))
    args = SimpleNamespace(
        train=train, validation=validation, plan=plan, output=tmp_path / "output",
        max_tokens=pilot.MAX_NATIVE_CONTEXT, activation_offload=True,
    )
    return args


def test_balanced_nine_is_deterministic_and_covers_grid_and_three_groups(tmp_path):
    args = _inputs(tmp_path)
    prepared = train_authored.preflight(args)
    selected = prepared["selected_train"]
    assert [row["id"] for row in selected] == [
        row["id"] for row in train_authored.select_balanced_nine(
            list(reversed(prepared["train_rows"])), split="train"
        )
    ]
    assert len(selected) == 9
    assert {(row["evidence_position"], row["gold_option_id"]) for row in selected} == {
        (position, label)
        for position in train_authored.POSITIONS
        for label in train_authored.LABELS
    }
    assert sorted({row["group_id"] for row in selected}) == sorted({
        row["group_id"] for row in prepared["train_rows"]
    })
    assert set(Counter(row["group_id"] for row in selected).values()) == {3}


def test_option_reorder_recomputes_numeric_target_from_semantic_gold(tmp_path):
    selected = train_authored.preflight(_inputs(tmp_path))["selected_train"]
    orders = []
    for row in selected:
        reordered, target = train_authored.reorder_and_target(row)
        option_ids = [option["id"] for option in reordered["options"]]
        assert option_ids[target] == row["gold_option_id"]
        assert reordered["gold_option_id"] == row["gold_option_id"]
        assert train_authored.reorder_and_target(row) == (reordered, target)
        orders.append(option_ids)
    assert any(order != [option["id"] for option in row["options"]]
               for order, row in zip(orders, selected))


@pytest.mark.parametrize("failure", ["split", "source", "group"])
def test_preflight_rejects_split_or_leakage_before_output_write(tmp_path, failure):
    args = _inputs(tmp_path)
    validation_rows = [json.loads(line) for line in args.validation.read_text().splitlines()]
    train_rows = [json.loads(line) for line in args.train.read_text().splitlines()]
    if failure == "split":
        validation_rows[0]["split"] = "train"
        match = "only rows assigned to validation"
    elif failure == "source":
        validation_rows[0]["source"]["source_id"] = train_rows[0]["source"]["source_id"]
        match = "source ID"
    else:
        validation_rows[0]["group_id"] = train_rows[0]["group_id"]
        match = "source group"
    _write_jsonl(args.validation, validation_rows)
    with pytest.raises(ValueError, match=match):
        train_authored.preflight(args)
    assert not args.output.exists()


def test_preflight_rejects_existing_output_without_touching_it(tmp_path):
    args = _inputs(tmp_path)
    args.output.mkdir()
    marker = args.output / "keep"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError, match="create-only"):
        train_authored.preflight(args)
    assert marker.read_text() == "unchanged"
