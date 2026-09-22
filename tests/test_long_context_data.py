import json
import re

import pytest

from experiments.long_context import authored_data
from openjev_phase1.core import direct_messages, validate_row


def test_labels_are_derived_from_source_facts_and_absence_is_not_negative():
    assert authored_data.derive_label(True) == "supported"
    assert authored_data.derive_label(False) == "contradicted"
    assert authored_data.derive_label(None) == "insufficient"
    assert authored_data.derive_label(True, claimed_outcome=False) == "contradicted"

    rows = authored_data.build_dataset(source_count=3, distractor_count=5)
    assert {row["gold_option_id"] for row in rows} == {
        "supported", "contradicted", "insufficient"
    }
    insufficient = [row for row in rows if row["gold_option_id"] == "insufficient"]
    contradicted = [row for row in rows if row["gold_option_id"] == "contradicted"]
    assert all("asset index assigns" in row["state"] for row in insufficient)
    assert all("no outcome" not in row["state"] for row in insufficient)
    for row in insufficient:
        target = "T-" + row["question"].split("T-", 1)[1].split()[0]
        target_record = next(line for line in row["state"].splitlines() if target in line)
        assert re.fullmatch(
            rf"Record \d+: The asset index assigns .* {target} archive reference AR-[0-9A-F]{{10}}\.",
            target_record,
        )
    assert all("explicitly says" in row["state"] for row in contradicted)

    targets = [re.search(r"T-([0-9A-F]+)", row["question"]).group(1) for row in rows]
    assert len(set(targets)) == len(targets)
    assert all(len(target) == 12 for target in targets)
    assert all(not re.fullmatch(r"\d{3}-\d{2}", target) for target in targets)


def test_positions_and_distractors_are_meaningful_and_configurable():
    rows = authored_data.build_dataset(source_count=3, distractor_count=7)
    assert {row["evidence_position"] for row in rows} == {"early", "middle", "late"}
    assert all(row["distractor_count"] == 7 for row in rows)
    for row in rows:
        records = row["state"].splitlines()
        assert len(records) == 8
        assert len(set(records)) == len(records)
        distractors = [record for record in records if "D-" in record]
        assert len(distractors) == 7
        assert len({record.split(":", 2)[1] for record in distractors}) > 1


def test_splits_hold_out_source_groups_and_generation_is_reproducible():
    first = authored_data.build_dataset(seed=41, source_count=9, distractor_count=6)
    again = authored_data.build_dataset(seed=41, source_count=9, distractor_count=6)
    assert first == again
    assert first != authored_data.build_dataset(seed=42, source_count=9, distractor_count=6)

    group_splits = {}
    source_splits = {}
    for row in first:
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
        source_splits.setdefault(row["source"]["source_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert all(len(splits) == 1 for splits in source_splits.values())
    split_groups = {
        split: {row["group_id"] for row in first if row["split"] == split}
        for split in authored_data.SPLITS
    }
    assert all(split_groups.values())
    assert split_groups["train"].isdisjoint(split_groups["validation"])
    assert split_groups["train"].isdisjoint(split_groups["test"])
    assert split_groups["validation"].isdisjoint(split_groups["test"])


def test_option_reordering_does_not_change_semantic_gold_or_leak_metadata():
    row = authored_data.build_dataset(seed=9, source_count=3, distractor_count=4)[0]
    reordered = next(
        candidate for option_seed in range(100)
        if (candidate := authored_data.reorder_options(row, option_seed))["options"] != row["options"]
    )
    assert reordered["gold_option_id"] == row["gold_option_id"]
    assert {option["id"] for option in reordered["options"]} == {
        option["id"] for option in row["options"]
    }
    assert reordered["gold_option_id"] in {option["id"] for option in reordered["options"]}
    rendered = json.dumps(direct_messages(reordered))
    assert "gold_option_id" not in rendered
    assert "provenance" not in rendered
    assert "source_id" not in rendered
    validate_row(reordered)


def test_writer_is_create_only_and_emits_split_jsonl(tmp_path):
    output = tmp_path / "authored"
    manifest = authored_data.write_dataset(output, seed=7, source_count=3, distractor_count=3)
    assert manifest["labels"] == {"contradicted": 9, "insufficient": 9, "supported": 9}
    assert set(manifest["files"]) == {
        "authored-long-context-train.jsonl",
        "authored-long-context-validation.jsonl",
        "authored-long-context-test.jsonl",
    }
    for name, details in manifest["files"].items():
        rows = [json.loads(line) for line in (output / name).read_text().splitlines()]
        assert len(rows) == details["rows"] == 9
        assert all(row["split"] in name for row in rows)
    with pytest.raises(FileExistsError):
        authored_data.write_dataset(output)


@pytest.mark.parametrize("field,value", [
    ("source_count", 2),
    ("distractor_count", 2),
    ("seed", True),
])
def test_invalid_builder_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        authored_data.build_dataset(**{field: value})
