import hashlib
import json

import pytest

from openjev_phase1 import datasets
from openjev_phase1.core import direct_messages


def _clinc_source(path):
    labels = [f"intent_{index:03d}" for index in range(150)]
    payload = {
        "train": [[f"train request {label}", label] for label in labels],
        "val": [[f"validation request {label}", label] for label in labels],
        "test": [[f"test request {label}", label] for label in labels],
        "oos_train": [["a train request outside the taxonomy", "oos"]],
        "oos_val": [["a validation request outside the taxonomy", "oos"]],
        "oos_test": [["a test request outside the taxonomy", "oos"]],
    }
    path.write_text(json.dumps(payload))


def test_clinc_converter_holds_out_labels_and_builds_bounded_options(tmp_path):
    source = tmp_path / "clinc.json"
    _clinc_source(source)
    rows = datasets.convert_clinc150(source)
    train_labels = {row["gold_option_id"] for row in rows if row["split"] == "train"}
    unseen_test = {row["gold_option_id"] for row in rows
                   if row["split"] == "test" and row["evaluation_slice"] == "unseen_label_test"}
    assert len(unseen_test) == 15
    assert train_labels.isdisjoint(unseen_test)
    train_option_ids = {option["id"] for row in rows if row["split"] == "train" for option in row["options"]}
    assert train_option_ids.isdisjoint(unseen_test)
    assert {len(row["options"]) for row in rows} == {2, 4, 8, 16}
    assert all(row["gold_option_id"] in {option["id"] for option in row["options"]} for row in rows)
    oos = [row for row in rows if row["gold_option_id"] == "none_of_above"]
    assert len(oos) == 3
    assert all(any(option["id"] == "none_of_above" for option in row["options"]) for row in oos)
    assert all(row["source"]["official_split"] not in {"test", "oos_test"}
               for row in rows if row["split"] == "train")
    rendered = json.dumps(direct_messages(rows[0]))
    assert "gold_option_id" not in rendered
    assert "official_split" not in rendered


def test_clinc_normalized_duplicates_are_kept_in_one_split(tmp_path):
    source = tmp_path / "clinc.json"
    _clinc_source(source)
    payload = json.loads(source.read_text())
    payload["train"][0][0] = "Same—Request!"
    payload["test"][0][0] = " same request "
    source.write_text(json.dumps(payload))
    rows = datasets.convert_clinc150(source)
    duplicates = [row for row in rows if datasets.normalize_text(row["state"]) == "same request"]
    assert len(duplicates) == 2
    assert {row["split"] for row in duplicates} == {"test"}
    assert len({row["group_id"] for row in duplicates}) == 1


def test_wanli_exclusion_removes_entire_related_component(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    raw = [
        {"id": 1, "premise": "selected premise", "hypothesis": "a", "gold": "neutral", "pairID": "p1"},
        {"id": 2, "premise": "bridge premise", "hypothesis": "b", "gold": "entailment", "pairID": "p1"},
        {"id": 3, "premise": "Bridge premise!", "hypothesis": "c", "gold": "contradiction", "pairID": "p2"},
        {"id": 4, "premise": "safe premise", "hypothesis": "d", "gold": "neutral", "pairID": "safe"},
    ]
    train.write_text("".join(json.dumps(row) + "\n" for row in raw))
    test = tmp_path / "test.jsonl"
    test_row = {"id": 99, "premise": "SELECTED PREMISE", "hypothesis": "external", "gold": "neutral", "pairID": "external"}
    test.write_text(json.dumps(test_row) + "\n")
    monkeypatch.setitem(datasets.WANLI_UPSTREAM_TEST, "sha256", hashlib.sha256(test.read_bytes()).hexdigest())
    selection = tmp_path / "selection.jsonl"
    selection.write_text(json.dumps({"source": "wanli", "group_id": "external-group",
                                     "upstream": {"source_id": 99, "seed_id": "external"}}) + "\n")
    rows = datasets.convert_wanli(train, exclusion_manifest=selection, exclusion_test_source=test)
    assert {row["source"]["source_id"] for row in rows} == {4}


def test_robustness_builder_only_uses_owned_rows():
    base = {"id": "owned", "task": "classification", "state": "record", "question": "Choose",
            "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
            "gold_option_id": "a", "group_id": "g", "split": "test",
            "source": {"dataset": "OpenJev"}}
    external = {**base, "id": "external", "source": {"dataset": "CLINC150"}}
    rows = datasets.build_robustness_examples([base, external])
    assert {row["perturbation"]["kind"] for row in rows} == {"option_order", "criterion_wrapper"}
    assert all(row["perturbation"]["base_id"] == "owned" for row in rows)
    assert all(row["gold_option_id"] == "a" for row in rows)


def test_prepare_is_create_only(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(FileExistsError, match="replace"):
        datasets.prepare_datasets(tmp_path / "missing-clinc", tmp_path / "missing-wanli", output)
