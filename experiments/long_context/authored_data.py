"""Deterministic project-authored data for distant-evidence experiments.

The records are fictional.  Labels are derived from the authored source fact,
not embedded in the evidence text or selected from an answer position.  This
module performs no tokenization, model loading, downloading, or training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DATASET_NAME = "OpenJev authored distant evidence"
DATASET_VERSION = "1"
SPLITS = ("train", "validation", "test")
POSITIONS = ("early", "middle", "late")
OUTCOMES = (True, False, None)
OPTIONS = (
    {"id": "supported", "description": "The records support the claim"},
    {"id": "contradicted", "description": "The records contradict the claim"},
    {"id": "insufficient", "description": "The records do not establish whether the claim is true"},
)


@dataclass(frozen=True)
class Scenario:
    key: str
    domain: str
    subject_kind: str
    claim: str
    positive_record: str
    negative_record: str


SCENARIOS = (
    Scenario("shipment", "general", "shipment", "arrived with its seal intact",
             "The signed receiving record confirms that {target} arrived with its seal intact.",
             "The signed receiving record explicitly says that {target} did not arrive with its seal intact."),
    Scenario("reservation", "general", "meeting room", "was reserved for the requested hour",
             "The final booking ledger confirms that {target} was reserved for the requested hour.",
             "The final booking ledger explicitly says that {target} was not reserved for the requested hour."),
    Scenario("calibration", "general", "sensor", "passed its calibration check",
             "The completed calibration certificate confirms that {target} passed its calibration check.",
             "The completed calibration certificate explicitly says that {target} did not pass its calibration check."),
    Scenario("refund", "general", "refund", "was issued to the customer",
             "The payment settlement record confirms that {target} was issued to the customer.",
             "The payment settlement record explicitly says that {target} was not issued to the customer."),
    Scenario("permit", "general", "permit", "received final approval",
             "The signed decision notice confirms that {target} received final approval.",
             "The signed decision notice explicitly says that {target} did not receive final approval."),
    Scenario("inspection", "general", "inspection", "was completed",
             "The inspector's final log confirms that {target} was completed.",
             "The inspector's final log explicitly says that {target} was not completed."),
    Scenario("patch", "defensive-cyber", "server", "received the required security patch",
             "The completed maintenance log confirms that {target} received the required security patch.",
             "The completed maintenance log explicitly says that {target} did not receive the required security patch."),
    Scenario("mfa", "defensive-cyber", "account", "had multi-factor authentication enabled",
             "The final access review confirms that {target} had multi-factor authentication enabled.",
             "The final access review explicitly says that {target} did not have multi-factor authentication enabled."),
    Scenario("signature", "defensive-cyber", "package", "passed signature verification",
             "The verification log confirms that {target} passed signature verification.",
             "The verification log explicitly says that {target} did not pass signature verification."),
)


DISTRACTOR_PATTERNS = (
    "Inventory record {record}: asset {asset} is assigned to storage zone {value}.",
    "Scheduling note {record}: courier route {asset} is planned for window {value}.",
    "Facilities log {record}: room {asset} has a capacity of {value} people.",
    "Catalog entry {record}: component {asset} has finish code {value}.",
    "Service record {record}: printer {asset} received routine cleaning in cycle {value}.",
    "Procurement note {record}: order {asset} contains {value} cartons.",
    "Archive index {record}: folder {asset} is retained under schedule {value}.",
    "Fleet record {record}: vehicle {asset} was assigned parking bay {value}.",
    "Training roster {record}: course {asset} lists {value} available seats.",
    "Network inventory {record}: switch {asset} is mounted in rack {value}.",
    "Backup catalog {record}: volume {asset} uses retention tier {value}.",
    "Certificate inventory {record}: certificate {asset} expires in quarter {value}.",
)


def derive_label(record_outcome: bool | None, claimed_outcome: bool = True) -> str:
    """Return the judgment entailed by a source fact and a boolean claim."""
    if record_outcome is None:
        return "insufficient"
    return "supported" if record_outcome is claimed_outcome else "contradicted"


def _stable(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def _distractors(source_index: int, count: int) -> list[str]:
    records = []
    for index in range(count):
        pattern = DISTRACTOR_PATTERNS[(source_index * 5 + index) % len(DISTRACTOR_PATTERNS)]
        records.append(pattern.format(
            record=f"D-{source_index:03d}-{index:04d}",
            asset=f"F-{(source_index * 97 + index * 13) % 10_000:04d}",
            value=10 + ((source_index * 31 + index * 17) % 89),
        ))
    return records


def _insert_record(records: list[str], relevant: str, position: str) -> list[str]:
    if position == "early":
        index = min(1, len(records))
    elif position == "middle":
        index = len(records) // 2
    elif position == "late":
        index = len(records)
    else:
        raise ValueError(f"Unknown evidence position: {position}")
    result = records.copy()
    result.insert(index, relevant)
    return result


def _ordered_options(seed: int, row_id: str) -> list[dict]:
    return sorted((dict(option) for option in OPTIONS),
                  key=lambda option: _stable("option-order", seed, row_id, option["id"]))


def reorder_options(row: dict, seed: int) -> dict:
    """Return a copied row with a deterministic option permutation.

    Semantic option IDs and the gold ID are retained, so the judgment is
    independent of answer position.
    """
    result = copy.deepcopy(row)
    result["options"] = sorted(result["options"],
                               key=lambda option: _stable("reorder", seed, row["id"], option["id"]))
    return result


def _relevant_record(scenario: Scenario, target: str, outcome: bool | None) -> str:
    if outcome is True:
        return scenario.positive_record.format(target=target)
    if outcome is False:
        return scenario.negative_record.format(target=target)
    archive_reference = _stable("archive-reference", target)[:10].upper()
    return f"The asset index assigns {target} archive reference AR-{archive_reference}."


def build_dataset(*, seed: int = 291_607, source_count: int = 9,
                  distractor_count: int = 24) -> list[dict]:
    """Build balanced rows with source-group-disjoint deterministic splits."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(source_count, int) or isinstance(source_count, bool) or source_count < len(SPLITS):
        raise ValueError(f"source_count must be at least {len(SPLITS)}")
    if not isinstance(distractor_count, int) or isinstance(distractor_count, bool) or distractor_count < 3:
        raise ValueError("distractor_count must be at least 3")

    assignments = list(range(source_count))
    random.Random(seed).shuffle(assignments)
    split_by_source = {source_index: SPLITS[rank % len(SPLITS)]
                       for rank, source_index in enumerate(assignments)}
    rows = []
    for source_index in range(source_count):
        scenario = SCENARIOS[source_index % len(SCENARIOS)]
        source_id = f"authored-lc-source-{source_index:04d}"
        group_id = f"authored-lc-group-{source_index:04d}"
        distractors = _distractors(source_index, distractor_count)
        for position_index, position in enumerate(POSITIONS):
            for outcome_index, outcome in enumerate(OUTCOMES):
                case_index = position_index * len(OUTCOMES) + outcome_index
                row_id = f"authored-lc-{source_index:04d}-{case_index:02d}"
                target_code = _stable("target", seed, source_id, position, outcome)[:12].upper()
                target = f"{scenario.subject_kind} T-{target_code}"
                evidence = _insert_record(
                    distractors,
                    _relevant_record(scenario, target, outcome),
                    position,
                )
                rows.append({
                    "id": row_id,
                    "task": "evidence",
                    "domain": scenario.domain,
                    "state": "\n".join(f"Record {index + 1}: {record}" for index, record in enumerate(evidence)),
                    "question": f"Using only the supplied records, assess this claim: {target} {scenario.claim}.",
                    "options": _ordered_options(seed, row_id),
                    "gold_option_id": derive_label(outcome),
                    "group_id": group_id,
                    "split": split_by_source[source_index],
                    "evaluation_slice": "authored_distant_evidence",
                    "evidence_position": position,
                    "distractor_count": distractor_count,
                    "authored": True,
                    "seed": seed,
                    "source": {
                        "dataset": DATASET_NAME,
                        "revision": DATASET_VERSION,
                        "source_id": source_id,
                        "license": "project-authored",
                    },
                    "provenance": {
                        "kind": "project-authored",
                        "generator": "experiments.long_context.authored_data",
                        "version": DATASET_VERSION,
                        "seed": seed,
                        "scenario": scenario.key,
                    },
                })
    return sorted(rows, key=lambda row: (row["split"], row["group_id"], row["id"]))


def write_dataset(output: Path, *, seed: int = 291_607, source_count: int = 9,
                  distractor_count: int = 24) -> dict:
    """Create split JSONL files and a compact manifest in a new directory."""
    rows = build_dataset(seed=seed, source_count=source_count, distractor_count=distractor_count)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split in SPLITS:
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows if row["split"] == split
        ).encode()
        path = output / f"authored-long-context-{split}.jsonl"
        path.write_bytes(payload)
        files[path.name] = {
            "rows": sum(row["split"] == split for row in rows),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest = {
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "authored": True,
        "seed": seed,
        "source_count": source_count,
        "distractor_count": distractor_count,
        "labels": dict(sorted(Counter(row["gold_option_id"] for row in rows).items())),
        "files": files,
        "limitations": [
            "Fictional templated records do not establish model quality on real documents.",
            "Character length is configurable, but token length depends on the selected tokenizer.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new directory for split JSONL files and manifest")
    parser.add_argument("--seed", type=int, default=291_607)
    parser.add_argument("--sources", type=int, default=9)
    parser.add_argument("--distractors", type=int, default=24)
    args = parser.parse_args(argv)
    manifest = write_dataset(args.output, seed=args.seed, source_count=args.sources,
                             distractor_count=args.distractors)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
