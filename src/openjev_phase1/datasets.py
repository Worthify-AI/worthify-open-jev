"""Pinned, deterministic converters for the Phase 1 training/evaluation data.

Third-party snapshots are inputs, never package data.  Conversion is create-only
and records the source and derived-file hashes in a manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
import urllib.request


CLINC_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
WANLI_REVISION = "61c95318fd71c55b6ba355d76253254615f387ec"
SOURCE_SNAPSHOTS = {
    "clinc150": {
        "filename": "clinc150-data_full.json",
        "url": f"https://raw.githubusercontent.com/clinc/oos-eval/{CLINC_REVISION}/data/data_full.json",
        "revision": CLINC_REVISION,
        "sha256": "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
        "license": "CC-BY-3.0",
        "attribution": "Larson et al., An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction",
    },
    "wanli": {
        "filename": "wanli-train.jsonl",
        "url": f"https://huggingface.co/datasets/alisawuffles/WANLI/resolve/{WANLI_REVISION}/train.jsonl",
        "revision": WANLI_REVISION,
        "sha256": "85058cf017a911e89242dc29fa0a4ddaad3664cb923dc0a82145fdda14b694e5",
        "license": "CC-BY-4.0",
        "attribution": "Liu et al., WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation",
    },
}
WANLI_UPSTREAM_TEST = {
    "url": f"https://huggingface.co/datasets/alisawuffles/WANLI/resolve/{WANLI_REVISION}/test.jsonl",
    "sha256": "4276e0af7fcdf657d1ab7beb54eaf025fda592a76c9ee86b63b7871953fc74fd",
    "use": "excluded; external evaluation selection is frozen separately",
}
NLI_OPTIONS = {
    "entailment": ("supported", "The evidence establishes the claim"),
    "neutral": ("insufficient", "The evidence does not establish either the claim or its opposite"),
    "contradiction": ("contradicted", "The evidence establishes the opposite of the claim"),
}
OPTION_COUNTS = (2, 4, 8, 16)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable(*parts: object) -> str:
    return _sha256("\x1f".join(map(str, parts)).encode())


def normalize_text(text: str) -> str:
    """Normalization used only for leakage grouping, never for model input."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w]+", " ", text).strip()


def _read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    return rows


def _option(intent: str) -> dict:
    return {"id": intent, "description": intent.replace("_", " ")}


def fetch_sources(output: Path, names: tuple[str, ...] = ("clinc150", "wanli")) -> dict[str, Path]:
    """Fetch verified training snapshots into a new directory."""
    unknown = set(names) - SOURCE_SNAPSHOTS.keys()
    if unknown:
        raise ValueError(f"Unknown sources: {sorted(unknown)}")
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    output.mkdir(parents=True)
    paths = {}
    for name in names:
        spec = SOURCE_SNAPSHOTS[name]
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "openjev-dataset-fetch/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(128 * 1024 * 1024 + 1)
        if len(data) > 128 * 1024 * 1024:
            raise ValueError(f"{name} exceeds the 128 MiB source limit")
        actual = _sha256(data)
        if actual != spec["sha256"]:
            raise ValueError(f"{name} changed: expected {spec['sha256']}, received {actual}")
        destination = output / spec["filename"]
        destination.write_bytes(data)
        paths[name] = destination
    return paths


def _verified_source(path: Path, name: str) -> None:
    actual = _sha256(path.read_bytes())
    expected = SOURCE_SNAPSHOTS[name]["sha256"]
    if actual != expected:
        raise ValueError(f"{name} snapshot hash mismatch: expected {expected}, received {actual}")


def convert_clinc150(source: Path, *, seed: int = 291607) -> list[dict]:
    """Convert CLINC150 with label-held-out validation/test partitions.

    Official test records always remain test records.  Fifteen deterministic
    intent labels are absent from training for each held-out split.
    """
    payload = json.loads(source.read_text())
    required = {"train", "val", "test", "oos_train", "oos_val", "oos_test"}
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise ValueError(f"CLINC source lacks splits: {sorted(required - set(payload or {}))}")
    labels = sorted({label for key in required for _, label in payload[key] if label != "oos"})
    if len(labels) != 150:
        raise ValueError(f"Expected 150 CLINC intents, found {len(labels)}")
    ordered = sorted(labels, key=lambda label: _stable("clinc-label", seed, label))
    unseen_validation, unseen_test = set(ordered[:15]), set(ordered[15:30])
    candidates = []
    split_map = {"train": "train", "val": "validation", "test": "test",
                 "oos_train": "train", "oos_val": "validation", "oos_test": "test"}
    for official_split in sorted(required):
        for source_index, pair in enumerate(payload[official_split]):
            if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(x, str) for x in pair):
                raise ValueError(f"Malformed CLINC row in {official_split} at {source_index}")
            text, label = pair
            split = split_map[official_split]
            if label in unseen_validation:
                split = "validation"
            elif label in unseen_test:
                split = "test"
            # The official test split is immutable and can never be promoted to train/validation.
            if official_split in {"test", "oos_test"}:
                split = "test"
            normalized = normalize_text(text)
            candidates.append({"text": text, "label": label, "official_split": official_split,
                               "source_index": source_index, "split": split, "normalized": normalized})

    # Exact/near-identical normalized strings form one leakage unit.  If upstream
    # splits disagree, use the most restrictive destination for the whole unit.
    rank = {"train": 0, "validation": 1, "test": 2}
    by_text = defaultdict(list)
    for item in candidates:
        by_text[item["normalized"]].append(item)
    for items in by_text.values():
        strictest = max((item["split"] for item in items), key=rank.__getitem__)
        for item in items:
            item["split"] = strictest

    rows = []
    for item in candidates:
        row_key = _stable("clinc150", item["official_split"], item["source_index"], item["text"], item["label"])
        count = OPTION_COUNTS[int(row_key[:8], 16) % len(OPTION_COUNTS)]
        visible_labels = (set(labels) - unseen_validation - unseen_test if item["split"] == "train"
                          else set(labels) - unseen_test if item["split"] == "validation"
                          else set(labels))
        if item["label"] == "oos":
            distractors = sorted(visible_labels, key=lambda label: _stable("clinc-options", seed, row_key, label))[: count - 1]
            option_ids = distractors + ["none_of_above"]
            gold = "none_of_above"
        else:
            distractors = sorted((label for label in visible_labels if label != item["label"]),
                                 key=lambda label: _stable("clinc-options", seed, row_key, label))[: count - 1]
            option_ids = [item["label"], *distractors]
            gold = item["label"]
        option_ids.sort(key=lambda value: _stable("clinc-order", seed, row_key, value))
        options = [{"id": "none_of_above", "description": "None of the listed intents"}
                   if value == "none_of_above" else _option(value) for value in option_ids]
        group_id = "clinc-text-" + _stable(item["normalized"])[:20]
        evaluation_slice = ("unseen_label_test" if item["label"] in unseen_test
                            else "unseen_label_validation" if item["label"] in unseen_validation
                            else "out_of_scope" if item["label"] == "oos" else "seen_label")
        rows.append({
            "id": "clinc-" + row_key[:24], "task": "classification", "state": item["text"],
            "question": "Which user intent best matches the request?", "options": options,
            "gold_option_id": gold, "group_id": group_id, "split": item["split"],
            "evaluation_slice": evaluation_slice, "source_text_untrusted": True,
            "source": {"dataset": "CLINC150", "revision": CLINC_REVISION,
                       "official_split": item["official_split"], "source_index": item["source_index"],
                       "license": "CC-BY-3.0"},
        })
    _validate_rows(rows)
    return sorted(rows, key=lambda row: (row["split"], row["id"]))


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _wanli_exclusions(path: Path | None) -> tuple[set[str], str | None]:
    if path is None:
        return set(), None
    tokens = set()
    for row in _read_jsonl(path):
        if row.get("source") != "wanli":
            continue
        tokens.add(str(row.get("group_id", "")))
        upstream = row.get("upstream", {})
        tokens.update(str(upstream.get(key, "")) for key in ("source_id", "seed_id", "pairID"))
    tokens.discard("")
    return tokens, _sha256(path.read_bytes())


def _wanli_related_exclusions(selection: Path | None, test_source: Path | None) -> tuple[set[str], set[str]]:
    """Resolve frozen external-test selections to pair and premise leakage keys."""
    if selection is None:
        return set(), set()
    if test_source is None:
        raise ValueError("WANLI selection exclusions require the pinned upstream test snapshot")
    actual = _sha256(test_source.read_bytes())
    if actual != WANLI_UPSTREAM_TEST["sha256"]:
        raise ValueError(f"WANLI test snapshot hash mismatch: {actual}")
    selected_ids = {
        str(row.get("upstream", {}).get("source_id"))
        for row in _read_jsonl(selection) if row.get("source") == "wanli"
    }
    test_rows = {str(row["id"]): row for row in _read_jsonl(test_source)}
    missing = selected_ids - test_rows.keys()
    if missing:
        raise ValueError(f"WANLI exclusion IDs absent from test snapshot: {sorted(missing)[:5]}")
    return ({str(test_rows[row_id]["pairID"]) for row_id in selected_ids},
            {normalize_text(test_rows[row_id]["premise"]) for row_id in selected_ids})


def convert_wanli(source: Path, *, exclusion_manifest: Path | None = None,
                  exclusion_test_source: Path | None = None, seed: int = 291607,
                  train_cap: int = 20_000, validation_cap: int = 1_500,
                  test_cap: int = 1_500) -> list[dict]:
    """Convert the WANLI train snapshot into group-disjoint internal splits."""
    raw = _read_jsonl(source)
    exclusions, _ = _wanli_exclusions(exclusion_manifest)
    excluded_pairs, excluded_premises = _wanli_related_exclusions(exclusion_manifest, exclusion_test_source)
    required = {"id", "premise", "hypothesis", "gold", "pairID"}
    filtered = []
    for row in raw:
        if not required <= row.keys() or row["gold"] not in NLI_OPTIONS:
            raise ValueError(f"Malformed WANLI row {row.get('id')}")
        filtered.append(row)

    uf = _UnionFind(len(filtered))
    seen_pair, seen_premise = {}, {}
    for index, row in enumerate(filtered):
        for lookup, key in ((seen_pair, str(row["pairID"])),
                            (seen_premise, normalize_text(row["premise"]))):
            if key in lookup:
                uf.union(index, lookup[key])
            else:
                lookup[key] = index
    components = defaultdict(list)
    for index, row in enumerate(filtered):
        components[uf.find(index)].append(row)

    components = {
        root: component for root, component in components.items()
        if not any(str(row["id"]) in exclusions or str(row["pairID"]) in exclusions
                   or str(row["pairID"]) in excluded_pairs
                   or normalize_text(row["premise"]) in excluded_premises for row in component)
    }

    caps = {"train": train_cap, "validation": validation_cap, "test": test_cap}
    selected = []
    remaining = [
        (min(_stable("wanli-component", row["id"], row["pairID"]) for row in component), component)
        for component in components.values()
    ]
    # Reserve balanced held-out populations first.  Training consumes the
    # remaining components, so no source group can be split to hit a quota.
    for split in ("validation", "test", "train"):
        # Whole components only. Greedy balancing keeps each label close while
        # respecting the hard total cap.
        label_cap = max(1, caps[split] // len(NLI_OPTIONS))
        counts = Counter()
        total = 0
        used = set()
        for key, component in sorted(remaining, key=lambda pair: _stable("wanli-select", seed, split, pair[0])):
            addition = Counter(row["gold"] for row in component)
            if total + len(component) > caps[split] or any(counts[label] + amount > label_cap for label, amount in addition.items()):
                continue
            group_id = "wanli-group-" + key[:20]
            for row in component:
                selected.append((split, group_id, row))
            counts.update(addition)
            total += len(component)
            used.add(key)
        remaining = [(key, component) for key, component in remaining if key not in used]

    rows = []
    option_ids = [value[0] for value in NLI_OPTIONS.values()]
    for split, group_id, item in selected:
        row_key = _stable("wanli", item["id"], item["pairID"])
        ordered = sorted(option_ids, key=lambda value: _stable("wanli-order", seed, row_key, value))
        gold_id = NLI_OPTIONS[item["gold"]][0]
        descriptions = {option_id: description for option_id, description in NLI_OPTIONS.values()}
        rows.append({
            "id": "wanli-" + row_key[:24], "task": "evidence", "state": item["premise"],
            "question": "Assess this claim using only the evidence: " + item["hypothesis"],
            "options": [{"id": value, "description": descriptions[value]} for value in ordered],
            "gold_option_id": gold_id, "group_id": group_id, "split": split,
            "evaluation_slice": "internal_group_holdout" if split != "train" else "training",
            "source_text_untrusted": True,
            "source": {"dataset": "WANLI", "revision": WANLI_REVISION, "official_split": "train",
                       "source_id": item["id"], "pair_id": item["pairID"], "license": "CC-BY-4.0"},
        })
    _validate_rows(rows)
    return sorted(rows, key=lambda row: (row["split"], row["id"]))


def _validate_rows(rows: list[dict]) -> None:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Converted row IDs are not unique")
    group_splits = defaultdict(set)
    normalized_splits = defaultdict(set)
    for row in rows:
        option_ids = [option["id"] for option in row["options"]]
        if not 2 <= len(option_ids) <= 16 or len(option_ids) != len(set(option_ids)):
            raise ValueError(f"Invalid options for {row['id']}")
        if row["gold_option_id"] not in option_ids:
            raise ValueError(f"Gold option absent for {row['id']}")
        if not isinstance(row.get("state"), str) or not row["state"].strip() or len(row["state"]) > 20_000:
            raise ValueError(f"State for {row['id']} must be nonempty and at most 20,000 characters")
        if not isinstance(row.get("question"), str) or not row["question"].strip() or len(row["question"]) > 20_000:
            raise ValueError(f"Question for {row['id']} must be nonempty and at most 20,000 characters")
        group_splits[row["group_id"]].add(row["split"])
        normalized_splits[normalize_text(str(row["state"]))].add(row["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("A source group crosses splits")
    if any(len(splits) != 1 for splits in normalized_splits.values()):
        raise ValueError("Normalized text leakage crosses splits")


def build_robustness_examples(rows: list[dict]) -> list[dict]:
    """Build bounded output-blind variants for project-owned examples only."""
    result = []
    for source in rows:
        owner = source.get("source", {}).get("dataset") or source.get("provenance", {}).get("source")
        if owner not in {"project-authored", "OpenJev", "openjev"}:
            continue
        for kind in ("option_order", "criterion_wrapper"):
            row = json.loads(json.dumps(source))
            row["id"] = "robust-" + _stable(source["id"], kind)[:24]
            row["split"] = "robustness"
            row["group_id"] = source["group_id"] + "/robustness"
            if kind == "option_order":
                row["options"].reverse()
            else:
                row["question"] = "Using only the supplied record, decide: " + row["question"]
            row["perturbation"] = {"kind": kind, "base_id": source["id"], "output_blind": True}
            result.append(row)
    _validate_rows(result)
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(payload)
    return _sha256(payload)


def prepare_datasets(clinc_source: Path, wanli_source: Path, output: Path, *,
                     wanli_exclusions: Path | None = None,
                     wanli_test_source: Path | None = None, seed: int = 291607,
                     verify_sources: bool = True) -> dict:
    """Create frozen split files and their audit manifest."""
    if output.exists():
        raise FileExistsError(f"Refusing to replace {output}")
    if verify_sources:
        _verified_source(clinc_source, "clinc150")
        _verified_source(wanli_source, "wanli")
    clinc = convert_clinc150(clinc_source, seed=seed)
    wanli = convert_wanli(wanli_source, exclusion_manifest=wanli_exclusions,
                          exclusion_test_source=wanli_test_source, seed=seed)
    output.mkdir(parents=True)
    files, counts = {}, {}
    for dataset, rows in (("clinc150", clinc), ("wanli", wanli)):
        counts[dataset] = Counter(row["split"] for row in rows)
        for split in ("train", "validation", "test"):
            selected = [row for row in rows if row["split"] == split]
            relative = f"{dataset}-{split}.jsonl"
            files[relative] = {"sha256": _write_jsonl(output / relative, selected), "rows": len(selected)}
    for slice_name, predicate in (
        ("unseen-label", lambda row: row.get("evaluation_slice") == "unseen_label_test"),
        ("seen-label", lambda row: row.get("evaluation_slice") == "seen_label"),
    ):
        selected = [row for row in clinc if row["split"] == "test" and predicate(row)]
        relative = f"clinc150-test-{slice_name}.jsonl"
        files[relative] = {"sha256": _write_jsonl(output / relative, selected), "rows": len(selected)}
    _, exclusion_hash = _wanli_exclusions(wanli_exclusions)
    manifest = {
        "version": "openjev-data-v1", "seed": seed, "create_only": True,
        "sources": {name: {**spec, "local_sha256": _sha256(path.read_bytes())}
                    for name, spec, path in (("clinc150", SOURCE_SNAPSHOTS["clinc150"], clinc_source),
                                             ("wanli", SOURCE_SNAPSHOTS["wanli"], wanli_source))},
        "wanli_upstream_test": WANLI_UPSTREAM_TEST,
        "wanli_external_selection_manifest_sha256": exclusion_hash,
        "wanli_external_test_source_sha256": _sha256(wanli_test_source.read_bytes()) if wanli_test_source else None,
        "splitting": {"group_disjoint": True, "normalized_text_disjoint": True,
                      "clinc_unseen_validation_labels": 15, "clinc_unseen_test_labels": 15,
                      "wanli_official_test_excluded": True, "max_train_rows_per_dataset": 20_000},
        "counts": {name: dict(value) for name, value in counts.items()}, "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="fetch pinned training snapshots")
    fetch.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="convert snapshots into frozen OpenJev rows")
    prepare.add_argument("--clinc-source", type=Path, required=True)
    prepare.add_argument("--wanli-source", type=Path, required=True)
    prepare.add_argument("--wanli-exclusions", type=Path)
    prepare.add_argument("--wanli-test-source", type=Path,
                         help="pinned upstream test snapshot from benchmarks/fetch_sources.py")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=291607)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        paths = fetch_sources(args.output)
        print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    else:
        manifest = prepare_datasets(args.clinc_source, args.wanli_source, args.output,
                                    wanli_exclusions=args.wanli_exclusions,
                                    wanli_test_source=args.wanli_test_source, seed=args.seed)
        print(json.dumps({"output": str(args.output), "counts": manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
