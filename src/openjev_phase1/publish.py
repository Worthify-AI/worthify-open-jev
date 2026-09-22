"""Validate and publish small OpenJev adapter releases.

This module intentionally accepts only adapter artefacts.  It is not a model
export tool and will refuse base-model weights, datasets, prediction records,
and credential-like files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable


SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_FILES = frozenset(
    {
        "adapter_model.safetensors",
        "adapter_config.json",
        "README.md",
        "LICENSE",
        "release-manifest.json",
        "training-manifest.json",
        "evaluation-manifest.json",
        "attribution.json",
        "NOTICE",
        "SHA256SUMS",
    }
)
SEED_RESULT_FILE = re.compile(r"^seed-[0-9]+-benchmark-results\.json$")
SEED_RELOAD_REFERENCE_FILE = re.compile(r"^seed-[0-9]+-reload-reference\.json$")
SEED_RELOAD_RECEIPT_FILE = re.compile(r"^seed-[0-9]+-fresh-reload-verification\.json$")
REQUIRED_FILES = frozenset(
    {
        "adapter_model.safetensors",
        "adapter_config.json",
        "README.md",
        "LICENSE",
        "release-manifest.json",
        "training-manifest.json",
        "evaluation-manifest.json",
        "attribution.json",
        "SHA256SUMS",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not SHA256.fullmatch(parts[0]):
            raise ValueError("SHA256SUMS has an invalid line")
        name = parts[1].removeprefix("*")
        if Path(name).name != name or not _allowed_name(name) or name == "SHA256SUMS":
            raise ValueError(f"SHA256SUMS names a forbidden file: {name}")
        if name in checksums:
            raise ValueError(f"SHA256SUMS repeats {name}")
        checksums[name] = parts[0]
    return checksums


def _allowed_name(name: str) -> bool:
    """Keep uploads closed while allowing one small evidence file per seed."""
    return name in ALLOWED_FILES or any(pattern.fullmatch(name) for pattern in (
        SEED_RESULT_FILE, SEED_RELOAD_REFERENCE_FILE, SEED_RELOAD_RECEIPT_FILE,
    ))


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {field}")
    return value


def _provenance(manifest: dict[str, Any], name: str) -> tuple[str, str]:
    item = manifest.get(name)
    if not isinstance(item, dict):
        raise ValueError(f"release-manifest.json missing {name}")
    return (
        _required_string(item.get("path"), f"{name}.path"),
        _required_string(item.get("sha256"), f"{name}.sha256"),
    )


def _validate_evaluation(record: dict, base: dict, adapter_sha: str, seed: int, *, recipe: str) -> None:
    if record.get("schema") != "openjev-phase1-evaluation-v1" or record.get("provenance_verified") is not True:
        raise ValueError("Evaluation must have verified prediction provenance and the evaluation-v1 schema")
    if record.get("base_model") != {"id": base["id"], "revision": base["revision"]} or record.get("adapter_sha256") != adapter_sha:
        raise ValueError("Evaluation provenance does not match the actual training base and adapter")
    if record.get("seed") != seed:
        raise ValueError("Evaluation report seed must match its training run")
    inputs = record.get("inputs", {})
    if not isinstance(inputs, dict) or any(not isinstance(inputs.get(key), str) or not SHA256.fullmatch(inputs[key])
                                          for key in ("gold_sha256", "predictions_sha256")):
        raise ValueError("Evaluation must identify its gold and prediction input hashes")
    if record.get("split") != "test":
        raise ValueError("Measured releases require a held-out test evaluation")
    if type(record.get("n")) is not int or record["n"] < 1 or any(
        not isinstance(record.get(key), (int, float)) or not math.isfinite(record[key]) or not 0 <= record[key] <= 1
        for key in ("accuracy", "mean_task_macro_f1")
    ):
        raise ValueError("Evaluation lacks finite measured metrics")
    tasks = record.get("task_results")
    if not isinstance(tasks, dict) or set(tasks) != {recipe} or tasks[recipe].get("n") != record["n"]:
        raise ValueError("Evaluation tasks do not match the release recipe")


def _validate_training_recipe(training: dict, recipe: str) -> None:
    declared = training.get("training_spec", {}).get("recipe")
    dataset = {"classification": "CLINC150", "evidence": "WANLI"}[recipe]
    if not isinstance(declared, dict) or declared.get("task") != recipe or declared.get("dataset") != dataset:
        raise ValueError("Training recipe does not match the requested release recipe")
    if declared.get("version") != "openjev-training-recipe-v1" or not isinstance(declared.get("digest"), str) or not SHA256.fullmatch(declared["digest"]):
        raise ValueError("Training recipe must preserve its version and content digest")


def _seeded_filename(pattern: re.Pattern[str], path: str, seed: int, field: str) -> None:
    if not pattern.fullmatch(path) or int(path.split("-", 2)[1]) != seed:
        raise ValueError(f"{field} must be an allowed file for seed {seed}")


def _validate_reload_reference(reference: dict[str, Any]) -> None:
    if set(reference) != {"schema", "max_tokens", "tolerance", "rows"} or reference.get("schema") != "openjev-phase1-adapter-reload-reference-v1":
        raise ValueError("Reload reference has an unsupported schema")
    if type(reference.get("max_tokens")) is not int or reference["max_tokens"] < 1 or reference.get("tolerance") != 1e-4:
        raise ValueError("Reload reference has invalid verification settings")
    rows = reference.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 8:
        raise ValueError("Reload reference must contain one to eight rows")
    row_ids: set[str] = set()
    for row in rows:
        if set(row) != {"id", "option_ids", "logits"} or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Reload reference must contain only row and option IDs with logits")
        if row["id"] in row_ids:
            raise ValueError("Reload reference row IDs must be unique")
        row_ids.add(row["id"])
        option_ids, logits = row.get("option_ids"), row.get("logits")
        if not isinstance(option_ids, list) or not 2 <= len(option_ids) <= 16 or len(set(option_ids)) != len(option_ids) or not all(isinstance(value, str) and value for value in option_ids):
            raise ValueError("Reload reference option IDs are invalid")
        if not isinstance(logits, list) or len(logits) != len(option_ids) or not all(type(value) in (int, float) and math.isfinite(value) for value in logits):
            raise ValueError("Reload reference logits are invalid")


def _validate_reload_receipt(receipt: dict[str, Any], reference_sha: str, *, seed: int, adapter_sha: str,
                             base: dict[str, Any], validation_sha: str, reference: dict[str, Any]) -> None:
    if receipt.get("schema") != "openjev-phase1-fresh-adapter-reload-v1" or receipt.get("passed") is not True:
        raise ValueError("Fresh reload receipt must report a passing verification")
    if receipt.get("seed") != seed or receipt.get("adapter_sha256") != adapter_sha:
        raise ValueError("Fresh reload receipt does not match its seed and adapter")
    if receipt.get("base_model") != {"source": base["id"], "revision": base["revision"]}:
        raise ValueError("Fresh reload receipt does not match the pinned base")
    if receipt.get("validation_sha256") != validation_sha or receipt.get("reference_sha256") != reference_sha:
        raise ValueError("Fresh reload receipt does not match its validation input and reference")
    if receipt.get("examples") != len(reference["rows"]) or receipt.get("tolerance") != reference["tolerance"]:
        raise ValueError("Fresh reload receipt does not match its reference settings")
    error = receipt.get("max_abs_logit_error")
    if type(error) not in (int, float) or not math.isfinite(error) or error < 0 or error > receipt["tolerance"]:
        raise ValueError("Fresh reload receipt has an invalid logit error")


def validate_release(artifact_dir: Path) -> dict[str, str]:
    """Validate a self-contained adapter release and return its file hashes."""
    artifact_dir = artifact_dir.resolve()
    if not artifact_dir.is_dir():
        raise ValueError(f"Artifact directory does not exist: {artifact_dir}")
    entries = list(artifact_dir.iterdir())
    if any(item.is_symlink() for item in entries):
        raise ValueError("Release content may not contain symlinks")
    if any(not item.is_file() and not item.is_dir() for item in entries):
        raise ValueError("Release content must contain regular files only")
    files = {item.name: item for item in entries if item.is_file()}
    nested = [item for item in entries if item.is_dir()]
    forbidden = sorted(name for name in files if not _allowed_name(name))
    if nested or forbidden:
        names = [item.name for item in nested] + forbidden
        raise ValueError(f"Forbidden release content: {', '.join(names)}")
    missing = sorted(REQUIRED_FILES - set(files))
    if missing:
        raise ValueError(f"Release is missing required files: {', '.join(missing)}")

    hashes = {name: sha256_file(path) for name, path in files.items()}
    checksums = _checksums(files["SHA256SUMS"])
    required_hashed = set(files) - {"SHA256SUMS"}
    if set(checksums) != required_hashed:
        raise ValueError("SHA256SUMS must cover every release file except itself")
    for name, expected in checksums.items():
        if hashes[name] != expected:
            raise ValueError(f"Checksum mismatch for {name}")

    if "Apache License" not in files["LICENSE"].read_text(encoding="utf-8", errors="replace"):
        raise ValueError("Adapter releases must include the Apache-2.0 license text")
    card = files["README.md"].read_text(encoding="utf-8", errors="replace")
    if "license: apache-2.0" not in card.lower():
        raise ValueError("README.md model card must declare license: apache-2.0")
    if "pipeline_tag" in card.lower():
        raise ValueError("Model cards must not declare a pipeline_tag for adapters")
    attribution = _load_json(files["attribution.json"])
    if attribution.get("schema") != "openjev-phase1-attribution-v1" or not isinstance(attribution.get("sources"), list):
        raise ValueError("Adapter package requires a structured attribution record")
    if not all(isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("license"), str)
               and isinstance(item.get("url"), str) and isinstance(item.get("revision"), str)
               and isinstance(item.get("modifications"), str) for item in attribution["sources"]):
        raise ValueError("Attribution sources need name, license, URL, revision, and modifications")

    manifest = _load_json(files["release-manifest.json"])
    if manifest.get("schema") != "openjev-phase1-adapter-release-v1" or manifest.get("kind") not in {"measured", "smoke"}:
        raise ValueError("Unsupported release manifest schema")
    if manifest.get("recipe") not in {"classification", "evidence"}:
        raise ValueError("Release recipe must be classification or evidence")
    selected_seed = manifest.get("selected_seed")
    if selected_seed not in (42, 43):
        raise ValueError("release-manifest.json needs selected_seed 42 or 43")
    base = manifest.get("base_model")
    if not isinstance(base, dict):
        raise ValueError("release-manifest.json missing base_model")
    base_id = _required_string(base.get("id"), "base_model.id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", base_id):
        raise ValueError("Released base identity must be a Hub model ID, not a local path")
    base_revision = _required_string(base.get("revision"), "base_model.revision")
    if not re.fullmatch(r"[0-9a-f]{40}", base_revision):
        raise ValueError("base_model.revision must be an immutable 40-character Hugging Face revision")
    adapter_path, adapter_sha = _provenance(manifest, "adapter")
    if adapter_path != "adapter_model.safetensors" or adapter_sha != hashes[adapter_path]:
        raise ValueError("Release manifest does not link the adapter hash")
    adapter_config = _load_json(files["adapter_config.json"])
    if adapter_config.get("base_model_name_or_path") != base_id or adapter_config.get("peft_type") != "LORA":
        raise ValueError("Adapter config must identify the same base model and LoRA type")
    for field, expected_name in (("training", "training-manifest.json"), ("evaluation", "evaluation-manifest.json")):
        path, digest = _provenance(manifest, field)
        if path != expected_name or digest != hashes[path]:
            raise ValueError(f"Release manifest does not link the {field} manifest")
        record = _load_json(files[path])
        linked = record.get("base_model")
        if not isinstance(linked, dict) or (linked.get("id"), linked.get("revision")) != (base_id, base_revision):
            raise ValueError(f"{path} does not link the same pinned base model")
        if record.get("adapter_sha256") != adapter_sha:
            raise ValueError(f"{path} does not link the adapter hash")
        if manifest["kind"] == "measured" and field == "evaluation":
            _validate_evaluation(record, base, adapter_sha, selected_seed, recipe=manifest["recipe"])
        if manifest["kind"] == "measured" and field == "training":
            _validate_training_recipe(record, manifest["recipe"])
    benchmarks = manifest.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("release-manifest.json must link one or more per-seed benchmark results")
    seen_benchmarks: set[str] = set()
    report_seeds: set[int] = set()
    selected_linked = False
    for item in benchmarks:
        if not isinstance(item, dict):
            raise ValueError("Each benchmark entry must be an object")
        path = _required_string(item.get("path"), "benchmarks.path")
        digest = _required_string(item.get("sha256"), "benchmarks.sha256")
        if not SEED_RESULT_FILE.fullmatch(path) or path in seen_benchmarks or digest != hashes.get(path):
            raise ValueError("Release manifest does not link an allowed per-seed benchmark result")
        seen_benchmarks.add(path)
        record = _load_json(files[path])
        if manifest["kind"] == "measured" and (not isinstance(record.get("n"), int) or record["n"] < 1 or not isinstance(record.get("accuracy"), (int, float))):
            raise ValueError(f"{path} lacks measured metrics")
        seed = int(path.split("-", 2)[1])
        if record.get("seed") != seed or record.get("kind") != manifest["kind"]:
            raise ValueError(f"{path} seed does not match its filename")
        linked = record.get("base_model")
        if not isinstance(linked, dict) or (linked.get("id"), linked.get("revision")) != (base_id, base_revision):
            raise ValueError(f"{path} does not link the same pinned base model")
        report_adapter_sha = record.get("adapter_sha256")
        if not isinstance(report_adapter_sha, str) or not SHA256.fullmatch(report_adapter_sha):
            raise ValueError(f"{path} must identify its adapter hash")
        if manifest["kind"] == "measured":
            _validate_evaluation(record, base, report_adapter_sha, seed, recipe=manifest["recipe"])
        if seed == selected_seed:
            selected_linked = report_adapter_sha == adapter_sha
        report_seeds.add(seed)
    if not selected_linked:
        raise ValueError("The selected seed report must link the exported adapter hash")
    if manifest["kind"] == "measured" and report_seeds != {42, 43}:
        raise ValueError("Measured releases require results for seeds 42 and 43")
    proofs = manifest.get("fresh_reload_proofs")
    if manifest["kind"] == "measured" and (not isinstance(proofs, list) or len(proofs) != 2):
        raise ValueError("Measured releases require fresh reload proofs for seeds 42 and 43")
    proof_seeds: set[int] = set()
    for proof in proofs or []:
        if not isinstance(proof, dict):
            raise ValueError("Each fresh reload proof must be an object")
        seed = proof.get("seed")
        if seed not in report_seeds or seed in proof_seeds:
            raise ValueError("Fresh reload proofs must cover each benchmark seed once")
        reference_path, reference_sha = _provenance(proof, "reference")
        receipt_path, receipt_sha = _provenance(proof, "receipt")
        _seeded_filename(SEED_RELOAD_REFERENCE_FILE, reference_path, seed, "reference.path")
        _seeded_filename(SEED_RELOAD_RECEIPT_FILE, receipt_path, seed, "receipt.path")
        if hashes.get(reference_path) != reference_sha or hashes.get(receipt_path) != receipt_sha:
            raise ValueError("Fresh reload proof does not link its evidence files")
        validation_sha = proof.get("validation_sha256")
        if not isinstance(validation_sha, str) or not SHA256.fullmatch(validation_sha):
            raise ValueError("Fresh reload proof must identify its validation input")
        reference = _load_json(files[reference_path])
        _validate_reload_reference(reference)
        receipt = _load_json(files[receipt_path])
        report = _load_json(files[f"seed-{seed}-benchmark-results.json"])
        _validate_reload_receipt(receipt, reference_sha, seed=seed, adapter_sha=report["adapter_sha256"],
                                 base=base, validation_sha=validation_sha, reference=reference)
        if seed == selected_seed and validation_sha != _load_json(files["training-manifest.json"])["training_spec"]["validation_sha256"]:
            raise ValueError("Selected fresh reload proof does not match the training validation input")
        if seed == selected_seed and _load_json(files["training-manifest.json"]).get("fresh_reload_reference") != {
            "path": reference_path, "sha256": reference_sha
        }:
            raise ValueError("Selected training manifest does not link its packaged fresh reload reference")
        proof_seeds.add(seed)
    if manifest["kind"] == "measured" and proof_seeds != {42, 43}:
        raise ValueError("Measured releases require fresh reload proofs for seeds 42 and 43")
    return hashes


def _hub_api(token: str | None = None) -> Any:
    from huggingface_hub import HfApi  # Imported lazily so validation has no network dependency.

    return HfApi(token=token if token is not None else os.environ.get("HF_TOKEN"))


def _repo_info(api: Any, repo_id: str) -> Any | None:
    try:
        return api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception as exc:
        if exc.__class__.__name__ == "RepositoryNotFoundError" or getattr(getattr(exc, "response", None), "status_code", None) == 404:
            return None
        raise


def _ensure_private(api: Any, repo_id: str) -> None:
    info = _repo_info(api, repo_id)
    if info is None:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=False)
    elif not bool(getattr(info, "private", False)):
        raise ValueError("Refusing to write to an existing public repository")


def _remote_names(api: Any, repo_id: str, revision: str | None = None) -> set[str]:
    return set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision))


def _assert_remote_exact(api: Any, repo_id: str, names: set[str], revision: str | None = None) -> None:
    # Hugging Face may create .gitattributes; no other untracked remote files are tolerated.
    found = _remote_names(api, repo_id, revision)
    if found - names - {".gitattributes"} or not names <= found:
        raise ValueError("Remote repository does not contain exactly the allowlisted release files")


def _commit_revision(commit: Any) -> str:
    revision = getattr(commit, "oid", None) or getattr(commit, "commit_oid", None)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Hub upload did not return an immutable commit revision")
    return revision


def upload_private(artifact_dir: Path, repo_id: str, *, api: Any | None = None) -> dict[str, Any]:
    """Create/update a private model repo, upload allowlisted files, and recheck them."""
    hashes = validate_release(artifact_dir)
    api = api or _hub_api()
    _ensure_private(api, repo_id)
    if _remote_names(api, repo_id) - set(hashes) - {".gitattributes"}:
        raise ValueError("Remote repository contains stale or forbidden files")
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(artifact_dir), allow_patterns=sorted(hashes))
    revision = _commit_revision(commit)
    verify_remote(repo_id, hashes, api=api, revision=revision)
    return {"repo_id": repo_id, "revision": revision, "hashes": hashes}


def smoke_upload(artifact_dir: Path, repo_id: str, *, api: Any | None = None) -> dict[str, Any]:
    """Private-only transport check for an inert checksummed payload, never a model claim."""
    allowed = {"README.md", "smoke-payload.json", "SHA256SUMS"}
    entries = list(artifact_dir.iterdir())
    if {entry.name for entry in entries} != allowed or any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise ValueError("Smoke upload accepts only README.md, smoke-payload.json, and SHA256SUMS")
    hashes = {entry.name: sha256_file(entry) for entry in entries}
    lines = (artifact_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    expected = {name: digest for name, digest in hashes.items() if name != "SHA256SUMS"}
    parsed = {}
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not SHA256.fullmatch(parts[0]):
            raise ValueError("Smoke upload has a malformed checksum line")
        name = parts[1].removeprefix("*")
        if name in parsed or name not in expected:
            raise ValueError("Smoke upload has duplicate or forbidden checksum entries")
        parsed[name] = parts[0]
    if parsed != expected:
        raise ValueError("Smoke upload checksum mismatch")
    api = api or _hub_api()
    _ensure_private(api, repo_id)
    if _remote_names(api, repo_id) - set(hashes) - {".gitattributes"}:
        raise ValueError("Remote repository contains stale or forbidden files")
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(artifact_dir), allow_patterns=sorted(hashes))
    revision = _commit_revision(commit)
    verify_remote(repo_id, hashes, api=api, revision=revision)
    return {"repo_id": repo_id, "revision": revision, "hashes": hashes}


def verify_remote(repo_id: str, hashes: dict[str, str], *, api: Any | None = None, revision: str | None = None) -> None:
    """Download each uploaded file through the Hub API and compare its digest."""
    api = api or _hub_api()
    _assert_remote_exact(api, repo_id, set(hashes), revision)
    for name, expected in hashes.items():
        if hasattr(api, "hf_hub_download"):
            downloaded_path = api.hf_hub_download(repo_id=repo_id, repo_type="model", filename=name, revision=revision)
        else:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(repo_id=repo_id, repo_type="model", filename=name, revision=revision, token=getattr(api, "token", None))
        downloaded = Path(downloaded_path)
        if sha256_file(downloaded) != expected:
            raise ValueError(f"Remote checksum mismatch for {name}")


def promote_public(artifact_dir: Path, private_repo_id: str, public_repo_id: str, *, release_public: bool,
                   private_revision: str, api: Any | None = None) -> dict[str, str]:
    """Recheck a private release, then publish the exact validated files on explicit request."""
    if not release_public:
        raise ValueError("Public promotion requires --release-public")
    if not re.fullmatch(r"[0-9a-f]{40}", private_revision):
        raise ValueError("Private candidate revision must be an immutable 40-character commit")
    if private_repo_id == public_repo_id:
        raise ValueError("Private and public repositories must differ")
    hashes = validate_release(artifact_dir)
    if _load_json(artifact_dir / "release-manifest.json").get("kind") != "measured":
        raise ValueError("Smoke or placeholder packages may not be promoted publicly")
    api = api or _hub_api()
    source_info = _repo_info(api, private_repo_id)
    if source_info is None or not bool(getattr(source_info, "private", False)):
        raise ValueError("Promotion source must be an existing private repository")
    verify_remote(private_repo_id, hashes, api=api, revision=private_revision)
    _ensure_private(api, public_repo_id)
    if _remote_names(api, public_repo_id) - set(hashes) - {".gitattributes"}:
        raise ValueError("Remote repository contains stale or forbidden files")
    commit = api.upload_folder(repo_id=public_repo_id, repo_type="model", folder_path=str(artifact_dir), allow_patterns=sorted(hashes))
    verify_remote(public_repo_id, hashes, api=api, revision=_commit_revision(commit))
    api.update_repo_settings(repo_id=public_repo_id, repo_type="model", private=False)
    return hashes


def publish_index(index_path: Path, work_dir: Path, *, release_public: bool, api: Any | None = None) -> None:
    """Download two pinned private candidates and promote only verified measured packages."""
    index = _load_json(index_path)
    candidates = index.get("candidates")
    if index.get("schema") != "openjev-phase1-release-index-v1" or index.get("state") != "verified" or not isinstance(candidates, list):
        raise ValueError("Release index is pending or invalid; no adapter may be published")
    if not release_public:
        raise ValueError("Public promotion requires --release-public")
    if len(candidates) != 2 or any(not isinstance(item, dict) for item in candidates):
        raise ValueError("Release index must contain two recipe candidates")
    recipes = [item.get("recipe", "") for item in candidates]
    if set(recipes) != {"classification", "evidence"}:
        raise ValueError("Release candidates must cover classification and evidence recipes")
    if any(item.get("seed") not in (42, 43) for item in candidates):
        raise ValueError("Each recipe must select seed 42 or 43")
    if len({item.get("public_repo_id") for item in candidates}) != 2:
        raise ValueError("Recipes must have distinct public repositories")
    api = api or _hub_api()
    work_dir.mkdir(parents=True, exist_ok=False)
    prepared = []
    base_models = []
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("Invalid release candidate")
        private_id = _required_string(item.get("private_repo_id"), "candidate.private_repo_id")
        public_id = _required_string(item.get("public_repo_id"), "candidate.public_repo_id")
        revision = _required_string(item.get("revision"), "candidate.revision")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Candidate revisions must be pinned 40-character Hub commits")
        destination = work_dir / item["recipe"]
        destination.mkdir()
        names = _remote_names(api, private_id, revision) - {".gitattributes"}
        if any(Path(name).name != name or not _allowed_name(name) for name in names):
            raise ValueError("Pinned candidate contains forbidden files")
        for name in sorted(names):
            if hasattr(api, "hf_hub_download"):
                downloaded = api.hf_hub_download(repo_id=private_id, repo_type="model", filename=name, revision=revision)
            else:
                from huggingface_hub import hf_hub_download
                downloaded = hf_hub_download(repo_id=private_id, repo_type="model", filename=name, revision=revision,
                                             token=getattr(api, "token", None))
            # Hub cache files may be symlinks. Copy their bytes into a fresh flat
            # bundle; never include local_dir's .cache metadata or cache symlinks.
            shutil.copyfile(downloaded, destination / name)
        manifest = _load_json(destination / "release-manifest.json")
        if manifest.get("kind") != "measured" or manifest.get("selected_seed") != item["seed"] or manifest.get("recipe") != item["recipe"]:
            raise ValueError("Candidate is not a measured selected-seed release")
        validate_release(destination)
        base_models.append(manifest["base_model"])
        prepared.append((destination, private_id, public_id, revision))
    if base_models[0] != base_models[1]:
        raise ValueError("Both recipes must use the same selected pinned base model")
    for destination, private_id, public_id, revision in prepared:
        promote_public(destination, private_id, public_id, release_public=release_public, private_revision=revision, api=api)


def package_release(run_dirs: list[Path], evaluations: list[Path], output: Path, selected_seed: int, *, recipe: str,
                    base_notice: Path | None = None) -> None:
    """Create a reproducible package from the two real training runs and eval reports."""
    if output.exists() or len(run_dirs) != 2 or len(evaluations) != 2 or selected_seed not in (42, 43):
        raise ValueError("Package output must be new and include seeds 42 and 43")
    if recipe not in {"classification", "evidence"}:
        raise ValueError("Recipe must be classification or evidence")
    records: dict[int, tuple[Path, dict[str, Any], dict[str, Any], str, dict[str, Path | str]]] = {}
    for run_dir, evaluation in zip(run_dirs, evaluations):
        training = _load_json(run_dir / "manifest.json")
        seed = training.get("seed")
        model = training.get("model")
        adapter = run_dir / "final-adapter" / "adapter_model.safetensors"
        if seed not in (42, 43) or not isinstance(model, dict) or not adapter.is_file() or seed in records:
            raise ValueError("Each run must be a distinct seed 42/43 training output with final adapter")
        base = {"id": _required_string(model.get("source"), "training model.source"), "revision": _required_string(model.get("revision"), "training model.revision")}
        if not re.fullmatch(r"[0-9a-f]{40}", base["revision"]):
            raise ValueError("Training model revision must be pinned SHA-40")
        report = _load_json(evaluation)
        adapter_sha = sha256_file(adapter)
        _validate_evaluation(report, base, adapter_sha, seed, recipe=recipe)
        _validate_training_recipe(training, recipe)
        spec = training.get("training_spec")
        if not isinstance(spec, dict) or spec.get("seed") != seed or (spec.get("model"), spec.get("revision")) != (base["id"], base["revision"]):
            raise ValueError("Training manifest must preserve its actual model/seed training specification")
        if any(not isinstance(spec.get(key), str) or not SHA256.fullmatch(spec[key]) for key in ("train_sha256", "validation_sha256")):
            raise ValueError("Training specification must identify train and validation input hashes")
        if report["inputs"]["gold_sha256"] in {spec["train_sha256"], spec["validation_sha256"]}:
            raise ValueError("Test evaluation input cannot be the training or validation input")
        reference_path = run_dir / "final-adapter" / "reload-reference.json"
        receipt_path = run_dir / "fresh-reload-verification.json"
        if not reference_path.is_file() or not receipt_path.is_file():
            raise ValueError("Each training run requires fresh reload reference and receipt")
        reference_sha = sha256_file(reference_path)
        fresh_reference = training.get("fresh_reload_reference")
        if fresh_reference != {"path": "final-adapter/reload-reference.json", "sha256": reference_sha}:
            raise ValueError("Training manifest does not link its fresh reload reference")
        reference = _load_json(reference_path)
        _validate_reload_reference(reference)
        if reference["max_tokens"] != training.get("max_tokens"):
            raise ValueError("Reload reference max tokens do not match training")
        receipt = _load_json(receipt_path)
        _validate_reload_receipt(receipt, reference_sha, seed=seed, adapter_sha=adapter_sha,
                                 base=base, validation_sha=spec["validation_sha256"], reference=reference)
        public_training = {key: value for key, value in training.items() if key != "validation_checkpoint"}
        records[seed] = (run_dir, {**public_training, "base_model": base, "adapter_sha256": adapter_sha}, {**report, "kind": "measured"}, adapter_sha,
                         {"reference": reference_path, "reference_sha256": reference_sha,
                          "receipt": receipt_path, "receipt_sha256": sha256_file(receipt_path),
                          "validation_sha256": spec["validation_sha256"]})
    if set(records) != {42, 43}:
        raise ValueError("Both seed 42 and seed 43 runs are required")
    if records[42][1]["base_model"] != records[43][1]["base_model"] or records[42][2]["inputs"]["gold_sha256"] != records[43][2]["inputs"]["gold_sha256"]:
        raise ValueError("Seed runs must share the same pinned base and evaluation input")
    specs = [{key: value for key, value in records[seed][1]["training_spec"].items() if key != "seed"} for seed in (42, 43)]
    if specs[0] != specs[1]:
        raise ValueError("Seed comparison requires the same training data and settings")
    if records[42][1].get("epochs") != records[43][1].get("epochs"):
        raise ValueError("Seed comparison requires the same training epoch budget")
    selected_f1 = records[selected_seed][1].get("best_validation_macro_f1")
    if any(not isinstance(item[1].get("best_validation_macro_f1"), (int, float)) or not math.isfinite(item[1]["best_validation_macro_f1"]) or not 0 <= item[1]["best_validation_macro_f1"] <= 1 for item in records.values()):
        raise ValueError("Training manifests must contain validation F1 for seed selection")
    if selected_f1 < max(item[1]["best_validation_macro_f1"] for item in records.values()):
        raise ValueError("selected_seed must be chosen by validation F1, not test results")
    output.mkdir(parents=True)
    selected = records[selected_seed]
    selected[1]["fresh_reload_reference"] = {
        "path": f"seed-{selected_seed}-reload-reference.json",
        "sha256": selected[4]["reference_sha256"],
    }
    shutil.copy2(selected[0] / "final-adapter" / "adapter_model.safetensors", output / "adapter_model.safetensors")
    shutil.copy2(selected[0] / "final-adapter" / "adapter_config.json", output / "adapter_config.json")
    for seed, (_, _, _, _, proof) in records.items():
        shutil.copy2(proof["reference"], output / f"seed-{seed}-reload-reference.json")
        shutil.copy2(proof["receipt"], output / f"seed-{seed}-fresh-reload-verification.json")
    (output / "LICENSE").write_text((Path(__file__).parents[2] / "LICENSES" / "Apache-2.0.txt").read_text(encoding="utf-8"), encoding="utf-8")
    if base_notice is not None:
        if base_notice.is_symlink() or not base_notice.is_file():
            raise ValueError("base NOTICE must be a regular file supplied from the base-model repository")
        shutil.copy2(base_notice, output / "NOTICE")
    data_sources = [
        {"name": "CLINC150", "license": "CC-BY-3.0", "url": "https://github.com/clinc/oos-eval", "revision": "828f8093932c8fe6ca7936c3d2e52903b1c523de", "modifications": "Converted into bounded-option OpenJev rows with governed split and leakage controls."},
        {"name": "WANLI", "license": "CC-BY-4.0", "url": "https://huggingface.co/datasets/alisawuffles/WANLI", "revision": "61c95318fd71c55b6ba355d76253254615f387ec", "modifications": "Converted from the pinned training snapshot into bounded-option evidence rows; governed external-test selections are excluded."},
        {"name": "OpenJev interface", "license": "MIT", "url": "https://github.com/bonsai/openjev", "revision": "53e3028363509f8533d90fe82d983770da1f6c02", "modifications": "Worthify fine-tuned a LoRA adapter for this interface; this does not disclose or reproduce Jev training data."},
    ]
    (output / "attribution.json").write_text(json.dumps({"schema": "openjev-phase1-attribution-v1", "recipe": recipe, "sources": data_sources}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_lines = []
    for seed in (42, 43):
        report = records[seed][2]
        report_lines.append(f"| {seed} | {report['accuracy']:.4f} | {report['mean_task_macro_f1']:.4f} | {report.get('task_results', {})} | {report.get('latency', {})} | {report.get('peak_memory_bytes', 'not reported')} |")
    (output / "README.md").write_text(
        f"---\nlicense: apache-2.0\nbase_model: {selected[1]['base_model']['id']}\n---\n"
        f"# OpenJev {recipe} LoRA adapter\n\nThis Worthify adapter fine-tunes the OpenJev interface; it is not Jev and does not claim access to Jev's undisclosed training data. "
        f"Base: `{selected[1]['base_model']['id']}` at `{selected[1]['base_model']['revision']}`. Seed {selected_seed} was selected by validation macro F1.\n\n"
        f"## Recipe and scope\n\nThe `{recipe}` recipe uses supplied decisions with 2–16 declared options. It is a candidate {recipe} adapter, not a 150-way intent classifier. "
        "The adapter package excludes base weights. Inspect the base-model repository for its current license and NOTICE; a supplied upstream NOTICE is copied here unchanged.\n\n"
        "## Measured held-out reports\n\n| seed | accuracy | mean task macro F1 | calibration details | latency details | peak memory bytes |\n| --- | ---: | ---: | --- | --- | --- |\n"
        + "\n".join(report_lines) + "\n\nScores are uncalibrated conditional option scores; Brier/ECE and reliability details remain in the linked per-seed reports. "
        "Expected reference environment: one NVIDIA A100 with NF4 quantization. This is an environment expectation, not a measured performance claim.\n\n"
        "## Usage\n\n```bash\nopenjev-score --mode direct --model " + selected[1]['base_model']['id'] + " --revision " + selected[1]['base_model']['revision']
        + " --adapter worthify/REPLACE_WITH_REPOSITORY --adapter-revision REPLACE_WITH_40_CHARACTER_HUB_COMMIT --input decisions.jsonl --output predictions.jsonl --quantization nf4\n```\n\n"
        "Replace the adapter repository and revision placeholder with the published repository and immutable 40-character Hub commit. `attribution.json` records CLINC150 (CC-BY-3.0), WANLI (CC-BY-4.0), and interface attribution, including modifications.\n",
        encoding="utf-8")
    for seed, (_, training, report, _, _) in records.items():
        if seed == selected_seed:
            (output / "training-manifest.json").write_text(json.dumps(training, indent=2, sort_keys=True) + "\n")
            (output / "evaluation-manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (output / f"seed-{seed}-benchmark-results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    base = selected[1]["base_model"]
    manifest = {"schema": "openjev-phase1-adapter-release-v1", "kind": "measured", "recipe": recipe, "selected_seed": selected_seed, "base_model": base,
                "adapter": {"path": "adapter_model.safetensors", "sha256": selected[3]},
                "training": {"path": "training-manifest.json", "sha256": sha256_file(output / "training-manifest.json")},
                "evaluation": {"path": "evaluation-manifest.json", "sha256": sha256_file(output / "evaluation-manifest.json")},
                "benchmarks": [{"path": f"seed-{seed}-benchmark-results.json", "sha256": sha256_file(output / f"seed-{seed}-benchmark-results.json")} for seed in (42, 43)],
                "fresh_reload_proofs": [{"seed": seed,
                                         "reference": {"path": f"seed-{seed}-reload-reference.json", "sha256": records[seed][4]["reference_sha256"]},
                                         "receipt": {"path": f"seed-{seed}-fresh-reload-verification.json", "sha256": records[seed][4]["receipt_sha256"]},
                                         "validation_sha256": records[seed][4]["validation_sha256"]} for seed in (42, 43)]}
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    names = sorted(item.name for item in output.iterdir() if item.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("".join(f"{sha256_file(output / name)}  {name}\n" for name in names), encoding="utf-8")
    validate_release(output)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    validate = actions.add_parser("validate")
    validate.add_argument("--artifact-dir", type=Path, required=True)
    private = actions.add_parser("upload-private")
    private.add_argument("--artifact-dir", type=Path, required=True)
    private.add_argument("--repo-id", required=True)
    smoke = actions.add_parser("smoke-upload")
    smoke.add_argument("--artifact-dir", type=Path, required=True)
    smoke.add_argument("--repo-id", required=True)
    promote = actions.add_parser("promote-public")
    promote.add_argument("--artifact-dir", type=Path, required=True)
    promote.add_argument("--private-repo-id", required=True)
    promote.add_argument("--public-repo-id", required=True)
    promote.add_argument("--private-revision", required=True)
    promote.add_argument("--release-public", action="store_true")
    index = actions.add_parser("publish-index")
    index.add_argument("--index", type=Path, required=True)
    index.add_argument("--work-dir", type=Path, required=True)
    index.add_argument("--release-public", action="store_true")
    package = actions.add_parser("package")
    package.add_argument("--run-dir", type=Path, action="append", required=True)
    package.add_argument("--evaluation", type=Path, action="append", required=True)
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--selected-seed", type=int, required=True)
    package.add_argument("--recipe", choices=("classification", "evidence"), required=True)
    package.add_argument("--base-notice", type=Path)
    args = parser.parse_args(argv)
    if args.command == "validate":
        validate_release(args.artifact_dir)
    elif args.command == "upload-private":
        print(json.dumps(upload_private(args.artifact_dir, args.repo_id), sort_keys=True))
    elif args.command == "smoke-upload":
        print(json.dumps(smoke_upload(args.artifact_dir, args.repo_id), sort_keys=True))
    elif args.command == "promote-public":
        promote_public(args.artifact_dir, args.private_repo_id, args.public_repo_id, release_public=args.release_public, private_revision=args.private_revision)
    elif args.command == "publish-index":
        publish_index(args.index, args.work_dir, release_public=args.release_public)
    else:
        package_release(args.run_dir, args.evaluation, args.output, args.selected_seed, recipe=args.recipe, base_notice=args.base_notice)


if __name__ == "__main__":
    main()
