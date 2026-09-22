import hashlib
import json
import shutil
from pathlib import Path

import pytest

from openjev_phase1.publish import package_release, publish_index, promote_public, smoke_upload, upload_private, validate_release


BASE = {
    "id": "example/base",
    "revision": "a" * 40,
}
COMMIT = "c" * 40


def recipe_record(recipe="classification"):
    return {"task": recipe, "dataset": "CLINC150" if recipe == "classification" else "WANLI",
            "version": "openjev-training-recipe-v1", "digest": "f" * 64}


def evaluation_record(seed, adapter_sha, recipe="classification"):
    return {"schema": "openjev-phase1-evaluation-v1", "provenance_verified": True,
            "inputs": {"gold_sha256": "d" * 64, "predictions_sha256": "e" * 64},
            "seed": seed, "kind": "measured", "split": "test", "n": 1, "accuracy": .5,
            "mean_task_macro_f1": .5, "base_model": BASE, "adapter_sha256": adapter_sha,
            "task_results": {recipe: {"n": 1, "macro_f1": .5}}}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_checksums(directory: Path) -> None:
    names = sorted(path.name for path in directory.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (directory / "SHA256SUMS").write_text("".join(f"{_sha(directory / name)}  {name}\n" for name in names))


def release_dir(tmp_path: Path, recipe="classification") -> Path:
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_model.safetensors").write_bytes(b"safe adapter only")
    (directory / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": BASE["id"], "peft_type": "LORA"}))
    (directory / "README.md").write_text("---\nlicense: apache-2.0\n---\n# Adapter\n")
    (directory / "LICENSE").write_text("Apache License\nVersion 2.0\n")
    (directory / "attribution.json").write_text(json.dumps({"schema": "openjev-phase1-attribution-v1", "sources": [{"name": "fixture", "license": "Apache-2.0", "url": "https://example.invalid", "revision": "fixture", "modifications": "fixture"}]}))
    adapter_sha = _sha(directory / "adapter_model.safetensors")
    for name in ("training-manifest.json", "evaluation-manifest.json"):
        record = evaluation_record(42, adapter_sha, recipe) if name == "evaluation-manifest.json" else {
            "base_model": BASE, "adapter_sha256": adapter_sha, "training_spec": {"recipe": recipe_record(recipe)}}
        (directory / name).write_text(json.dumps(record))
    benchmark_names = ["seed-42-benchmark-results.json", "seed-43-benchmark-results.json"]
    for benchmark_name in benchmark_names:
        seed = int(benchmark_name.split("-")[1])
        (directory / benchmark_name).write_text(json.dumps(evaluation_record(seed, adapter_sha, recipe)))
    manifest = {
        "schema": "openjev-phase1-adapter-release-v1", "kind": "measured", "recipe": recipe, "selected_seed": 42,
        "base_model": BASE,
        "adapter": {"path": "adapter_model.safetensors", "sha256": adapter_sha},
        "training": {"path": "training-manifest.json", "sha256": _sha(directory / "training-manifest.json")},
        "evaluation": {"path": "evaluation-manifest.json", "sha256": _sha(directory / "evaluation-manifest.json")},
        "benchmarks": [{"path": name, "sha256": _sha(directory / name)} for name in benchmark_names],
    }
    (directory / "release-manifest.json").write_text(json.dumps(manifest))
    _rewrite_checksums(directory)
    return directory


class FakeHub:
    def __init__(self, root: Path):
        self.root = root
        self.private: dict[str, bool] = {}
        self.downloads = []

    def create_repo(self, *, repo_id, repo_type, private, exist_ok):
        self.private[repo_id] = private
        (self.root / repo_id.replace("/", "_")).mkdir(parents=True, exist_ok=True)

    def repo_info(self, *, repo_id, repo_type):
        if repo_id not in self.private:
            raise type("RepositoryNotFoundError", (Exception,), {})()
        return type("Info", (), {"private": self.private[repo_id]})()

    def list_repo_files(self, *, repo_id, repo_type, revision=None):
        target = self.root / repo_id.replace("/", "_")
        return [path.name for path in target.iterdir()] if target.exists() else []

    def upload_folder(self, *, repo_id, repo_type, folder_path, allow_patterns):
        target = self.root / repo_id.replace("/", "_")
        for name in allow_patterns:
            source = Path(folder_path) / name
            if source.exists():
                shutil.copy2(source, target / name)
        return type("Commit", (), {"oid": COMMIT})()

    def hf_hub_download(self, *, repo_id, repo_type, filename, revision=None):
        self.downloads.append((repo_id, filename, revision))
        return str(self.root / repo_id.replace("/", "_") / filename)

    def update_repo_settings(self, *, repo_id, repo_type, private):
        self.private[repo_id] = private


def test_validate_release_links_checksums_and_provenance(tmp_path):
    artifact = release_dir(tmp_path)
    hashes = validate_release(artifact)
    assert hashes["adapter_model.safetensors"] == _sha(artifact / "adapter_model.safetensors")


def test_validate_release_rejects_checksum_tampering(tmp_path):
    artifact = release_dir(tmp_path)
    (artifact / "adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_release(artifact)


def test_validate_release_rejects_base_weights_and_bad_manifest_link(tmp_path):
    artifact = release_dir(tmp_path)
    (artifact / "pytorch_model.bin").write_bytes(b"base weights")
    with pytest.raises(ValueError, match="Forbidden release content"):
        validate_release(artifact)
    (artifact / "pytorch_model.bin").unlink()
    record = json.loads((artifact / "evaluation-manifest.json").read_text())
    record["adapter_sha256"] = "0" * 64
    (artifact / "evaluation-manifest.json").write_text(json.dumps(record))
    manifest = json.loads((artifact / "release-manifest.json").read_text())
    manifest["evaluation"]["sha256"] = _sha(artifact / "evaluation-manifest.json")
    (artifact / "release-manifest.json").write_text(json.dumps(manifest))
    _rewrite_checksums(artifact)
    with pytest.raises(ValueError, match="does not link the adapter hash"):
        validate_release(artifact)


def test_private_upload_then_explicit_public_promotion(tmp_path):
    artifact = release_dir(tmp_path)
    hub = FakeHub(tmp_path / "hub")
    upload_private(artifact, "worthify/private-adapter", api=hub)
    assert hub.private["worthify/private-adapter"] is True
    with pytest.raises(ValueError, match="--release-public"):
        promote_public(artifact, "worthify/private-adapter", "worthify/public-adapter", release_public=False, private_revision=COMMIT, api=hub)
    promote_public(artifact, "worthify/private-adapter", "worthify/public-adapter", release_public=True, private_revision=COMMIT, api=hub)
    assert hub.private["worthify/public-adapter"] is False
    assert all(revision == COMMIT for _, _, revision in hub.downloads)


def test_private_upload_refuses_existing_public_and_stale_files(tmp_path):
    artifact, hub = release_dir(tmp_path), FakeHub(tmp_path / "hub")
    hub.create_repo(repo_id="org/public", repo_type="model", private=False, exist_ok=False)
    with pytest.raises(ValueError, match="existing public"):
        upload_private(artifact, "org/public", api=hub)
    hub.create_repo(repo_id="org/private", repo_type="model", private=True, exist_ok=False)
    (hub.root / "org_private" / "stale.bin").write_bytes(b"stale")
    with pytest.raises(ValueError, match="stale"):
        upload_private(artifact, "org/private", api=hub)


def test_private_upload_accepts_initial_gitattributes_only(tmp_path):
    artifact, hub = release_dir(tmp_path), FakeHub(tmp_path / "hub")
    hub.create_repo(repo_id="org/private", repo_type="model", private=True, exist_ok=False)
    (hub.root / "org_private" / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    upload_private(artifact, "org/private", api=hub)


def test_release_rejects_symlink(tmp_path):
    artifact = release_dir(tmp_path)
    readme = artifact / "README.md"
    target = tmp_path / "README.md"
    readme.rename(target)
    readme.symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        validate_release(artifact)


def test_publish_index_materializes_two_recipes_with_same_seed(tmp_path):
    artifact, hub = release_dir(tmp_path), FakeHub(tmp_path / "hub")
    candidates = []
    for recipe in ("classification", "evidence"):
        (tmp_path / recipe).mkdir()
        artifact = release_dir(tmp_path / recipe, recipe)
        private_id, public_id = f"org/private-{recipe}", f"org/{recipe}"
        manifest = json.loads((artifact / "release-manifest.json").read_text())
        manifest["recipe"] = recipe
        (artifact / "release-manifest.json").write_text(json.dumps(manifest))
        _rewrite_checksums(artifact)
        upload_private(artifact, private_id, api=hub)
        # Hub snapshots routinely return symlinks. They must be copied to regular files.
        cached = hub.root / private_id.replace("/", "_") / "README.md"
        cached.unlink()
        cached.symlink_to(artifact / "README.md")
        candidates.append({"recipe": recipe, "seed": 42, "private_repo_id": private_id,
                           "public_repo_id": public_id, "revision": COMMIT})
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"schema": "openjev-phase1-release-index-v1", "state": "verified", "candidates": candidates}))
    destination = tmp_path / "materialized"
    publish_index(index, destination, release_public=True, api=hub)
    for recipe in ("classification", "evidence"):
        assert hub.private[f"org/{recipe}"] is False
        assert not (destination / recipe / "README.md").is_symlink()
        validate_release(destination / recipe)
    assert all(revision == COMMIT for _, _, revision in hub.downloads)


def training_runs(tmp_path):
    runs, evaluations = [], []
    for seed in (42, 43):
        run = tmp_path / f"run-{seed}"
        adapter = run / "final-adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(f"adapter-{seed}".encode())
        (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": BASE["id"], "peft_type": "LORA"}))
        (run / "manifest.json").write_text(json.dumps({"seed": seed, "model": {"source": BASE["id"], "revision": BASE["revision"]},
                                                      "epochs": 2, "validation_checkpoint": "/private/operator/run/checkpoint-epoch-00",
                                                      "training_spec": {"seed": seed, "model": BASE["id"], "revision": BASE["revision"],
                                                                        "recipe": recipe_record(),
                                                                        "train_sha256": "1" * 64, "validation_sha256": "2" * 64},
                                                      "best_validation_macro_f1": .8 if seed == 42 else .7}))
        evaluation = tmp_path / f"evaluation-{seed}.json"
        evaluation.write_text(json.dumps(evaluation_record(seed, _sha(adapter / "adapter_model.safetensors"))))
        runs.append(run)
        evaluations.append(evaluation)
    return runs, evaluations


def test_package_preserves_actual_evaluation_provenance(tmp_path):
    runs, evaluations = training_runs(tmp_path)
    destination = tmp_path / "release"
    package_release(runs, evaluations, destination, 42, recipe="classification")
    validate_release(destination)
    assert json.loads((destination / "evaluation-manifest.json").read_text()) == json.loads(evaluations[0].read_text())
    assert _sha(destination / "adapter_model.safetensors") == _sha(runs[0] / "final-adapter/adapter_model.safetensors")
    assert "validation_checkpoint" not in json.loads((destination / "training-manifest.json").read_text())
    card = (destination / "README.md").read_text()
    assert "not Jev" in card and "uncalibrated conditional option scores" in card and "REPLACE_WITH_40_CHARACTER_HUB_COMMIT" in card
    attribution = json.loads((destination / "attribution.json").read_text())
    assert {source["name"] for source in attribution["sources"]} >= {"CLINC150", "WANLI"}


def test_package_copies_only_supplied_regular_base_notice(tmp_path):
    runs, evaluations = training_runs(tmp_path)
    notice = tmp_path / "base-NOTICE"
    notice.write_text("upstream notice")
    destination = tmp_path / "release"
    package_release(runs, evaluations, destination, 42, recipe="classification", base_notice=notice)
    assert (destination / "NOTICE").read_text() == "upstream notice"
    assert "NOTICE" in validate_release(destination)


@pytest.mark.parametrize("field,value", [("adapter_sha256", "0" * 64), ("base_model", {"id": "wrong", "revision": "a" * 40}),
                                        ("provenance_verified", False), ("inputs", {}), ("seed", 43), ("accuracy", float("nan"))])
def test_package_refuses_mismatched_or_unverified_report(tmp_path, field, value):
    runs, evaluations = training_runs(tmp_path)
    report = json.loads(evaluations[0].read_text())
    report[field] = value
    evaluations[0].write_text(json.dumps(report))
    with pytest.raises(ValueError):
        package_release(runs, evaluations, tmp_path / "release", 42, recipe="classification")
    assert not (tmp_path / "release").exists()


def test_package_selection_uses_validation_not_test(tmp_path):
    runs, evaluations = training_runs(tmp_path)
    with pytest.raises(ValueError, match="validation F1"):
        package_release(runs, evaluations, tmp_path / "release", 43, recipe="classification")


def test_package_rejects_incomparable_training_runs(tmp_path):
    runs, evaluations = training_runs(tmp_path)
    manifest_path = runs[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_spec"]["train_sha256"] = "3" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="same training data"):
        package_release(runs, evaluations, tmp_path / "release", 42, recipe="classification")


def test_package_rejects_recipe_relabeling(tmp_path):
    runs, evaluations = training_runs(tmp_path)
    with pytest.raises(ValueError, match="recipe"):
        package_release(runs, evaluations, tmp_path / "release", 42, recipe="evidence")
    # Relabeling the reports still cannot relabel the actual recorded training recipe.
    for path in evaluations:
        report = json.loads(path.read_text())
        report["task_results"] = {"evidence": report["task_results"]["classification"]}
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Training recipe"):
        package_release(runs, evaluations, tmp_path / "release", 42, recipe="evidence")


def test_smoke_receipt_is_pinned_and_checksum_lines_are_strict(tmp_path):
    artifact = tmp_path / "smoke"
    artifact.mkdir()
    (artifact / "README.md").write_text("Private transport smoke only")
    (artifact / "smoke-payload.json").write_text('{"inert": true}')
    _rewrite_checksums(artifact)
    hub = FakeHub(tmp_path / "hub")
    receipt = smoke_upload(artifact, "org/smoke", api=hub)
    assert receipt["revision"] == COMMIT
    assert hub.private["org/smoke"] is True
    assert all(revision == COMMIT for _, _, revision in hub.downloads)
    checksums = artifact / "SHA256SUMS"
    checksums.write_text(checksums.read_text() + "ignored-garbage\n")
    with pytest.raises(ValueError, match="malformed"):
        smoke_upload(artifact, "org/other", api=hub)
    assert "org/other" not in hub.private
