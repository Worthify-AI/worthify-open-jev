import importlib.util
import json
from pathlib import Path
import sys

import pytest


GAME_DIR = Path(__file__).resolve().parents[1] / "game-demo"
sys.path.insert(0, str(GAME_DIR))
spec = importlib.util.spec_from_file_location("game_hf_packaging", GAME_DIR / "package_hf_gameplay.py")
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


def _release(path: Path) -> Path:
    path.mkdir()
    (path / "README.md").write_text("---\nlicense: apache-2.0\n---\n# Adapter\n")
    (path / "release-manifest.json").write_text(json.dumps({
        "kind": "measured", "recipe": "classification", "selected_seed": 42,
        "base_model": {"id": "google/gemma-4-12B-it", "revision": "a" * 40},
        "adapter": {"sha256": "b" * 64},
    }))
    (path / "training-manifest.json").write_text(json.dumps({"max_tokens": 2048}))
    (path / "SHA256SUMS").write_text("")
    return path


def _gallery(path: Path, *, smoke=False, omit=None) -> Path:
    for game in packaging.GAMES:
        for seed in packaging.SEEDS:
            if (game, seed) == omit:
                continue
            run = path / "runs" / f"{game}-seed{seed}"
            run.mkdir(parents=True)
            (run / "episode.json").write_text(json.dumps({
                "game": game, "seed": seed,
                "controller": {"kind": "smoke" if smoke else "model", "model": {
                    "source": "google/gemma-4-12B-it", "revision": "a" * 40,
                    "adapter_sha256": "b" * 64,
                }},
                "decisions": [{"inference_seconds": 0.25}],
            }))
            (run / "playback.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 32)
    checksums = []
    for video in sorted(path.glob("runs/*/playback.mp4")):
        checksums.append(f"{packaging.sha256_file(video)}  {video.relative_to(path).as_posix()}")
    (path / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    return path


def test_packages_exactly_six_verified_model_videos(tmp_path, monkeypatch):
    adapter = _release(tmp_path / "adapter")
    gallery = _gallery(tmp_path / "gallery")
    monkeypatch.setattr(packaging, "validate_release", lambda _path: {})
    monkeypatch.setattr(packaging, "verify_replay", lambda path, require_model: {
        "episode_sha256": "c" * 64, "decisions": 3, "summary": {"score": 1},
    })
    output = tmp_path / "assets"
    manifest = packaging.package_assets(gallery, adapter, output)
    assert len(manifest["runs"]) == 6
    assert {item["path"] for item in manifest["runs"]} == {
        f"{game}-seed{seed}.mp4" for game in packaging.GAMES for seed in packaging.SEEDS
    }
    assert "not a gameplay benchmark" in manifest["purpose"]
    assert len((output / "SHA256SUMS").read_text().splitlines()) == 10
    assert "Redistribution and use" in (output / "FREEDOOM-BSD-3-CLAUSE.txt").read_text()


def test_package_rejects_incomplete_recipe(tmp_path, monkeypatch):
    adapter = _release(tmp_path / "adapter")
    gallery = _gallery(tmp_path / "gallery", omit=("doom", 19))
    monkeypatch.setattr(packaging, "validate_release", lambda _path: {})
    monkeypatch.setattr(packaging, "verify_replay", lambda *args, **kwargs: {
        "episode_sha256": "c" * 64, "decisions": 3, "summary": {},
    })
    with pytest.raises(ValueError, match="all six"):
        packaging.package_assets(gallery, adapter, tmp_path / "assets")


def test_package_rejects_video_substituted_after_gallery_close(tmp_path, monkeypatch):
    adapter = _release(tmp_path / "adapter")
    gallery = _gallery(tmp_path / "gallery")
    (gallery / "runs/doom-seed7/playback.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42substituted")
    monkeypatch.setattr(packaging, "validate_release", lambda _path: {})
    monkeypatch.setattr(packaging, "verify_replay", lambda *args, **kwargs: {
        "episode_sha256": "c" * 64, "decisions": 3, "summary": {},
    })
    with pytest.raises(ValueError, match="closed gallery"):
        packaging.package_assets(gallery, adapter, tmp_path / "assets")


def test_card_uses_pinned_hf_video_urls_and_recloses_checksums(tmp_path, monkeypatch):
    adapter = _release(tmp_path / "adapter")
    assets = tmp_path / "assets"
    assets.mkdir()
    runs = [{
        "game": game, "seed": seed, "path": f"{game}-seed{seed}.mp4",
        "video_sha256": hashlib, "decisions": seed,
        "inference_seconds": {"mean_per_decision": 0.25},
    } for game in packaging.GAMES for seed in packaging.SEEDS for hashlib in ["d" * 64]]
    (assets / "gameplay-manifest.json").write_text(json.dumps({
        "schema": "worthify-gameplay-assets-v1",
        "adapter": {"sha256": "b" * 64},
        "base_model": {"id": "google/gemma-4-12B-it", "revision": "a" * 40},
        "runs": runs,
    }))
    calls = []
    monkeypatch.setattr(packaging, "validate_release", lambda path: calls.append(path))
    revision = "e" * 40
    packaging.patch_model_card(adapter, assets, "Worthify/worthify-jev-gameplay", revision)
    card = (adapter / "README.md").read_text()
    assert card.count("<video controls") == 6
    assert card.count(f"/resolve/{revision}/") == 12  # embed plus fallback link
    assert "all six predeclared illustrative runs" in card
    assert "maximum prompt length of 2048 tokens" in card
    assert "no 1M-context support claim" in card
    assert "mean recorded inference 0.250 s/decision" in card
    assert calls == [adapter, adapter]
    assert any(line.endswith("  README.md") for line in (adapter / "SHA256SUMS").read_text().splitlines())


def test_card_patch_rolls_back_if_final_release_validation_fails(tmp_path, monkeypatch):
    adapter = _release(tmp_path / "adapter")
    assets = tmp_path / "assets"
    assets.mkdir()
    runs = [{
        "game": game, "seed": seed, "path": f"{game}-seed{seed}.mp4",
        "video_sha256": "d" * 64, "decisions": 1,
        "inference_seconds": {"mean_per_decision": 0.1},
    } for game in packaging.GAMES for seed in packaging.SEEDS]
    (assets / "gameplay-manifest.json").write_text(json.dumps({
        "schema": "worthify-gameplay-assets-v1", "adapter": {"sha256": "b" * 64},
        "base_model": {"id": "google/gemma-4-12B-it", "revision": "a" * 40}, "runs": runs,
    }))
    original_card = (adapter / "README.md").read_bytes()
    original_sums = (adapter / "SHA256SUMS").read_bytes()
    calls = 0

    def validate(_path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("final validation failed")

    monkeypatch.setattr(packaging, "validate_release", validate)
    with pytest.raises(ValueError, match="final validation"):
        packaging.patch_model_card(adapter, assets, "Worthify/worthify-jev-gameplay", "e" * 40)
    assert (adapter / "README.md").read_bytes() == original_card
    assert (adapter / "SHA256SUMS").read_bytes() == original_sums


def test_remote_verification_requires_exact_pinned_bundle(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "gameplay-manifest.json").write_text("{}")
    for game in packaging.GAMES:
        for seed in packaging.SEEDS:
            (assets / f"{game}-seed{seed}.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + bytes([seed]))
    (assets / "README.md").write_text("companion")
    (assets / "SHA256SUMS").write_text("checksums")

    class Hub:
        def list_repo_files(self, **_kwargs):
            return [".gitattributes", *(path.name for path in assets.iterdir())]

        def hf_hub_download(self, *, filename, **_kwargs):
            return str(assets / filename)

    receipt = packaging.verify_remote_assets(
        assets, "Worthify/worthify-jev-gameplay", "e" * 40, api=Hub()
    )
    assert receipt["revision"] == "e" * 40
    assert len(receipt["verified_files"]) == 9
