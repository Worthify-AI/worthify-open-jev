#!/usr/bin/env python3
"""Package six verified gameplay videos and link them from an adapter model card.

Gameplay media stays in a companion Hub repository because the adapter release
validator intentionally permits only the small, flat adapter evidence bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

from openjev_phase1.publish import sha256_file, validate_release
from verify_replay import verify_replay


GAMES = ("doom", "tetris")
SEEDS = (7, 19, 42)
CARD_START = "<!-- verified-gameplay-start -->"
CARD_END = "<!-- verified-gameplay-end -->"
HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _check_mp4(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 16:
        raise ValueError(f"Missing regular gameplay video: {path}")
    header = path.read_bytes()[:64]
    if b"ftyp" not in header:
        raise ValueError(f"Gameplay video is not an MP4 container: {path}")


def _gallery_checksums(gallery: Path) -> dict[str, str]:
    checksums = {}
    try:
        lines = (gallery / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError("Gallery is missing SHA256SUMS") from error
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise ValueError("Gallery SHA256SUMS contains an invalid line")
        name = parts[1].removeprefix("*")
        candidate = Path(name)
        if candidate.is_absolute() or ".." in candidate.parts or name in checksums:
            raise ValueError("Gallery SHA256SUMS contains an unsafe or duplicate path")
        checksums[name] = parts[0]
    return checksums


def package_assets(gallery: Path, adapter_dir: Path, output: Path) -> dict:
    """Create a content-hashed companion bundle from exactly six model runs."""
    if output.exists():
        raise ValueError("Gameplay asset output must be a new directory")
    validate_release(adapter_dir)
    release = _load(adapter_dir / "release-manifest.json")
    if release.get("kind") != "measured" or release.get("recipe") != "classification":
        raise ValueError("Gameplay assets require the measured classification release")
    adapter_sha = release.get("adapter", {}).get("sha256")
    base = release.get("base_model")
    expected = {(game, seed) for game in GAMES for seed in SEEDS}
    runs = gallery / "runs"
    gallery_checksums = _gallery_checksums(gallery)
    actual = set()
    records = []
    for episode_path in sorted(runs.glob("*/episode.json")):
        episode = _load(episode_path)
        identity = (episode.get("game"), episode.get("seed"))
        if identity in actual:
            raise ValueError(f"Duplicate gameplay run: {identity}")
        actual.add(identity)
        if identity not in expected:
            raise ValueError(f"Unexpected gameplay run: {identity}")
        receipt = verify_replay(episode_path, require_model=True)
        model = episode.get("controller", {}).get("model", {})
        if model.get("adapter_sha256") != adapter_sha:
            raise ValueError("Gameplay adapter hash differs from the packaged release")
        if {"id": model.get("source"), "revision": model.get("revision")} != base:
            raise ValueError("Gameplay base model differs from the packaged release")
        source_video = episode_path.parent / "playback.mp4"
        _check_mp4(source_video)
        relative_video = source_video.relative_to(gallery).as_posix()
        if gallery_checksums.get(relative_video) != sha256_file(source_video):
            raise ValueError(f"Gameplay video differs from the closed gallery: {relative_video}")
        timings = [item.get("inference_seconds") for item in episode.get("decisions", [])]
        if not timings or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in timings
        ):
            raise ValueError("Verified model gameplay must retain per-decision inference timing")
        name = f"{identity[0]}-seed{identity[1]}.mp4"
        records.append({
            "game": identity[0],
            "seed": identity[1],
            "path": name,
            "video_sha256": sha256_file(source_video),
            "video_bytes": source_video.stat().st_size,
            "episode_sha256": receipt["episode_sha256"],
            "decisions": receipt["decisions"],
            "inference_seconds": {
                "mean_per_decision": sum(timings) / len(timings),
                "minimum": min(timings),
                "maximum": max(timings),
                "total": sum(timings),
            },
            "summary": receipt["summary"],
        })
    if actual != expected:
        missing = sorted(expected - actual)
        raise ValueError(f"Gameplay bundle requires all six predeclared model runs; missing={missing}")

    output.mkdir(parents=True, exist_ok=False)
    for record in records:
        source = runs / f"{record['game']}-seed{record['seed']}" / "playback.mp4"
        shutil.copyfile(source, output / record["path"])
    manifest = {
        "schema": "worthify-gameplay-assets-v1",
        "purpose": "illustrative recorded model play; not a gameplay benchmark",
        "adapter": {"sha256": adapter_sha, "selected_seed": release.get("selected_seed")},
        "base_model": base,
        "attribution": {
            "doom_assets": "Freedoom 0.13 artwork supplied through ViZDoom",
            "license": "BSD-3-Clause",
            "license_path": "FREEDOOM-BSD-3-CLAUSE.txt",
            "notice_path": "FREEDOOM-ATTRIBUTION.md",
        },
        "runs": records,
        "limitations": [
            "All six predeclared runs are included; none was selected by game outcome.",
            "Episodes are illustrative and are not a gameplay benchmark or quality metric.",
            "Playback is fixed at four decisions per second and is not real-time inference footage.",
            "Doom uses text telemetry and privileged engine object labels; the model does not see pixels.",
            "Falling blocks uses sampled straight-drop placements rather than full real-time controls.",
            "Conditional option probabilities are uncalibrated.",
        ],
    }
    (output / "gameplay-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Worthify OpenJev recorded gameplay\n\n"
        "This companion contains all six predeclared recorded-model episodes: Doom/Freedoom and "
        "falling blocks at seeds 7, 19, and 42. They are illustrative episodes, not a gameplay "
        "benchmark or model-quality metric. See `gameplay-manifest.json` for hashes, provenance, "
        "outcomes, and limitations. Playback is time-scaled to four decisions per second. Doom "
        "frames reuse Freedoom 0.13 artwork supplied through ViZDoom; preserve "
        "`FREEDOOM-BSD-3-CLAUSE.txt` and `FREEDOOM-ATTRIBUTION.md` with these assets.\n",
        encoding="utf-8",
    )
    licenses = Path(__file__).resolve().parent / "LICENSES"
    shutil.copyfile(licenses / "FREEDOOM-BSD-3-CLAUSE.txt", output / "FREEDOOM-BSD-3-CLAUSE.txt")
    shutil.copyfile(licenses / "README.md", output / "FREEDOOM-ATTRIBUTION.md")
    names = sorted(path.name for path in output.iterdir() if path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(output / name)}  {name}\n" for name in names), encoding="utf-8"
    )
    return manifest


def _card_section(asset_repo_id: str, asset_revision: str, manifest: dict, *, training_max_tokens: int) -> str:
    if not HUB_ID.fullmatch(asset_repo_id):
        raise ValueError("Asset repository must be an owner/repository Hugging Face ID")
    if not SHA40.fullmatch(asset_revision):
        raise ValueError("Asset revision must be an immutable 40-character Hub commit")
    runs = manifest.get("runs")
    expected = {(game, seed) for game in GAMES for seed in SEEDS}
    if (
        not isinstance(runs, list)
        or len(runs) != 6
        or {(item.get("game"), item.get("seed")) for item in runs} != expected
        or any(not re.fullmatch(r"[0-9a-f]{64}", item.get("video_sha256", "")) for item in runs)
    ):
        raise ValueError("Gameplay manifest must contain exactly the six predeclared runs")
    lines = [
        CARD_START,
        "## Recorded Doom and falling-block play",
        "",
        "These are all six predeclared illustrative runs, not selected highlights. They are not a "
        "gameplay benchmark or a model-quality metric. The model chose from text-defined options; "
        "it did not see pixels. Doom uses privileged engine object labels. Falling blocks uses "
        "sampled straight-drop placements. Playback is fixed at four decisions per second, so it "
        "does not show inference speed; the per-decision timings below are the recorded inference "
        "latencies. Conditional option probabilities are uncalibrated.",
        "",
        f"This adapter was trained with a maximum prompt length of {training_max_tokens} tokens. "
        "The base model's native 262,144-token context is a separate upstream capability. Any "
        "isolated long-context memory/gradient probe is a synthetic systems measurement, not "
        "adapter training or a quality result. This release makes no 1M-context support claim.",
        "",
    ]
    for game in GAMES:
        lines.extend((f"### {'Doom / Freedoom' if game == 'doom' else 'Falling blocks'}", ""))
        for item in sorted((item for item in runs if item["game"] == game), key=lambda x: x["seed"]):
            url = f"https://huggingface.co/datasets/{asset_repo_id}/resolve/{asset_revision}/{item['path']}"
            lines.extend((
                f"Seed {item['seed']} · {item['decisions']} decisions · "
                f"mean recorded inference {item['inference_seconds']['mean_per_decision']:.3f} s/decision · "
                f"[open MP4]({url}) · video SHA-256 `{item['video_sha256']}`",
                "",
                f'<video controls preload="metadata" src="{url}"></video>',
                "",
            ))
    lines.extend((
        "The companion `gameplay-manifest.json` records the episode hashes, outcomes, exact adapter "
        "hash, base revision, and limitations for every run.",
        CARD_END,
    ))
    return "\n".join(lines)


def patch_model_card(adapter_dir: Path, assets: Path, asset_repo_id: str, asset_revision: str) -> None:
    """Add pinned video embeds, then restore the adapter bundle's checksum closure."""
    validate_release(adapter_dir)
    manifest = _load(assets / "gameplay-manifest.json")
    release = _load(adapter_dir / "release-manifest.json")
    if manifest.get("schema") != "worthify-gameplay-assets-v1":
        raise ValueError("Gameplay manifest has an unsupported schema")
    if manifest.get("adapter", {}).get("sha256") != release.get("adapter", {}).get("sha256"):
        raise ValueError("Gameplay manifest does not match this adapter release")
    if manifest.get("base_model") != release.get("base_model"):
        raise ValueError("Gameplay manifest base model does not match this adapter release")
    training = _load(adapter_dir / "training-manifest.json")
    training_max_tokens = training.get("max_tokens")
    if training_max_tokens != 2048:
        raise ValueError("Gameplay card text currently requires the verified 2048-token adapter recipe")
    card_path = adapter_dir / "README.md"
    card = card_path.read_text(encoding="utf-8")
    if CARD_START in card or CARD_END in card:
        raise ValueError("Model card already contains a verified gameplay section")
    updated_card = card.rstrip() + "\n\n" + _card_section(
        asset_repo_id, asset_revision, manifest, training_max_tokens=training_max_tokens
    ) + "\n"
    checksum_path = adapter_dir / "SHA256SUMS"
    original_checksums = checksum_path.read_bytes()
    names = sorted(path.name for path in adapter_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    try:
        card_path.write_text(updated_card, encoding="utf-8")
        checksum_path.write_text(
            "".join(f"{sha256_file(adapter_dir / name)}  {name}\n" for name in names), encoding="utf-8"
        )
        validate_release(adapter_dir)
    except BaseException:
        card_path.write_text(card, encoding="utf-8")
        checksum_path.write_bytes(original_checksums)
        raise


def verify_remote_assets(assets: Path, repo_id: str, revision: str, *, api=None) -> dict:
    """Re-download the pinned Hub dataset bundle and verify every local byte."""
    if not HUB_ID.fullmatch(repo_id) or not SHA40.fullmatch(revision):
        raise ValueError("Remote gameplay verification requires a Hub ID and immutable SHA-40 revision")
    local = {path.name: sha256_file(path) for path in assets.iterdir() if path.is_file()}
    if "gameplay-manifest.json" not in local or len([name for name in local if name.endswith(".mp4")]) != 6:
        raise ValueError("Local gameplay bundle must contain its manifest and six MP4 files")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    remote = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision))
    if remote - {".gitattributes"} != set(local):
        raise ValueError("Pinned Hub gameplay repository does not exactly match the local bundle")
    verified = {}
    for name, digest in sorted(local.items()):
        if hasattr(api, "hf_hub_download"):
            downloaded = api.hf_hub_download(
                repo_id=repo_id, repo_type="dataset", filename=name, revision=revision
            )
        else:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id, name, repo_type="dataset", revision=revision, token=getattr(api, "token", None)
            )
        if sha256_file(Path(downloaded)) != digest:
            raise ValueError(f"Pinned Hub gameplay checksum mismatch: {name}")
        verified[name] = digest
    return {
        "schema": "worthify-gameplay-hub-verification-v1",
        "repo_id": repo_id,
        "revision": revision,
        "verified_files": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument("--gallery", type=Path, required=True)
    package.add_argument("--adapter-dir", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    card = commands.add_parser("patch-card")
    card.add_argument("--adapter-dir", type=Path, required=True)
    card.add_argument("--assets", type=Path, required=True)
    card.add_argument("--asset-repo-id", required=True)
    card.add_argument("--asset-revision", required=True)
    verify = commands.add_parser("verify-remote")
    verify.add_argument("--assets", type=Path, required=True)
    verify.add_argument("--asset-repo-id", required=True)
    verify.add_argument("--asset-revision", required=True)
    args = parser.parse_args()
    if args.command == "package":
        print(json.dumps(package_assets(args.gallery, args.adapter_dir, args.output), indent=2))
    elif args.command == "patch-card":
        patch_model_card(args.adapter_dir, args.assets, args.asset_repo_id, args.asset_revision)
    else:
        print(json.dumps(verify_remote_assets(
            args.assets, args.asset_repo_id, args.asset_revision
        ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
