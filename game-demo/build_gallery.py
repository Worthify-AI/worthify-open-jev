"""Build a standalone static gallery from verified recorded episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from verify_replay import verify_replay


def preflight_video() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Video export requires ffmpeg on PATH")
    for flag, required in (("-filters", ("drawtext", "scale")), ("-encoders", ("libx264",))):
        available = subprocess.check_output(["ffmpeg", "-hide_banner", flag], text=True, stderr=subprocess.DEVNULL)
        if any(name not in available for name in required):
            raise RuntimeError(f"ffmpeg lacks required video capabilities: {required}")


def export_video(directory: Path, episode: dict, *, fps: int = 4) -> None:
    """An explicitly time-scaled decision replay, not a real-time game capture."""
    preflight_video()
    label = "RECORDED MODEL PLAY" if episode["controller"]["kind"] == "model" else "SMOKE TEST - NOT MODEL PLAY"
    frames = [episode["initial_frame"], *(item["frame_after"] for item in episode["decisions"])]
    # A transient numbered sequence avoids passing arbitrary names to ffmpeg's
    # concat demuxer. It also supports environments that reuse the final frame.
    sequence = directory / "video-frames"
    sequence.mkdir(exist_ok=False)
    try:
        for index, name in enumerate(frames):
            shutil.copyfile(directory / name, sequence / f"{index:06d}.png")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-framerate", str(fps), "-i", str(sequence / "%06d.png"),
            "-vf", "scale=960:-2,"
            f"drawtext=text='{label}':fontcolor=white:fontsize=22:box=1:boxcolor=black@0.8:x=10:y=10,"
            f"drawtext=text='Replay - {fps} decisions per second':fontcolor=white:fontsize=20:box=1:boxcolor=black@0.8:x=10:y=h-35",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(directory / "playback.mp4"),
        ], check=True)
    finally:
        shutil.rmtree(sequence)


def build_gallery(episodes: list[Path], output: Path, *, allow_smoke=False, video=False) -> dict:
    if output.exists():
        raise ValueError("Gallery output must be new")
    if not episodes:
        raise ValueError("At least one episode is required")
    if video:
        preflight_video()
    verified = []
    names = set()
    for path in episodes:
        receipt = verify_replay(path, require_model=not allow_smoke)
        episode = json.loads(path.read_text())
        if type(episode.get("seed")) is not int:
            raise ValueError("Episode seed must be an integer")
        name = f"{episode['game']}-seed{episode['seed']}"
        if name in names:
            raise ValueError("Duplicate game and seed in gallery")
        names.add(name)
        verified.append((path, episode, name, receipt))
    source = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=False)
    for name in ("index.html", "replay.js", "replay.css"):
        shutil.copyfile(source / name, output / name)
    shutil.copytree(source / "LICENSES", output / "LICENSES")
    (output / "runs").mkdir()
    index = {"schema": "worthify-game-index-v1", "episodes": []}
    receipts = []
    for path, episode, name, receipt in verified:
        destination = output / "runs" / name
        destination.mkdir()
        for asset in ["episode.json", "decisions.jsonl", *episode["frame_sha256"]]:
            target = destination / asset
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path.parent / asset, target)
        if video:
            export_video(destination, episode)
        prefix = "MODEL" if episode["controller"]["kind"] == "model" else "SMOKE TEST"
        index["episodes"].append({
            "title": f"{prefix} · {episode['game'].title()} · seed {episode['seed']}",
            "game": episode["game"], "href": f"{name}/episode.json",
        })
        receipts.append(receipt)
    (output / "runs" / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    report = {"schema": "worthify-game-gallery-v1", "episodes": receipts,
              "videos": video, "playback_decisions_per_second": 4 if video else None}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    checksums = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}"
                 for path in sorted(output.rglob("*")) if path.is_file()]
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true", help="For local plumbing tests only")
    parser.add_argument("--video", action="store_true", help="Also export labelled MP4 decision replays using ffmpeg")
    args = parser.parse_args()
    print(json.dumps(build_gallery(args.episode, args.output, allow_smoke=args.allow_smoke, video=args.video), indent=2))
