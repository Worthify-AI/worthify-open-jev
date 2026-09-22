"""Verify decision/frame alignment and content hashes before publishing a replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from openjev_phase1.core import softmax, validate_row


def _asset(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Replay assets must be relative paths")
    parts = PurePosixPath(name)
    if parts.is_absolute() or ".." in parts.parts or not parts.parts:
        raise ValueError("Replay asset escapes recording directory")
    path = directory / parts
    if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Replay asset is missing or escapes recording directory")
    return path


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_replay(path: Path, *, require_model=False) -> dict:
    episode = json.loads(path.read_text())
    if episode.get("schema") != "worthify-game-replay-v1" or episode.get("game") not in {"doom", "tetris"}:
        raise ValueError("Unknown replay schema or game")
    environment = episode.get("environment", {})
    if type(episode.get("seed")) is not int or environment.get("game_id") != episode["game"] or environment.get("seed") != episode["seed"]:
        raise ValueError("Replay identity differs from its environment")
    controller = episode.get("controller", {})
    model_run = controller.get("kind") == "model"
    if require_model and not model_run:
        raise ValueError("Scripted smoke tests cannot be published as model playback")
    if controller.get("kind") not in {"model", "smoke"}:
        raise ValueError("Unknown controller")
    if model_run:
        model = controller.get("model", {})
        if controller.get("selection") != "argmax" or not model.get("source") or not model.get("adapter"):
            raise ValueError("Model decisions need complete source and selection provenance")
        for value, length in ((model.get("revision"), 40), (model.get("adapter_sha256"), 64)):
            if not re.fullmatch(r"[0-9a-f]{%d}" % length, value or ""):
                raise ValueError("Model and adapter must be content-pinned")
        if not model.get("adapter_revision"):
            raise ValueError("Missing adapter revision")
    elif controller.get("model") is not None:
        raise ValueError("Smoke controls cannot claim model provenance")
    decisions = episode.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("No recorded decisions")
    frame = episode["initial_frame"]
    frames = {frame}
    for step, item in enumerate(decisions):
        validate_row(item)
        if item["id"] != f"{episode['game']}-{episode['seed']}-{step}":
            raise ValueError("Decision identity differs from its episode")
        if item.get("step") != step or item.get("frame_before") != frame:
            raise ValueError("Decision/frame order is broken")
        ids = [option["id"] for option in item["options"]]
        if item.get("selected_option_id") not in ids:
            raise ValueError("Recorded choice was not offered to the model")
        if type(item.get("terminated")) is not bool or (item["terminated"] and step != len(decisions) - 1):
            raise ValueError("Decisions continue after episode termination")
        if not isinstance(item.get("reward"), (int, float)) or not math.isfinite(item["reward"]):
            raise ValueError("Reward must be finite")
        if model_run:
            logits, probabilities = item.get("option_logits", []), item.get("probabilities", [])
            if len(logits) != len(ids) or len(probabilities) != len(ids):
                raise ValueError("Option scores are misaligned")
            expected = softmax(logits)
            if any(not isinstance(p, (int, float)) or not math.isfinite(p) or abs(p - q) > 1e-6
                   for p, q in zip(probabilities, expected)):
                raise ValueError("Invalid option probabilities")
            if ids[max(range(len(ids)), key=lambda i: probabilities[i])] != item["selected_option_id"]:
                raise ValueError("Recorded action differs from the model argmax")
            if not isinstance(item.get("inference_seconds"), (int, float)) or not math.isfinite(item["inference_seconds"]) or item["inference_seconds"] < 0:
                raise ValueError("Missing measured model timing")
            if not re.fullmatch(r"[0-9a-f]{64}", item.get("prompt_sha256", "")):
                raise ValueError("Missing prompt hash")
            if item.get("prompt_version") != "direct-options-v1" or not 1 <= item.get("input_tokens", 0) <= 2048:
                raise ValueError("Unexpected prompt or token limit")
        elif item.get("probabilities") or item.get("option_logits") or item.get("inference_seconds") is not None:
            raise ValueError("Smoke controls cannot claim model probabilities or timing")
        frame = item["frame_after"]
        frames.add(frame)
    bounded = decisions[-1].get("summary", {}).get("termination_reason") in {"max_steps", "step_limit"}
    expected_end = "step_limit" if bounded or not decisions[-1]["terminated"] else "terminated"
    if episode.get("ended_reason") != expected_end or episode.get("summary") != decisions[-1].get("summary"):
        raise ValueError("Final outcome differs from the final recorded decision")
    if set(episode.get("frame_sha256", {})) != frames:
        raise ValueError("Frame inventory is incomplete")
    for name, digest in episode["frame_sha256"].items():
        if _hash(_asset(path.parent, name)) != digest:
            raise ValueError(f"Frame checksum mismatch: {name}")
    log = _asset(path.parent, "decisions.jsonl")
    if _hash(log) != episode.get("decisions_sha256") or [json.loads(line) for line in log.read_text().splitlines()] != decisions:
        raise ValueError("Decision log differs from replay")
    return {"verified": True, "game": episode["game"], "controller": controller["kind"],
            "decisions": len(decisions), "frames": len(frames), "episode_sha256": _hash(path),
            "summary": episode["summary"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--require-model", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_replay(args.episode, require_model=args.require_model), indent=2))
