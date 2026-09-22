"""Record real option-logit decisions in a bounded game; never overwrite a run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess

from openjev_phase1.core import load_causal_model, validate_row
from openjev_phase1.direct import score
from verify_replay import verify_replay

BASE_MODEL = "google/gemma-4-12B-it"
BASE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SmokeController:
    """Deterministic random controls for plumbing tests, explicitly not a model."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.metadata = {"kind": "smoke", "selection": "seeded_random", "model": None}

    def decide(self, row):
        return {
            "selected_option_id": self.rng.choice(row["options"])["id"],
            "probabilities": [], "option_logits": [], "inference_seconds": None,
        }


class ModelController:
    def __init__(self, args):
        self.model, self.tokenizer, metadata = load_causal_model(
            args.model, args.revision, adapter=args.adapter,
            adapter_revision=args.adapter_revision, quantization="nf4",
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        # A local adapter is identified by its content, never a private host path.
        metadata = dict(metadata)
        if Path(args.adapter).is_dir():
            metadata["adapter"] = "local-sha256:" + metadata["adapter_sha256"]
        self.metadata = {"kind": "model", "selection": "argmax", "model": metadata}

    def decide(self, row):
        result = score(self.model, self.tokenizer, row, self.metadata["model"], max_tokens=2048)
        index = max(range(len(row["options"])), key=lambda i: result["probabilities"][i])
        return {
            "selected_option_id": result["option_ids"][index],
            "probabilities": result["probabilities"], "option_logits": result["option_logits"],
            "inference_seconds": result["total_seconds"],
            "prompt_sha256": result["prompt_sha256"], "prompt_version": result["prompt_version"],
            "input_tokens": result["input_tokens"],
        }


def record_episode(game, controller, output_dir: Path, *, seed: int, max_steps: int) -> dict:
    if (output_dir / "episode.json").exists() or (output_dir / "decisions.jsonl").exists():
        raise ValueError("Recording output must be new")
    source_dir = Path(__file__).resolve().parent
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_dir, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    episode = {
        "schema": "worthify-game-replay-v1", "game": game.game_id, "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "controller": controller.metadata, "environment": game.metadata,
        "code_revision": revision,
        "source_sha256": {name: sha256(source_dir / name) for name in
                          ("record.py", "verify_replay.py", f"{game.game_id}_env.py")},
        "initial_frame": game.frame(), "decisions": [],
        "limitations": [
            "Recorded decisions, not a live model session. Replay speed is adjustable and is not measured inference speed.",
            "The model receives structured text state, not screenshot pixels. Engine state can include privileged information.",
            "Option probabilities are conditional on the supplied choices and are uncalibrated.",
            "A bounded demonstration, not a game benchmark or evidence of general game-playing ability.",
        ],
    }
    frame = episode["initial_frame"]
    with (output_dir / "decisions.jsonl").open("x") as log:
        for step in range(max_steps):
            observation = game.observation()
            row = {"id": f"{game.game_id}-{seed}-{step}", **observation}
            validate_row(row)
            decision = controller.decide(row)
            if decision["selected_option_id"] not in {item["id"] for item in row["options"]}:
                raise ValueError("Controller selected an unavailable action")
            outcome = game.step(decision["selected_option_id"])
            next_frame = game.frame()
            item = {
                "step": step, **row, **decision, **outcome,
                "frame_before": frame, "frame_after": next_frame,
            }
            log.write(json.dumps(item, allow_nan=False) + "\n")
            log.flush()
            episode["decisions"].append(item)
            frame = next_frame
            if outcome["terminated"]:
                break
    episode["summary"] = game.summary
    bounded = episode["summary"].get("termination_reason") in {"max_steps", "step_limit"}
    episode["ended_reason"] = "step_limit" if bounded or not episode["decisions"][-1]["terminated"] else "terminated"
    frames = {episode["initial_frame"]}
    for item in episode["decisions"]:
        frames.update((item["frame_before"], item["frame_after"]))
    episode["frame_sha256"] = {name: sha256(output_dir / name) for name in sorted(frames)}
    episode["decisions_sha256"] = sha256(output_dir / "decisions.jsonl")
    with (output_dir / "episode.json").open("x") as destination:
        destination.write(json.dumps(episode, indent=2, allow_nan=False) + "\n")
    return verify_replay(output_dir / "episode.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("doom", "tetris"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller", choices=("model", "smoke"), default="model")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--revision", default=BASE_REVISION)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or not 1 <= args.max_steps <= 1000:
        parser.error("Output must not exist; max-steps must be between 1 and 1000")
    if args.controller == "model" and (not args.adapter or not args.adapter_revision):
        parser.error("Model playback requires --adapter and --adapter-revision")
    if args.controller == "smoke" and (args.adapter or args.adapter_revision):
        parser.error("Smoke controls must not be labelled with an adapter")
    controller = ModelController(args) if args.controller == "model" else SmokeController(args.seed)
    if args.game == "doom":
        from doom_env import DoomGame
        environment = DoomGame
    else:
        from tetris_env import TetrisGame
        environment = TetrisGame
    args.output.mkdir(parents=True, exist_ok=False)
    game = environment(args.seed, args.output, max_steps=args.max_steps)
    try:
        report = record_episode(game, controller, args.output, seed=args.seed, max_steps=args.max_steps)
    finally:
        game.close()
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
