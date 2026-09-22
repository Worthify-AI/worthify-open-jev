import importlib.util
import json
from pathlib import Path
import sys

import pytest


GAME_DIR = Path(__file__).resolve().parents[1] / "game-demo"
sys.path.insert(0, str(GAME_DIR))
spec = importlib.util.spec_from_file_location("game_record", GAME_DIR / "record.py")
record = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record)
from verify_replay import verify_replay


class TinyGame:
    game_id = "tetris"
    metadata = {"name": "test-only environment", "game_id": "tetris", "seed": 7}

    def __init__(self, root):
        self.root = root
        self.steps = 0

    @property
    def summary(self):
        return {"steps": self.steps}

    def frame(self):
        name = f"{self.steps}.png"
        (self.root / name).write_bytes(b"test-frame" + bytes([self.steps]))
        return name

    def observation(self):
        return {"state": "A test board", "question": "Choose an action", "options": [
            {"id": "left", "description": "Move left"},
            {"id": "right", "description": "Move right"},
        ]}

    def step(self, action):
        self.steps += 1
        return {"reward": 1.0, "terminated": self.steps == 2, "summary": self.summary}


def smoke(tmp_path):
    record.record_episode(TinyGame(tmp_path), record.SmokeController(7), tmp_path, seed=7, max_steps=4)
    return tmp_path / "episode.json"


def test_replay_stops_at_terminal_and_keeps_frames_aligned(tmp_path):
    path = smoke(tmp_path)
    report = verify_replay(path)
    assert report["decisions"] == 2
    assert report["frames"] == 3
    assert report["controller"] == "smoke"
    with pytest.raises(ValueError, match="cannot be published"):
        verify_replay(path, require_model=True)


def test_changed_frame_is_rejected(tmp_path):
    path = smoke(tmp_path)
    (tmp_path / "1.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_replay(path)


def test_reordered_options_cannot_relabel_the_recorded_model_action(tmp_path):
    path = smoke(tmp_path)
    episode = json.loads(path.read_text())
    episode["controller"] = {"kind": "model", "selection": "argmax", "model": {
        "source": record.BASE_MODEL, "revision": record.BASE_REVISION,
        "adapter": "test-fixture", "adapter_revision": "a" * 40, "adapter_sha256": "b" * 64,
    }}
    for decision in episode["decisions"]:
        decision.update(selected_option_id="right", probabilities=[0.25, 0.75],
                        option_logits=[0.0, 1.0986122886681098], inference_seconds=0.1,
                        prompt_sha256="c" * 64, prompt_version="direct-options-v1", input_tokens=20)
    # Swap descriptions/IDs without changing the probability positions.
    episode["decisions"][0]["options"].reverse()
    path.write_text(json.dumps(episode))
    with pytest.raises(ValueError, match="differs from the model argmax"):
        verify_replay(path)


def test_smoke_cannot_claim_model_probabilities(tmp_path):
    path = smoke(tmp_path)
    episode = json.loads(path.read_text())
    episode["decisions"][0]["probabilities"] = [0.5, 0.5]
    path.write_text(json.dumps(episode))
    with pytest.raises(ValueError, match="cannot claim model probabilities"):
        verify_replay(path)


def test_relative_asset_cannot_escape_run(tmp_path):
    from verify_replay import _asset
    with pytest.raises(ValueError, match="escapes"):
        _asset(tmp_path, "../private.png")


def test_recording_never_overwrites_existing_run(tmp_path):
    smoke(tmp_path)
    with pytest.raises(ValueError, match="must be new"):
        record.record_episode(TinyGame(tmp_path), record.SmokeController(7), tmp_path, seed=7, max_steps=4)


def test_bounded_environment_reports_step_limit(tmp_path):
    class BoundedGame(TinyGame):
        @property
        def summary(self):
            return {"steps": self.steps, "termination_reason": "max_steps"}

    record.record_episode(BoundedGame(tmp_path), record.SmokeController(7), tmp_path, seed=7, max_steps=4)
    episode = json.loads((tmp_path / "episode.json").read_text())
    assert episode["ended_reason"] == "step_limit"
    assert episode["decisions"][-1]["terminated"] is True


def test_replay_cannot_be_relabelled_as_another_seed(tmp_path):
    path = smoke(tmp_path)
    episode = json.loads(path.read_text())
    episode["seed"] = 19
    path.write_text(json.dumps(episode))
    with pytest.raises(ValueError, match="identity differs"):
        verify_replay(path)
