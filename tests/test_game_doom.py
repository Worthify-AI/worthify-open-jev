"""CPU-only smoke coverage for the optional Doom recording environment.

The fixed action below verifies engine integration.  It is not model gameplay
and must not be presented as an OpenJev decision trace.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


pytest.importorskip("vizdoom", reason="game-demo/requirements.txt is an optional env")
pytest.importorskip("PIL", reason="game-demo/requirements.txt is an optional env")


def _doom_class():
    module_path = Path(__file__).parents[1] / "game-demo" / "doom_env.py"
    spec = importlib.util.spec_from_file_location("openjev_doom_env", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DoomGame


def test_scripted_engine_smoke_not_model_gameplay(tmp_path: Path) -> None:
    DoomGame = _doom_class()

    with DoomGame(seed=17, output_dir=tmp_path, max_steps=1) as game:
        observation = game.observation()
        assert game.game_id == "doom"
        assert "DoomPlayer" not in observation["state"]
        assert "Crosshair x=160" in observation["state"]
        assert len(observation["options"]) == 4
        assert {option["id"] for option in observation["options"]} == {
            "move_left",
            "move_right",
            "attack",
            "idle",
        }
        assert game.metadata["engine_version"] == "1.3.1"
        assert len(game.metadata["assets"]["basic.wad"]) == 64

        initial_frame = game.frame()
        assert initial_frame == "frames/000000.png"
        assert not Path(initial_frame).is_absolute()
        assert (tmp_path / initial_frame).is_file()

        result = game.step("idle")
        assert result["terminated"] is True
        assert result["summary"]["termination_reason"] == "max_steps"
        assert result["summary"]["steps"] == 1

        final_frame = game.frame()
        assert final_frame == "frames/000001.png"
        assert (tmp_path / final_frame).is_file()


def test_output_directory_must_be_path(tmp_path: Path) -> None:
    DoomGame = _doom_class()
    with pytest.raises(TypeError, match="pathlib.Path"):
        DoomGame(seed=1, output_dir=str(tmp_path), max_steps=1)
