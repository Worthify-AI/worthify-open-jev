import importlib.util
import random
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "game-demo" / "tetris_env.py"
SPEC = importlib.util.spec_from_file_location("tetris_env", MODULE_PATH)
assert SPEC and SPEC.loader
tetris_env = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tetris_env)
TetrisGame = tetris_env.TetrisGame


def _action_trace(game, limit=40):
    trace = []
    while not game.terminated and len(trace) < limit:
        action = game.observation()["options"][0]["id"]
        trace.append((game.current_piece, action, game.step(action)))
    return trace


def test_seed_is_deterministic(tmp_path):
    first = TetrisGame(17, tmp_path / "a", max_steps=40)
    second = TetrisGame(17, tmp_path / "b", max_steps=40)
    assert first.observation() == second.observation()
    assert _action_trace(first) == _action_trace(second)
    assert first.board == second.board


def test_piece_source_uses_seeded_seven_bags(tmp_path):
    game = TetrisGame(17, tmp_path)
    pieces = [game.current_piece] + [game._draw_piece() for _ in range(13)]
    assert set(pieces[:7]) == set(tetris_env.SHAPES)
    assert set(pieces[7:]) == set(tetris_env.SHAPES)


def test_options_match_collision_free_landing_cells(tmp_path):
    game = TetrisGame(3, tmp_path)
    observation = game.observation()
    assert 1 <= len(observation["options"]) <= 16
    for public in observation["options"]:
        option = game._options[public["id"]]
        assert all(game.board[y][x] == "." for x, y in option["cells"])
        assert all(f"(r{y + 1},c{x + 1})" in public["description"] for x, y in option["cells"])
        assert any(y == tetris_env.HEIGHT - 1 or game.board[y + 1][x] != "." for x, y in option["cells"])


def test_row_clear_and_score(tmp_path):
    game = TetrisGame(0, tmp_path)
    game.board[-1] = ["J"] * tetris_env.WIDTH
    game.board[-1][3:5] = [".", "."]
    game.current_piece = "O"
    game._refresh_options()
    matching = [
        option["id"]
        for option in game._options.values()
        if {(x, y) for x, y in option["cells"] if y == tetris_env.HEIGHT - 1} == {(3, 19), (4, 19)}
    ]
    assert matching
    result = game.step(matching[0])
    assert result["reward"] == 101
    assert game.lines == 1
    assert game.score == 100
    assert game.board[-1][3:5] == ["O", "O"]


def test_collision_excludes_blocked_columns(tmp_path):
    game = TetrisGame(1, tmp_path)
    game.board[0][0] = "Z"
    game.current_piece = "I"
    game._refresh_options()
    assert game._landing_y(tetris_env.SHAPES["I"][0], 0) is None
    assert game._landing_y(tetris_env.SHAPES["I"][0], 1) is not None
    for option in game._options.values():
        assert (0, 0) not in option["cells"]
        assert game._landing_y(option["shape"], min(x for x, _ in option["cells"])) is not None


def test_terminal_state_has_no_single_choice(tmp_path):
    game = TetrisGame(9, tmp_path)
    game.board = [["J"] * tetris_env.WIDTH for _ in range(tetris_env.HEIGHT)]
    game.board[0][0:4] = ["."] * 4
    game.current_piece = "I"
    game._refresh_options()
    assert len(game._options) == 1
    game._terminate_if_no_choice()
    assert game.terminated
    assert game.termination_reason == "fewer_than_two_legal_choices"
    assert game.observation()["options"] == []


def test_random_action_smoke_and_frame(tmp_path):
    game = TetrisGame(29, tmp_path, max_steps=220)
    chooser = random.Random(99)
    for _ in range(220):
        if game.terminated:
            break
        options = game.observation()["options"]
        assert options
        result = game.step(chooser.choice(options)["id"])
        assert set(result) == {"reward", "terminated", "summary"}
    assert game.steps > 10
    assert game.summary["terminated"] is game.terminated
    pytest.importorskip("PIL", reason="PNG rendering uses optional game-demo dependencies")
    filename = game.frame()
    assert filename.startswith("frames/")
    assert not Path(filename).is_absolute()
    assert (tmp_path / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_rejects_invalid_or_late_actions(tmp_path):
    game = TetrisGame(4, tmp_path, max_steps=1)
    with pytest.raises(ValueError):
        game.step("not-an-option")
    game.step(game.observation()["options"][0]["id"])
    assert game.observation()["options"] == []
    with pytest.raises(RuntimeError):
        game.step("anything")
