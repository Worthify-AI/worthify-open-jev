"""Deterministic, bounded falling-block environment for recorded model play.

This is an original, dependency-light implementation.  It deliberately exposes a
sampled placement action set rather than claiming to be a complete Tetris client.
"""

from __future__ import annotations

import random
from pathlib import Path


WIDTH = 10
HEIGHT = 20
_BASE_SHAPES = {
    "I": ((0, 0), (1, 0), (2, 0), (3, 0)),
    "O": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "T": ((0, 0), (1, 0), (2, 0), (1, 1)),
    "S": ((1, 0), (2, 0), (0, 1), (1, 1)),
    "Z": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "J": ((0, 0), (0, 1), (1, 1), (2, 1)),
    "L": ((2, 0), (0, 1), (1, 1), (2, 1)),
}
_LINE_SCORES = (0, 100, 300, 500, 800)
_COLORS = {
    "I": "#35c9d0",
    "O": "#e6c84a",
    "T": "#a868d2",
    "S": "#55b969",
    "Z": "#dd5b63",
    "J": "#5687d8",
    "L": "#e69645",
}


def _normalise(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return tuple(sorted(((x - min_x, y - min_y) for x, y in cells), key=lambda p: (p[1], p[0])))


def _rotations(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    result = []
    current = _normalise(cells)
    for _ in range(4):
        if current not in result:
            result.append(current)
        current = _normalise(tuple((-y, x) for x, y in current))
    return tuple(result)


SHAPES = {name: _rotations(cells) for name, cells in _BASE_SHAPES.items()}


class TetrisGame:
    """A placement-based falling-block game with deterministic seven-bag pieces."""

    game_id = "tetris"

    def __init__(self, seed: int, output_dir: Path, max_steps: int = 120):
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir = self.output_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self._bag: list[str] = []
        self.board = [["." for _ in range(WIDTH)] for _ in range(HEIGHT)]
        self.steps = 0
        self.score = 0
        self.lines = 0
        self.terminated = False
        self.termination_reason: str | None = None
        self.current_piece = self._draw_piece()
        self._options: dict[str, dict] = {}
        self._refresh_options()
        self._terminate_if_no_choice()

    def _draw_piece(self) -> str:
        if not self._bag:
            self._bag = list(_BASE_SHAPES)
            self._rng.shuffle(self._bag)
        return self._bag.pop()

    def _collides(self, shape: tuple[tuple[int, int], ...], x: int, y: int) -> bool:
        for dx, dy in shape:
            bx, by = x + dx, y + dy
            if bx < 0 or bx >= WIDTH or by >= HEIGHT:
                return True
            if by >= 0 and self.board[by][bx] != ".":
                return True
        return False

    def _landing_y(self, shape: tuple[tuple[int, int], ...], x: int) -> int | None:
        if self._collides(shape, x, 0):
            return None
        y = 0
        while not self._collides(shape, x, y + 1):
            y += 1
        return y

    @staticmethod
    def _sample_evenly(values: list[int], count: int) -> list[int]:
        if len(values) <= count:
            return values
        if count == 1:
            return [values[len(values) // 2]]
        indexes = [round(i * (len(values) - 1) / (count - 1)) for i in range(count)]
        return [values[i] for i in indexes]

    def _refresh_options(self) -> None:
        self._options = {}
        rotations = SHAPES[self.current_piece]
        # Four columns per rotation yields 8 or 16 choices for most pieces.  The
        # square has one unique rotation, so sample eight columns to retain choice.
        per_rotation = 8 if len(rotations) == 1 else 4
        if len(rotations) == 4:
            per_rotation = 4
        for rotation, shape in enumerate(rotations):
            shape_width = max(x for x, _ in shape) + 1
            legal = [x for x in range(WIDTH - shape_width + 1) if self._landing_y(shape, x) is not None]
            for x in self._sample_evenly(legal, per_rotation):
                y = self._landing_y(shape, x)
                assert y is not None
                cells = tuple(sorted(((x + dx, y + dy) for dx, dy in shape), key=lambda p: (p[1], p[0])))
                action_id = f"r{rotation}-c{x}"
                cell_text = ", ".join(f"(r{cy + 1},c{cx + 1})" for cx, cy in cells)
                self._options[action_id] = {
                    "id": action_id,
                    "description": (
                        f"Place {self.current_piece}, rotation {rotation * 90} degrees, "
                        f"left edge column {x + 1}; landing cells: {cell_text}."
                    ),
                    "shape": shape,
                    "cells": cells,
                }

    def _terminate_if_no_choice(self) -> None:
        if len(self._options) < 2:
            self.terminated = True
            self.termination_reason = "fewer_than_two_legal_choices"
            self._options = {}

    def _board_text(self) -> str:
        border = "+" + "-" * WIDTH + "+"
        rows = ["|" + "".join(row) + "|" for row in self.board]
        return "\n".join([border, *rows, border])

    def observation(self) -> dict:
        options = [
            {"id": option["id"], "description": option["description"]}
            for option in self._options.values()
        ]
        state = (
            f"Piece: {self.current_piece} | score: {self.score} | lines: {self.lines} "
            f"| placements: {self.steps}/{self.max_steps}\n"
            "Coordinates are 1-based from the top-left: row increases downward, column increases right.\n"
            f"{self._board_text()}"
        )
        return {
            "state": state,
            "question": (
                "Choose a placement to maximize cleared lines and avoid reaching the top. "
                "The piece drops straight down and locks immediately."
            ),
            "options": options,
        }

    def _clear_rows(self) -> int:
        remaining = [row for row in self.board if "." in row]
        cleared = HEIGHT - len(remaining)
        self.board = [["." for _ in range(WIDTH)] for _ in range(cleared)] + remaining
        return cleared

    def step(self, action_id: str) -> dict:
        if self.terminated:
            raise RuntimeError("game has terminated")
        option = self._options.get(action_id)
        if option is None:
            raise ValueError(f"unknown or unavailable action: {action_id!r}")
        for x, y in option["cells"]:
            self.board[y][x] = self.current_piece
        cleared = self._clear_rows()
        score_delta = _LINE_SCORES[cleared]
        self.score += score_delta
        self.lines += cleared
        self.steps += 1
        reward = 1 + score_delta

        if self.steps >= self.max_steps:
            self.terminated = True
            self.termination_reason = "step_limit"
            self._options = {}
        else:
            self.current_piece = self._draw_piece()
            self._refresh_options()
            self._terminate_if_no_choice()
        return {"reward": reward, "terminated": self.terminated, "summary": self.summary}

    @property
    def summary(self) -> dict:
        occupied = [(x, y) for y, row in enumerate(self.board) for x, cell in enumerate(row) if cell != "."]
        height = 0 if not occupied else HEIGHT - min(y for _, y in occupied)
        holes = 0
        for x in range(WIDTH):
            seen_block = False
            for y in range(HEIGHT):
                if self.board[y][x] != ".":
                    seen_block = True
                elif seen_block:
                    holes += 1
        return {
            "score": self.score,
            "lines_cleared": self.lines,
            "pieces_placed": self.steps,
            "board_height": height,
            "holes": holes,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
        }

    @property
    def metadata(self) -> dict:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "board": {"width": WIDTH, "height": HEIGHT},
            "piece_randomizer": "seeded seven-bag",
            "action_space": "sampled straight-drop placements (8-16 when available)",
        }

    def frame(self) -> str:
        from PIL import Image, ImageDraw, ImageFont

        cell = 24
        margin = 20
        panel = 180
        width = margin * 3 + WIDTH * cell + panel
        height = margin * 2 + HEIGHT * cell
        image = Image.new("RGB", (width, height), "#151820")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        board_x, board_y = margin, margin
        draw.rectangle(
            (board_x - 2, board_y - 2, board_x + WIDTH * cell + 1, board_y + HEIGHT * cell + 1),
            outline="#ccd4df",
            width=2,
        )
        for y, row in enumerate(self.board):
            for x, value in enumerate(row):
                left, top = board_x + x * cell, board_y + y * cell
                draw.rectangle((left, top, left + cell - 1, top + cell - 1), fill="#242a35", outline="#343c49")
                if value != ".":
                    draw.rectangle((left + 2, top + 2, left + cell - 3, top + cell - 3), fill=_COLORS[value])

        text_x = margin * 2 + WIDTH * cell
        draw.text((text_x, margin), "FALLING BLOCKS", fill="#f2f5f8", font=font)
        draw.text((text_x, margin + 28), f"Current: {self.current_piece}", fill="#f2f5f8", font=font)
        shape = SHAPES[self.current_piece][0]
        for x, y in shape:
            left, top = text_x + x * cell, margin + 55 + y * cell
            draw.rectangle((left + 2, top + 2, left + cell - 3, top + cell - 3), fill=_COLORS[self.current_piece])
        draw.text((text_x, margin + 145), f"Score: {self.score}", fill="#d8dee8", font=font)
        draw.text((text_x, margin + 165), f"Lines: {self.lines}", fill="#d8dee8", font=font)
        draw.text((text_x, margin + 185), f"Pieces: {self.steps}", fill="#d8dee8", font=font)
        if self.terminated:
            draw.text((text_x, margin + 225), "GAME OVER", fill="#ff8b8b", font=font)

        filename = f"frames/{self.steps:06d}.png"
        image.save(self.output_dir / filename)
        return filename

    def close(self) -> None:
        """Release resources (the implementation holds no external resources)."""
