"""Bounded, headless ViZDoom environment for recorded model play.

The environment exposes engine telemetry as text.  It does not infer anything
from pixels and it does not choose actions for the model.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any


GAME_ID = "doom"
ENGINE_VERSION = "1.3.1"
ENGINE_REVISION = "f771231811cd3be417f97230d009e0aa9d983ed6"
TICS_PER_STEP = 8

_OPTIONS = (
    {"id": "move_left", "description": "Strafe left; the target shifts right on screen."},
    {"id": "move_right", "description": "Strafe right; the target shifts left on screen."},
    {"id": "attack", "description": "Fire the weapon."},
    {"id": "idle", "description": "Hold position and do not fire."},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DoomGame:
    """A single seeded episode of ViZDoom's bundled ``basic`` scenario."""

    game_id = GAME_ID

    def __init__(self, seed: int, output_dir: Path, max_steps: int = 120) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if isinstance(output_dir, str):
            raise TypeError("output_dir must be a pathlib.Path")
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")

        try:
            import vizdoom as vzd
        except ImportError as exc:  # Keep importing this module dependency-free.
            raise RuntimeError(
                "Doom playback requires the game-only dependencies from "
                "game-demo/requirements.txt (ViZDoom 1.3.1)."
            ) from exc

        actual_version = importlib.metadata.version("vizdoom")
        if actual_version != ENGINE_VERSION:
            raise RuntimeError(
                f"ViZDoom {ENGINE_VERSION} is required; found {actual_version}."
            )

        self._vzd = vzd
        self.seed = int(seed)
        self.output_dir = output_dir
        self.max_steps = int(max_steps)
        self._frames_dir = self.output_dir / "frames"
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._step_count = 0
        self._frame_count = 0
        self._total_reward = 0.0
        self._terminated = False
        self._termination_reason: str | None = None
        self._last_screen: Any | None = None
        self._last_frame_is_preterminal = False
        self._telemetry: dict[str, int] = {}

        package_dir = Path(vzd.__file__).resolve().parent
        self._scenario_path = Path(vzd.scenarios_path) / "basic.wad"
        self._game_wad_path = package_dir / "freedoom2.wad"
        for asset in (self._scenario_path, self._game_wad_path):
            if not asset.is_file():
                raise RuntimeError(f"Required ViZDoom asset is missing: {asset.name}")

        self._asset_hashes = {
            "basic.wad": _sha256(self._scenario_path),
            "freedoom2.wad": _sha256(self._game_wad_path),
        }

        game = vzd.DoomGame()
        self._game = game
        try:
            game.set_doom_game_path(str(self._game_wad_path))
            game.set_doom_scenario_path(str(self._scenario_path))
            game.set_doom_config_path(str(self.output_dir / "vizdoom.ini"))
            game.set_doom_map("map01")
            game.set_seed(self.seed)
            game.set_mode(vzd.Mode.PLAYER)
            game.set_window_visible(False)
            game.set_sound_enabled(False)
            game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
            game.set_screen_format(vzd.ScreenFormat.RGB24)
            game.set_labels_buffer_enabled(True)
            game.set_available_buttons(
                [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK]
            )
            game.set_available_game_variables(
                [vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2, vzd.GameVariable.KILLCOUNT]
            )
            game.set_episode_start_time(14)
            game.set_episode_timeout(self.max_steps * TICS_PER_STEP + 1)
            game.set_living_reward(-1.0)
            game.set_doom_skill(5)
            game.init()
            game.new_episode()
            self._update_telemetry()
            self._cache_screen()
        except Exception:
            game.close()
            raise

    @property
    def metadata(self) -> dict[str, Any]:
        """Version and asset identity needed to reproduce this episode."""

        return {
            "game_id": GAME_ID,
            "engine": "ViZDoom",
            "engine_version": ENGINE_VERSION,
            "engine_revision": ENGINE_REVISION,
            "dependencies": {
                "numpy": importlib.metadata.version("numpy"),
                "Pillow": importlib.metadata.version("Pillow"),
            },
            "scenario": "basic/map01",
            "seed": self.seed,
            "max_steps": self.max_steps,
            "tics_per_step": TICS_PER_STEP,
            "assets": dict(self._asset_hashes),
            "asset_source": "ViZDoom 1.3.1 Python distribution",
            "content": "Freedoom 0.13 (bundled by ViZDoom; BSD-3-Clause)",
        }

    def _update_telemetry(self) -> None:
        """Capture declared engine variables while a state is available."""

        variables = {
            "health": self._vzd.GameVariable.HEALTH,
            "ammo": self._vzd.GameVariable.AMMO2,
            "kills": self._vzd.GameVariable.KILLCOUNT,
        }
        self._telemetry = {
            name: int(round(self._game.get_game_variable(variable)))
            for name, variable in variables.items()
        }

    def _cache_screen(self) -> None:
        if self._game.is_episode_finished():
            return
        state = self._game.get_state()
        if state is not None and state.screen_buffer is not None:
            self._last_screen = state.screen_buffer.copy()

    def _visible_labels(self) -> str:
        if self._game.is_episode_finished():
            return "none reported"
        state = self._game.get_state()
        labels = () if state is None or state.labels is None else state.labels
        descriptions: list[str] = []
        for label in labels:
            if str(label.object_category) != "Monster":
                continue
            width = int(label.width)
            center = int(label.x) + (width - 1) / 2
            direction = "left" if center < 128 else "right" if center > 192 else "center"
            name = str(label.object_name).replace("_", " ")
            left = int(label.x)
            right = left + max(width - 1, 0)
            descriptions.append(
                f"{name} x={left}..{right}, center={center:.0f} ({direction})"
            )
        return ", ".join(descriptions[:6]) if descriptions else "none reported"

    def observation(self) -> dict[str, Any]:
        """Return concise text derived only from ViZDoom telemetry."""

        if self._terminated:
            state_text = f"Episode ended: {self._termination_reason}."
        else:
            health = self._telemetry["health"]
            ammo = self._telemetry["ammo"]
            kills = self._telemetry["kills"]
            state_text = (
                f"Step {self._step_count}/{self.max_steps}. Health {health}; "
                f"ammo {ammo}; kills {kills}. Crosshair x=160. "
                f"Engine target boxes: {self._visible_labels()}."
            )
        return {
            "state": state_text,
            "question": (
                "Defeat the labeled target while conserving ammo: strafe until its "
                "x range covers the crosshair at x=160, then fire. Choose one input "
                "for the next 8 game tics."
            ),
            "options": [dict(option) for option in _OPTIONS],
        }

    def step(self, action_id: str) -> dict[str, Any]:
        if self._terminated:
            raise RuntimeError("Cannot step a terminated Doom episode")
        action_vectors = {
            "move_left": [1, 0, 0],
            "move_right": [0, 1, 0],
            "attack": [0, 0, 1],
            "idle": [0, 0, 0],
        }
        if action_id not in action_vectors:
            allowed = ", ".join(action_vectors)
            raise ValueError(f"Unknown Doom action {action_id!r}; expected one of: {allowed}")

        reward = float(self._game.make_action(action_vectors[action_id], TICS_PER_STEP))
        self._step_count += 1
        self._total_reward += reward
        engine_finished = self._game.is_episode_finished()
        bounded = self._step_count >= self.max_steps
        self._terminated = engine_finished or bounded

        if engine_finished:
            self._last_frame_is_preterminal = True
            if self._game.is_episode_timeout_reached():
                self._termination_reason = "engine_timeout"
            elif self._game.is_player_dead():
                self._termination_reason = "player_dead"
            else:
                # In basic.wad, the only remaining ACS terminal is target defeat.
                self._termination_reason = "target_killed"
        else:
            self._update_telemetry()
            self._cache_screen()
            if bounded:
                self._termination_reason = "max_steps"

        return {
            "reward": reward,
            "terminated": self._terminated,
            "summary": self.summary,
        }

    def frame(self) -> str:
        """Save the latest engine screen and return a path relative to output_dir."""

        if self._last_screen is None:
            raise RuntimeError("No ViZDoom screen buffer is available")
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "PNG recording requires Pillow from game-demo/requirements.txt."
            ) from exc

        relative = Path("frames") / f"{self._frame_count:06d}.png"
        Image.fromarray(self._last_screen).save(self.output_dir / relative)
        self._frame_count += 1
        return relative.as_posix()

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "game_id": GAME_ID,
            "seed": self.seed,
            "steps": self._step_count,
            "max_steps": self.max_steps,
            "total_reward": self._total_reward,
            "health": self._telemetry["health"],
            "ammo": self._telemetry["ammo"],
            "kills": self._telemetry["kills"],
            "terminated": self._terminated,
            "termination_reason": self._termination_reason,
            "last_frame_is_preterminal": self._last_frame_is_preterminal,
            "telemetry_snapshot": (
                "pre_terminal" if self._last_frame_is_preterminal else "current"
            ),
        }

    def close(self) -> None:
        game = getattr(self, "_game", None)
        if game is not None:
            game.close()

    def __enter__(self) -> "DoomGame":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
