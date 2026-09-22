# Recorded Doom and falling-block play

These examples connect the same `state`, `question`, and supplied `options` API
to games. The classification adapter selects an action using its option logits;
the environment executes it and records the result. No game-specific training or
reinforcement learning is performed.

Actual trained-model recordings are pending the two-seed Gemma 4 12B run. The
checked-in replay index is intentionally empty until verified recordings exist.

## What the model sees

- **Doom:** ViZDoom 1.3.1's `basic` scenario with its bundled Freedoom graphics.
  The model receives text describing engine variables and object labels, then
  chooses left, right, fire, or idle. The engine waits during inference and each
  action advances eight game tics. This is a small aiming scenario, not a full
  Doom campaign. The screenshots are for human playback; the model does not see
  their pixels. Object labels are privileged engine information.
  ViZDoom can discard its state at termination; those endings explicitly retain
  and label the last available pre-terminal frame and telemetry.
- **Falling blocks:** an original Tetris-style placement game, with a 10×20 board,
  seven-bag pieces, collisions, line clearing, and scoring. The model receives an
  ASCII board and up to 16 legal straight-drop placements. Columns are sampled
  evenly per rotation without ranking their quality. This reduces the action
  space: it does not implement full real-time Tetris controls, wall kicks, or
  hold. An episode also ends if fewer than two placements remain, matching the
  API's minimum choice count.

The release recipe uses **seeds 7, 19, and 42 for both games**, at most 120
decisions each, and includes every run. Do not select a winning video and present
it as typical performance. These are illustrative episodes, not benchmark
claims. Bad moves, losses, and early termination remain in the playback.

## Record and verify

Use a separate environment so demo dependencies cannot alter a running trainer:

```bash
python -m venv .game-venv
.game-venv/bin/pip install -e '.[test,train]' -r game-demo/requirements.txt
```

For a published adapter, use the immutable commit from its release record:

```bash
CUDA_VISIBLE_DEVICES=<one-GPU-UUID> .game-venv/bin/python game-demo/record.py \
  --game doom --seed 7 --max-steps 120 \
  --adapter Worthify/worthify-jev-classification \
  --adapter-revision <full-Hub-commit> --cache-dir /path/to/model-cache \
  --output artifacts/doom-seed7
.game-venv/bin/python game-demo/verify_replay.py \
  artifacts/doom-seed7/episode.json --require-model
```

Use `--game tetris` for falling blocks. The base defaults to the project's frozen
Gemma 4 12B revision. The classification adapter must be downloaded or trained
first; the public repository above is a planned destination until release.
The loader requires exactly one visible CUDA GPU. Keep both training A100s free
of demo inference until training and evaluation finish.

Each create-only output contains `episode.json`, an append-only decision log,
frames, and their checksums. It records the exact base revision, adapter digest,
game environment, code revision and source digests, every offered action, chosen
action, logits, prompt hash, measured decision latency, rewards, and end state.
Verification checks action alignment, softmax/argmax, frame continuity, and hashes.
This protects against accidental inconsistency; a recording is not a signed
attestation of model execution.

For a local engine/plumbing test without a GPU:

```bash
.game-venv/bin/python game-demo/record.py --game tetris --controller smoke \
  --seed 7 --output artifacts/tetris-smoke
```

Smoke runs use seeded random controls. They contain no model probabilities and
are visibly marked **SMOKE TEST — NOT MODEL PLAY**. The public gallery builder
rejects them unless the explicit local-testing flag `--allow-smoke` is supplied.

## Browser and video playback

```bash
.game-venv/bin/python game-demo/build_gallery.py \
  --episode artifacts/doom-seed7/episode.json \
  --episode artifacts/tetris-seed7/episode.json \
  --output artifacts/game-playback --video
python -m http.server 8090 --directory artifacts/game-playback
```

Open `http://localhost:8090`. The gallery is static HTML/CSS/JavaScript, with play,
pause, scrub, speed controls, and the model's action scores alongside frames.
It needs no account token, live model server, or browser GPU. Serve it over HTTP;
opening `index.html` as a local file can block JSON fetches.

The optional `--video` export requires `ffmpeg` with libx264 and drawtext support.
It creates `runs/<game>-seed<seed>/playback.mp4`, labelled as recorded model play
or a smoke test. Videos show four decisions per second; browser speed is
adjustable. Neither is a real-time performance measurement. Raw option
probabilities are uncalibrated and conditional on the offered choices.

After release packaging, run the complete six-episode recipe:

```bash
CUDA_VISIBLE_DEVICES=<one-GPU-UUID> PYTHON=.game-venv/bin/python \
  bash game-demo/run-release-demos.sh /path/to/release-classification \
  /path/to/new-gameplay-output /path/to/model-cache
```

This verifies the packaged adapter first and builds a gallery only after all six
episodes pass verification. Model selection uses validation metrics from the
training project, never game outcomes.

## Reuse and attribution

ViZDoom supplies the Doom engine, scenario and renderer. We explicitly select
the bundled Freedoom WAD; no original commercial Doom data is loaded. Preserve
[the Freedoom notice and component notes](LICENSES/README.md) with replays.
Engine and WAD binaries remain external dependencies. Falling-block code and
drawings are original and use this repository's MIT license; it contains no
official Tetris artwork, music, or branding. Pillow renders the frames.

The existing `demo/` benchmark replay and browser-inference `webgpu-demo/` are
separate examples.
