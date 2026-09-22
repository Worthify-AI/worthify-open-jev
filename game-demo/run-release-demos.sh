#!/usr/bin/env bash
# Record every predeclared seed with the validation-selected classification adapter.
set -euo pipefail
cd "$(dirname "$0")/.."
task_adapter=${1:?Pass the validated classification release directory}
task_output=${2:?Pass a new output directory}
task_cache=${3:?Pass the base-model cache directory}
task_python=${PYTHON:-python}
test ! -e "$task_output"
test -s "$task_adapter/adapter_model.safetensors"
"$task_python" -m openjev_phase1.publish validate --artifact-dir "$task_adapter"
"$task_python" - "$task_adapter" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'game-demo')
from record import BASE_MODEL, BASE_REVISION
from build_gallery import preflight_video
preflight_video()
manifest = json.loads((Path(sys.argv[1]) / 'release-manifest.json').read_text())
if manifest.get('kind') != 'measured' or manifest.get('recipe') != 'classification':
    raise ValueError('Game release recordings require the measured classification adapter')
if manifest.get('base_model') != {'id': BASE_MODEL, 'revision': BASE_REVISION}:
    raise ValueError('Game release adapter must use the pinned Gemma 4 12B base')
PY
task_revision=$(sha256sum "$task_adapter/adapter_model.safetensors")
task_revision=${task_revision%% *}
mkdir -p "$task_output"
task_episodes=()
for task_game in doom tetris; do
  for task_seed in 7 19 42; do
    task_run="$task_output/$task_game-seed$task_seed"
    "$task_python" game-demo/record.py --game "$task_game" --seed "$task_seed" \
      --max-steps 120 --controller model --adapter "$task_adapter" \
      --adapter-revision "$task_revision" --cache-dir "$task_cache" --output "$task_run"
    "$task_python" game-demo/verify_replay.py "$task_run/episode.json" --require-model
    task_episodes+=(--episode "$task_run/episode.json")
  done
done
"$task_python" game-demo/build_gallery.py "${task_episodes[@]}" --output "$task_output/site" --video
date -u +%FT%TZ > "$task_output/completed.txt"
