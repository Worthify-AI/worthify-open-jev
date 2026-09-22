#!/usr/bin/env bash
# Validate the evidence bundle before a release tag can publish adapter files.
set -euo pipefail

tag=${1:-${CI_COMMIT_TAG:-}}
if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.]+)?$ ]]; then
  echo "A semantic vMAJOR.MINOR.PATCH release tag is required" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "refs/tags/$tag^{commit}")" ]]; then
  echo "Release checks must run on the exact tagged commit" >&2
  exit 2
fi
(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py

# The release notes must point to verified committed metrics.  This rejects a
# placeholder or an unverified tag before any external publication job runs.
if ! git show "$tag:results/phase1-summary.json" >/dev/null 2>&1; then
  echo "Release tag lacks the verified metrics summary" >&2
  exit 1
fi
python - "$tag" <<'PY'
import json
import pathlib
import sys

index = json.loads(pathlib.Path("release", sys.argv[1] + ".json").read_text())
if index.get("schema") != "openjev-phase1-release-index-v1" or index.get("state") != "verified":
    raise SystemExit("Release index is not verified; adapter publication is blocked")
PY
