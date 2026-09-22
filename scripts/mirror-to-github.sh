#!/usr/bin/env bash
# Mirror the protected GitLab default branch or a release tag to GitHub.
# This script never uses --force, --mirror, or a deletion refspec.
set -euo pipefail

remote_url=${GITHUB_MIRROR_URL:?Set GITHUB_MIRROR_URL to the public GitHub SSH repository URL}
ref=${1:?Pass destination refs/heads/<default-branch> or refs/tags/v<version>}
source_sha=${CI_COMMIT_SHA:?CI_COMMIT_SHA is required for detached GitLab jobs}

if [[ "$ref" != refs/heads/* && ! "$ref" =~ ^refs/tags/v[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.]+)?$ ]]; then
  echo "Ref must be the default branch or a vMAJOR.MINOR.PATCH tag" >&2
  exit 2
fi

if [[ "$ref" == refs/heads/* && "${CI_DEFAULT_BRANCH:-}" != "${ref#refs/heads/}" ]]; then
  echo "Only the configured default branch may be mirrored" >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "$source_sha" ]]; then
  echo "Checkout does not match CI_COMMIT_SHA" >&2
  exit 2
fi
source_ref=$source_sha
if [[ "$ref" == refs/tags/* ]]; then
  if [[ "$(git rev-parse "$ref^{commit}")" != "$source_sha" ]]; then
    echo "Release tag does not identify this CI commit" >&2
    exit 2
  fi
  source_ref=$ref  # Preserve annotated tag objects.
fi
if [[ "${CI_COMMIT_REF_PROTECTED:-false}" != "true" ]]; then
  echo "Refusing to mirror an unprotected GitLab ref" >&2
  exit 2
fi

ssh_dir=$(mktemp -d)
trap 'rm -rf "$ssh_dir"' EXIT
install -m 700 -d "$ssh_dir"
install -m 600 "${GITHUB_DEPLOY_KEY:?Use a protected file variable}" "$ssh_dir/id_ed25519"
install -m 600 "${GITHUB_KNOWN_HOSTS:?Use a protected file variable}" "$ssh_dir/known_hosts"
export GIT_SSH_COMMAND="ssh -i $ssh_dir/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$ssh_dir/known_hosts -o StrictHostKeyChecking=yes"

# A plain refspec cannot delete or force-update remote refs.
git push "$remote_url" "$source_ref:$ref"
