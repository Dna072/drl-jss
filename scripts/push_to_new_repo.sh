#!/usr/bin/env bash
# Push the improved industrial DRL codebase to a brand-new empty GitHub repo.
#
# Prerequisites:
#   1. Create an empty GitHub repository (no README / .gitignore / license).
#   2. Have `gh` authenticated as an account with write access, OR set REMOTE_URL
#      to an HTTPS URL that includes a PAT with `repo` scope.
#
# Usage:
#   ./scripts/push_to_new_repo.sh <owner/new-repo-name>
#   ./scripts/push_to_new_repo.sh Dna072/industrial-drl-job-shop-scheduling
#
# Optional:
#   REMOTE_URL=https://<token>@github.com/OWNER/REPO.git ./scripts/push_to_new_repo.sh OWNER/REPO

set -euo pipefail

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
  echo "Usage: $0 <owner/repo>"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse --short HEAD)"
REMOTE_URL="${REMOTE_URL:-}"

if [[ -z "${REMOTE_URL}" ]]; then
  REMOTE_URL="https://github.com/${TARGET}.git"
fi

echo "Pushing ${BRANCH} @ ${COMMIT} -> ${TARGET}"
echo "Remote: ${REMOTE_URL}"

if git remote get-url new-origin >/dev/null 2>&1; then
  git remote set-url new-origin "${REMOTE_URL}"
else
  git remote add new-origin "${REMOTE_URL}"
fi

# Publish current improved tip as master and main for compatibility.
git push -u new-origin "HEAD:master"
git push new-origin "HEAD:main"

# Also push the feature branch name for traceability.
git push new-origin "HEAD:refs/heads/${BRANCH}"

echo
echo "Done."
echo "Repo: https://github.com/${TARGET}"
echo "Default branches updated: master, main"
