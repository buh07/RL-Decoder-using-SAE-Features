#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[bootstrap] repo: $REPO_ROOT"

git config --global --add safe.directory "$REPO_ROOT"

echo "[bootstrap] added safe.directory for this user"

# Keep credential handling user-local; do not write shared tokens.
if ! git config --global --get credential.helper >/dev/null 2>&1; then
  git config --global credential.helper cache
  echo "[bootstrap] set credential.helper=cache for this user"
fi

FETCH_OK=0
if GIT_TERMINAL_PROMPT=0 git ls-remote origin >/dev/null 2>&1; then
  FETCH_OK=1
fi

PUSH_OK=0
if GIT_TERMINAL_PROMPT=0 git push --dry-run origin HEAD >/dev/null 2>&1; then
  PUSH_OK=1
fi

echo "[bootstrap] fetch_check=$FETCH_OK push_check=$PUSH_OK"

if [[ "$FETCH_OK" -ne 1 ]]; then
  echo "[next] Fetch still blocked. Verify network/VPN and remote visibility." >&2
fi

if [[ "$PUSH_OK" -ne 1 ]]; then
  cat >&2 <<MSG
[next] Push is not authorized for this user yet.
Use one of:
  1) gh auth login
  2) git credential-manager-core configure
  3) Personal Access Token (classic or fine-grained) with repo write scope
Then re-run:
  git push --dry-run origin HEAD
MSG
fi

echo "[bootstrap] done"
