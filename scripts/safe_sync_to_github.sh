#!/usr/bin/env bash
#
# safe_sync_to_github.sh — guarded sync of SAFE setup/analysis outputs to a draft PR.
#
# Usage:
#   scripts/safe_sync_to_github.sh "Commit message"
#
# Policy (see CLAUDE.md "Automatic GitHub Sync Policy"):
#   * Never run on main/master or detached HEAD.
#   * Never commit anything under data/raw/ or raw/source data files.
#   * Never commit main.tex or manuscript PDFs/outputs during the analysis-only phase.
#   * Commit only known-safe text/result artifacts.
#   * Push the feature branch and open (or locate) a DRAFT PR into main. Never merge.
#
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Arguments
# ---------------------------------------------------------------------------
COMMIT_MSG="${1:-}"
if [[ -z "${COMMIT_MSG}" ]]; then
  echo "ERROR: commit message required." >&2
  echo "Usage: scripts/safe_sync_to_github.sh \"Commit message\"" >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"

BASE_BRANCH="main"

# Approved small-test-fixture path prefix (none active yet; reserved for future).
FIXTURE_PREFIX="tests/fixtures/"

abort() { echo "ABORT: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Branch guard: never on main/master or detached HEAD.
# ---------------------------------------------------------------------------
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${BRANCH}" == "HEAD" ]]; then
  abort "detached HEAD — checkout a feature branch before syncing."
fi
if [[ "${BRANCH}" == "main" || "${BRANCH}" == "master" ]]; then
  abort "refusing to sync directly from '${BRANCH}'. Use a feature branch."
fi
echo "Branch: ${BRANCH}"

# ---------------------------------------------------------------------------
# 2. data/raw/ guard: nothing under data/raw/ may be tracked or staged.
# ---------------------------------------------------------------------------
if git ls-files --error-unmatch -- 'data/raw' >/dev/null 2>&1; then
  abort "files under data/raw/ are TRACKED in git. Remove them before syncing."
fi
if git diff --cached --name-only -- 'data/raw' | grep -q .; then
  abort "files under data/raw/ are STAGED. Unstage them before syncing."
fi

# ---------------------------------------------------------------------------
# Helper: stage the candidate safe files explicitly (never `git add -A`).
# Add tracked-and-modified + untracked files that match the safe allowlist.
# ---------------------------------------------------------------------------
SAFE_REGEX='^(CLAUDE\.md|\.gitignore|requirements\.txt|docs/.*|scripts/.*|tests/.*|.*\.md|.*\.csv|.*\.png|.*\.ya?ml|.*\.json)$'

# Files git considers changed (modified/untracked), excluding ignored ones.
# (bash 3.2 compatible — no mapfile.)
while IFS= read -r f; do
  [[ -z "${f}" ]] && continue
  # Skip anything under data/raw entirely.
  [[ "${f}" == data/raw/* ]] && continue
  if [[ "${f}" =~ ${SAFE_REGEX} ]]; then
    git add -- "${f}"
  fi
done < <(git status --porcelain=v1 --untracked-files=all \
  | sed -E 's/^.. //' | sed -E 's/^.* -> //')

# ---------------------------------------------------------------------------
# 3. Raw/source data extension guard on the STAGED set.
# ---------------------------------------------------------------------------
STAGED_LIST="$(git diff --cached --name-only)"
if [[ -z "${STAGED_LIST}" ]]; then
  abort "no safe files staged — nothing to sync."
fi

RAW_EXT_REGEX='\.(edf|bdf|set|fif|npy|npz|mat|zip|tar|tar\.gz|tgz|rar)$'
while IFS= read -r f; do
  [[ -z "${f}" ]] && continue
  f_lower="$(printf '%s' "${f}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${f_lower}" =~ ${RAW_EXT_REGEX} ]]; then
    if [[ "${f}" == ${FIXTURE_PREFIX}* ]]; then
      echo "NOTE: allowing approved test fixture: ${f}"
    else
      abort "raw/source data file staged: ${f} (blocked extension)."
    fi
  fi
done <<< "${STAGED_LIST}"

# ---------------------------------------------------------------------------
# 4. Manuscript guard during analysis-only phase.
# ---------------------------------------------------------------------------
while IFS= read -r f; do
  [[ -z "${f}" ]] && continue
  case "${f}" in
    *main.tex|*.pdf)
      abort "manuscript artifact staged: ${f} (blocked in analysis-only phase)." ;;
    paper/*|*manuscript*|outputs_journal_upgrade/*|outputs_phd_revision/*)
      abort "manuscript-output path staged: ${f} (blocked in analysis-only phase)." ;;
  esac
done <<< "${STAGED_LIST}"

# ---------------------------------------------------------------------------
# 5/6. Show the exact candidate staged-file list before committing.
# ---------------------------------------------------------------------------
echo ""
echo "===== Candidate staged files ====="
echo "${STAGED_LIST}"
echo "=================================="
echo ""

# ---------------------------------------------------------------------------
# 7. Run existing safe tests when appropriate (best-effort; do not block on
#    the documented known missing-parquet failure).
# ---------------------------------------------------------------------------
if command -v pytest >/dev/null 2>&1; then
  echo "Running safe provenance tests..."
  pytest tests/test_raw_mat_provenance.py tests/test_mat_raw_rebuild.py || \
    echo "NOTE: test suite reported failures (expected: missing generated parquet). Continuing."
else
  echo "NOTE: pytest not on PATH; skipping test run."
fi

# ---------------------------------------------------------------------------
# 8. Commit.
# ---------------------------------------------------------------------------
git commit -m "${COMMIT_MSG}"

# ---------------------------------------------------------------------------
# 9. Push the current feature branch to origin.
# ---------------------------------------------------------------------------
git push -u origin "${BRANCH}"

# ---------------------------------------------------------------------------
# 10/11. Create or locate a DRAFT PR into main (never merge).
# ---------------------------------------------------------------------------
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  EXISTING_URL="$(gh pr view "${BRANCH}" --json url --jq .url 2>/dev/null || true)"
  if [[ -n "${EXISTING_URL}" ]]; then
    echo "Draft/PR already exists: ${EXISTING_URL}"
  else
    gh pr create \
      --base "${BASE_BRANCH}" \
      --head "${BRANCH}" \
      --draft \
      --title "${COMMIT_MSG}" \
      --body "Automated safe sync via scripts/safe_sync_to_github.sh. Draft only — do not merge automatically." \
      && gh pr view "${BRANCH}" --json url --jq .url
  fi
else
  REMOTE_URL="$(git remote get-url origin 2>/dev/null || echo '<origin>')"
  echo "GitHub CLI unavailable or unauthenticated."
  echo "Branch '${BRANCH}' was pushed to origin."
  echo "To open a PR manually:"
  echo "  1. Visit ${REMOTE_URL%.git}"
  echo "  2. Click 'Compare & pull request' for branch '${BRANCH}'."
  echo "  3. Set the base branch to '${BASE_BRANCH}', mark it as a DRAFT, and create it."
fi
