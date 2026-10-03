#!/usr/bin/env bash
# Push AMR-Q to a new GitHub repository.
# Prerequisite: run `gh auth login` once (or set GH_TOKEN), then run this script.
set -euo pipefail

REPO_NAME="${1:-AMR-Q}"
VISIBILITY="${2:-public}"   # public | private

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required: https://cli.github.com/"
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Not authenticated. Run:  gh auth login"
  exit 1
fi

git init -b main 2>/dev/null || true
git add -A
git commit -m "AMR-Q v1.0.0 — quantum-assisted antibiotic resistance screening platform" || true

USERNAME="$(gh api user -q .login)"
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/${USERNAME}/${REPO_NAME}.git"

gh repo create "${REPO_NAME}" --${VISIBILITY} --source . --remote origin --push \
  --description "Quantum-assisted antibiotic resistance fighter: VQE + ML screening of resistance-breaking antibiotic candidates"

echo ""
echo "Published: https://github.com/${USERNAME}/${REPO_NAME}"
