#!/usr/bin/env bash
set -euo pipefail

PART="${1:-patch}"

if ! git diff --stat --exit-code; then
    echo "error: working tree has uncommitted changes; aborting." >&2
    exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "master" ] && [ "$BRANCH" != "main" ]; then
    echo "warning: releasing from branch '$BRANCH' (not master/main)"
fi

echo "==> bumpversion $PART"
bumpversion "$PART" --tag --verbose --commit

echo "==> push commit + tags"
git push
git push --tags

echo "==> build"
python -m build

echo "==> gh release"
TAG=$(git describe --tags --abbrev=0)
gh release create "$TAG" --generate-notes

echo "==> done: $TAG released"
