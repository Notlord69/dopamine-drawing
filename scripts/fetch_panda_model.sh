#!/usr/bin/env bash
# Fetch the Franka Emika Panda model from MuJoCo Menagerie (Apache-2.0, not redistributed here).
#
#   scripts/fetch_panda_model.sh [target_dir]      default: <repo>/mujoco_menagerie
#
# dopamine_drawing/config.py looks in <repo>/mujoco_menagerie, or in $DD_MENAGERIE if that is set.
# Pinned to the commit closest to the copy the experiments ran with (2026-10-14). Its
# franka_emika_panda/ folder matches that copy except one whitespace-only difference in panda.xml.
set -euo pipefail
COMMIT=62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35
URL=https://github.com/google-deepmind/mujoco_menagerie.git
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-$REPO_ROOT/mujoco_menagerie}"

if [ -f "$TARGET/franka_emika_panda/scene.xml" ]; then
  echo "already there: $TARGET/franka_emika_panda"
  exit 0
fi
mkdir -p "$TARGET"
cd "$TARGET"
git init -q
git remote add origin "$URL" 2>/dev/null || true
git sparse-checkout set franka_emika_panda          # only the Panda (~36 MB), not all of Menagerie
git fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git checkout -q FETCH_HEAD
echo "Panda model at $TARGET/franka_emika_panda (menagerie $COMMIT)"
