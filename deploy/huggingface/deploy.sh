#!/usr/bin/env bash
# Publish the current commit as a Hugging Face Space (free CPU hardware is enough).
#
#   1. Create the Space once at https://huggingface.co/new-space  (SDK: Docker, hardware: CPU basic)
#   2. Log in once:  pip install -U huggingface_hub && huggingface-cli login   (a "write" token)
#   3. From the repository root:  bash deploy/huggingface/deploy.sh <your-hf-username>/<space-name>
#
# Only files tracked by git are sent. The burn-scar weights are left out: the web app does not
# use them, and Spaces need Git LFS for files over 10 MB.
set -euo pipefail

SPACE="${1:?usage: deploy.sh <hf-username>/<space-name>}"
ROOT="$(git rev-parse --show-toplevel)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

git clone --depth 1 "${SPACE_GIT_URL:-https://huggingface.co/spaces/${SPACE}}" "$WORK/space"
find "$WORK/space" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
git -C "$ROOT" archive HEAD | tar -x -C "$WORK/space"
rm -rf "$WORK/space/models/burn_scars" "$WORK/space/deploy"
cp "$ROOT/deploy/huggingface/README.md" "$WORK/space/README.md"
# load the search models while the Space starts, so the first question is fast
printf '\nENV PARALI_WARMUP=1\n' >> "$WORK/space/Dockerfile"

cd "$WORK/space"
git add -A
git commit -q -m "Deploy $(git -C "$ROOT" rev-parse --short HEAD)"
git push
echo "Pushed. Build logs and the app: https://huggingface.co/spaces/${SPACE}"
