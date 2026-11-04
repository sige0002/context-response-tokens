#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
revision=e9594ce1c732d97440f0332fdc43170a2294dbfa
if [[ ! -d vendor/mamba/.git ]]; then
  git clone --no-checkout https://github.com/state-spaces/mamba.git vendor/mamba
  git -C vendor/mamba checkout --detach "$revision"
fi
[[ "$(git -C vendor/mamba rev-parse HEAD)" == "$revision" ]]
readout_revision=59249bf83311bc34bae277e2e8adec287ffe5d0f
if [[ ! -d vendor/frozen-video/.git ]]; then
  git clone --no-checkout https://github.com/tue-mps/towards-video-image-frozen.git vendor/frozen-video
  git -C vendor/frozen-video checkout --detach "$readout_revision"
fi
[[ "$(git -C vendor/frozen-video rev-parse HEAD)" == "$readout_revision" ]]
uv sync --locked
uv run --no-sync python fix_arm64_wheel_tag.py
uv pip check
