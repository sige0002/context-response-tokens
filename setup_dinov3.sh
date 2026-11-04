#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
dinov3_revision=6876159a11b4df116f30f667f8c9888617df0751
if [[ ! -d vendor/dinov3/.git ]]; then
  git clone --no-checkout https://github.com/facebookresearch/dinov3.git vendor/dinov3
  git -C vendor/dinov3 checkout --detach "$dinov3_revision"
fi
[[ "$(git -C vendor/dinov3 rev-parse HEAD)" == "$dinov3_revision" ]]
