"""Checkpoint diagnostic using upstream PyTorch scan instead of native CUDA.

Never used for benchmark training or timing. Only the scan implementation is
substituted, to test resume logic independently of native atomic reductions.
"""
import torch

from mamba_ssm.modules import mamba_simple
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

import train_multidepth_video


def reference_scan(*args, **kwargs):
    with torch.autocast("cuda", enabled=False):
        return selective_scan_ref(*args, **kwargs)


if __name__ == "__main__":
    original = mamba_simple.selective_scan_fn
    mamba_simple.selective_scan_fn = reference_scan
    try:
        train_multidepth_video.main()
    finally:
        mamba_simple.selective_scan_fn = original
