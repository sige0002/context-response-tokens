import json
import random

import av
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import SiglipImageProcessor

from train_temporal import check_split
from video_data import RawVideoClips, temporal_indices


def test_ordered_sampling_covers_short_and_long_clips():
    for length in (1, 2, 7, 8, 29, 100):
        for frames in (1, 6, 8, 16):
            for rng in (None, random.Random(91)):
                selected = temporal_indices(length, frames, rng)
                assert len(selected) == frames
                assert selected == sorted(selected)
                assert min(selected) >= 0 and max(selected) < length
                assert selected[-1] == length - 1
                # Every sample belongs to its assigned segment, even if the
                # segment is narrower than one decoded frame.
                for i, index in enumerate(selected[:-1]):
                    past_length = max(1, length - 1)
                    lo = i * past_length // (frames - 1)
                    hi = max(lo + 1, (i + 1) * past_length // (frames - 1))
                    assert lo <= index < hi
    with pytest.raises(ValueError):
        temporal_indices(0, 8)


def test_real_video_decode_sampling_and_worker_reproducibility(tmp_path):
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for t in range(24):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            pixels[:, :, 0] = t * 10
            pixels[:, :, 1] = np.arange(64)[None, :] * 3
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(dict(id="clip", group="source", video=path.name,
                                        label=1, bytes=path.stat().st_size)) + "\n")
    processor = SiglipImageProcessor(size={"height": 32, "width": 32})
    full = RawVideoClips(manifest, processor, classes=2)
    pixels, label, info = full.read(0)
    assert pixels.shape == (8, 3, 32, 32) and label == 1
    assert info["decoded_frames"] == 24
    assert pixels[:, 0].mean((1, 2)).tolist() == sorted(pixels[:, 0].mean((1, 2)).tolist())
    torch.testing.assert_close(pixels, full[0][0], atol=0, rtol=0)
    current = RawVideoClips(manifest, processor, classes=2, ablation="current")[0][0]
    reverse = RawVideoClips(manifest, processor, classes=2, ablation="reverse_history")[0][0]
    torch.testing.assert_close(current, pixels[-1:].expand_as(pixels), atol=0, rtol=0)
    torch.testing.assert_close(reverse[:-1], pixels[:-1].flip(0), atol=0, rtol=0)
    torch.testing.assert_close(reverse[-1], pixels[-1], atol=0, rtol=0)
    train = RawVideoClips(manifest, processor, classes=2, training=True)
    first, _, first_info = train.read(0)
    worker_values, _ = next(iter(DataLoader(train, batch_size=1, num_workers=1,
                                           multiprocessing_context="spawn", timeout=30)))
    torch.testing.assert_close(first, worker_values[0], atol=0, rtol=0)
    train.set_epoch(1)
    _, _, next_info = train.read(0)
    assert (first_info["indices"], first_info["crop"]) != (next_info["indices"], next_info["crop"])
    with pytest.raises(ValueError, match="overlap"):
        check_split(train, full)
