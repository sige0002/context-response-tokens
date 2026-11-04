import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch
from transformers import SiglipImageProcessor

from kubric_data import KubricPointClips, load_points, save_points


def test_float32_storage_is_bit_exact_and_does_not_invert_occlusion(tmp_path):
    rng = np.random.default_rng(52)
    points = (rng.normal(size=(19, 12, 2)) * 700).astype(np.float32)
    points[0, 0] = [-0., np.nextafter(np.float32(0), np.float32(1))]
    occluded = rng.random((19, 12)) < .3
    path = tmp_path / "points.npz"
    save_points(path, points, occluded)
    restored, mask = load_points(path)
    assert np.array_equal(restored.view(np.uint8), points.view(np.uint8))
    assert np.array_equal(mask, occluded)
    points[0, 0, 0] = np.nan
    occluded[0, 0] = False
    with pytest.raises(ValueError, match="upstream-visible"):
        save_points(path, points, occluded)
    occluded[0, 0] = True
    points[1, 3] = [-np.inf,np.inf]
    occluded[1, 3] = True
    save_points(path,points,occluded)
    restored, mask = load_points(path)
    assert np.array_equal(restored.view(np.uint8),points.view(np.uint8))
    assert np.array_equal(mask,occluded)
    # A corrupted visibility mask cannot turn undefined coordinates into a
    # query/supervision target merely by bypassing save_points.
    with np.load(path) as saved:
        raw=saved['points_bytes'].copy()
    occluded[1,3]=False
    np.savez_compressed(path,format_version=np.array(1),points_bytes=raw,occluded=occluded)
    with pytest.raises(ValueError,match="upstream-visible"):
        load_points(path)


def make_scene(tmp_path):
    directory = tmp_path / "scene"
    directory.mkdir()
    points = np.full((8, 6, 2), [128., 256.], dtype=np.float32)
    occluded = np.zeros((8, 6), dtype=bool)
    occluded[0, :2] = True
    points[0, 3] = [-np.inf,np.inf]
    occluded[0, 3] = True
    points[1, :3, 0] = -3.  # Outside the image, even if upstream mask says visible.
    occluded[-1, :-1] = True  # Only visible on the last frame: cannot supervise.
    occluded[-2] = True
    save_points(directory / "points.npz", points, occluded)
    with zipfile.ZipFile(directory / "images.zip", "w") as archive:
        for t in range(6):
            image = Image.fromarray(np.full((224, 224, 3), t * 30, dtype=np.uint8))
            encoded = io.BytesIO()
            image.save(encoded, format="PNG")
            archive.writestr(f"{t:03d}.png", encoded.getvalue())
    manifest = tmp_path / "clips.jsonl"
    manifest.write_text(json.dumps(dict(id="scene", directory="scene", frames=6, native_height=512, native_width=512,
        files={name:dict(bytes=(directory / name).stat().st_size) for name in ("images.zip", "points.npz")})) + "\n")
    return manifest


def test_visible_queries_normalization_and_exact_frame_alignment(tmp_path):
    path = make_scene(tmp_path)
    processor = SiglipImageProcessor(size={"height": 224, "width": 224})
    dataset = KubricPointClips(path, processor, frames=6, queries=6)
    sample, info = dataset.read(0)
    assert set(info["track_indices"]) == set(range(6))
    assert info["indices"] == list(range(6))
    for index, track in enumerate(info["track_indices"]):
        expected_first = 2 if track == 0 else 3 if track == 1 else 0
        assert int(sample["query_frames"][index]) == expected_first
        assert not sample["occluded"][index, expected_first]
        torch.testing.assert_close(sample["query_xy"][index], torch.tensor([.25, .5]))
        torch.testing.assert_close(sample["query_xy"][index], sample["target_xy"][index, expected_first])
    expected = torch.arange(6).float() * 30 / 127.5 - 1
    torch.testing.assert_close(sample["video"].mean((1, 2, 3)), expected)
    # The label field contains True for occlusion, never visibility.
    assert sample["occluded"][info["track_indices"].index(0), :2].all()
    assert sample["occluded"][info["track_indices"].index(1), :3].all()
    bad=info["track_indices"].index(0)
    assert torch.isinf(sample["target_xy"][bad,3]).all() and sample["occluded"][bad,3]
    assert torch.isfinite(sample["query_xy"]).all()
    repeated, repeated_info = dataset.read(0)
    assert info == repeated_info
    assert all(torch.equal(value, repeated[key]) for key, value in sample.items())
    dataset.set_epoch(3)
    assert dataset.read(0)[1] == info  # Validation is independent of epoch.
    with pytest.raises(ValueError, match="Too few"):
        KubricPointClips(path, processor, frames=6, queries=7).read(0)


def test_training_sampling_is_epoch_deterministic_and_forward_only(tmp_path):
    path = make_scene(tmp_path)
    dataset = KubricPointClips(path, SiglipImageProcessor(size={"height":224,"width":224}),
                               frames=3, queries=2, training=True)
    first, info = dataset.read(0)
    repeated, repeated_info = dataset.read(0)
    assert info == repeated_info
    assert all(torch.equal(value, repeated[key]) for key, value in first.items())
    variants = set()
    for epoch in range(5):
        dataset.set_epoch(epoch)
        sample, meta = dataset.read(0)
        assert all(a < b for a, b in zip(meta["indices"], meta["indices"][1:]))
        assert sample["query_frames"].max() < 2
        variants.add((tuple(meta["indices"]), tuple(meta["track_indices"])))
    assert len(variants) > 1


def test_occluded_nonfinite_targets_do_not_change_visible_loss_or_official_metrics(tmp_path):
    from point_readout import point_tracking_loss
    from tapvid_metrics import compute_tapvid_metrics
    path=make_scene(tmp_path)
    sample,_=KubricPointClips(path,SiglipImageProcessor(size={"height":224,"width":224}),frames=6,queries=6).read(0)
    target=sample['target_xy'][None]
    occluded=sample['occluded'][None]
    first=sample['query_frames'][None]
    points=torch.full_like(target,.4,requires_grad=True)
    logits=torch.zeros_like(occluded,dtype=torch.float32,requires_grad=True)
    output=dict(points=points,visibility_logits=logits)
    loss=point_tracking_loss(output,target,occluded,first)['loss']
    finite_target=torch.where(torch.isfinite(target),target,torch.zeros_like(target))
    expected=point_tracking_loss(output,finite_target,occluded,first)['loss']
    torch.testing.assert_close(loss,expected,atol=0,rtol=0)
    loss.backward()
    assert torch.isfinite(points.grad).all() and torch.isfinite(logits.grad).all()
    xy=sample['query_xy'].numpy()
    queries=np.stack((first.numpy()[0],xy[:,1]*256,xy[:,0]*256),-1)[None]
    original=compute_tapvid_metrics(queries,occluded.numpy(),target.numpy()*256,
        (logits.detach().numpy()<0),points.detach().numpy()*256,query_mode='first')
    replaced=compute_tapvid_metrics(queries,occluded.numpy(),finite_target.numpy()*256,
        (logits.detach().numpy()<0),points.detach().numpy()*256,query_mode='first')
    for name in original:
        assert np.isfinite(original[name]).all()
        np.testing.assert_array_equal(original[name],replaced[name])
