import json

import torch

from diagnose_video_history import history_view, sample_manifest


def test_history_interventions_preserve_current_frame_and_length_without_mutation():
    video = torch.arange(2 * 8 * 3 * 4 * 4).reshape(2, 8, 3, 4, 4)
    before = video.clone()
    current = history_view(video, "current")
    reverse = history_view(video, "reverse_history")
    assert torch.equal(video, before)
    assert current.shape == reverse.shape == video.shape
    for name in ("full", "current", "reverse_history"):
        assert torch.equal(history_view(video, name)[:, -1], video[:, -1])
    assert torch.equal(current[:, 0], video[:, -1])
    assert torch.equal(reverse[:, 0], video[:, -2])
    assert torch.equal(reverse[:, -2], video[:, 0])
    assert torch.equal(history_view(video[:, :1], "reverse_history"), video[:, :1])


def test_sampling_is_fixed_before_model_results_and_resolves_paths(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(dict(id=str(i), group=str(i), label=i % 3,
                                             video=f"videos/{i}.mp4")) + "\n" for i in range(30)))
    first = sample_manifest(source, 12, 17, tmp_path / "a.jsonl")
    second = sample_manifest(source, 12, 17, tmp_path / "b.jsonl")
    assert first == second and len({row["id"] for row in first}) == 12
    assert all(row["video"] == str(tmp_path / "videos" / f"{row['id']}.mp4") for row in first)
    assert first != sample_manifest(source, 12, 18, tmp_path / "c.jsonl")
