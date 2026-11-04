import copy
from pathlib import Path

import numpy as np
import pytest

from gmmix_experiment import SHARED_KEYS
from paired_metrics import paired_mean_interval
from point_experiment import write_json
from summarize_full_video import paired_correctness, summarize as summarize_video, verify_shared_protocol
from summarize_points import align_reports, compare_reports, length_description, read_completed


def test_paired_bootstrap_preserves_video_weights_and_duplicate_clusters():
    scores = [1.] * 100 + [0.]
    groups = ["same_rgb"] * 100 + ["other_rgb"]
    clustered = paired_mean_interval(scores, np.zeros(101), groups)
    independent = paired_mean_interval(scores, np.zeros(101), list(map(str, range(101))))
    assert clustered["a_mean"] == clustered["difference"] == 100/101
    assert clustered["clusters"] == 2 and clustered["examples"] == 101
    assert clustered["conditional_95_interval"] == [0., 1.]
    assert independent["conditional_95_interval"][0] > .9
    assert paired_mean_interval([.3,.6,.9], [.3,.6,.9], ["a","b","c"])["conditional_95_interval"] == [0.,0.]
    with pytest.raises(ValueError, match="two independent"):
        paired_mean_interval([.2], [.3], ["a"])
    with pytest.raises(ValueError, match="finite"):
        paired_mean_interval([.2,np.nan], [.3,.4], ["a","b"])


def test_five_way_video_comparability_allows_core_changes_but_rejects_shared_changes():
    ours = {key:key for key in SHARED_KEYS}
    ours["source_sha256"] = {"shared.py":"same", "ours.py":"ours"}
    prior = dict(ours, kind="gmmix", encoder_architecture={"layers":4},
                 source_sha256={"shared.py":"same", "prior.py":"baseline"})
    verify_shared_protocol([ours, prior])
    for key in ("model", "frames", "steps", "val_sha256", "readout_sha256", "spatial_precision", "reproducibility"):
        with pytest.raises(ValueError, match="protocol differs"):
            verify_shared_protocol([ours, dict(prior, **{key:"different"})])
    with pytest.raises(ValueError, match="Shared implementation"):
        verify_shared_protocol([ours, dict(prior, source_sha256={"shared.py":"changed"})])


def test_paired_video_predictions_use_labels_and_reject_misalignment():
    a = dict(ids=["a","b","c"], groups=["a","b","c"], labels=[0,1,1],
             predictions={("full","stream"):[0,1,1]})
    b = dict(a, predictions={("full","stream"):[1,1,1]})
    result = paired_correctness(a, b)
    assert result["a_only_correct"] == 1 and result["b_only_correct"] == 0
    assert result["difference"] == 1/3
    with pytest.raises(ValueError, match="different examples"):
        paired_correctness(a, dict(b, ids=["b","a","c"]))


def tracking_report():
    rows = []
    for name, group, frames, metric in (("a","rgb1",128,.2),("b","rgb1",128,.4),("c","rgb2",513,.8)):
        rows.append(dict(id=name, video_sha256=group, frames=frames, tracks=8,
                         metrics=dict(average_jaccard=metric, average_pts_within_thresh=metric, occlusion_accuracy=metric),
                         max_encoder_state_bytes=100, max_query_state_bytes=frames))
    return dict(rows=rows)


def test_tracking_pairs_annotations_clusters_duplicate_rgb_and_separates_state():
    a = tracking_report()
    b = copy.deepcopy(a)
    b["rows"].reverse()
    for row in b["rows"]:
        row["metrics"] = {key:value - .1 for key,value in row["metrics"].items()}
    interval = compare_reports(a, b)["average_jaccard"]
    assert interval["clusters"] == 2 and interval["examples"] == 3
    assert np.isclose(interval["difference"], .1)
    groups = length_description(a)
    assert [row["videos"] for row in groups] == [2,0,0,1]
    assert groups[0]["max_per_video_encoder_state"]["max_bytes"] == 100
    assert groups[-1]["max_per_video_query_state"]["max_bytes"] == 513
    b["rows"][0]["tracks"] = 9
    with pytest.raises(ValueError, match="example differs"):
        align_reports(a,b)


def test_pending_real_queues_cannot_be_reported_as_completed(tmp_path):
    # Refuse a waiting queue before reading any nonexistent result/checkpoint.
    video, baseline, point = (tmp_path / name for name in ("video", "baseline", "point"))
    for path in (video, baseline, point):
        path.mkdir()
        write_json(path / "plan.json", {})
        write_json(path / "status.json", {"status":"waiting_for_dependencies"})
    with pytest.raises(ValueError, match="must finish"):
        summarize_video(video,baseline)
    with pytest.raises(ValueError, match="not complete"):
        read_completed(point)
