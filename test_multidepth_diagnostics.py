"""Preserve paired RGB selection and fixed-state accounting for the sixth model."""
import copy

import pytest

import run_multidepth_diagnostics as queue
from point_checkpoint import file_hash
from point_experiment import write_json


def test_plan_reuses_exact_original_video_selection_and_point_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "verify_point_plan", lambda plan: None)
    monkeypatch.setattr(queue, "verify_integration", lambda path: None)
    points, costs, integration = [tmp_path / name for name in ("points", "costs", "integration")]
    for path in (points, costs, integration):
        path.mkdir()
    original = str(tmp_path / "original_points")
    videos = [dict(id=name, frames=length, video_sha256=name, shard="fixture", selection=label)
              for name,length,label in (("a",52,"shortest"), ("b",245,"median"), ("c",1337,"longest"))]
    point = dict(point_queue=original, point_plan_sha256="original-points", robotap_root="robotap",
        robotap_extraction_sha256="data", runs=[dict(mode=mode, encoder_type="gmmix_multidepth", name=mode)
                                                for mode in ("probe", "adapt")])
    cost = dict(point_queue=original, point_plan_sha256="original-points", data_root="robotap",
        extraction_sha256="data", repeats=3, warmup_passes=1, videos=videos, selection="No scores used")
    write_json(points / "plan.json", point)
    write_json(costs / "plan.json", cost)
    write_json(integration / "verification.json", {})
    plan = queue.build_plan(points, costs, integration, tmp_path / "output")
    assert plan["videos"] == videos and plan["periods"] == [1,24]
    assert plan["probe"]["mode"] == "probe" and plan["repeats"] == 3
    assert plan["point_plan_sha256"] == file_hash(points / "plan.json")
    for field,value in (("repeats", 2), ("extraction_sha256", "other"), ("point_queue", "other")):
        write_json(costs / "plan.json", dict(cost, **{field:value}))
        with pytest.raises(ValueError, match="comparisons differ"):
            queue.build_plan(points, costs, integration, tmp_path / "output")


@pytest.mark.parametrize("failure", [None, "growing", "empty"])
def test_fixed_state_gate_covers_multidepth_kind(monkeypatch, failure):
    measurement = dict(max_state_bytes=96337920,
        bins=[dict(state_bytes_min=96337920, state_bytes_max=96337920)])
    broken = copy.deepcopy(measurement)
    if failure == "growing":
        broken["bins"][0]["state_bytes_min"] = 90000000
    elif failure == "empty":
        broken["max_state_bytes"] = 0
    monkeypatch.setattr(queue, "verify_measurement", lambda *args: broken)
    plan = dict(repeats=3, warmup_passes=1, extraction_sha256="fixture")
    if failure is None:
        assert queue.fixed_state_measurement("fixture", {}, {}, {}, plan) == measurement
    else:
        with pytest.raises(ValueError, match="state accounting"):
            queue.fixed_state_measurement("fixture", {}, {}, {}, plan)
