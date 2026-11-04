"""Joint reporting must preserve checkpoint identity, mode and JSON semantics."""
import copy
import csv
import json
from pathlib import Path

import pytest

import summarize_six_models as summary
from point_checkpoint import file_hash
from point_experiment import read_json, write_json


def joint_fixture(tmp_path):
    video = dict(seed=42, runs=[])
    points = dict(runs={}, reused_controls={})
    costs = dict(measurements=[])
    for name in summary.MODEL_NAMES:
        checkpoint = name+"-video"
        video["runs"].append(dict(name=name, checkpoint_sha256=checkpoint, trainable_parameters=100,
                                 conditions={"full/stream":dict(accuracy=.6)}))
        for mode in ("probe", "adapt"):
            key = f"{name}_{mode}_seed42"
            if name == "none_interleaved" and mode == "adapt":
                points["reused_controls"][key] = dict(reference=f"{name}_probe_seed42")
                continue
            path = tmp_path / (key+".json")
            write_json(path, dict(checkpoint=dict(metadata=dict(config=dict(
                encoder_initialization=dict(checkpoint_sha256=checkpoint))))))
            points["runs"][key] = dict(mode=mode, report=str(path), trainable_parameters=50,
                checkpoint_sha256=key, metrics=dict(average_jaccard=.25 if mode == "probe" else .4))
        for selection in ("shortest", "median", "longest"):
            costs["measurements"].append(dict(model=name, video=dict(selection=selection),
                                              measurement=dict(parameter_count=123)))
    return video, points, costs


def test_joint_table_keeps_probe_adaptation_and_percentages_separate(tmp_path):
    video, points, costs = joint_fixture(tmp_path)
    rows = summary.joint_rows(video, points, costs)
    assert len(rows) == 6 and rows[0]["model"] == "none_interleaved"
    assert all(row["ssv2_stream_top1_percent"] == 60 and row["robotap_probe_aj_percent"] == 25 for row in rows)
    assert rows[0]["robotap_adapt_aj_percent"] == 25 and rows[0]["adapt_reuses_probe"]
    assert all(row["robotap_adapt_aj_percent"] == 40 and not row["adapt_reuses_probe"] for row in rows[1:])


def test_identical_scores_cannot_hide_different_initial_encoder(tmp_path):
    video, points, costs = joint_fixture(tmp_path)
    path = Path(points["runs"]["mamba_interleaved_probe_seed42"]["report"])
    record = read_json(path)
    record["checkpoint"]["metadata"]["config"]["encoder_initialization"]["checkpoint_sha256"] = "different"
    write_json(path, record)
    with pytest.raises(ValueError, match="different encoders"):
        summary.joint_rows(video, points, costs)


@pytest.mark.parametrize("mutation", ["missing_model", "missing_cost", "mode", "reuse"])
def test_rejects_partial_or_mixed_comparisons(tmp_path, mutation):
    video, points, costs = joint_fixture(tmp_path)
    if mutation == "missing_model":
        video["runs"].pop()
    elif mutation == "missing_cost":
        costs["measurements"].pop()
    elif mutation == "mode":
        points["runs"]["mamba_late_adapt_seed42"]["mode"] = "probe"
    else:
        points["reused_controls"]["gmmix_after_encoder_adapt_seed42"] = dict(reference="gmmix_after_encoder_probe_seed42")
    with pytest.raises(ValueError):
        summary.joint_rows(video, points, costs)


def test_unfinished_queue_is_rejected_before_any_score_is_accepted(tmp_path):
    write_json(tmp_path / "status.json", dict(status="waiting_for_reference"))
    with pytest.raises(ValueError, match="not complete"):
        summary.extend_video_summary("missing", "missing", tmp_path)
    with pytest.raises(ValueError, match="not complete"):
        summary.extend_point_summary("missing", tmp_path)


def test_history_recount_accepts_json_array_roundtrip_but_not_changed_values(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(); new.mkdir()
    # Synthetic records test serialization/joins only; native raw-label analysis
    # is covered by the saved 296-frame engineering regression.
    analyses, records, point_runs = {}, {}, {}
    for index,name in enumerate(summary.MODEL_NAMES):
        key = f"{name}_probe_seed42"
        path = tmp_path / f"analysis{index}.json"
        value = dict(scope="Full RoboTAP paired encoder-state intervention", videos=265,
            age_bins=((1,8),(9,None)), checkpoint_sha256=key, strata=dict(all_post_query={}))
        analyses[str(path)] = value
        write_json(path,value)
        records[key] = dict(analysis=str(path), analysis_sha256=file_hash(path))
        point_runs[key] = dict(mode="probe", checkpoint_sha256=key, report=str(path))
    additional = f"{summary.MULTIDEPTH}_probe_seed42"
    write_json(old / "status.json", dict(status="complete", results={k:v for k,v in records.items() if k != additional}))
    write_json(old / "plan.json", dict(dataset_extraction_sha256="data", periods=[1,24]))
    write_json(new / "plan.json", dict(extraction_sha256="data", periods=[1,24], data_root="fixture", probe=dict(name=additional)))
    write_json(new / "summary.json", dict(history=records[additional]))
    monkeypatch.setattr(summary, "analyze_history", lambda directory,baseline,root: analyses[baseline])
    result = summary.verify_histories(old,new,dict(runs=point_runs))
    assert len(result["runs"]) == 6
    analyses[records[additional]["analysis"]]["age_bins"] = ((1,9),(10,None))
    with pytest.raises(ValueError, match="differs from original"):
        summary.verify_histories(old,new,dict(runs=point_runs))


def test_table_artifacts_preserve_values_without_an_omnibus_score(tmp_path):
    video, points, costs = joint_fixture(tmp_path)
    joint = summary.joint_rows(video, points, costs)
    cost = [dict(model="mamba_interleaved", selection="longest", video_id="fixture", frames=1337, repeats=3,
        gpu_median_ms=12.34, pipeline_median_ms=15.67, max_encoder_state_mib=75., torch_peak_allocated_mib=500.,
        repeated_final_outputs_equal=True)]
    summary.write_tables(tmp_path,joint,cost)
    with (tmp_path / "recognition_transfer.csv").open() as stream:
        parsed = list(csv.DictReader(stream))
    assert len(parsed) == 6 and float(parsed[0]["robotap_adapt_aj_percent"]) == 25
    assert not any("overall" in key for key in parsed[0])
    assert "12.340" in (tmp_path / "COMPARISON.md").read_text()
