"""Fixed-seed execution and reporting must not silently change the experiment."""
import copy
from pathlib import Path

import pytest

import replication_protocol as protocol
import replication_summary as summary
from point_checkpoint import file_hash
from point_experiment import write_json


def fixture_plan(tmp_path, monkeypatch):
    # This fixture exercises real command builders with synthetic manifests;
    # source/native qualification is separately checked against the live plan.
    names = ("diagnostics", "multi_points", "points", "video", "multi_video", "robotap")
    paths = {name:tmp_path/name for name in names}
    for path in paths.values():
        path.mkdir()
    write_json(paths["robotap"]/"extraction.json", {"fixture": True})
    config = dict(protocol.PROTOCOL, train="train.jsonl", val="val.jsonl", classes=174,
                  readout="attention", kind="mamba", placement="interleaved",
                  nested={"precision": "float32"})
    config.pop("queries")
    base = dict(config, frames=8, batch_size=4, accumulation=8)
    multi = dict(base,kind="gmmix_multidepth",placement="after_encoder")
    write_json(paths["video"]/"plan.json", dict(reference="seed42/main", reference_config=base,
        controls=[dict(kind=k,placement=p,run=f"seed42/{k}_{p}")
                  for k,p in (("attention","interleaved"),("mamba","late"))]))
    write_json(paths["multi_video"]/"plan.json", dict(run="seed42/multi", config=multi))
    points = [dict(mode="probe",encoder_config=dict(base,kind=k,placement=p))
              for k,p in (("mamba","interleaved"),("attention","interleaved"),("mamba","late"))]
    write_json(paths["points"]/"plan.json", dict(video_queue=str(paths["video"]),
        protocol=protocol.PROTOCOL,runs=points,data_root="fixed-kubric",data_plan={"split_seed":42},
        robotap_root=str(paths["robotap"]),robotap_extraction_sha256=file_hash(paths["robotap"]/"extraction.json")))
    write_json(paths["multi_points"]/"plan.json", dict(point_queue=str(paths["points"]),
        video_queue=str(paths["multi_video"]), protocol=protocol.PROTOCOL,
        runs=[dict(mode="probe",encoder_config=multi)]))
    write_json(paths["diagnostics"]/"plan.json", dict(point_queue=str(paths["multi_points"])))
    monkeypatch.setattr(protocol,"verify_diagnostic_plan",lambda _:None)
    monkeypatch.setattr(protocol,"verify_inputs",lambda _:None)
    return protocol.build_plan(paths["diagnostics"],tmp_path/"replication",sources=())


def test_replication_preserves_conditions_and_starts_fresh(tmp_path,monkeypatch):
    plan = fixture_plan(tmp_path,monkeypatch)
    protocol.verify_plan(plan)
    assert len(plan["runs"]) == 8
    assert plan["data_plan"] == {"split_seed":42}
    for row in plan["runs"]:
        original = plan["reference_runs"][row["model"]]["video_config"]
        assert {k for k in original if original[k] != row["video_config"][k]} == {"seed"}
        for command in (row["train_video"],row["train_point"]):
            assert command[command.index("--seed")+1] == str(row["seed"])
            assert command[command.index("--steps")+1] == "20000"
            assert "--resume" not in command and "--development-run" not in command
        assert "--checkpoint" not in row["train_video"]
        point = row["train_point"]
        assert point[point.index("--encoder-checkpoint")+1] == str(Path(row["video_run"])/"last.pt")
        assert point[point.index("--mode")+1] == "probe"
        assert point[point.index("--train")+1] == "fixed-kubric/train.jsonl"
    plan["runs"][0]["video_config"]["nested"]["precision"] = "float16"
    assert plan["reference_runs"][protocol.MODELS[0]]["video_config"]["nested"]["precision"] == "float32"
    with pytest.raises(ValueError,match="beyond seed"):
        protocol.verify_plan(plan)


@pytest.mark.parametrize("mutation",["duplicate","missing","point_budget","changed_dependency"])
def test_incomplete_or_drifted_replication_is_rejected(tmp_path,monkeypatch,mutation):
    plan = fixture_plan(tmp_path,monkeypatch)
    if mutation == "duplicate":
        plan["runs"][-1] = copy.deepcopy(plan["runs"][0])
    elif mutation == "missing":
        plan["runs"].pop()
    elif mutation == "point_budget":
        plan["runs"][0]["point_protocol"]["steps"] = 1000
    else:
        write_json(Path(plan["dependencies"]["video"])/"plan.json", {"changed":True})
    with pytest.raises(ValueError):
        protocol.verify_plan(plan)


def scores():
    return [dict(model=model,seed=seed,ssv2_stream_accuracy=.5+.1*i-.01*j,
                 robotap_probe_aj=.2+.05*i,history_full_minus_reset1_aj=-.05+.01*i)
            for j,model in enumerate(protocol.MODELS) for i,seed in enumerate((42,43,44))]


def test_seed_statistics_use_sample_sd_and_paired_differences():
    result = summary.aggregate(list(reversed(scores())))
    model = result["models"][protocol.MODELS[0]]["ssv2_stream_accuracy"]
    assert model["n_seeds"] == 3 and model["seeds"] == [42,43,44]
    assert model["mean_percentage_points"] == pytest.approx(60)
    assert model["sample_sd_percentage_points"] == pytest.approx(10)
    contrast = result["reference_minus_other"][protocol.MODELS[1]]["ssv2_stream_accuracy"]
    # Individual scores vary by 10 pp, but their same-seed difference is 1 pp.
    assert contrast["n_pairs"] == 3
    assert contrast["paired_differences_pp"] == pytest.approx([1,1,1])
    assert contrast["sample_sd_difference_pp"] == pytest.approx(0,abs=1e-12)


@pytest.mark.parametrize("mutation",["missing","duplicate","percent_as_fraction","nan"])
def test_bad_seed_scores_cannot_be_summarized(mutation):
    rows = scores()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0].copy()
    else:
        rows[0]["ssv2_stream_accuracy"] = 50 if mutation == "percent_as_fraction" else float("nan")
    with pytest.raises(ValueError):
        summary.aggregate(rows)


def test_seed42_report_loading_checks_weights_and_converts_percent(tmp_path):
    report = tmp_path/"report"
    report.mkdir()
    write_json(tmp_path/"plan.json",dict(paths=dict(diagnostics="fixed-chain")))
    joint,history = [],{}
    for model in protocol.MODELS:
        joint.append(dict(model=model,ssv2_stream_top1_percent=60,robotap_probe_aj_percent=25,
            video_checkpoint_sha256=model+"-video",point_probe_checkpoint_sha256=model+"-probe"))
        history[model+"_probe_seed42"] = dict(checkpoint_sha256=model+"-probe",strata=dict(
            all_post_query=dict(full_minus_reset=dict(reset_1=dict(average_jaccard=dict(difference=.03))))))
    write_json(report/"joint.json",dict(rows=joint))
    write_json(report/"history.json",dict(runs=history))

    def refresh():
        write_json(report/"manifest.json",dict(reports={name:file_hash(report/name) for name in ("joint.json","history.json")}))
        write_json(tmp_path/"status.json",dict(status="complete",report=str(report/"manifest.json"),
                                               report_sha256=file_hash(report/"manifest.json")))

    refresh()
    plan = dict(seed42_summary=str(tmp_path),seed42_summary_plan_sha256=file_hash(tmp_path/"plan.json"),
                dependencies=dict(diagnostics="fixed-chain"))
    rows,_ = summary.load_reference(plan)
    assert len(rows) == 4 and all(r["ssv2_stream_accuracy"] == .6 and r["robotap_probe_aj"] == .25 for r in rows)
    history[protocol.MODELS[0]+"_probe_seed42"]["checkpoint_sha256"] = "wrong-weights"
    write_json(report/"history.json",dict(runs=history))
    refresh()
    with pytest.raises(ValueError,match="different weights"):
        summary.load_reference(plan)
