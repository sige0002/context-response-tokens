"""Joint six-encoder report from completed, independently scored experiments.

Extends the original five-way report without changing its primary contrasts.
New multi-depth comparisons remain exploratory. No partial learning curve or
engineering fixture can replace a full fixed-budget experiment here.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from analyze_point_history import analyze as analyze_history
from encoder_costs import verify_measurement, paired_costs
from multidepth_experiment import verify_completion as verify_multidepth_video, verify_evaluation
from multidepth_point_experiment import verify_point_run, verify_dataset, verify_robotap
from point_checkpoint import file_hash
from point_experiment import read_json, write_json
from run_multidepth_comparison import verify_plan as verify_video_plan
from run_multidepth_points import verify_plan as verify_point_plan
from run_multidepth_diagnostics import verify_plan as verify_diagnostic_plan
from summarize_full_video import summarize as summarize_five_video, verify_shared_protocol, paired_correctness
from summarize_points import summarize as summarize_original_points, compare_reports, length_description, METRICS
from summarize_video import read_run


ROOT = Path(__file__).resolve().parent
MULTIDEPTH = "gmmix_multidepth_after_encoder"
MODEL_NAMES = ("none_interleaved", "attention_interleaved", "mamba_late", "mamba_interleaved",
               "gmmix_after_encoder", MULTIDEPTH)


def require_complete(path):
    if read_json(Path(path) / "status.json")["status"] != "complete":
        raise ValueError(f"Experiment is not complete: {path}")


def trainable_parameters(run):
    events = [json.loads(line) for line in (Path(run) / "progress.jsonl").read_text().splitlines() if line]
    counts = {row["trainable_parameters"] for row in events if row["event"] == "start"}
    if len(counts) != 1:
        raise ValueError("Missing or inconsistent training parameter count")
    return counts.pop()


def extend_video_summary(video_queue, gmmix_queue, multidepth_queue):
    video_queue, multidepth_queue = Path(video_queue).resolve(), Path(multidepth_queue).resolve()
    require_complete(multidepth_queue)
    plan = read_json(multidepth_queue / "plan.json")
    if Path(plan["video_queue"]) != video_queue or plan["video_plan_sha256"] != file_hash(video_queue / "plan.json"):
        raise ValueError("Multi-depth video comparison belongs to a different chain")
    verify_video_plan(plan)
    report = summarize_five_video(video_queue, gmmix_queue)
    original = read_json(video_queue / "plan.json")
    config = plan["config"]
    verify_shared_protocol([original["reference_config"], config])
    completed = verify_multidepth_video(plan["run"], config)
    run = Path(plan["run"])
    verify_evaluation(run / "final_clip_stream.json", config, completed, ["full"], ["clip", "stream"])
    verify_evaluation(run / "final_history_ablations.json", config, completed, ["current", "reverse_history"], ["clip"])
    values = read_run(run)
    reference = read_run(original["reference"])
    if len(values["ids"]) != 24777 or values["checkpoint_sha256"] != completed["checkpoint_sha256"]:
        raise ValueError("Additional video evaluation is partial or uses a different checkpoint")
    report["runs"].append(dict(name=MULTIDEPTH, path=str(run), checkpoint_sha256=completed["checkpoint_sha256"],
        evaluation_sha256=values["evaluation_sha256"], trainable_parameters=trainable_parameters(run),
        conditions={f"{ablation}/{mode}":dict(accuracy=metric["accuracy"], macro_recall=metric["macro_recall_present_classes"])
                    for (ablation,mode),metric in values["scores"].items()},
        clip_stream_prediction_disagreements=values["disagreements"],
        history_diagnostics={name:paired_correctness(values, values, ("full","clip"), (name,"clip"))
                             for name in ("current", "reverse_history")}))
    report["differences_against_mamba_interleaved"][MULTIDEPTH] = paired_correctness(reference, values)
    report.update(multidepth_plan_sha256=file_hash(multidepth_queue / "plan.json"),
        additional_contrast="Exploratory interleaved Mamba-3 minus multi-depth GMMix; original primary contrast unchanged")
    return report


def check_audit(evaluation, record, extraction):
    path = Path(record["report"])
    if file_hash(path) != record["report_sha256"]:
        raise ValueError("Raw-label audit changed")
    audit = read_json(path)
    if (audit["scope"] != "Full RoboTAP raw-label metric audit" or audit["videos"] != 265
            or audit["report_sha256"] != file_hash(evaluation)
            or audit["dataset_extraction_sha256"] != extraction):
        raise ValueError("Raw-label audit is partial or belongs to different inputs")
    report = read_json(evaluation)
    indexed = {row["id"]:row for row in audit["rows"]}
    if len(indexed) != 265 or set(indexed) != {r["id"] for r in report["rows"]} or audit["metrics"] != report["metrics"]:
        raise ValueError("Raw-label audit coverage or mean differs")
    for row in report["rows"]:
        if any(indexed[row["id"]][key] != row[key] for key in ("metrics", "video_sha256", "predictions_sha256")):
            raise ValueError("Raw-label audit differs from per-video predictions")
    return report


def extend_point_summary(original_queue, additional_queue):
    original_queue, additional_queue = Path(original_queue).resolve(), Path(additional_queue).resolve()
    require_complete(additional_queue)
    plan = read_json(additional_queue / "plan.json")
    if Path(plan["point_queue"]) != original_queue or plan["point_plan_sha256"] != file_hash(original_queue / "plan.json"):
        raise ValueError("Additional point comparison belongs to a different chain")
    verify_point_plan(plan)
    report = summarize_original_points(original_queue)
    original_plan = read_json(original_queue / "plan.json")
    original_inputs = read_json(original_queue / "inputs.json")
    if plan["protocol"] != original_plan["protocol"]:
        raise ValueError("Point protocol differs between original and additional comparisons")
    dataset = verify_dataset(plan["data_root"], plan["data_plan"])
    if dataset != original_inputs["dataset"]:
        raise ValueError("Point training data differs")
    status = read_json(additional_queue / "status.json")
    if len(plan["runs"]) != 2 or set(status["results"]) != {row["name"] for row in plan["runs"]}:
        raise ValueError("Both additional probe/adapt results are required")
    if {r["mode"] for r in plan["runs"]} != {"probe", "adapt"}:
        raise ValueError("Expected distinct probe and adaptation")
    for row in plan["runs"]:
        encoder = verify_multidepth_video(row["encoder_run"], row["encoder_config"])
        completion = verify_point_run(row, plan, dataset, encoder)
        result = status["results"][row["name"]]
        if result["training"] != completion:
            raise ValueError("Additional point training record changed")
        path = Path(result["evaluation"]["report"])
        if verify_robotap(path.parent, completion, plan["robotap_extraction_sha256"]) != result["evaluation"]:
            raise ValueError("Additional point evaluation changed")
        values = check_audit(path, result["raw_label_audit"], plan["robotap_extraction_sha256"])
        if values["checkpoint"]["metadata"]["config"] != read_json(Path(row["run"]) / "config.json"):
            raise ValueError("Additional point evaluation configuration differs")
        report["runs"][row["name"]] = dict(mode=row["mode"], kind="gmmix_multidepth", placement="after_encoder",
            trainable_parameters=trainable_parameters(row["run"]), report=str(path), report_sha256=file_hash(path),
            checkpoint_sha256=completion["checkpoint_sha256"], raw_label_audit=result["raw_label_audit"],
            metrics={name:values["metrics"][name] for name in METRICS}, video_length_strata=length_description(values))
        contrast = report["contrasts"][row["mode"]]
        reference = report["runs"][contrast["reference"]]
        if file_hash(reference["report"]) != reference["report_sha256"]:
            raise ValueError("Reference point report changed")
        contrast["comparisons"][row["name"]] = compare_reports(read_json(reference["report"]), values)
    report.update(additional_plan_sha256=file_hash(additional_queue / "plan.json"),
        additional_contrast="Exploratory multi-depth baseline; original primary probe contrast unchanged")
    return report


def verify_costs(diagnostics, video_report):
    diagnostics = Path(diagnostics)
    require_complete(diagnostics)
    plan = read_json(diagnostics / "plan.json")
    verify_diagnostic_plan(plan)
    report = read_json(diagnostics / "cost_summary.json")
    parent = read_json(diagnostics / "summary.json")
    if (parent["cost_summary_sha256"] != file_hash(diagnostics / "cost_summary.json")
            or report["plan_sha256"] != file_hash(diagnostics / "plan.json")):
        raise ValueError("Cost summary changed or refers to a different plan")
    indexed = {row["name"]:row for row in video_report["runs"]}
    expected = {(name,video["id"]) for name in MODEL_NAMES for video in plan["videos"]}
    measurements = report["measurements"]
    if len(measurements) != 18 or {(r["model"],r["video"]["id"]) for r in measurements} != expected:
        raise ValueError("Require all six models on the same three videos")
    for row in measurements:
        entry = indexed[row["model"]]
        directory = Path(row["measurement"]["directory"])
        config = read_json(Path(entry["path"]) / "config.json")
        checkpoint = dict(step=video_report["steps"], checkpoint_sha256=entry["checkpoint_sha256"])
        result = verify_measurement(directory, config, checkpoint, row["video"], 3, 1, plan["extraction_sha256"])
        if result != row["measurement"]:
            raise ValueError("Joint cost summary differs from frame measurements")
        if row["model"] == MULTIDEPTH and (result["max_state_bytes"] != 96337920 or any(
                b["state_bytes_min"] != 96337920 or b["state_bytes_max"] != 96337920 for b in result["bins"])):
            raise ValueError("Multi-depth state is not the expected fixed size")
    recalculated = paired_costs(measurements)
    if any(report[key] != value for key,value in recalculated.items()):
        raise ValueError("Paired cost ratios differ from measurements")
    return report


def verify_histories(original_history, diagnostics, point_report):
    original_history, diagnostics = Path(original_history), Path(diagnostics)
    require_complete(original_history)
    old_plan = read_json(original_history / "plan.json")
    old_status = read_json(original_history / "status.json")
    new_plan = read_json(diagnostics / "plan.json")
    new_result = read_json(diagnostics / "summary.json")["history"]
    if (old_plan["dataset_extraction_sha256"] != new_plan["extraction_sha256"]
            or old_plan["periods"] != [1,24] or new_plan["periods"] != [1,24]):
        raise ValueError("History experiments use different data or reset periods")
    records = dict(old_status["results"])
    name = new_plan["probe"]["name"]
    if name in records:
        raise ValueError("Repeated additional history result")
    records[name] = new_result
    expected = {name for name,row in point_report["runs"].items() if row["mode"] == "probe"}
    if len(records) != 6 or set(records) != expected:
        raise ValueError("Full history results for all six probes are required")
    result = {}
    for name,record in records.items():
        path = Path(record["analysis"])
        if file_hash(path) != record["analysis_sha256"]:
            raise ValueError("History analysis changed")
        ordinary = point_report["runs"][name]
        checked = analyze_history(path.parent, ordinary["report"], new_plan["data_root"])
        if json.loads(json.dumps(checked)) != read_json(path) or checked["videos"] != 265 or checked["scope"] != "Full RoboTAP paired encoder-state intervention":
            raise ValueError("Full history analysis differs from original RGB/labels/predictions")
        if checked["checkpoint_sha256"] != ordinary["checkpoint_sha256"]:
            raise ValueError("History and ordinary point results use different checkpoints")
        result[name] = dict(analysis=str(path), analysis_sha256=file_hash(path),
            checkpoint_sha256=checked["checkpoint_sha256"], strata=checked["strata"])
    return dict(scope="Conditional encoder-state intervention given common full-history query anchors",
        primary="Interleaved Mamba probe: all-post-query AJ, full minus reset-1",
        other_contrasts="All controls, reset-24 and age/reappearance strata are exploratory",
        limits="One seed; shared query anchors retain history; reset changes feature distributions; no multiplicity correction",
        runs=result)


def joint_rows(video, points, costs):
    """Join checkpoints explicitly; do not join scores only by display names."""
    video_runs = {r["name"]:r for r in video["runs"]}
    if len(video["runs"]) != 6 or set(video_runs) != set(MODEL_NAMES) or len(points["runs"]) != 11:
        raise ValueError("Joint reporting requires six video models and eleven point experiments")
    seed = video["seed"]
    output = []
    for name in MODEL_NAMES:
        probe_name, adapt_name = [f"{name}_{mode}_seed{seed}" for mode in ("probe","adapt")]
        if adapt_name in points["reused_controls"]:
            reuse = points["reused_controls"][adapt_name]
            if name != "none_interleaved" or reuse["reference"] != probe_name:
                raise ValueError("Only the predefined no-time adaptation alias may reuse its probe")
            adapt_name = reuse["reference"]
        probe, adapt = points["runs"][probe_name], points["runs"][adapt_name]
        if probe["mode"] != "probe" or (name != "none_interleaved" and adapt["mode"] != "adapt"):
            raise ValueError("Probe/adaptation modes were mixed")
        entry = video_runs[name]
        selected = [r for r in costs["measurements"] if r["model"] == name]
        if len(selected) != 3 or {r["video"]["selection"] for r in selected} != {"shortest","median","longest"}:
            raise ValueError("Missing per-video costs")
        parameters = {r["measurement"]["parameter_count"] for r in selected}
        if len(parameters) != 1:
            raise ValueError("Encoder parameter count differs across cost videos")
        for point in (probe, adapt):
            saved = read_json(point["report"])
            if saved["checkpoint"]["metadata"]["config"]["encoder_initialization"]["checkpoint_sha256"] != entry["checkpoint_sha256"]:
                raise ValueError("Point transfer and video score start from different encoders")
        output.append(dict(model=name, video_checkpoint_sha256=entry["checkpoint_sha256"],
            ssv2_stream_top1_percent=100*entry["conditions"]["full/stream"]["accuracy"],
            robotap_probe_aj_percent=100*probe["metrics"]["average_jaccard"],
            robotap_adapt_aj_percent=100*adapt["metrics"]["average_jaccard"],
            adapt_reuses_probe=name == "none_interleaved", encoder_parameters=parameters.pop(),
            video_trainable_parameters=entry["trainable_parameters"],
            point_probe_trainable_parameters=probe["trainable_parameters"],
            point_adapt_trainable_parameters=adapt["trainable_parameters"],
            point_probe_checkpoint_sha256=probe["checkpoint_sha256"], point_adapt_checkpoint_sha256=adapt["checkpoint_sha256"]))
    return output


def cost_rows(costs):
    rows = []
    for entry in costs["measurements"]:
        measurement = entry["measurement"]
        raw = [json.loads(line) for line in (Path(measurement["directory"]) / "timings.jsonl").read_text().splitlines() if line]
        rows.append(dict(model=entry["model"], selection=entry["video"]["selection"],
            video_id=entry["video"]["id"], frames=entry["video"]["frames"], repeats=measurement["repeats"],
            gpu_median_ms=float(np.median([r["gpu_ms"] for r in raw])),
            pipeline_median_ms=float(np.median([r["pipeline_wall_ms"] for r in raw])),
            max_encoder_state_mib=measurement["max_state_bytes"]/2**20,
            torch_peak_allocated_mib=measurement["torch_peak_allocated_bytes"]/2**20,
            repeated_final_outputs_equal=measurement["repeated_final_outputs_equal"]))
    return rows


def write_tables(output, joint, costs):
    output = Path(output)
    for filename,rows in (("recognition_transfer.csv",joint), ("cost_by_video.csv",costs)):
        with (output / filename).open("w",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# 同条件の6方式比較", "", "全学習・全評価が完了した1 seedの結果。未使用test、複数seed、SOTAやIROS採択の証明ではない。", "",
        "動画分類はSSv2全24,777例の逐次top-1、点追跡はRoboTAP全265動画のAJ。probeとadaptは別条件。", "",
        "| 方式 | SSv2 top-1 (%) | probe AJ (%) | adapt AJ (%) | エンコーダパラメータ (M) | 動画学習対象 (M) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in joint:
        lines.append(f"| {row['model']} | {row['ssv2_stream_top1_percent']:.2f} | {row['robotap_probe_aj_percent']:.2f} | {row['robotap_adapt_aj_percent']:.2f} | {row['encoder_parameters']/1e6:.3f} | {row['video_trainable_parameters']/1e6:.3f} |")
    lines += ["", "履歴なしのadapt欄は、同じ学習対象となるprobeの再利用。モデル間のパラメータ数と学習時間は一致させていない。", "",
        "## 実測費用", "", "同じRGB動画、batch 1、1回warmup後3回測定。各動画の全フレーム・全測定回の中央値。", "",
        "| 方式 | 動画 | フレーム数 | GPU中央値 (ms) | 前処理込み中央値 (ms) | 状態最大 (MiB) | PyTorch割当ピーク (MiB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in costs:
        lines.append(f"| {row['model']} | {row['selection']} | {row['frames']} | {row['gpu_median_ms']:.3f} | {row['pipeline_median_ms']:.3f} | {row['max_encoder_state_mib']:.3f} | {row['torch_peak_allocated_mib']:.1f} |")
    lines += ["", "GPU時間は前処理・入力転送を除外。前処理込み時間も動画デコードと検査を除外し、各フレームで同期する。", "",
        "PyTorch割当ピークには重み・入力・状態・活性値・一時領域を含むが、driver/contextと非PyTorch割当は含まない。状態量だけを総メモリと呼ばない。", "",
        "対応付き差と条件付き95%区間は[動画](video.json)・[点追跡](points.json)、状態消去の結果は[履歴](history.json)に記録する。", "",
        "区間は同じ学習済み重みに条件付けたもので、学習seed間の変動を含まない。通常履歴のquery記述子は状態消去後も残る。", "",
        "機械可読表：[認識・転移](recognition_transfer.csv)、[動画別費用](cost_by_video.csv)。"]
    (output / "COMPARISON.md").write_text("\n".join(lines)+"\n")


def summarize_all(paths, output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    video = extend_video_summary(paths["video"], paths["gmmix"], paths["multidepth_video"])
    write_json(output / "video.json", video)
    points = extend_point_summary(paths["points"], paths["multidepth_points"])
    write_json(output / "points.json", points)
    costs = verify_costs(paths["diagnostics"], video)
    write_json(output / "costs.json", costs)
    history = verify_histories(paths["history"], paths["diagnostics"], points)
    write_json(output / "history.json", history)
    joint = joint_rows(video, points, costs)
    write_json(output / "joint.json", dict(scope="Six models, one seed; full fixed-step results with explicit checkpoint joins", rows=joint,
        limits="No averaged omnibus score or automatic novelty/SOTA/publication verdict; more seeds and substantive contribution assessment remain"))
    write_tables(output, joint, cost_rows(costs))
    filenames = ("video.json", "points.json", "costs.json", "history.json", "joint.json", "COMPARISON.md",
                 "recognition_transfer.csv", "cost_by_video.csv")
    manifest = dict(status="complete_single_seed_report", reports={name:file_hash(output / name) for name in filenames},
        inputs={key:dict(path=str(Path(path).resolve()), plan_sha256=file_hash(Path(path) / "plan.json"),
                        status_sha256=file_hash(Path(path) / "status.json")) for key,path in paths.items()},
        next="Replicate performance across seeds and assess contribution before making publication-level claims")
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("video", "gmmix", "multidepth-video", "points", "multidepth-points", "history", "diagnostics"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    paths = {key:value for key,value in vars(args).items() if key != "output"}
    summarize_all(paths,args.output)
    print("Completed six-model single-seed comparison; publication claim remains unproven",flush=True)


if __name__ == "__main__":
    main()
