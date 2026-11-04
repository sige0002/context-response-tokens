"""Across-seed reporting without treating repeated benchmark clips as new data."""
import csv
import json
import math
from pathlib import Path
import statistics

from analyze_point_history import analyze
from multidepth_experiment import verify_completion as verify_multi_video
from multidepth_point_experiment import verify_point_run as verify_multi_point
from point_checkpoint import file_hash
from point_experiment import read_json,write_json,verify_dataset,verify_point_run,verify_robotap
from replication_protocol import MODELS,SEEDS,point_row_for,point_plan_for,validate_video_evaluations
from run_video_comparisons import verify_completion as verify_temporal_video
from summarize_six_models import check_audit


METRICS = ("ssv2_stream_accuracy","robotap_probe_aj","history_full_minus_reset1_aj")


def aggregate(rows):
    seeds = (42,*SEEDS)
    indexed = {(r["model"],r["seed"]):r for r in rows}
    expected = {(m,s) for m in MODELS for s in seeds}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError("All four models and all three predeclared seeds are required exactly once")
    for row in rows:
        for metric in METRICS:
            value = row[metric]
            lower = -1 if metric.startswith("history_") else 0
            if not math.isfinite(value) or not lower <= value <= 1:
                raise ValueError("Invalid replication metric")
    models = {}
    contrasts = {}
    for model in MODELS:
        models[model] = {}
        for metric in METRICS:
            values = [100*indexed[model,seed][metric] for seed in seeds]
            models[model][metric] = dict(seeds=list(seeds),values_percentage_points=values,
                mean_percentage_points=statistics.mean(values),sample_sd_percentage_points=statistics.stdev(values),n_seeds=len(values))
        if model != MODELS[0]:
            contrasts[model] = {}
            for metric in METRICS:
                differences = [100*(indexed[MODELS[0],seed][metric]-indexed[model,seed][metric]) for seed in seeds]
                contrasts[model][metric] = dict(seeds=list(seeds),paired_differences_pp=differences,
                    mean_difference_pp=statistics.mean(differences),sample_sd_difference_pp=statistics.stdev(differences),n_pairs=len(seeds))
    return dict(models=models,reference_minus_other=contrasts,
        primary_video="Interleaved Mamba-3 minus interleaved time attention: streaming top-1",
        primary_point="Same contrast on frozen-encoder probe AJ",
        scope="Three paired training seeds on a fixed benchmark; values and sample SD, not a precise training-distribution confidence interval",
        limits="No pooling seed-by-clip rows as independent observations. Three seeds offer limited precision. Adaptation and two other architectures remain single-seed; history uses shared full-history query anchors.")


def load_reference(plan):
    directory = Path(plan["seed42_summary"])
    if file_hash(directory/"plan.json") != plan["seed42_summary_plan_sha256"]:
        raise ValueError("Seed42 report plan changed")
    queue = read_json(directory/"plan.json")
    status = read_json(directory/"status.json")
    if status["status"] != "complete" or queue["paths"]["diagnostics"] != plan["dependencies"]["diagnostics"]:
        raise ValueError("Full seed42 report is incomplete or refers to a different experiment chain")
    manifest_path = Path(status["report"])
    if file_hash(manifest_path) != status["report_sha256"]:
        raise ValueError("Seed42 report manifest changed")
    manifest = read_json(manifest_path)
    for name,digest in manifest["reports"].items():
        if file_hash(manifest_path.parent/name) != digest:
            raise ValueError("Seed42 report artifact changed")
    joint = read_json(manifest_path.parent/"joint.json")["rows"]
    history = read_json(manifest_path.parent/"history.json")["runs"]
    rows = []
    for model in MODELS:
        matches = [r for r in joint if r["model"] == model]
        if len(matches) != 1:
            raise ValueError("Missing seed42 model in joint report")
        value = matches[0]
        h = history[f"{model}_probe_seed42"]
        if h["checkpoint_sha256"] != value["point_probe_checkpoint_sha256"]:
            raise ValueError("Seed42 history and point results use different weights")
        rows.append(dict(model=model,seed=42,ssv2_stream_accuracy=value["ssv2_stream_top1_percent"]/100,
            robotap_probe_aj=value["robotap_probe_aj_percent"]/100,
            history_full_minus_reset1_aj=h["strata"]["all_post_query"]["full_minus_reset"]["reset_1"]["average_jaccard"]["difference"],
            video_checkpoint_sha256=value["video_checkpoint_sha256"],point_checkpoint_sha256=value["point_probe_checkpoint_sha256"]))
    return rows,dict(path=str(manifest_path),sha256=file_hash(manifest_path))


def collect_additional(plan,status):
    names = {row["name"] for row in plan["runs"]}
    if len(names) != 8 or any(set(status[key]) != names for key in ("video_results","point_results","history_results")):
        raise ValueError("Additional-seed result grid is incomplete")
    dataset = verify_dataset(plan["data_root"],plan["data_plan"])
    rows = []
    for row in plan["runs"]:
        is_multi = row["model"] == MODELS[-1]
        completed = (verify_multi_video if is_multi else verify_temporal_video)(row["video_run"],row["video_config"])
        expected_video = dict(completion=completed,evaluation=validate_video_evaluations(row,completed))
        if expected_video != status["video_results"][row["name"]]:
            raise ValueError("Replication video record changed")
        point = (verify_multi_point if is_multi else verify_point_run)(point_row_for(row),point_plan_for(plan,row),dataset,completed)
        point_record = status["point_results"][row["name"]]
        if point != point_record["training"]:
            raise ValueError("Replication point record changed")
        report_path = Path(point_record["evaluation"]["report"])
        if verify_robotap(report_path.parent,point,plan["robotap_extraction_sha256"]) != point_record["evaluation"]:
            raise ValueError("Replication point evaluation changed")
        report = check_audit(report_path,point_record["raw_label_audit"],plan["robotap_extraction_sha256"])
        if report["checkpoint"]["metadata"]["config"] != read_json(Path(row["point_run"])/"config.json"):
            raise ValueError("Replication point evaluation configuration differs")
        h = status["history_results"][row["name"]]
        analysis_path = Path(h["analysis"])
        if file_hash(analysis_path) != h["analysis_sha256"]:
            raise ValueError("Replication history record changed")
        history = analyze(analysis_path.parent,report_path,plan["robotap_root"])
        if json.loads(json.dumps(history)) != read_json(analysis_path):
            raise ValueError("Replication history differs from raw predictions/labels")
        rows.append(dict(model=row["model"],seed=row["seed"],
            ssv2_stream_accuracy=expected_video["evaluation"]["stream_accuracy"],robotap_probe_aj=report["metrics"]["average_jaccard"],
            history_full_minus_reset1_aj=history["strata"]["all_post_query"]["full_minus_reset"]["reset_1"]["average_jaccard"]["difference"],
            video_checkpoint_sha256=completed["checkpoint_sha256"],point_checkpoint_sha256=point["checkpoint_sha256"]))
    return rows


def write_report(plan,status,output):
    output = Path(output)
    reference,provenance = load_reference(plan)
    rows = reference+collect_additional(plan,status)
    result = dict(rows=rows,seed42_report=provenance,**aggregate(rows))
    write_json(output/"seed_summary.json",result)
    with (output/"seed_scores.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    lines = ["# 固定した3 seedの反復結果", "",
        "SSv2の逐次top-1とRoboTAP probe AJ。平均 ± 標本標準偏差で、学習分布の95%信頼区間ではない。", "",
        "| 方式 | SSv2 top-1 (%) | probe AJ (%) | 履歴効果 ΔAJ (pp) |",
        "| --- | ---: | ---: | ---: |"]
    for model,metrics in result["models"].items():
        values = [f"{metrics[k]['mean_percentage_points']:.2f} ± {metrics[k]['sample_sd_percentage_points']:.2f}" for k in METRICS]
        lines.append("| "+" | ".join([model,*values])+" |")
    lines += ["", "履歴効果は通常履歴のAJからreset-1のAJを引いた値。queryの記述子には通常履歴が残る。", "",
        "[各seedの生値](seed_scores.csv)と[同じseed同士の差](seed_summary.json)を併記する。", "",
        "3 seedは推定精度が限られる。adapt、履歴なし、最終層のみGMMixは1 seedの探索的結果のままである。SOTAやIROS採択を自動判定しない。"]
    (output/"SEED_COMPARISON.md").write_text("\n".join(lines)+"\n")
    return {name:file_hash(output/name) for name in ("seed_summary.json","seed_scores.csv","SEED_COMPARISON.md")}
