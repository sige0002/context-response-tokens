"""Run the fixed additional-seed grid after the existing GPU experiment chain."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

import torch

from benchmark_streaming import idle_compute_processes
from multidepth_experiment import verify_completion as verify_multi_video
from multidepth_point_experiment import verify_point_run as verify_multi_point, assert_training_config as assert_multi_config
from point_checkpoint import file_hash
from point_experiment import (read_json,write_json,verify_dataset,verify_point_run,verify_robotap,assert_training_config)
from replication_protocol import ROOT, UV, MODELS, build_plan, verify_plan, point_plan_for, point_row_for, validate_video_evaluations
from run_point_comparisons import bind_process
from run_video_comparisons import verify_completion as verify_temporal_video
from train_points import SOURCES as POINT_SOURCES
from train_multidepth_points import SOURCES as MULTI_POINT_SOURCES


SOURCES = tuple(dict.fromkeys((*POINT_SOURCES,*MULTI_POINT_SOURCES,"replication_protocol.py","run_replication.py","replication_summary.py","summarize_six_models.py",
    "run_video_comparisons.py","multidepth_experiment.py","run_multidepth_comparison.py","run_multidepth_points.py",
    "run_multidepth_diagnostics.py","multidepth_point_experiment.py","point_experiment.py","run_point_comparisons.py",
    "evaluate_video.py","evaluate_temporal.py","evaluate_multidepth_video.py","evaluate_point_tracks.py",
    "evaluate_multidepth_points.py","evaluate_point_history.py","evaluate_multidepth_history.py","point_history.py",
    "analyze_point_history.py","audit_robotap_predictions.py","robotap_data.py","summarize_video.py",
    "summarize_points.py","summarize_temporal.py","paired_metrics.py","benchmark_streaming.py")))


def video_verifier(row):
    return verify_multi_video if row["model"] == MODELS[-1] else verify_temporal_video


def point_verifier(row):
    return verify_multi_point if row["model"] == MODELS[-1] else verify_point_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--after-diagnostics-pid", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed42-summary",required=True)
    parser.add_argument("--workers", type=int,default=2)
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("Workers must be nonnegative")
    os.chdir(ROOT)
    torch.set_num_threads(2)
    output = Path(args.output).resolve()
    plan = build_plan(args.diagnostics,output,SOURCES,args.workers)
    plan["seed42_summary"] = str(Path(args.seed42_summary).resolve())
    plan["seed42_summary_plan_sha256"] = file_hash(Path(plan["seed42_summary"])/"plan.json")
    if args.resume:
        if read_json(output/"plan.json") != plan:
            raise ValueError("Replication plan/source changed")
        status = read_json(output/"status.json")
        status.pop("error",None)
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/"plan.json",plan)
        for name in SOURCES:
            target = output/"source"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
        status = dict(status="planned",operations=[],video_results={},point_results={},history_results={})
    status["dependency_pid"] = args.after_diagnostics_pid
    write_json(output/"status.json",status)
    if args.plan_only:
        print("Saved eight video, eight probe and eight history replications; no jobs started",flush=True)
        return
    child,fd,cancelled = None,None,False

    def save():
        write_json(output/"status.json",status)

    def stop(signum,frame):
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)

    def check_inputs():
        if cancelled:
            raise InterruptedError("Replication queue cancelled")
        verify_plan(plan)
        for filename,key in (("train.jsonl","train_sha256"),("validation.jsonl","val_sha256"),("complete.json","complete_sha256")):
            if file_hash(Path(plan["data_root"])/filename) != dataset[key]:
                raise ValueError("Completed synthetic data changed")

    def launch(command,name,cpu=False):
        nonlocal child
        check_inputs()
        if not cpu:
            idle_compute_processes()
        operation = dict(name=name,command=command,started=time.time(),status="running",
                         log=f"{len(status['operations'])+1:03d}_{name}.log")
        status["operations"].append(operation)
        status["status"] = "running"
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV",None)
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        with (output/operation["log"]).open("w") as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            operation["pid"] = child.pid
            save()
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"{name}: process {child.pid} is running",flush=True)
        code,child = child.returncode,None
        operation.update(returncode=code,finished=time.time(),status="complete" if code == 0 else "failed")
        save()
        if cancelled or code:
            raise RuntimeError(f"{name} stopped/failed ({code}); inspect {operation['log']}")

    def finished(run,steps):
        path = Path(run)/"progress.jsonl"
        if not path.exists():
            return False
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        return bool(records and records[-1].get("event") == "budget_complete" and records[-1]["step"] == steps)

    def next_directory(parent,label):
        index = 1
        while (Path(parent)/f"{label}_{index:02d}").exists():
            index += 1
        return Path(parent)/f"{label}_{index:02d}"

    try:
        if args.after_diagnostics_pid is not None:
            fd = bind_process(args.after_diagnostics_pid,"run_multidepth_diagnostics.py","--output",plan["dependencies"]["diagnostics"])
            poller = select.poll()
            poller.register(fd,select.POLLIN)
            status["status"] = "waiting_for_seed42_gpu_pipeline"
            save()
            while not poller.poll(1000):
                if cancelled:
                    raise InterruptedError("Replication cancelled while waiting; dependencies unchanged")
            os.close(fd)
            fd = None
        verify_plan(plan)
        for path in plan["dependencies"].values():
            if read_json(Path(path)/"status.json")["status"] != "complete":
                raise ValueError("Seed42 experiments must complete before replication")
        dataset = verify_dataset(plan["data_root"],plan["data_plan"])
        initial = dict(dataset=dataset,reference_checkpoints={})
        for model,reference in plan["reference_runs"].items():
            row = dict(model=model)
            complete = video_verifier(row)(reference["video_run"],reference["video_config"])
            initial["reference_checkpoints"][model] = complete
        if (output/"inputs.json").exists() and read_json(output/"inputs.json") != initial:
            raise ValueError("Replication input data/checkpoints changed across attempts")
        write_json(output/"inputs.json",initial)
        # Finish recognition replications first, then run transfer on exactly
        # those final checkpoints. No score controls whether a run is retained.
        for row in plan["runs"]:
            check_inputs()
            run = Path(row["video_run"])
            resume = (run/"last.pt").exists()
            if resume and read_json(run/"config.json") != row["video_config"]:
                raise ValueError("Partial video run differs from replication plan")
            if not finished(run,row["video_config"]["steps"]):
                launch(row["train_video"]+(["--resume"] if resume else []),row["name"]+"_video")
            complete = video_verifier(row)(run,row["video_config"])
            for index,command in enumerate(row["evaluate_video"]):
                if file_hash(run/"last.pt") != complete["checkpoint_sha256"]:
                    raise ValueError("Final video checkpoint changed")
                destination = Path(command[command.index("--output")+1])
                if not destination.exists():
                    launch(command,row["name"]+f"_video_eval{index}")
            result = dict(completion=complete,evaluation=validate_video_evaluations(row,complete))
            if row["name"] in status["video_results"] and status["video_results"][row["name"]] != result:
                raise ValueError("Previously completed replication video result changed")
            status["video_results"][row["name"]] = result
            save()
        for row in plan["runs"]:
            check_inputs()
            run = Path(row["point_run"])
            encoder = status["video_results"][row["name"]]["completion"]
            if file_hash(Path(row["video_run"])/"last.pt") != encoder["checkpoint_sha256"]:
                raise ValueError("Point initialization checkpoint changed")
            pr,pp = point_row_for(row),point_plan_for(plan,row)
            resume = (run/"last.pt").exists()
            if resume:
                checker = assert_multi_config if row["model"] == MODELS[-1] else assert_training_config
                checker(read_json(run/"config.json"),pr,pp,dataset,encoder)
            if not finished(run,row["point_protocol"]["steps"]):
                launch(row["train_point"]+(["--resume"] if resume else []),row["name"]+"_probe")
            complete = point_verifier(row)(pr,pp,dataset,encoder)
            if row["name"] in status["point_results"]:
                prior = status["point_results"][row["name"]]
                if prior["training"] != complete:
                    raise ValueError("Previously completed replication point result changed")
                destination = Path(prior["evaluation"]["report"]).parent
            else:
                destination = next_directory(run,"robotap")
                launch(UV+[row["point_evaluator"],"--checkpoint",str(run/"last.pt"),"--root",plan["robotap_root"],
                    "--output",str(destination)],row["name"]+"_robotap")
            evaluation = verify_robotap(destination,complete,plan["robotap_extraction_sha256"])
            audit_path = destination/"raw_label_metric_audit.json"
            if not audit_path.exists():
                launch(UV+["audit_robotap_predictions.py","--report",evaluation["report"],"--root",plan["robotap_root"],
                    "--output",str(audit_path)],row["name"]+"_point_audit",cpu=True)
            audit = read_json(audit_path)
            if (audit["scope"] != "Full RoboTAP raw-label metric audit" or audit["videos"] != 265
                    or audit["report_sha256"] != evaluation["report_sha256"]
                    or audit["dataset_extraction_sha256"] != plan["robotap_extraction_sha256"]
                    or audit["metrics"] != evaluation["metrics"]):
                raise ValueError("Replication point metric audit differs")
            status["point_results"][row["name"]] = dict(training=complete,evaluation=evaluation,
                raw_label_audit=dict(report=str(audit_path),report_sha256=file_hash(audit_path)))
            save()
            if row["name"] in status["history_results"]:
                history_path = Path(status["history_results"][row["name"]]["directory"])
            else:
                history_path = next_directory(run,"history")
                launch(UV+[row["history_evaluator"],"--checkpoint",str(run/"last.pt"),"--root",plan["robotap_root"],
                    "--output",str(history_path),"--periods","1","24"],row["name"]+"_history")
                status["history_results"][row["name"]] = dict(directory=str(history_path))
                save()
            analysis = history_path/"analysis.json"
            if not analysis.exists():
                launch(UV+["analyze_point_history.py","--interventions",str(history_path),"--baseline",evaluation["report"],
                    "--root",plan["robotap_root"],"--output",str(analysis)],row["name"]+"_history_audit",cpu=True)
            else:
                from analyze_point_history import analyze
                if json.loads(json.dumps(analyze(history_path,evaluation["report"],plan["robotap_root"]))) != read_json(analysis):
                    raise ValueError("Replication history recount differs")
            report = read_json(analysis)
            if (report["scope"] != "Full RoboTAP paired encoder-state intervention" or report["videos"] != 265
                    or report["checkpoint_sha256"] != complete["checkpoint_sha256"]
                    or report["verification"]["baseline_report_sha256"] != evaluation["report_sha256"]):
                raise ValueError("Replication history analysis is partial or inconsistent")
            status["history_results"][row["name"]].update(analysis=str(analysis),analysis_sha256=file_hash(analysis))
            save()
        check_inputs()
        names = {r["name"] for r in plan["runs"]}
        if any(set(status[key]) != names for key in ("video_results","point_results","history_results")):
            raise ValueError("Replication result grid is incomplete")
        write_json(output/"results.json",dict(plan_sha256=file_hash(output/"plan.json"),
            video=status["video_results"],points=status["point_results"],history=status["history_results"],
            scope=plan["reporting"],limits=plan["limits"]))
        status["status"] = "summarizing_across_seeds"
        save()
        from replication_summary import write_report
        status["summary_artifacts"] = write_report(plan,status,output)
        check_inputs()
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="stopped" if cancelled else "failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if fd is not None:
            os.close(fd)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        save()


if __name__ == "__main__":
    main()
