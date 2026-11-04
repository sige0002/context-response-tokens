"""Extend the pinned cost/history comparisons to the multi-depth baseline.

Wait for completed point-transfer runs, then use the original score-independent
real-video selection and the same conditional query-anchor history intervention.
"""
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
from encoder_costs import verify_measurement, paired_costs
from multidepth_experiment import verify_completion
from multidepth_point_experiment import verify_point_run, verify_robotap, verify_dataset, UV
from point_checkpoint import file_hash
from point_experiment import read_json, write_json
from run_multidepth_points import verify_plan as verify_point_plan
from run_point_comparisons import bind_process


ROOT = Path(__file__).resolve().parent
SOURCES = ("run_multidepth_diagnostics.py", "benchmark_multidepth_video.py", "benchmark_real_video.py",
    "benchmark_streaming.py", "encoder_costs.py", "evaluate_multidepth_history.py", "evaluate_point_history.py",
    "point_history.py", "analyze_point_history.py", "audit_robotap_predictions.py", "robotap_data.py",
    "tapvid_metrics.py", "summarize_points.py", "paired_metrics.py", "multidepth_checkpoint.py",
    "multidepth_point_checkpoint.py", "multidepth_point_experiment.py", "multidepth_experiment.py",
    "run_multidepth_points.py", "run_point_comparisons.py", "point_experiment.py", "point_checkpoint.py",
    "point_readout.py", "gmmix_multidepth.py", "gmmix_siglip.py", "temporal_siglip.py", "train_video.py",
    "pyproject.toml", "uv.lock", "baselines/gmmix/pyproject.toml", "baselines/gmmix/uv.lock",
    "baselines/gmmix/source_manifest.json")


def verify_integration(integration):
    path = Path(integration)
    record = read_json(path / "verification.json")
    if (record.get("status") != "passed_scoped_checks" or record.get("history_scope") != "Engineering subset"
            or record.get("history_videos") != 1 or record.get("history_periods") != [1,24]
            or not record["cost"]["repeated_final_outputs_equal"]
            or record["cost"]["max_state_bytes"] != 96337920):
        raise ValueError("Missing native cost/history integration checks")
    expected = {"check_multidepth_diagnostics.py", "benchmark_multidepth_video.py", "benchmark_real_video.py",
        "evaluate_multidepth_history.py", "evaluate_point_history.py", "analyze_point_history.py", "point_history.py",
        "multidepth_point_checkpoint.py", "multidepth_checkpoint.py", "gmmix_multidepth.py", "encoder_costs.py"}
    if set(record["sources"]) != expected:
        raise ValueError("Diagnostic integration source inventory differs")
    for name,digest in record["sources"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Diagnostic integration source changed: {name}")
    if file_hash(record["history_analysis"]) != record["history_analysis_sha256"]:
        raise ValueError("History integration evidence changed")
    cost_plan = read_json(path / "cost/plan.json")
    checked = verify_measurement(path / "cost", cost_plan["checkpoint"]["config"],
        cost_plan["checkpoint"], record["cost_video"], 2, 1, cost_plan["dataset_extraction_sha256"])
    if checked != record["cost"]:
        raise ValueError("Cost integration evidence changed")


def build_plan(points, costs, integration, output):
    point_plan, cost_plan = read_json(points / "plan.json"), read_json(costs / "plan.json")
    verify_point_plan(point_plan)
    verify_integration(integration)
    if (cost_plan["point_queue"] != point_plan["point_queue"]
            or cost_plan["point_plan_sha256"] != point_plan["point_plan_sha256"]
            or cost_plan["data_root"] != point_plan["robotap_root"]
            or cost_plan["extraction_sha256"] != point_plan["robotap_extraction_sha256"]
            or cost_plan["repeats"] != 3 or cost_plan["warmup_passes"] != 1):
        raise ValueError("Original cost and additional point comparisons differ")
    probes = [row for row in point_plan["runs"] if row["mode"] == "probe"]
    if len(probes) != 1 or probes[0]["encoder_type"] != "gmmix_multidepth":
        raise ValueError("Expected the additional multi-depth frozen probe")
    return dict(point_queue=str(points), point_plan_sha256=file_hash(points / "plan.json"),
        cost_queue=str(costs), cost_plan_sha256=file_hash(costs / "plan.json"),
        integration=str(integration), integration_sha256=file_hash(integration / "verification.json"),
        data_root=cost_plan["data_root"], extraction_sha256=cost_plan["extraction_sha256"],
        videos=cost_plan["videos"], selection=cost_plan["selection"], repeats=3, warmup_passes=1,
        probe=probes[0], periods=[1,24],
        source_sha256={name:file_hash(ROOT / name) for name in SOURCES},
        scope="Additional final video encoder on the same three real videos; full RoboTAP history intervention on its frozen point probe. One seed; shared query anchors retain history; no accuracy claim from fixed state size.")


def verify_plan(plan):
    for name,digest in plan["source_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f"Additional diagnostic source changed: {name}")
    for path,key in ((Path(plan["point_queue"]) / "plan.json", "point_plan_sha256"),
            (Path(plan["cost_queue"]) / "plan.json", "cost_plan_sha256"),
            (Path(plan["integration"]) / "verification.json", "integration_sha256"),
            (Path(plan["data_root"]) / "extraction.json", "extraction_sha256")):
        if file_hash(path) != plan[key]:
            raise ValueError(f"Additional diagnostic dependency changed: {path}")
    verify_point_plan(read_json(Path(plan["point_queue"]) / "plan.json"))
    verify_integration(plan["integration"])


def fixed_state_measurement(directory, config, checkpoint, video, plan):
    result = verify_measurement(directory, config, checkpoint, video,
        plan["repeats"], plan["warmup_passes"], plan["extraction_sha256"])
    if (result["max_state_bytes"] != 96337920 or any(
            row["state_bytes_min"] != 96337920 or row["state_bytes_max"] != 96337920 for row in result["bins"])):
        raise ValueError("Multi-depth recurrent state accounting differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-comparisons", required=True)
    parser.add_argument("--cost-comparisons", required=True)
    parser.add_argument("--integration", default="results/multidepth_diagnostics_integration")
    parser.add_argument("--after-point-pid", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    points, costs, integration, output = [Path(p).resolve() for p in
        (args.point_comparisons, args.cost_comparisons, args.integration, args.output)]
    plan = build_plan(points, costs, integration, output)
    if args.resume:
        if read_json(output / "plan.json") != plan:
            raise ValueError("Additional diagnostic plan changed")
        status = read_json(output / "status.json")
        status.pop("error", None)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        for name in SOURCES:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        status = dict(status="planned", operations=[], measurements={})
    status["dependency_pid"] = args.after_point_pid
    write_json(output / "status.json", status)
    if args.plan_only:
        print("Saved three cost measurements and one full history intervention; no jobs started", flush=True)
        return
    fd, child, cancelled = None, None, False

    def save():
        write_json(output / "status.json", status)

    def stop(signum, frame):
        nonlocal cancelled
        cancelled = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def launch(command, name, cpu=False):
        nonlocal child
        if cancelled:
            raise InterruptedError("Additional diagnostics cancelled")
        verify_plan(plan)
        if not cpu:
            idle_compute_processes()
        operation = dict(name=name, command=command, started=time.time(), status="running",
                         log=f"{len(status['operations'])+1:03d}_{name}.log")
        status["operations"].append(operation)
        status["status"] = "running"
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        with (output / operation["log"]).open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            operation["pid"] = child.pid
            save()
            code = child.wait()
            child = None
        operation.update(returncode=code, finished=time.time(), status="complete" if code == 0 else "failed")
        save()
        if cancelled or code:
            raise RuntimeError(f"{name} stopped/failed ({code}); inspect its log")

    def unused_destination(base):
        index = 1
        while (output / f"{base}_{index:02d}").exists():
            index += 1
        return output / f"{base}_{index:02d}"

    try:
        if args.after_point_pid:
            fd = bind_process(args.after_point_pid, "run_multidepth_points.py", "--output", points)
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            status["status"] = "waiting_for_multidepth_points"
            save()
            while not poller.poll(1000):
                if cancelled:
                    raise InterruptedError("Cancelled while waiting; dependency unchanged")
            os.close(fd)
            fd = None
        verify_plan(plan)
        if any(read_json(p / "status.json")["status"] != "complete" for p in (points, costs)):
            raise ValueError("Preceding point/cost experiments did not finish normally")
        point_plan = read_json(points / "plan.json")
        row = plan["probe"]
        config = row["encoder_config"]
        encoder = verify_completion(row["encoder_run"], config)
        dataset = verify_dataset(point_plan["data_root"], point_plan["data_plan"])
        probe = verify_point_run(row, point_plan, dataset, encoder)
        point_result = read_json(points / "status.json")["results"][row["name"]]
        if point_result["training"] != probe:
            raise ValueError("Completed probe changed")
        ordinary = verify_robotap(Path(point_result["evaluation"]["report"]).parent, probe, plan["extraction_sha256"])
        audit = point_result["raw_label_audit"]
        if file_hash(audit["report"]) != audit["report_sha256"]:
            raise ValueError("Ordinary point metric audit changed")
        checkpoint = Path(row["encoder_run"]) / "last.pt"
        for video in plan["videos"]:
            label = video["selection"]
            if file_hash(checkpoint) != encoder["checkpoint_sha256"]:
                raise ValueError("Completed encoder changed")
            if label in status["measurements"]:
                destination = Path(status["measurements"][label]["measurement"]["directory"])
            else:
                destination = unused_destination("cost_"+label)
                launch(UV + ["benchmark_multidepth_video.py", "--checkpoint", str(checkpoint),
                    "--root", plan["data_root"], "--video-id", video["id"], "--repeats", "3",
                    "--warmup-passes", "1", "--output", str(destination)], "cost_"+label)
            result = fixed_state_measurement(destination, config, encoder, video, plan)
            status["measurements"][label] = dict(model="gmmix_multidepth_after_encoder", video=video, measurement=result)
            save()
        # Recount existing measurements before extending their paired ratios.
        original_plan = read_json(costs / "plan.json")
        original_summary = read_json(costs / "summary.json")
        if original_summary["plan_sha256"] != file_hash(costs / "plan.json"):
            raise ValueError("Original cost summary refers to a different plan")
        original_checkpoints = read_json(costs / "completed_encoders.json")
        existing = original_summary["measurements"]
        expected_pairs = {(entry["config"]["kind"]+"_"+entry["config"]["placement"], video["id"])
                          for entry in original_plan["encoders"] for video in plan["videos"]}
        if len(existing) != 15 or {(r["model"],r["video"]["id"]) for r in existing} != expected_pairs:
            raise ValueError("Original cost measurements are incomplete")
        for observed in existing:
            entry = next(e for e in original_plan["encoders"]
                         if e["config"]["kind"]+"_"+e["config"]["placement"] == observed["model"])
            checked = verify_measurement(observed["measurement"]["directory"], entry["config"], original_checkpoints[entry["run"]],
                observed["video"], 3, 1, plan["extraction_sha256"])
            if checked != observed["measurement"]:
                raise ValueError("Original cost summary differs from measurements")
        combined = existing + list(status["measurements"].values())
        write_json(output / "cost_summary.json", dict(plan_sha256=file_hash(output / "plan.json"),
            original_cost_summary_sha256=file_hash(costs / "summary.json"), measurements=combined, **paired_costs(combined)))
        point_checkpoint = Path(row["run"]) / "last.pt"
        if file_hash(point_checkpoint) != probe["checkpoint_sha256"]:
            raise ValueError("Completed point checkpoint changed")
        if "history" in status:
            destination = Path(status["history"]["directory"])
        else:
            destination = unused_destination("history")
            launch(UV + ["evaluate_multidepth_history.py", "--checkpoint", str(point_checkpoint),
                "--root", plan["data_root"], "--output", str(destination), "--periods", "1", "24"], "history")
            status["history"] = dict(directory=str(destination))
            save()
        analysis = destination / "analysis.json"
        if not analysis.exists():
            launch(UV + ["analyze_point_history.py", "--interventions", str(destination), "--baseline", ordinary["report"],
                "--root", plan["data_root"], "--output", str(analysis)], "history_analysis", cpu=True)
        else:
            # A resumed queue must verify raw prediction archives again, not
            # infer completion from an analysis filename or its scope string.
            from analyze_point_history import analyze
            # The analyzer's age/gap bins are tuples in memory and JSON arrays
            # on disk. Compare their serialized values, not Python containers.
            recounted = json.loads(json.dumps(analyze(destination, ordinary["report"], plan["data_root"])))
            if recounted != read_json(analysis):
                raise ValueError("Saved history analysis differs from a raw-label recount")
        report = read_json(analysis)
        if (report["scope"] != "Full RoboTAP paired encoder-state intervention" or report["videos"] != 265
                or report["checkpoint_sha256"] != probe["checkpoint_sha256"]
                or report["verification"]["baseline_report_sha256"] != ordinary["report_sha256"]
                or report["dataset_extraction_sha256"] != plan["extraction_sha256"]
                or any(plan["source_sha256"].get(name) != digest for name,digest in report["source_sha256"].items())):
            raise ValueError("Full history analysis differs from plan/checkpoint/data")
        status["history"].update(analysis=str(analysis), analysis_sha256=file_hash(analysis))
        verify_plan(plan)
        write_json(output / "summary.json", dict(scope=plan["scope"], plan_sha256=file_hash(output / "plan.json"),
            cost_summary=str(output / "cost_summary.json"), cost_summary_sha256=file_hash(output / "cost_summary.json"),
            history=status["history"], remaining="Joint recognition/transfer comparison, multiple seeds and publication assessment"))
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="stopped" if cancelled else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if fd is not None:
            os.close(fd)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        save()


if __name__ == "__main__":
    main()
