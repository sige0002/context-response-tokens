"""Queue multi-depth GMMix probe/adapt runs after its full video comparison.

The existing synthetic split, point head, losses and final RoboTAP evaluation
are reused. Published DINO scores are not substituted for measured results.
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

from multidepth_experiment import verify_completion as verify_video_completion
from multidepth_point_experiment import (PROTOCOL, UV, SOURCES, assert_training_config,
    read_json, write_json, train_command, verify_dataset, verify_point_run, verify_robotap)
from point_checkpoint import file_hash
from run_multidepth_comparison import verify_plan as verify_video_plan
from run_point_comparisons import bind_process, verify_plan_inputs as verify_original_point_plan


ROOT = Path(__file__).resolve().parent
QUEUE_SOURCES = tuple(dict.fromkeys((*SOURCES, "multidepth_point_experiment.py", "run_multidepth_points.py",
    "evaluate_multidepth_points.py", "evaluate_point_tracks.py", "robotap_data.py", "audit_robotap_predictions.py",
    "point_experiment.py", "run_point_comparisons.py", "run_multidepth_comparison.py", "multidepth_experiment.py")))


def verify_integration(path):
    path = Path(path)
    evidence = read_json(path / "verification.json")
    if (evidence.get("status") != "passed_scoped_checks" or not evidence.get("probe_resume_checked")
            or not evidence.get("probe_encoder_bitwise_unchanged")
            or not evidence.get("adapt_backbone_bitwise_unchanged")
            or evidence.get("native_adapt_bitwise_resume_claim") is not False
            or evidence.get("adapt_batchnorm_updates") != [2]*4
            or (evidence.get("steps"), evidence.get("train_scenes"), evidence.get("validation_scenes")) != (3, 3, 1)
            or evidence.get("robotap_videos") != 1):
        raise ValueError("Missing scoped native point-training and evaluation checks")
    expected_sources = set((*SOURCES, "check_multidepth_points.py", "evaluate_multidepth_points.py",
        "evaluate_point_tracks.py", "robotap_data.py", "audit_robotap_predictions.py", "pause_training.py"))
    if set(evidence["source_sha256"]) != expected_sources:
        raise ValueError("Point integration source inventory differs")
    for name,expected in evidence["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Point integration source changed: {name}")
    if file_hash(evidence["encoder_checkpoint"]) != evidence["encoder_sha256"]:
        raise ValueError("Point integration initial encoder changed")
    for name,key in (("direct/last.pt", "probe_checkpoint_sha256"), ("adapt/last.pt", "adapt_checkpoint_sha256"),
                     ("robotap/summary.json", "evaluation_sha256"),
                     ("robotap/raw_label_metric_audit.json", "raw_label_audit_sha256")):
        if file_hash(path / name) != evidence[key]:
            raise ValueError(f"Point integration evidence changed: {name}")
    thresholds = dict(weights=5e-6, optimizer=5e-6, loss=1e-3, normalized_xy=1e-3)
    if evidence.get("tolerances") != thresholds or set(evidence["comparisons"]) != {"repeat", "resumed"}:
        raise ValueError("Integration numerical protocol differs")
    fields = dict(weights="max_weight_difference", optimizer="max_optimizer_difference",
                  loss="max_loss_difference", normalized_xy="max_normalized_xy_difference")
    for comparison in evidence["comparisons"].values():
        if (not comparison["rng_and_integer_buffers_exact"]
                or any(not 0 <= comparison[field] < thresholds[key] for key,field in fields.items())):
            raise ValueError("Frozen point-probe restart/repeat check failed")
    return evidence


def build_plan(video_queue, point_queue, integration, output, workers):
    video = read_json(video_queue / "plan.json")
    original = read_json(point_queue / "plan.json")
    history = read_json(Path(video["history_queue"]) / "plan.json")
    if (Path(history["point_queue"]) != point_queue
            or history["point_plan_sha256"] != file_hash(point_queue / "plan.json")
            or video["video_plan_sha256"] != original["video_plan_sha256"]):
        raise ValueError("Multi-depth video and original point comparisons belong to different chains")
    if original["protocol"] != PROTOCOL or original["modes"] != ["probe", "adapt"]:
        raise ValueError("Point comparison protocol differs")
    verify_video_plan(video)
    verify_original_point_plan(original)
    evidence = verify_integration(integration)
    runs = []
    for mode in ("probe", "adapt"):
        name = f"gmmix_multidepth_after_encoder_{mode}_seed{PROTOCOL['seed']}"
        run = output / name
        runs.append(dict(name=name, run=str(run), mode=mode, encoder_type="gmmix_multidepth",
            encoder_run=video["run"], encoder_config=video["config"],
            command=train_command(PROTOCOL, video["run"], original["data_root"], run, mode, workers)))
    return dict(video_queue=str(video_queue), video_plan_sha256=file_hash(video_queue / "plan.json"),
        point_queue=str(point_queue), point_plan_sha256=file_hash(point_queue / "plan.json"),
        integration=str(integration), integration_sha256=file_hash(integration / "verification.json"),
        data_root=original["data_root"], data_plan=original["data_plan"],
        data_plan_sha256=original["data_plan_sha256"], robotap_root=original["robotap_root"],
        robotap_extraction_sha256=original["robotap_extraction_sha256"],
        protocol=PROTOCOL, modes=["probe", "adapt"], workers=workers, runs=runs,
        source_sha256={name:file_hash(ROOT / name) for name in QUEUE_SOURCES},
        numerical_scope=dict(probe_resume_checked=evidence["probe_resume_checked"],
                             native_adapt_bitwise_resume_claim=False),
        selection=original["selection"],
        scope="Additional multi-depth SigLIP2/GMMix baseline, same point protocol. One seed; probe/adapt separate. Native adaptation backward is nondeterministic. Parameters and elapsed time are not matched.")


def verify_plan(plan):
    for name,expected in plan["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Multi-depth point source changed: {name}")
    for path,key in ((Path(plan["video_queue"]) / "plan.json", "video_plan_sha256"),
            (Path(plan["point_queue"]) / "plan.json", "point_plan_sha256"),
            (Path(plan["integration"]) / "verification.json", "integration_sha256"),
            (Path(plan["data_root"]) / "plan.json", "data_plan_sha256"),
            (Path(plan["robotap_root"]) / "extraction.json", "robotap_extraction_sha256")):
        if file_hash(path) != plan[key]:
            raise ValueError(f"Multi-depth point dependency changed: {path}")
    verify_video_plan(read_json(Path(plan["video_queue"]) / "plan.json"))
    verify_original_point_plan(read_json(Path(plan["point_queue"]) / "plan.json"))
    verify_integration(plan["integration"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-comparison", required=True)
    parser.add_argument("--point-comparisons", required=True)
    parser.add_argument("--integration", default="results/multidepth_point_integration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--after-video-pid", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("Workers must be nonnegative")
    os.chdir(ROOT)
    torch.set_num_threads(2)
    video, points, integration, output = [Path(p).resolve() for p in
        (args.video_comparison, args.point_comparisons, args.integration, args.output)]
    plan = build_plan(video, points, integration, output, args.workers)
    if args.resume:
        if read_json(output / "plan.json") != plan:
            raise ValueError("Resume plan/source/protocol changed")
        status = read_json(output / "status.json")
        status.pop("error", None)
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        for name in plan["source_sha256"]:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        status = dict(status="planned", operations=[], results={})
    status["dependency_pid"] = args.after_video_pid
    write_json(output / "status.json", status)
    if args.plan_only:
        print("Saved two point-transfer runs; no jobs started", flush=True)
        return
    child, cancelled, fd = None, False, None

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

    def check_inputs():
        if cancelled:
            raise InterruptedError("Additional point queue cancelled")
        verify_plan(plan)
        for filename,key in (("train.jsonl", "train_sha256"), ("validation.jsonl", "val_sha256"),
                             ("complete.json", "complete_sha256")):
            if file_hash(Path(plan["data_root"]) / filename) != dataset[key]:
                raise ValueError("Completed synthetic data changed")
        if file_hash(Path(video_plan["run"]) / "last.pt") != encoder["checkpoint_sha256"]:
            raise ValueError("Initial encoder checkpoint changed")

    def launch(command, name):
        nonlocal child
        check_inputs()
        pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        if any(p.strip().isdigit() and Path(f"/proc/{p.strip()}").exists() for p in pids.splitlines()):
            raise RuntimeError("GPU compute process present; refusing overlapping experiment")
        row = dict(name=name, command=command, started=time.time(), status="running",
                   log=f"{len(status['operations'])+1:03d}_{name}.log")
        status["operations"].append(row)
        status["status"] = "running"
        with (output / row["log"]).open("w") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            row["pid"] = child.pid
            save()
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"{name}: process {child.pid} is running", flush=True)
        code, child = child.returncode, None
        row.update(exit_code=code, finished=time.time(),
                   status="stopped" if cancelled else "complete" if code == 0 else "failed")
        save()
        if cancelled or code:
            raise RuntimeError(f"{name} stopped/failed ({code}); inspect {row['log']}")

    try:
        if args.after_video_pid:
            fd = bind_process(args.after_video_pid, "run_multidepth_comparison.py", "--output", video)
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            status["status"] = "waiting_for_multidepth_video"
            save()
            while not poller.poll(1000):
                if cancelled:
                    raise InterruptedError("Cancelled while waiting; predecessor was not signalled")
            os.close(fd)
            fd = None
        verify_plan(plan)
        if any(read_json(p / "status.json")["status"] != "complete" for p in (video, points)):
            raise ValueError("Preceding video/point comparisons did not finish normally")
        video_plan = read_json(video / "plan.json")
        encoder = verify_video_completion(video_plan["run"], video_plan["config"])
        dataset = verify_dataset(plan["data_root"], plan["data_plan"])
        inputs = dict(encoder=encoder, dataset=dataset)
        if (output / "inputs.json").exists() and read_json(output / "inputs.json") != inputs:
            raise ValueError("Completed inputs changed across attempts")
        write_json(output / "inputs.json", inputs)
        for row in plan["runs"]:
            check_inputs()
            run = Path(row["run"])
            resume = (run / "last.pt").exists()
            complete = False
            if resume:
                assert_training_config(read_json(run / "config.json"), row, plan, dataset, encoder)
                records = [json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines() if line]
                complete = bool(records and records[-1] == dict(event="budget_complete", step=PROTOCOL["steps"]))
            if not complete:
                launch(row["command"] + (["--resume"] if resume else []), row["name"])
            result = verify_point_run(row, plan, dataset, encoder)
            if row["name"] in status["results"]:
                previous = status["results"][row["name"]]
                if previous["training"] != result:
                    raise ValueError("Completed point checkpoint changed")
                evaluation_dir = Path(previous["evaluation"]["report"]).parent
            else:
                index = 1
                while (run / f"robotap_{index:02d}").exists():
                    index += 1
                evaluation_dir = run / f"robotap_{index:02d}"
                launch(UV + ["evaluate_multidepth_points.py", "--checkpoint", str(run / "last.pt"),
                    "--root", plan["robotap_root"], "--output", str(evaluation_dir)], "evaluate_" + row["name"])
            evaluation = verify_robotap(evaluation_dir, result, plan["robotap_extraction_sha256"])
            report = read_json(evaluation_dir / "summary.json")
            if any(plan["source_sha256"].get(name) != value for name,value in report["source_sha256"].items()):
                raise ValueError("Point evaluator source differs")
            audit_path = evaluation_dir / "raw_label_metric_audit.json"
            if not audit_path.exists():
                launch(UV + ["audit_robotap_predictions.py", "--report", evaluation["report"],
                    "--root", plan["robotap_root"], "--output", str(audit_path)], "audit_" + row["name"])
            audit = read_json(audit_path)
            if (audit["scope"] != "Full RoboTAP raw-label metric audit" or audit["videos"] != 265
                    or audit["report_sha256"] != evaluation["report_sha256"]
                    or audit["dataset_extraction_sha256"] != plan["robotap_extraction_sha256"]
                    or audit["metrics"] != evaluation["metrics"]
                    or any(plan["source_sha256"].get(name) != value for name,value in audit["source_sha256"].items())):
                raise ValueError("RoboTAP raw-label audit is partial or inconsistent")
            status["results"][row["name"]] = dict(training=result, evaluation=evaluation,
                raw_label_audit=dict(report=str(audit_path), report_sha256=file_hash(audit_path)))
            save()
        status["status"] = "complete"
        write_json(output / "results.json", dict(scope=plan["scope"], selection=plan["selection"], results=status["results"],
            remaining="Combine with original comparisons; paired uncertainty, cost/history evaluation and multiple seeds remain"))
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
