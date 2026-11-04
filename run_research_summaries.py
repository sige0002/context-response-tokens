"""Wait for the verified comparison processes, then produce CPU-only summaries."""
import argparse
import os
from pathlib import Path
import select
import signal
import time

import torch

from point_checkpoint import file_hash
from point_experiment import write_json
from run_point_comparisons import bind_process
from summarize_full_video import summarize as video_summary
from summarize_points import summarize as point_summary


ROOT = Path(__file__).resolve().parent
SOURCES = ("run_research_summaries.py", "summarize_full_video.py", "summarize_video.py", "paired_metrics.py",
           "summarize_points.py", "point_checkpoint.py", "point_experiment.py", "run_point_comparisons.py",
           "run_video_comparisons.py", "gmmix_experiment.py", "evaluate_temporal.py", "summarize_temporal.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-comparisons", required=True)
    parser.add_argument("--gmmix-comparison", required=True)
    parser.add_argument("--point-comparisons", required=True)
    parser.add_argument("--after-gmmix-pid", type=int)
    parser.add_argument("--after-point-pid", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    video, gmmix, points, output = [Path(p).resolve() for p in
                                   (args.video_comparisons, args.gmmix_comparison, args.point_comparisons, args.output)]
    output.mkdir(parents=True, exist_ok=False)
    hashes = {name:file_hash(ROOT / name) for name in SOURCES}
    plans = {str(path / "plan.json"):file_hash(path / "plan.json") for path in (video,gmmix,points)}
    for name in hashes:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    write_json(output / "plan.json", dict(video_queue=str(video), gmmix_queue=str(gmmix), point_queue=str(points),
        plan_sha256=plans, source_sha256=hashes, primary_video_contrast="Mamba-3 interleaved minus time attention interleaved",
        primary_point_contrast="Same contrast with frozen encoders (probe), average Jaccard",
        bootstrap=dict(draws=10000, seed=17, interval="paired cluster percentile 95%", conditional_on="fitted weights")))
    status = dict(status="starting", started=time.time(), results={},
                  dependency_pids=dict(video=args.after_gmmix_pid, points=args.after_point_pid))
    descriptors, cancelled = {}, False

    def save():
        write_json(output / "status.json", status)

    def stop(signum, frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for name, pid, script, directory in (
            ("video", args.after_gmmix_pid, "run_gmmix_comparison.py", gmmix),
            ("points", args.after_point_pid, "run_point_comparisons.py", points)):
            if pid is not None:
                descriptors[name] = bind_process(pid, script, "--output", directory)
        for name in ("video", "points"):
            if name in descriptors:
                status["status"] = "waiting_for_" + name
                save()
                poller = select.poll()
                poller.register(descriptors[name], select.POLLIN)
                while not poller.poll(1000):
                    if cancelled:
                        raise InterruptedError("Summary queue cancelled; training dependencies were not signalled")
                os.close(descriptors.pop(name))
            if cancelled:
                raise InterruptedError("Summary queue cancelled")
            for source, digest in hashes.items():
                if file_hash(ROOT / source) != digest:
                    raise ValueError(f"Summary implementation changed: {source}")
            for path, digest in plans.items():
                if file_hash(path) != digest:
                    raise ValueError(f"Comparison plan changed: {path}")
            status["status"] = "summarizing_" + name
            save()
            result = video_summary(video,gmmix) if name == "video" else point_summary(points)
            path = output / (name + ".json")
            write_json(path, result)
            status["results"][name] = dict(report=str(path), sha256=file_hash(path), finished=time.time())
            save()
            print(f"Completed verified {name} comparison summary", flush=True)
        status["status"] = "complete"
    except BaseException as exc:
        status.update(status="stopped" if cancelled else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for fd in descriptors.values():
            os.close(fd)
        save()


if __name__ == "__main__":
    main()
