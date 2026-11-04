"""Predeclared seed replication for the primary and architecture contrasts.

Training code, split, budgets and model definitions are unchanged. Seeds 43/44
add four architectures each to the seed-42 study. Point probes and their state
interventions are repeated; point adaptation remains explicitly single seed.
"""
import copy
from pathlib import Path

from point_checkpoint import file_hash
from point_experiment import PROTOCOL, read_json, train_command as point_command
from multidepth_point_experiment import train_command as multi_point_command
from run_video_comparisons import control_command, evaluation_commands as temporal_evaluations, verify_inputs
from multidepth_experiment import train_command as multi_command, evaluation_commands as multi_evaluations
from run_multidepth_diagnostics import verify_plan as verify_diagnostic_plan


ROOT = Path(__file__).resolve().parent
UV = ["uv", "run", "--project", "baselines/gmmix", "--locked", "python"]
MODELS = ("mamba_interleaved", "attention_interleaved", "mamba_late", "gmmix_multidepth_after_encoder")
SEEDS = (43,44)


def change_seed(config, seed):
    if type(seed) is not int or seed < 0 or seed == config["seed"]:
        raise ValueError("Replication requires a distinct nonnegative integer seed")
    result = copy.deepcopy(config)
    result["seed"] = seed
    return result


def point_plan_for(plan, row):
    return dict(protocol=row["point_protocol"], data_root=plan["data_root"], source_sha256=plan["source_sha256"])


def point_row_for(row):
    return dict(name=row["name"]+"_probe", run=row["point_run"], mode="probe",
        encoder_type="gmmix_multidepth" if row["model"].startswith("gmmix_multidepth") else "temporal",
        encoder_run=row["video_run"], encoder_config=row["video_config"])


def build_plan(diagnostics, output, sources, workers=2):
    diagnostics, output = Path(diagnostics).resolve(), Path(output).resolve()
    diagnostic = read_json(diagnostics / "plan.json")
    verify_diagnostic_plan(diagnostic)
    multi_points = Path(diagnostic["point_queue"])
    mp = read_json(multi_points / "plan.json")
    points = Path(mp["point_queue"])
    pp = read_json(points / "plan.json")
    video = Path(pp["video_queue"])
    vp = read_json(video / "plan.json")
    multi_video = Path(mp["video_queue"])
    mv = read_json(multi_video / "plan.json")
    if (pp["protocol"] != PROTOCOL or PROTOCOL["seed"] != 42 or vp["reference_config"]["seed"] != 42
            or mv["config"]["seed"] != 42 or mp["protocol"] != PROTOCOL):
        raise ValueError("Replication is tied to the original seed-42 fixed protocol")
    controls = {(r["kind"],r["placement"]):r for r in vp["controls"]}
    reference = vp["reference_config"]
    base = {"mamba_interleaved":dict(video_run=vp["reference"], video_config=reference)}
    for kind,placement in (("attention","interleaved"),("mamba","late")):
        base[kind+"_"+placement] = dict(video_run=controls[kind,placement]["run"],
                                       video_config=dict(reference,kind=kind,placement=placement))
    base[MODELS[-1]] = dict(video_run=mv["run"], video_config=mv["config"])
    for row in pp["runs"]+mp["runs"]:
        key = row["encoder_config"]["kind"]+"_"+row["encoder_config"]["placement"]
        if key in base and row["mode"] == "probe":
            base[key]["point_row"] = row
            base[key]["point_queue"] = str(multi_points if key == MODELS[-1] else points)
    if set(base) != set(MODELS) or any("point_row" not in b for b in base.values()):
        raise ValueError("Missing seed-42 probe reference")
    runs = []
    for seed in SEEDS:
        for model in MODELS:
            template = base[model]["video_config"]
            verify_inputs(template)
            config = change_seed(template,seed)
            name = f"{model}_seed{seed}"
            directory = output / name
            video_run, point_run = directory / "video", directory / "probe"
            is_multi = model == MODELS[-1]
            train_video = multi_command(config,video_run,workers) if is_multi else control_command(
                config,config["kind"],config["placement"],video_run,workers)
            evaluations = multi_evaluations(config,video_run,workers) if is_multi else temporal_evaluations(video_run,config,workers)
            point_protocol = dict(PROTOCOL,seed=seed)
            train_point = (multi_point_command(point_protocol,video_run,pp["data_root"],point_run,"probe",workers)
                if is_multi else point_command(point_protocol,video_run,pp["data_root"],point_run,"probe",workers,"temporal"))
            runs.append(dict(name=name, model=model, seed=seed, video_run=str(video_run), point_run=str(point_run),
                video_config=config, point_protocol=point_protocol, train_video=train_video, evaluate_video=evaluations,
                train_point=train_point, point_evaluator="evaluate_multidepth_points.py" if is_multi else "evaluate_point_tracks.py",
                history_evaluator="evaluate_multidepth_history.py" if is_multi else "evaluate_point_history.py"))
    dependencies = dict(diagnostics=str(diagnostics), video=str(video), points=str(points),
                        multidepth_video=str(multi_video), multidepth_points=str(multi_points))
    return dict(reference_seed=42, additional_seeds=list(SEEDS), models=list(MODELS), reference_runs=base, runs=runs,
        dependencies=dependencies, dependency_plans={k:file_hash(Path(p)/"plan.json") for k,p in dependencies.items()},
        data_root=pp["data_root"], data_plan=pp["data_plan"], robotap_root=pp["robotap_root"],
        robotap_extraction_sha256=pp["robotap_extraction_sha256"], workers=workers, history_periods=[1,24],
        source_sha256={name:file_hash(ROOT/name) for name in sources},
        primary_video="Interleaved Mamba-3 minus interleaved temporal attention, streaming top-1",
        primary_point="Same contrast on frozen-encoder probe Average Jaccard",
        secondary="Insertion position: interleaved versus late Mamba-3; multi-depth GMMix; conditional state interventions",
        initialization="Fresh temporal components/readout per seed; same frozen pretrained SigLIP2 revision; no warm start from seed42",
        changed_randomness="Initialization, clip/crop/query sampling and minibatch order. Native GMMix atomic backward also has nondeterministic variation.",
        selection="All predeclared seeds and models retained, including negative results; fixed final 20000 updates; no score-based queue gate",
        reporting="Per-seed scores and paired differences, mean and sample SD over three seeds. No treating clips across seeds as independent extra data.",
        limits="Only four selected architectures and probe/history are replicated. History-free, final-layer GMMix and point adaptation remain seed42 only. Three seeds do not provide a precise estimate of training uncertainty.")


def verify_plan(plan):
    if plan["additional_seeds"] != list(SEEDS) or plan["models"] != list(MODELS):
        raise ValueError("Predeclared replication seeds/models changed")
    if {(r["model"],r["seed"]) for r in plan["runs"]} != {(m,s) for m in MODELS for s in SEEDS} or len(plan["runs"]) != 8:
        raise ValueError("Replication grid is missing or duplicates a model/seed")
    for name,expected in plan["source_sha256"].items():
        if file_hash(ROOT/name) != expected:
            raise ValueError(f"Replication source changed: {name}")
    for key,path in plan["dependencies"].items():
        if file_hash(Path(path)/"plan.json") != plan["dependency_plans"][key]:
            raise ValueError(f"Replication dependency plan changed: {key}")
    if "seed42_summary" in plan and file_hash(Path(plan["seed42_summary"])/"plan.json") != plan["seed42_summary_plan_sha256"]:
        raise ValueError("Seed42 summary plan changed")
    verify_diagnostic_plan(read_json(Path(plan["dependencies"]["diagnostics"])/"plan.json"))
    if file_hash(Path(plan["robotap_root"])/"extraction.json") != plan["robotap_extraction_sha256"]:
        raise ValueError("Replication RoboTAP data changed")
    for row in plan["runs"]:
        if row["video_config"] != change_seed(plan["reference_runs"][row["model"]]["video_config"],row["seed"]):
            raise ValueError("Replication changed video conditions beyond seed")
        if row["point_protocol"] != dict(PROTOCOL,seed=row["seed"]):
            raise ValueError("Replication changed point conditions beyond seed")
        verify_inputs(row["video_config"])


def validate_video_evaluations(row, completion):
    from summarize_video import read_run
    run = Path(row["video_run"])
    values = read_run(run)
    if (values["config"] != row["video_config"] or values["checkpoint_sha256"] != completion["checkpoint_sha256"]
            or len(values["ids"]) != 24777):
        raise ValueError("Replication video evaluation is partial or mismatched")
    if set(values["predictions"]) != {("full","clip"),("full","stream"),("current","clip"),("reverse_history","clip")}:
        raise ValueError("Replication video evaluation conditions differ")
    return dict(checkpoint_sha256=completion["checkpoint_sha256"],
        evaluation_sha256=values["evaluation_sha256"], stream_accuracy=values["scores"]["full","stream"]["accuracy"],
        samples=len(values["ids"]))
