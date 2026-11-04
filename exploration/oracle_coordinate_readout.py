"""Control memory resampling using identical adjacent-frame oracle flow maps.

Repeated: H_t = W_t H_(t-1).
Address: C_t = W_t C_(t-1), H_read_t = sample(H_0, C_t).
The latter preserves H_0 but adds two coordinates per grid cell. Coordinates
still accumulate correspondence/interpolation errors. No learned model result.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from kubric_data import load_points
from exploration.oracle_transport import (FEATURES, bilinear_flow, digest,
    errors, memory_values, sample, track_partition)


def centers(grid):
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    return (torch.stack((xx, yy), -1).reshape(-1, 2).double() + .5) / grid


def evaluate(directory, config, original):
    metadata = json.loads((directory / "metadata.json").read_text())
    for name in ("images.zip", "points.npz"):
        if digest(directory / name) != metadata["files"][name]["sha256"]:
            raise ValueError("Prepared data changed")
    points, occluded = load_points(directory / "points.npz")
    visible = (~occluded) & np.isfinite(points).all(-1)
    visible &= (points[..., 0] >= 0) & (points[..., 0] <= metadata["native_width"] - 1)
    visible &= (points[..., 1] >= 0) & (points[..., 1] <= metadata["native_height"] - 1)
    points = points / np.array([metadata["native_width"], metadata["native_height"]], dtype=np.float32)
    fit, query = track_partition(len(points), config["seed"] + int(directory.name),
                                 config["fit_tracks"], config["query_tracks"])
    grid = config["grid"]
    memory = memory_values(directory, grid, config["channels"], config["seed"] + int(directory.name))
    start_visible = visible[query, 0]
    origin = np.where(start_visible[:, None], points[query, 0], 0)
    target = sample(memory, torch.from_numpy(origin).double(), grid)
    rows = []
    for stride in config["strides"]:
        coordinates = centers(grid)
        repeated = memory.clone()
        for update in range(1, config["frames"]):
            a, b = (update - 1) * stride, update * stride
            accepted = visible[fit, a] & visible[fit, b]
            if accepted.sum() < 2:
                raise ValueError("Insufficient fitting tracks")
            warp, coverage = bilinear_flow(torch.from_numpy(points[fit[accepted], a]).double(),
                                           torch.from_numpy(points[fit[accepted], b]).double(), grid)
            repeated = warp @ repeated
            coordinates = warp @ coordinates
            addressed = sample(memory, coordinates, grid)
            valid = start_visible & visible[query, b]
            continuous = visible[query, :b+1].all(1)
            xy = np.where(valid[:, None], points[query, b], 0)
            distance = np.linalg.norm((xy-origin)*224, axis=1)
            strata = dict(all_visible=valid, continuous=valid & continuous,
                          after_occlusion=valid & ~continuous, moving_over_2px=valid & (distance > 2))
            old = next(r for r in original["rows"] if r["stride"] == stride and r["frame"] == b)
            if old["fit"]["fit_tracks"] != int(accepted.sum()) or old["fit"]["covered_cells"] != coverage:
                raise ValueError("Flow fitting cohort changed")
            row = dict(stride=stride, frame=b, updates=update, fit_tracks=int(accepted.sum()),
                       covered_cells=coverage, extra_coordinate_scalars=grid*grid*2, methods={})
            for method, values in (("repeated_bilinear", repeated), ("coordinate_readout", addressed)):
                read = sample(values, torch.from_numpy(xy).double(), grid)
                row["methods"][method] = {}
                for i, feature in enumerate(FEATURES):
                    region = slice(i*config["channels"], (i+1)*config["channels"])
                    metrics = {key: errors(read[:, region], target[:, region], torch.from_numpy(mask))
                               for key, mask in strata.items()}
                    for key, current in metrics.items():
                        baseline = old["methods"]["bilinear_flow"][feature]["retained"][key]
                        if (baseline is None) != (current is None):
                            raise ValueError("Query cohort changed")
                        if baseline is not None:
                            if (current["points"] != baseline["points"] or
                                    abs(current["target_energy"] - baseline["target_energy"]) > 1e-8):
                                raise ValueError("Target or query cohort changed")
                            if method == "repeated_bilinear" and abs(current["normalized_mse"] - baseline["normalized_mse"]) > 1e-10:
                                raise ValueError("Repeated baseline failed to reproduce")
                    row["methods"][method][feature] = dict(retained=metrics)
            rows.append(row)
    return dict(scene=directory.name, rows=rows)


def summarize(reports, config):
    rows, groups = [], []
    for scene in reports:
        for row in scene["rows"]:
            if row["updates"] != config["frames"] - 1:
                continue
            for feature in FEATURES:
                for stratum, baseline in row["methods"]["repeated_bilinear"][feature]["retained"].items():
                    if baseline is None:
                        continue
                    current = row["methods"]["coordinate_readout"][feature]["retained"][stratum]
                    rows.append(dict(scene=scene["scene"], stride=row["stride"], frame=row["frame"],
                        feature=feature, stratum=stratum, points=current["points"],
                        repeated_normalized_mse=baseline["normalized_mse"], coordinate_normalized_mse=current["normalized_mse"],
                        difference=current["normalized_mse"]-baseline["normalized_mse"]))
    for stride in config["strides"]:
        for feature in FEATURES:
            for stratum in ("all_visible", "continuous", "after_occlusion", "moving_over_2px"):
                values = [r for r in rows if (r["stride"], r["feature"], r["stratum"]) == (stride, feature, stratum)]
                groups.append(dict(stride=stride, feature=feature, stratum=stratum, scenes=len(values),
                    points=sum(r["points"] for r in values),
                    mean_repeated_normalized_mse=float(np.mean([r["repeated_normalized_mse"] for r in values])) if values else None,
                    mean_coordinate_normalized_mse=float(np.mean([r["coordinate_normalized_mse"] for r in values])) if values else None,
                    mean_difference=float(np.mean([r["difference"] for r in values])) if values else None,
                    scenes_with_lower_error=sum(r["difference"] < 0 for r in values)))
    return dict(rows=rows, summary=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    original, output = Path(args.original).resolve(), Path(args.output).resolve()
    source = json.loads((original/"plan.json").read_text())
    baseline = json.loads((original/"summary.json").read_text())
    if baseline["plan_sha256"] != digest(original/"plan.json"):
        raise ValueError("Original plan changed")
    for name, value in source["sources"].items():
        if digest(name) != value:
            raise ValueError("Original source changed")
    config = source["config"]
    plan = dict(original=str(original), original_summary_sha256=digest(original/"summary.json"), config=config,
                source_sha256=digest(__file__), scope=__doc__,
                limits="Oracle adjacent flow in both arms; no new observations. Same maps, fit/query cohorts, fixed prior, dense output grid. Address method uses 2 extra scalars per cell. Coordinate averaging still loses multimodal associations. Not recognition performance or fixed-budget victory.")
    if args.resume:
        if json.loads((output/"plan.json").read_text()) != plan:
            raise ValueError("Protocol/source changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    if args.plan_only:
        print("Same-flow coordinate control fixed; no fitting run", flush=True)
        return
    started, reports = time.monotonic(), []
    for scene in config["scenes"]:
        if digest(original/(scene+".json")) != baseline["scenes"][scene]:
            raise ValueError("Original report changed")
        if digest(Path(source["root"])/scene/"metadata.json") != source["metadata"][scene]:
            raise ValueError("Input metadata changed")
        result = evaluate(Path(source["root"])/scene, config, json.loads((original/(scene+".json")).read_text()))
        reports.append(result)
        (output/(scene+".json")).write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(dict(scene=scene, completed=len(reports), elapsed_seconds=time.monotonic()-started)), flush=True)
    result = dict(plan_sha256=digest(output/"plan.json"), **summarize(reports, config),
                  scene_sha256={s:digest(output/(s+".json")) for s in config["scenes"]}, elapsed_seconds=time.monotonic()-started)
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
