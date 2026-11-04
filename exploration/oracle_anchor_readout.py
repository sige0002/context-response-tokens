"""Follow-up diagnostic: read once from an immutable initial memory.

This uses privileged frame-zero/current GT correspondences. It is not a causal
correspondence estimator and does not establish an implementable architecture's
accuracy. It tests whether repeated regridding is a useful failure mode to study.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from kubric_data import load_points
from exploration.oracle_transport import (FEATURES,bilinear_flow,cross_gram,digest,errors,
    memory_values,sample,track_partition)


def evaluate(directory,config):
    scene=directory.name
    metadata=json.loads((directory/"metadata.json").read_text())
    for name in ("images.zip","points.npz"):
        if digest(directory/name)!=metadata["files"][name]["sha256"]:
            raise ValueError("Prepared scene changed")
    points,occluded=load_points(directory/"points.npz")
    visible=(~occluded)&np.isfinite(points).all(-1)
    visible&=(points[...,0]>=0)&(points[...,0]<=metadata["native_width"]-1)
    visible&=(points[...,1]>=0)&(points[...,1]<=metadata["native_height"]-1)
    points=points/np.array([metadata["native_width"],metadata["native_height"]],dtype=np.float32)
    fit,query=track_partition(len(points),config["seed"]+int(scene),config["fit_tracks"],config["query_tracks"])
    memory=memory_values(directory,config["grid"],config["channels"],config["seed"]+int(scene))
    initial_visible=visible[query,0]
    origin=np.where(initial_visible[:,None],points[query,0],0)
    target=sample(memory,torch.from_numpy(origin).double(),config["grid"])
    rows=[]
    for stride in config["strides"]:
        for update in range(1,config["frames"]):
            frame=update*stride
            accepted=visible[fit,0]&visible[fit,frame]
            previous=torch.from_numpy(points[fit[accepted],0]).double()
            current=torch.from_numpy(points[fit[accepted],frame]).double()
            warp,coverage=bilinear_flow(previous,current,config["grid"])
            cross=cross_gram(current,previous,config["grid"])
            gram=cross_gram(current,current,config["grid"])
            regularizer=config["prior_ratio"]*gram.trace()/len(gram)
            identity=torch.eye(len(gram),dtype=gram.dtype)
            ridge=torch.linalg.solve(gram+regularizer*identity,cross+regularizer*identity)
            valid=initial_visible&visible[query,frame]
            continuous=visible[query,:frame+1].all(1)
            xy=np.where(valid[:,None],points[query,frame],0)
            distance=np.linalg.norm((xy-origin)*224,axis=1)
            strata=dict(all_visible=valid,continuous=valid&continuous,
                after_occlusion=valid&~continuous,moving_over_2px=valid&(distance>2))
            row=dict(stride=stride,frame=frame,updates=update,fit_tracks=int(accepted.sum()),covered_cells=coverage,
                     persistent_memory_energy_ratio=1.,methods={})
            for method,matrix in (("anchor_bilinear",warp),("anchor_ridge",ridge)):
                output=matrix@memory
                read=sample(output,torch.from_numpy(xy).double(),config["grid"])
                row["methods"][method]={}
                for i,feature in enumerate(FEATURES):
                    region=slice(i*config["channels"],(i+1)*config["channels"])
                    row["methods"][method][feature]=dict(retained={key:errors(read[:,region],target[:,region],torch.from_numpy(mask)) for key,mask in strata.items()},
                        readout_energy_ratio=float(output[:,region].square().sum()/memory[:,region].square().sum()))
            rows.append(row)
    return dict(scene=scene,rows=rows)


def paired_summary(original,additional,config):
    rows=[]
    for scene in config["scenes"]:
        old=original[scene]["rows"]
        new=additional[scene]["rows"]
        if len(old)!=len(new):
            raise ValueError("Mismatched diagnostic grids")
        for a,b in zip(old,new):
            if any(a[key]!=b[key] for key in ("stride","frame","updates")):
                raise ValueError("Mismatched diagnostic timeline")
            if a["updates"]!=config["frames"]-1:
                continue
            for feature in FEATURES:
                for stratum in ("all_visible","continuous","after_occlusion","moving_over_2px"):
                    baseline=a["methods"]["bilinear_flow"][feature]["retained"][stratum]
                    for method in b["methods"]:
                        current=b["methods"][method][feature]["retained"][stratum]
                        if (baseline is None)!=(current is None):
                            raise ValueError("Selection changed")
                        if baseline is None:
                            continue
                        if (current["points"]!=baseline["points"] or
                                abs(current["target_energy"]-baseline["target_energy"])>1e-8):
                            raise ValueError("Held-out query cohort or target memory changed")
                        rows.append(dict(scene=scene,stride=a["stride"],frame=a["frame"],feature=feature,
                            stratum=stratum,method=method,points=current["points"],
                            repeated_normalized_mse=baseline["normalized_mse"],
                            anchor_normalized_mse=current["normalized_mse"],
                            difference=current["normalized_mse"]-baseline["normalized_mse"]))
    groups=[]
    for stride in config["strides"]:
        for feature in FEATURES:
            for stratum in ("all_visible","continuous","after_occlusion","moving_over_2px"):
                for method in ("anchor_bilinear","anchor_ridge"):
                    values=[r for r in rows if (r["stride"],r["feature"],r["stratum"],r["method"])==(stride,feature,stratum,method)]
                    groups.append(dict(stride=stride,feature=feature,stratum=stratum,method=method,scenes=len(values),
                        points=sum(r["points"] for r in values),
                        mean_repeated_normalized_mse=float(np.mean([r["repeated_normalized_mse"] for r in values])) if values else None,
                        mean_anchor_normalized_mse=float(np.mean([r["anchor_normalized_mse"] for r in values])) if values else None,
                        mean_difference=float(np.mean([r["difference"] for r in values])) if values else None,
                        scenes_with_lower_error=sum(r["difference"]<0 for r in values)))
    return dict(rows=rows,summary=groups)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    original,output=Path(args.original).resolve(),Path(args.output).resolve()
    source=json.loads((original/"plan.json").read_text())
    baseline=json.loads((original/"summary.json").read_text())
    if baseline["plan_sha256"]!=digest(original/"plan.json"):
        raise ValueError("Original plan changed")
    for name,value in source["sources"].items():
        if digest(name)!=value:
            raise ValueError("Original source changed")
    config=source["config"]
    plan=dict(original=str(original),original_summary_sha256=digest(original/"summary.json"),config=config,
        source_sha256=digest(__file__),scope=__doc__,
        hypothesis="A fixed anchor avoids accumulated interpolation; GT long-range correspondence may be much harder to learn than adjacent-frame flow.",
        limits="Oracle initial-to-current associations are privileged. Fitting-track visibility coverage differs from adjacent-frame fitting. No trained model, recognition score, SOTA claim, or deployable fixed-budget victory.")
    if args.resume:
        if json.loads((output/"plan.json").read_text())!=plan:
            raise ValueError("Anchor protocol/source changed")
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    if args.plan_only:
        print("Anchor-readout follow-up plan fixed; no fitting run",flush=True)
        return
    old,new={},{}
    started=time.monotonic()
    for scene in config["scenes"]:
        if digest(original/(scene+".json"))!=baseline["scenes"][scene]:
            raise ValueError("Original scene report changed")
        if digest(Path(source["root"])/scene/"metadata.json")!=source["metadata"][scene]:
            raise ValueError("Input metadata changed")
        old[scene]=json.loads((original/(scene+".json")).read_text())
        new[scene]=evaluate(Path(source["root"])/scene,config)
        (output/(scene+".json")).write_text(json.dumps(new[scene],indent=2)+"\n")
        print(json.dumps(dict(scene=scene,completed=len(new),elapsed_seconds=time.monotonic()-started)),flush=True)
    report=dict(plan_sha256=digest(output/"plan.json"),**paired_summary(old,new,config),
        scene_sha256={s:digest(output/(s+".json")) for s in config["scenes"]},elapsed_seconds=time.monotonic()-started)
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
