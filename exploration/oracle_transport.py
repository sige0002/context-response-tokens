"""CPU diagnostic of spatial memory transport using oracle Kubric tracks.

No trained recognition result: observations are 224 RGB, but memory values are
random spatial fingerprints and untrained RGB-patch projections. Oracle maps
use disjoint fitting tracks at t-1/t; held-out tracks evaluate old-memory reads.
The dense SO(N) fit is an optimistic expressivity diagnostic for local rotations,
not a deployable architecture, nor a bound on recognition accuracy.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image
import torch

from kubric_data import load_points


METHODS = ("fixed", "bilinear_flow", "orthogonal_oracle", "ridge_oracle")
FEATURES = ("random_fingerprints", "rgb_patch_projection")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def basis(xy,grid):
    """Bilinear point sampling on align_corners=False grid, border padding.

    xy is normalized by native width/height, matching the point protocol.
    Invalid invisible points must be excluded by the caller.
    """
    if xy.ndim != 2 or xy.shape[1] != 2 or not torch.isfinite(xy).all():
        raise ValueError("Finite [points,xy] positions required")
    if grid < 2:
        raise ValueError("Grid must have at least two cells per axis")
    pixel = (xy*grid-.5).clamp(0,grid-1)
    lo = pixel.floor().long()
    hi = (lo+1).clamp(max=grid-1)
    fraction = pixel-lo
    indices,weights = [],[]
    for yy,wy in ((lo[:,1],1-fraction[:,1]),(hi[:,1],fraction[:,1])):
        for xx,wx in ((lo[:,0],1-fraction[:,0]),(hi[:,0],fraction[:,0])):
            indices.append(yy*grid+xx)
            weights.append(wy*wx)
    return torch.stack(indices,1),torch.stack(weights,1)


def sample(memory,xy,grid):
    indices,weights = basis(xy,grid)
    return (memory[indices]*weights[...,None]).sum(1)


def cross_gram(a,b,grid):
    """B(a)^T B(b), avoiding a dense points-by-grid array."""
    ia,wa = basis(a,grid)
    ib,wb = basis(b,grid)
    n = grid*grid
    index = (ia[:,:,None]*n+ib[:,None,:]).reshape(-1)
    weight = (wa[:,:,None]*wb[:,None,:]).reshape(-1)
    result = torch.zeros(n*n,dtype=a.dtype)
    result.index_add_(0,index,weight)
    return result.reshape(n,n)/max(1,len(a))


def orthogonal_fit(cross,regularizer):
    """Minimize ||B_current Q-B_previous||_F with Q in SO(N), plus I prior."""
    identity = torch.eye(len(cross),dtype=cross.dtype)
    u,_,vh = torch.linalg.svd(cross+regularizer*identity)
    sign = torch.ones(len(cross),dtype=cross.dtype)
    sign[-1] = torch.linalg.det(u@vh).sign()
    return (u*sign)@vh


def bilinear_flow(previous,current,grid):
    n = grid*grid
    indices,weights = basis(current,grid)
    count = torch.zeros(n,dtype=current.dtype)
    flow = torch.zeros(n,2,dtype=current.dtype)
    for index in range(4):
        count.index_add_(0,indices[:,index],weights[:,index])
        flow.index_add_(0,indices[:,index],weights[:,index,None]*(previous-current))
    flow = flow/count.clamp_min(1e-12)[:,None]
    yy,xx = torch.meshgrid(torch.arange(grid),torch.arange(grid),indexing="ij")
    centers = (torch.stack((xx,yy),-1).reshape(n,2).to(current.dtype)+.5)/grid
    source_indices,source_weights = basis(centers+flow,grid)
    warp = torch.zeros(n,n,dtype=current.dtype)
    warp.scatter_add_(1,source_indices,source_weights)
    return warp,int((count>0).sum())


def operators(previous,current,grid,prior_ratio):
    if len(previous) < 2 or previous.shape != current.shape:
        raise ValueError("Need at least two jointly visible fitting tracks")
    cross = cross_gram(current,previous,grid)
    gram = cross_gram(current,current,grid)
    n = grid*grid
    identity = torch.eye(n,dtype=gram.dtype)
    scale = float(gram.trace())/n
    regularizer = prior_ratio*scale
    if regularizer <= 0:
        raise ValueError("Positive fixed identity prior required")
    orthogonal = orthogonal_fit(cross,regularizer)
    ridge = torch.linalg.solve(gram+regularizer*identity,cross+regularizer*identity)
    warp,coverage = bilinear_flow(previous,current,grid)
    return dict(fixed=identity,bilinear_flow=warp,orthogonal_oracle=orthogonal,ridge_oracle=ridge),dict(
        fit_tracks=len(previous),covered_cells=coverage,cells=n,regularizer=regularizer,
        orthogonality_error=float((orthogonal.T@orthogonal-identity).abs().max()),
        orthogonal_constant_field_error=float((orthogonal@torch.ones(n,dtype=gram.dtype)-1).square().mean()))


def memory_values(directory,grid,channels,seed):
    with zipfile.ZipFile(directory/"images.zip") as archive:
        rgb = np.asarray(Image.open(io.BytesIO(archive.read("000.png"))).convert("RGB"))
    if rgb.shape != (224,224,3) or 224%grid:
        raise ValueError("Expected prepared 224 RGB and a dividing grid")
    size = 224//grid
    patches = torch.from_numpy(rgb.copy()).double()/255
    patches = patches.reshape(grid,size,grid,size,3).permute(0,2,1,3,4).reshape(grid*grid,-1)
    generator = torch.Generator().manual_seed(seed)
    fingerprints = torch.randn(grid*grid,channels,generator=generator,dtype=torch.float64)
    projection = torch.randn(patches.shape[1],channels,generator=generator,dtype=torch.float64)/patches.shape[1]**.5
    visual = patches@projection
    visual = (visual-visual.mean(0))/visual.std(0).clamp_min(1e-6)
    return torch.cat((fingerprints,visual),dim=1)


def track_partition(count,seed,fit_count,query_count):
    if fit_count+query_count>count:
        raise ValueError("Insufficient disjoint tracks")
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    return order[:fit_count],order[fit_count:fit_count+query_count]


def errors(prediction,target,mask):
    if not mask.any():
        return None
    prediction,target = prediction[mask],target[mask]
    energy = float(target.square().sum())
    return dict(points=len(target),squared_error=float((prediction-target).square().sum()),
        target_energy=energy,normalized_mse=float((prediction-target).square().sum())/max(energy,1e-12))


def evaluate_scene(directory,config):
    scene = directory.name
    metadata = json.loads((directory/"metadata.json").read_text())
    for name in ("images.zip","points.npz"):
        if digest(directory/name)!=metadata["files"][name]["sha256"]:
            raise ValueError("Prepared data differs from source conversion record")
    points,occluded = load_points(directory/"points.npz")
    visible = ~occluded
    visible &= np.isfinite(points).all(-1)
    visible &= (points[...,0]>=0)&(points[...,0]<=metadata["native_width"]-1)
    visible &= (points[...,1]>=0)&(points[...,1]<=metadata["native_height"]-1)
    points = points/np.array([metadata["native_width"],metadata["native_height"]],dtype=np.float32)
    fit,query = track_partition(len(points),config["seed"]+int(scene),config["fit_tracks"],config["query_tracks"])
    initial = memory_values(directory,config["grid"],config["channels"],config["seed"]+int(scene))
    frames = config["frames"]
    results = []
    for stride in config["strides"]:
        indices = np.arange(frames)*stride
        if indices[-1]>=metadata["frames"]:
            raise ValueError("Requested horizon exceeds actual data")
        memories = {key:initial.clone() for key in METHODS}
        # Query positions are evaluated only when visible at frame zero/current.
        start = points[query,0]
        valid_start = visible[query,0]
        safe_start = np.where(valid_start[:,None],start,0)
        target = sample(initial,torch.from_numpy(safe_start).double(),config["grid"])
        continuous = valid_start.copy()
        for index in range(1,len(indices)):
            a,b = int(indices[index-1]),int(indices[index])
            valid_fit = visible[fit,a]&visible[fit,b]
            transforms,info = operators(torch.from_numpy(points[fit[valid_fit],a]).double(),
                torch.from_numpy(points[fit[valid_fit],b]).double(),config["grid"],config["prior_ratio"])
            valid = valid_start&visible[query,b]
            # Continuity includes intermediate original frames, not just sampled frames.
            continuous &= visible[query,a+1:b+1].all(1)
            safe_now = np.where(valid[:,None],points[query,b],0)
            current = torch.from_numpy(safe_now).double()
            displacement = np.linalg.norm((safe_now-safe_start)*224,axis=1)
            strata = dict(all_visible=valid,continuous=valid&continuous,
                after_occlusion=valid&~continuous,moving_over_2px=valid&(displacement>2))
            row = dict(stride=stride,frame=b,updates=index,fit=info,methods={})
            valid_pair = visible[query,a]&visible[query,b]
            previous_xy = torch.from_numpy(points[query[valid_pair],a]).double()
            current_xy = torch.from_numpy(points[query[valid_pair],b]).double()
            # Fresh independent field isolates single-step alignment from accumulation.
            prior_samples = sample(initial,previous_xy,config["grid"])
            for method,matrix in transforms.items():
                memories[method] = matrix@memories[method]
                predicted = sample(memories[method],current,config["grid"])
                one_step = sample(matrix@initial,current_xy,config["grid"])
                row["methods"][method] = {}
                for fi,feature in enumerate(FEATURES):
                    region = slice(fi*config["channels"],(fi+1)*config["channels"])
                    row["methods"][method][feature] = dict(
                        state_energy_ratio=float(memories[method][:,region].square().sum()/initial[:,region].square().sum()),
                        one_step=errors(one_step[:,region],prior_samples[:,region],torch.ones(len(prior_samples),dtype=torch.bool)),
                        retained={key:errors(predicted[:,region],target[:,region],torch.from_numpy(mask)) for key,mask in strata.items()})
            results.append(row)
    return dict(scene=scene,metadata_sha256=digest(directory/"metadata.json"),source_sha256=metadata["source_sha256"],
        fit_indices_sha256=hashlib.sha256(fit.tobytes()).hexdigest(),query_indices_sha256=hashlib.sha256(query.tobytes()).hexdigest(),
        rows=results)


def summarize(scenes,config):
    output=[]
    for stride in config["strides"]:
        end=(config["frames"]-1)*stride
        for feature in FEATURES:
            for stratum in ("all_visible","continuous","after_occlusion","moving_over_2px"):
                for method in METHODS:
                    selected=[next(r for r in s["rows"] if r["stride"]==stride and r["frame"]==end)["methods"][method][feature] for s in scenes]
                    metrics=[r["retained"][stratum] for r in selected if r["retained"][stratum] is not None]
                    output.append(dict(stride=stride,end_frame=end,feature=feature,stratum=stratum,method=method,
                        scenes=len(metrics),points=sum(r["points"] for r in metrics),
                        mean_scene_normalized_mse=float(np.mean([r["normalized_mse"] for r in metrics])) if metrics else None,
                        mean_state_energy_ratio=float(np.mean([r["state_energy_ratio"] for r in selected]))))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default="data/cotracker3_kubric/scenes")
    parser.add_argument("--output",required=True)
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    root,output=Path(args.root).resolve(),Path(args.output).resolve()
    config=dict(scenes=["0016","0020","0030","0061","0065","0081","0123","0128"],
        grid=14,channels=32,frames=24,strides=[1,4],seed=17,fit_tracks=8192,query_tracks=8192,prior_ratio=1e-3,
        computation="CPU float64, 2 threads; no CUDA, training or checkpoint changes",
        scope="Oracle correspondence diagnostic on eight synthetic rendered validation scenes; not recognition or tracking accuracy",
        selection="First eight prepared fixed-validation scene IDs, sorted before examining outcomes",
        limitations="Current/previous GT coordinates estimate operators; held-out tracks evaluate memory. Dense oracle maps are not deployable. No new evidence is written after frame zero. Untrained RGB projections and random state values do not measure learned visual representations.")
    ids=[f"{i:04d}" for i in range(5869)];random.Random(42).shuffle(ids)
    if not set(config["scenes"])<=set(ids[:300]):
        raise ValueError("All selected scenes must belong to the fixed validation split")
    source_files=[Path(__file__),Path(__file__).resolve().parents[1]/"kubric_data.py"]
    plan=dict(config=config,root=str(root),sources={str(p.resolve()):digest(p) for p in source_files},
              metadata={s:digest(root/s/"metadata.json") for s in config["scenes"]})
    if args.resume:
        if json.loads((output/"plan.json").read_text())!=plan:
            raise ValueError("Oracle inputs/protocol/source changed")
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    if args.plan_only:
        print("Oracle-only plan saved; eight scenes, no fitting or training run",flush=True)
        return
    start=time.monotonic()
    reports=[]
    for scene in config["scenes"]:
        path=output/(scene+".json")
        # Do not silently accept a partial cached scene or score-selected subset.
        if path.exists():
            report=json.loads(path.read_text())
            if report["metadata_sha256"]!=plan["metadata"][scene] or len(report["rows"])!=46:
                raise ValueError("Cached scene is partial or mismatched")
        else:
            report=evaluate_scene(root/scene,config)
            temporary=path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report,indent=2)+"\n");temporary.replace(path)
        reports.append(report)
        print(json.dumps(dict(scene=scene,completed=len(reports),elapsed_seconds=time.monotonic()-start)),flush=True)
    result=dict(config=config,plan_sha256=digest(output/"plan.json"),scenes={s["scene"]:digest(output/(s["scene"]+".json")) for s in reports},
                summary=summarize(reports,config),elapsed_seconds=time.monotonic()-start)
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
