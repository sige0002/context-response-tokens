"""Independent RGB-edge and real-data repeatability audit after the input fix."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import av
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import SiglipImageProcessor

from point_experiment import write_json
from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from video_checkpoint import file_hash
from video_data import RawVideoClips, frame_to_rgb, RGB_INTERPOLATION, temporal_indices


def hashes(loader):
    values=[]
    for pixels,labels in loader:
        for image,label in zip(pixels,labels):
            values.append(dict(pixels_sha256=hashlib.sha256(image.numpy().tobytes()).hexdigest(),label=int(label)))
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--samples-per-split',type=int,default=32)
    args=parser.parse_args()
    destination=Path(args.output)
    if destination.exists() or args.samples_per_split<1:
        raise ValueError('Choose new output and positive sample count')
    torch.set_num_threads(4)
    processor=SiglipImageProcessor.from_pretrained(DEFAULT_MODEL,revision=DEFAULT_REVISION,local_files_only=True)
    known=av.VideoFrame(427,240,'yuv420p')
    known.colorspace=5;known.color_range=1
    for index,plane in enumerate(known.planes):
        plane.update(bytes([100 if index==0 else 128])*plane.buffer_size)
    old=np.asarray(known.to_image()).copy()
    corrected=np.asarray(frame_to_rgb(known)).copy()
    constant=dict(yuv=[100,128,128],expected_limited_bt601_rgb=[98,98,98],
        default_last_column_unique=np.unique(old[:,-1],axis=0).tolist(),
        default_max_error=int(np.abs(old.astype(int)-98).max()),
        corrected_unique=np.unique(corrected.reshape(-1,3),axis=0).tolist(),
        corrected_max_error=int(np.abs(corrected.astype(int)-98).max()))
    assert constant['corrected_max_error']<=1 and (corrected==corrected[0,0]).all()
    reports={}
    for split in ('train','val'):
        dataset=RawVideoClips(f'data/ssv2/videos_trainval/{split}.jsonl',processor,8,174,training=split=='train')
        indices=sorted(random.Random(17).sample(range(len(dataset)),args.samples_per_split))
        subset=Subset(dataset,indices)
        observed=[]
        for workers in (0,0,2):
            loader=DataLoader(subset,batch_size=4,num_workers=workers,
                multiprocessing_context='spawn' if workers else None)
            observed.append(hashes(loader))
        assert observed[0]==observed[1]==observed[2],split
        reports[split]=dict(samples=len(indices),training_augmentation=split=='train',workers=[0,0,2],
            exact_repeated_pixels_and_labels=True,indices=indices,
            rows=[dict(id=dataset.rows[i]['id'],source_sha256=dataset.rows[i]['sha256'],**v)
                  for i,v in zip(indices,observed[0])])
        print(json.dumps(dict(split=split,samples=len(indices),exact_repeats=True,spawn_workers_match=True)),flush=True)
    # Bind the original three failing examples; compare visible decoded planes
    # before RGB conversion to distinguish decoding from conversion.
    failing=RawVideoClips('results/ssv2_history_diagnostic_interim_seed17/subset.jsonl',processor,8,174)
    diagnostics=[]
    for index in (0,10,42):
        row=failing.rows[index]
        decoded=[]
        for _ in range(2):
            with av.open(str(row['video'])) as container:
                container.streams.video[0].thread_count=1
                decoded.append(list(container.decode(video=0)))
        selected=temporal_indices(len(decoded[0]),8)
        planes_equal=True
        raw_bad=0;changed_columns=set();fixed_rgb_equal=True
        images=[[],[]]
        for time_index in selected:
            a,b=decoded[0][time_index],decoded[1][time_index]
            assert a.format.name==b.format.name=='yuv420p'
            for p,q in zip(a.planes,b.planes):
                x=np.frombuffer(p,dtype=np.uint8).reshape(p.height,p.line_size)[:,:p.width]
                y=np.frombuffer(q,dtype=np.uint8).reshape(q.height,q.line_size)[:,:q.width]
                planes_equal &= np.array_equal(x,y)
            old_a,old_b=np.asarray(a.to_image()).copy(),np.asarray(b.to_image()).copy()
            changed=(old_a!=old_b).any(-1)
            raw_bad+=int(changed.sum());changed_columns.update(np.nonzero(changed)[1].tolist())
            new_a,new_b=frame_to_rgb(a),frame_to_rgb(b)
            fixed_rgb_equal &= np.array_equal(np.asarray(new_a),np.asarray(new_b))
            images[0].append(new_a);images[1].append(new_b)
        x,y=[processor(images=v,return_tensors='pt').pixel_values for v in images]
        assert planes_equal and fixed_rgb_equal and torch.equal(x,y)
        diagnostics.append(dict(id=row['id'],width=decoded[0][0].width,height=decoded[0][0].height,
            source_sha256=file_hash(row['video']),decoded_visible_planes_equal=planes_equal,
            default_rgb_changed_pixels=raw_bad,default_changed_columns=sorted(changed_columns),
            corrected_rgb_equal=fixed_rgb_equal,corrected_tensor_max_abs=float((x-y).abs().max()),
            corrected_tensor_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest()))
    report=dict(scope='Known-color edge correctness and selected real-video preprocessing audit; no recognition accuracy',
        environment=dict(pyav=av.__version__,libraries=av.library_versions,torch=torch.__version__),
        rgb_interpolation=int(RGB_INTERPOLATION),rgb_flags=['BILINEAR','ACCURATE_RND','BITEXACT','FULL_CHR_H_INT'],
        rgb_conversion_threads=1,constant_yuv_reference=constant,real_data=reports,original_failing_examples=diagnostics,
        source_sha256={name:file_hash(name) for name in ('video_data.py','audit_video_rgb.py','test_video_rgb.py','uv.lock')},
        limitations='Sampled 64 real videos, not a complete decoded-pixel audit of all SSv2. Default converter corruption can depend on allocation reuse; absence on a particular call is not a correctness proof.')
    write_json(destination,report)


if __name__=='__main__':
    main()
