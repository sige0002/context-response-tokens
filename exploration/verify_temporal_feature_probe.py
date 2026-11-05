"""Re-run the CPU readout from saved features and independently recount metrics."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch
from transformers import SiglipImageProcessor

from exploration.continuous_matching import continuous_match
from exploration.oracle_transport import digest,sample


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    args=parser.parse_args(); run=Path(args.run)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    plan=json.loads((run/'plan.json').read_text()); config=plan['config']
    report=json.loads((run/'summary.json').read_text())
    assert digest(run/'plan.json')==report['plan_sha256']
    assert digest(run/'checkpoint.pt')==report['checkpoint_sha256']==plan['checkpoint']['checkpoint_sha256']
    for source,expected in plan['sources'].items(): assert digest(source)==expected,source
    cache=Path(plan['cache']); receipt=json.loads((cache/'summary.json').read_text())
    assert digest(cache/'summary.json')==plan['cache_summary_sha256']
    root=Path(plan['rgb_root'])
    processor=SiglipImageProcessor.from_pretrained(config['model'],revision=config['revision'],local_files_only=True)
    files=readout_calls=0; by_key={}; feature_changes={}
    for scene in config['val']:
        original_path=cache/(scene+'.pt'); assert digest(original_path)==receipt['scenes'][scene]['sha256']
        original=torch.load(original_path,map_location='cpu',weights_only=True)
        candidates=torch.where(original['visible'][:,0])[0]
        generator=torch.Generator().manual_seed(config['seed']+int(scene))
        selected=candidates[torch.randperm(len(candidates),generator=generator)[:config['evaluation_queries']]]
        truth,visible=original['points'][selected].numpy(),original['visible'][selected].numpy()
        feature_path=run/(scene+'_features.pt'); assert digest(feature_path)==report['features_sha256'][scene]
        payload=torch.load(feature_path,map_location='cpu',weights_only=True)
        assert torch.equal(payload['selected'],selected) and payload['frames']==original['frames']
        features=payload['features']; frames=len(payload['frames'])
        metadata=root/scene/'metadata.json'; archive_path=root/scene/'images.zip'
        assert digest(metadata)==plan['images'][scene]['metadata_sha256']
        assert digest(archive_path)==plan['images'][scene]['images_sha256']
        with zipfile.ZipFile(archive_path) as archive:
            rgb=[Image.open(io.BytesIO(archive.read(f'{frame:03d}.png'))).convert('RGB') for frame in payload['frames']]
        pixels=processor(images=rgb,return_tensors='pt').pixel_values
        assert hashlib.sha256(pixels.numpy().tobytes()).hexdigest()==payload['pixels_sha256']
        del rgb,pixels,original
        state_path=run/(scene+'_states.json'); assert digest(state_path)==report['states_sha256'][scene]
        states=json.loads(state_path.read_text()); feature_changes[scene]={}
        for condition in config['conditions']:
            sequence=states[condition]; assert len(sequence)==frames
            assert len({s['bytes'] for s in sequence})==1
            for frame,state in enumerate(sequence):
                assert state['offset']==(1 if condition=='mamba_reset1' else frame+1)
                if condition=='siglip_only':
                    assert state['bytes']==0 and not state['layer_states']
                else:
                    assert set(state['layer_states'])=={'3','7','11'}
                    assert all(len(value)==4 for value in state['layer_states'].values())
                    byte_count=sum(math.prod(t['shape'])*{'torch.float32':4,'torch.bfloat16':2,'torch.float16':2}[t['dtype']]
                        for values in state['layer_states'].values() for t in values)
                    assert byte_count==state['bytes']
            for depth in ('4','12'):
                current=features[condition][depth]
                assert current.shape==(frames,196,768) and current.dtype==torch.float32 and torch.isfinite(current).all()
                path=run/(scene+'_'+condition+'_'+depth+'.npz')
                assert digest(path)==report['predictions_sha256'][path.name]
                with np.load(path) as raw: data={k:raw[k] for k in raw.files}
                assert np.array_equal(data['target_xy'],truth) and np.array_equal(data['visible'],visible)
                assert np.array_equal(data['selected'],selected.numpy())
                assert np.array_equal(data['static_query'],np.broadcast_to(truth[:,:1],truth.shape))
                anchor=sample(current[0],torch.from_numpy(truth[:,0]),config['grid'])
                for frame in range(frames):
                    repeated=continuous_match(anchor,current[frame],config['grid'],depth=='12',config['topk'],config['refinement_steps'])
                    for method,result in repeated.items():
                        assert np.array_equal(result['points'].numpy(),data[method][:,frame]),(scene,condition,depth,frame,method)
                        assert np.array_equal(result['scores'].numpy(),data[method+'_score'][:,frame])
                    readout_calls+=1
                for method in config['methods']:
                    assert np.isfinite(data[method]).all() and (data[method]>=0).all() and (data[method]<=1).all()
                by_key[(scene,condition,int(depth))]=data; files+=1
        for depth in ('4','12'):
            full,reset=features['mamba_full'][depth],features['mamba_reset1'][depth]
            assert torch.equal(full[0],reset[0])
            assert np.array_equal(by_key[(scene,'mamba_full',int(depth))]['center'][:,0],
                                  by_key[(scene,'mamba_reset1',int(depth))]['center'][:,0])
            feature_changes[scene][depth]=dict(max_abs=float((full[1:]-reset[1:]).abs().max()),
                relative_l2=float(torch.linalg.vector_norm(full[1:]-reset[1:])/torch.linalg.vector_norm(reset[1:]).clamp_min(1e-12)))
        print(json.dumps(dict(scene=scene,readout_calls=readout_calls,files=files)),flush=True)
    rows=0
    for row in report['rows']:
        data=by_key[(row['scene'],row['condition'],row['depth'])]
        truth,visible=data['target_xy'],data['visible']; prediction=data[row['method']]
        squared=(((prediction-truth)*256)**2).sum(-1)
        if row['task']=='identity':
            errors=np.sqrt(squared[:,0])
        else:
            move=np.sqrt((((truth[:,1:]-truth[:,:1])*256)**2).sum(-1))
            mask=visible[:,1:].copy()
            if row['stratum']=='moving_gt2': mask &= move>2
            if row['stratum']=='moving_gt16': mask &= move>16
            errors=np.sqrt(squared[:,1:])[mask]
        assert len(errors)==row['points']
        for threshold in (1,2,4,8,16): assert float((errors<threshold).mean())==row[f'pck_{threshold}']
        assert abs(float(errors.mean())-row['mean_error'])<2e-5
        metrics=row['full_sequence_metrics']
        if metrics:
            gt=visible[:,1:]; count=int(gt.sum()); total=gt.size; js=[]; pcks=[]
            for threshold in (1,2,4,8,16):
                correct=int(((squared[:,1:]<threshold**2)&gt).sum())
                jaccard=correct/(count+total-correct); pck=correct/count
                assert abs(jaccard-metrics[f'jaccard_{threshold}'])<1e-12
                assert abs(pck-metrics[f'pts_within_{threshold}'])<1e-12
                js.append(jaccard); pcks.append(pck)
            assert abs(float(np.mean(js))-metrics['average_jaccard'])<1e-12
            assert abs(float(np.mean(pcks))-metrics['average_pts_within_thresh'])<1e-12
            assert abs(count/total-metrics['occlusion_accuracy'])<1e-12
        rows+=1
    for summary in report['summary']:
        keys=('condition','depth','method','task','stratum')
        matching=[r for r in report['rows'] if all(r[k]==summary[k] for k in keys) and r['points']]
        assert len(matching)==summary['scenes'] and sum(r['points'] for r in matching)==summary['points']
        for key in ('mean_error','pck_1','pck_2','pck_4','pck_8','pck_16'):
            assert float(np.mean([r[key] for r in matching]))==summary[key]
        if summary['full_sequence_metrics']:
            for key,value in summary['full_sequence_metrics'].items():
                assert float(np.mean([r['full_sequence_metrics'][key] for r in matching]))==value
    output=dict(status='passed',source_sha256=digest(__file__),summary_sha256=digest(run/'summary.json'),
        prediction_files=files,rerun_matching_calls=readout_calls,recounted_rows=rows,
        original_rgb_preprocessing_queries_and_annotations_match=True,
        full_reset_query_features_identical=True,state_offsets_and_four_native_buffers_verified=True,
        feature_changes=feature_changes,
        scope='CPU readout rerun from saved features, independent metric recount, RGB preprocessing rerun. GPU encoder inference is not rerun.')
    (run/'verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='feature_changes'}),flush=True)


if __name__=='__main__': main()
