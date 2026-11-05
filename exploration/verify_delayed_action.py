"""Re-decode qualification RGB and independently recount its artifact invariants."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import SiglipImageProcessor

from exploration.delayed_action_data import sha,verify_plan
from exploration.evaluate_video_history import tensor_hash
from video_checkpoint import file_hash
from video_data import RawVideoClips


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    args=p.parse_args(); root=Path(args.run)
    output=root/'verification.json'
    if output.exists(): raise FileExistsError('Verification already exists')
    torch.set_num_threads(2)
    plan=json.loads((root/'plan.json').read_text()); result=json.loads((root/'summary.json').read_text())
    if not plan['qualification'] or result['scope']!='qualification_only':
        raise ValueError('This verifier is for the two-prefix qualification only')
    verify_plan(plan['data'])
    for field,path in [('plan_sha256','plan.json'),('features_sha256','features.npz'),('inputs_sha256','inputs.json')]:
        if result[field]!=sha(root/path): raise ValueError('Artifact changed: '+path)
    if file_hash(plan['checkpoint_path'])!=plan['checkpoint_info']['checkpoint_sha256']:
        raise ValueError('Qualification checkpoint changed')
    for path,digest in plan['sources'].items():
        if sha(path)!=digest: raise ValueError('Extraction source changed: '+path)
    config=plan['checkpoint_info']['config']
    processor=SiglipImageProcessor.from_pretrained(config['model'],revision=config['revision'],local_files_only=True)
    source=plan['data']['manifests']['validation']
    if sha(source['path'])!=source['sha256']: raise ValueError('Original validation manifest changed')
    dataset=RawVideoClips(source['path'],processor,frames=8,classes=174)
    index={r['id']:i for i,r in enumerate(dataset.rows)}
    block=plan['data']['splits']['validation'][0]
    originals=block['suffix']+block['evidence'][:2]
    records=json.loads((root/'inputs.json').read_text())
    if len(originals)!=len(records): raise ValueError('Wrong number of decoded clips')
    decoded={}
    for expected,record in zip(originals,records):
        i=index[expected['id']]; row=dataset.rows[i]
        if row['label']!=expected['label'] or file_hash(row['video'])!=expected['sha256']:
            raise ValueError('Source content or target differs')
        pixels,target,metadata=dataset.read(i)
        actual=dict(split='validation',video_sha256=expected['sha256'],pixels_sha256=tensor_hash(pixels),**metadata)
        if record!=actual: raise ValueError('Re-decoded RGB differs: '+expected['id'])
        decoded[expected['id']]=pixels
    suffix=torch.cat([decoded[r['id']] for r in block['suffix']])
    delays=plan['data']['delays']
    with np.load(root/'features.npz') as arrays:
        labels=[r['label'] for r in block['evidence'][:2]]
        ids=[r['id'] for r in block['evidence'][:2]]
        if arrays['validation_labels'].tolist()!=labels or arrays['validation_ids'].tolist()!=ids:
            raise ValueError('Cached labels or IDs differ')
        full=arrays['validation_full_memory']; reset=arrays['validation_suffix_only']; saved=arrays['validation_saved_evidence']
        if full.shape!=(2,len(delays),768) or reset.shape!=(2,len(delays)-1,768):
            raise ValueError('Wrong feature shape')
        if not np.isfinite(full).all() or not np.isfinite(reset).all() or not np.array_equal(reset[0],reset[1]):
            raise ValueError('Nonfinite or evidence-dependent suffix control')
        if not np.array_equal(saved,np.broadcast_to(full[:,:1],full.shape)):
            raise ValueError('Saved-evidence control differs')
        distances=np.linalg.norm(full[0]-full[1],axis=-1).tolist()
        if distances!=result['between_prefix_feature_l2_by_delay']:
            raise ValueError('Feature difference recount failed')
    for state,record in zip(result['states'],block['evidence'][:2]):
        expected=[tensor_hash(decoded[record['id']][-1])]+[tensor_hash(suffix[d-1]) for d in delays[1:]]
        if state['id']!=record['id'] or state['current_rgb_sha256']!=expected:
            raise ValueError('Current observations do not match source pixels')
        if set(state['full_bytes']+state['suffix_only_bytes'])!={79027200}:
            raise ValueError('Unexpected state size in the fixed qualification model')
    receipt=dict(passed=True,summary_sha256=sha(root/'summary.json'),
        decoded_clips=len(records),decoded_sampled_frames=sum(len(v) for v in decoded.values()),
        current_images_recomputed=True,original_labels_match=True,feature_controls_recounted=True,
        checkpoint_sha256=plan['checkpoint_info']['checkpoint_sha256'],source_sha256=sha(__file__),
        scope='CPU re-decode of all 18 qualification videos and saved-feature checks; no independent GPU rerun or recognition measurement')
    output.write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
