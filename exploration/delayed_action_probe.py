"""Fit CPU ridge readouts on frozen delayed-action features, without val tuning."""
import argparse
import json
from pathlib import Path

import numpy as np

from exploration.delayed_action_data import sha, verify_plan


def fit_ridge(x, labels, classes, alpha):
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    if (x.ndim != 2 or len(x) != len(labels) or alpha <= 0 or not np.isfinite(x).all()
            or set(labels.tolist()) != set(range(classes))):
        raise ValueError('Invalid or incomplete training features')
    mean = x.mean(0)
    target = np.eye(classes)[labels]
    bias = target.mean(0)
    centered = x-mean
    weights = np.linalg.solve(centered.T@centered + alpha*np.eye(x.shape[1]),
                              centered.T@(target-bias))
    return weights, bias, mean


def evaluate(features, plan):
    verify_plan(plan)
    delays = plan['delays']; classes = plan['classes']
    output = {}; fitted = {}
    for condition in plan['conditions']:
        offset = 1 if condition == 'suffix_only' else 0
        arrays = {}
        for split in ('train','validation'):
            labels = features[split+'_labels']; blocks = features[split+'_blocks']
            expected = len(plan['splits'][split])*classes
            x = features[split+'_'+condition]
            if x.ndim != 3 or x.shape[:2] != (expected,len(delays)-offset) or not np.isfinite(x).all():
                raise ValueError('Wrong feature population or delay grid: '+condition)
            # Class balance must hold independently for each shared suffix.
            if len(labels) != expected or len(blocks) != expected:
                raise ValueError('Wrong labels or block count')
            expected_rows=[(b['block'],r['id'],r['label']) for b in plan['splits'][split] for r in b['evidence']]
            actual_rows=list(zip(blocks.tolist(),features[split+'_ids'].tolist(),labels.tolist()))
            if actual_rows!=expected_rows:
                raise ValueError('Features and original evidence labels/IDs are not aligned')
            if set(blocks.tolist()) != set(range(len(plan['splits'][split]))):
                raise ValueError('Missing suffix block')
            for b in set(blocks.tolist()):
                if sorted(labels[blocks==b].tolist()) != list(range(classes)):
                    raise ValueError('Unbalanced suffix block')
                if condition == 'suffix_only' and not np.array_equal(x[blocks==b],
                        np.broadcast_to(x[blocks==b][0],x[blocks==b].shape)):
                    raise ValueError('Suffix-only features leak evidence identity')
            arrays[split]=(x,labels,blocks)
        x,labels,_=arrays['train']
        w,b,m=fit_ridge(x.reshape(-1,x.shape[-1]),np.repeat(labels,x.shape[1]),
                        classes,plan['ridge_alpha'])
        fitted.update({condition+'_weight':w,condition+'_bias':b,condition+'_mean':m})
        output[condition]={}
        for split,(x,labels,blocks) in arrays.items():
            predicted=((x-m)@w+b).argmax(-1)
            rows=[]
            for j,delay in enumerate(delays[offset:]):
                correct=predicted[:,j]==labels
                rows.append(dict(delay_observations=delay,total=len(labels),correct=int(correct.sum()),
                    accuracy=float(correct.mean()),by_block={str(k):float(correct[blocks==k].mean())
                    for k in sorted(set(blocks.tolist()))},predictions=predicted[:,j].tolist()))
                if condition=='suffix_only' and int(correct.sum()) != len(plan['splits'][split]):
                    raise AssertionError('Class-independent suffix exceeds its exact balanced ceiling')
            output[condition][split]=rows
    return output,fitted


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args(); cache=Path(args.cache); output=Path(args.output)
    if output.exists(): raise FileExistsError('Use a fresh probe output')
    receipt=json.loads((cache/'summary.json').read_text())
    plan=json.loads((cache/'plan.json').read_text())
    if receipt['scope']!='complete_cache' or receipt['plan_sha256']!=sha(cache/'plan.json'):
        raise ValueError('Only a complete cache can train a probe')
    if receipt['features_sha256']!=sha(cache/'features.npz'):
        raise ValueError('Feature artifact changed')
    with np.load(cache/'features.npz') as features:
        results,weights=evaluate(features,plan['data'])
    output.mkdir(parents=True)
    np.savez_compressed(output/'readouts.npz',**weights)
    summary=dict(cache=str(cache.resolve()),cache_summary_sha256=sha(cache/'summary.json'),
        results=results,readouts_sha256=sha(output/'readouts.npz'),source_sha256=sha(__file__),
        interpretation='Linear decodability of the first action after controlled distractors; not natural-video SOTA',
        validation_used_for_fitting=False,encoder_updated=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({c:[r['accuracy'] for r in scores['validation']] for c,scores in results.items()}))


if __name__=='__main__': main()
