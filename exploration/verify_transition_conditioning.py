"""CPU recount of the matched pilot, using original manifest labels.

This checks recorded artifacts and the trainer's complete export replays; it
does not itself execute a new GPU forward or re-decode all source videos.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from exploration.temporal_density import paired, read, write
from exploration.train_transition_conditioning import checked_plan, digest
from video_checkpoint import file_hash


def check_score(logits, targets, reported):
    logits=np.asarray(logits);targets=np.asarray(targets)
    if logits.shape!=(len(targets),174) or not np.isfinite(logits).all(): raise ValueError('Invalid logits')
    correct=int(np.count_nonzero(logits.argmax(1)==targets))
    if reported['correct']!=correct or reported['total']!=len(targets) or abs(reported['accuracy']-correct/len(targets))>1e-12:
        raise ValueError('Original-label recount differs')


def verify_arm(output,arm,plan,targets):
    folder=output/'training'/arm
    summary=read(folder/'summary.json')
    for name,key in [('predictions.npz','predictions_sha256'),('model.pt','checkpoint_sha256')]:
        if file_hash(folder/name)!=summary[key]: raise ValueError('Changed model/prediction artifact')
    with np.load(folder/'predictions.npz',allow_pickle=False) as saved:
        if not np.array_equal(saved['targets'],targets): raise ValueError('Wrong target order')
        values={k:saved[k].copy() for k in ('initial','final','initial_stream','final_stream')}
    for name,v in values.items(): check_score(v,targets,summary[name])
    for key,candidate,reference in [('vs_initial','final','initial'),('stream_vs_initial','final_stream','initial_stream'),('stream_vs_clip','final_stream','final')]:
        if paired(values[candidate],values[reference],targets)!=summary[key]: raise ValueError('Paired metric differs')
    if summary['replay_score']!=summary['final'] or summary['replay_stream_score']!=summary['final_stream']:
        raise ValueError('Reported export replay differs')
    if (summary['all_validation_export_replay_max_abs']!=0 or summary['all_validation_stream_export_replay_max_abs']!=0
            or not summary['frozen_backbone_unchanged'] or summary['steps']!=plan['steps']):
        raise ValueError('Training/replay requirement not met')
    for field in ('input_metadata_sha256','input_pixels_sha256'):
        if len({summary[name][field] for name in values})!=1: raise ValueError('Evaluation inputs differ')
    records=[json.loads(s) for s in (output/'progress.jsonl').read_text().splitlines()]
    steps=[r['step'] for r in records if r['arm']==arm]
    if steps!=list(range(32,plan['steps']+1,32)): raise ValueError('Missing or duplicate training progress')
    return summary,values


def verify(output,arm=None):
    plan=checked_plan(output);config=plan['checkpoint_info']['config']
    tables={}
    for split in ('train','val'):
        path=Path(config[split])
        if file_hash(path)!=config[split+'_sha256']: raise ValueError('Original manifest changed')
        tables[split]=[json.loads(s) for s in path.read_text().splitlines()]
    selected=plan['validation'];training=read(output/'training_schedule.json')
    if len(selected)!=696 or len({r['id'] for r in selected})!=696: raise ValueError('Wrong validation cohort')
    if len(training)!=8192 or len({r['id'] for r in training})!=8192: raise ValueError('Wrong training cohort')
    if digest(training)!=plan['training_schedule_sha256']: raise ValueError('Changed training schedule')
    if {r['id'] for r in selected}&{r['id'] for r in training}: raise ValueError('Split overlap')
    for split,entries in [('train',training),('val',selected)]:
        for item in entries:
            row=tables[split][item['index']]
            if row['id']!=item['id'] or row['label']!=item['label']: raise ValueError('Manifest label/order drift')
    targets=np.array([tables['val'][r['index']]['label'] for r in selected])
    if np.bincount(targets,minlength=174).tolist()!=[4]*174: raise ValueError('Wrong class balance')
    names=[arm] if arm else plan['arms'];results={};values={}
    for name in names: results[name],values[name]=verify_arm(output,name,plan,targets)
    if arm is None:
        aggregate=read(output/'summary.json')
        if aggregate['plan_sha256']!=file_hash(output/'plan.json') or aggregate['arms']!=results:
            raise ValueError('Aggregate provenance differs')
        for field in ('initial_model_sha256','training_input_metadata_sha256','training_pixels_sha256','trainable_parameters'):
            if len({r[field] for r in results.values()})!=1: raise ValueError(f'Arms not matched: {field}')
        for mode in ('initial','initial_stream'):
            if not np.array_equal(values['appearance'][mode],values['difference'][mode]):
                raise ValueError('Initial predictions differ across arms')
        primary=paired(values['difference']['final_stream'],values['appearance']['final_stream'],targets)
        if primary!=aggregate['difference_vs_appearance']: raise ValueError('Primary streaming comparison differs')
    return dict(passed=True,arms=names,plan_sha256=file_hash(output/'plan.json'),
        summary_sha256=file_hash(output/'summary.json') if arm is None else file_hash(output/'training'/arm/'summary.json'),
        predictions_recounted=len(names)*696*4,
        scope='CPU source/artifact and original-manifest recount of clip and streaming predictions; checks recorded full export replays, not a new GPU inference.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--arm',choices=('appearance','difference'))
    args=parser.parse_args();output=Path(args.output);result=verify(output,args.arm)
    destination=output/'verification.json' if args.arm is None else output/'training'/args.arm/'verification.json'
    write(destination,result);print(json.dumps(result))
