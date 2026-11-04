"""Audit same-video state interventions and describe age/reappearance strata.

Only saved predictions and raw annotations are read. No encoder is run or fit.
Query anchors remain shared full-history descriptors. This measures the effect
of resetting ongoing encoder state, not a guarantee of semantic memory.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_robotap_predictions import audit_example, official_metric_matches
from paired_metrics import paired_mean_interval
from point_checkpoint import file_hash
from point_experiment import read_json, write_json
from robotap_data import iter_videos
from summarize_points import align_reports, METRICS


AGE_BINS = ((1,8),(9,24),(25,64),(65,128),(129,None))
GAP_BINS = ((1,7),(8,23),(24,None))


def strata_masks(first, occluded):
    first,occluded=np.asarray(first),np.asarray(occluded)
    if (occluded.ndim!=2 or occluded.dtype!=np.bool_ or first.shape!=(len(occluded),)
            or not np.issubdtype(first.dtype,np.integer)
            or ((first<0)|(first>=occluded.shape[1])).any()):
        raise ValueError('Invalid query times or occlusion array')
    age=np.arange(occluded.shape[1])[None]-first[:,None]
    masks={'all_post_query':age>0}
    for low,high in AGE_BINS:
        masks[f'age_{low}_{high or "plus"}']=(age>=low)&(True if high is None else age<=high)
    gap=np.zeros(occluded.shape,dtype=np.int32)
    for query,start in enumerate(first):
        run=0
        for frame in range(int(start)+1,occluded.shape[1]):
            if occluded[query,frame]:run+=1
            else:
                gap[query,frame]=run;run=0
    for low,high in GAP_BINS:
        masks[f'reappearance_gap_{low}_{high or "plus"}']=(gap>=low)&(True if high is None else gap<=high)
    return masks


def masked_metrics(target,occluded,predicted,logits,mask):
    """TAP-Vid count formulas with a restricted post-query evaluation mask.

    An eligible video has at least one GT-visible point in this stratum. This
    GT-only rule keeps the paired cohort independent of model predictions and
    avoids defining empty/all-occluded denominators as either zero or success.
    """
    target,occluded,predicted,logits,mask=map(np.asarray,(target,occluded,predicted,logits,mask))
    if (target.shape!=predicted.shape or target.shape!=occluded.shape+(2,)
            or logits.shape!=occluded.shape or mask.shape!=occluded.shape
            or mask.dtype!=np.bool_ or occluded.dtype!=np.bool_
            or not all(np.isfinite(x).all() for x in (target,predicted,logits))):
        raise ValueError('Invalid masked metric arrays')
    visible=~occluded;pred_visible=logits>=0
    counts=dict(evaluated_positions=int(mask.sum()),visible_positions=int((visible&mask).sum()),
                occluded_positions=int((occluded&mask).sum()))
    if not counts['visible_positions']:
        return dict(counts=counts,metrics=None,eligibility='No GT-visible selected target positions')
    distance=np.square(predicted*256-target*256).sum(-1)
    point_rates,jaccards=[],[]
    for threshold in (1,2,4,8,16):
        near=distance<threshold**2
        correct=near&visible
        point_rates.append(float((correct&mask).sum()/counts['visible_positions']))
        tp=int((correct&pred_visible&mask).sum())
        fp=int(((~visible|~near)&pred_visible&mask).sum())
        jaccards.append(tp/(counts['visible_positions']+fp))
    metrics=dict(average_jaccard=float(np.mean(jaccards)),average_pts_within_thresh=float(np.mean(point_rates)),
                 occlusion_accuracy=float(((pred_visible==visible)&mask).sum()/counts['evaluated_positions']))
    return dict(counts=counts,metrics=metrics,eligibility='Has GT-visible selected target positions')


def verify_reports(directory, baseline, allow_subset=False):
    directory,baseline=Path(directory),Path(baseline)
    parent=read_json(directory/'summary.json');original=read_json(baseline)
    if set(parent['conditions'])!={'full','reset_1','reset_24'}:
        raise ValueError('Expected the prespecified full/reset-1/reset-24 comparison')
    subset=original['scope']=='engineering subset'
    if subset and not allow_subset:
        raise ValueError('Engineering reports require explicit --allow-subset')
    if not subset and (original['scope']!='full RoboTAP first-visible online evaluation' or original['videos']!=265):
        raise ValueError('Expected full official baseline evaluation')
    if len(original['rows'])!=original['videos']:
        raise ValueError('Baseline video count differs from its rows')
    if parent['videos']!=original['videos'] or parent['checkpoint_sha256']!=original['checkpoint']['checkpoint_sha256']:
        raise ValueError('Intervention checkpoint or example count differs from baseline')
    reports={}
    for name,record in parent['conditions'].items():
        path=directory/name/'summary.json';report=read_json(path)
        if (file_hash(path)!=record['report_sha256'] or report['metrics']!=record['metrics']
                or report['checkpoint']!=original['checkpoint']
                or report['dataset_extraction_sha256']!=original['dataset_extraction_sha256']
                or report['scope']!=original['scope'] or report['videos']!=original['videos']
                or len(report['rows'])!=original['videos']
                or report['evaluation_reproducibility']!=original['evaluation_reproducibility']):
            raise ValueError('Intervention report, checkpoint, data, or precision differs')
        for source in ('point_history.py','evaluate_point_history.py'):
            if report['source_sha256'].get(source)!=file_hash(source):
                raise ValueError('Intervention implementation differs')
        a,b=align_reports(original,report)
        period=None if name=='full' else int(name.split('_')[1])
        for x,y in zip(a,b,strict=True):
            expected=[] if period is None else list(range(period,y['frames'],period))
            if y['reset_frames']!=expected:
                raise ValueError('Unexpected state-reset schedule')
        reports[name]=report
    # The normal branch must reproduce the already independently scored model.
    ordinary,full=align_reports(original,reports['full'])
    maximum_xy=maximum_logits=0.
    for a,b in zip(ordinary,full,strict=True):
        if file_hash(baseline.parent/a['predictions'])!=a['predictions_sha256']:
            raise ValueError('Ordinary prediction archive changed')
        with np.load(baseline.parent/a['predictions'],allow_pickle=False) as x,np.load(directory/'full'/b['predictions'],allow_pickle=False) as y:
            for key in ('valid_track_indices','query_frames'):
                if not np.array_equal(x[key],y[key]):raise ValueError('Query order changed in normal branch')
            maximum_xy=max(maximum_xy,float(np.max(np.abs(x['predicted_xy']-y['predicted_xy']))))
            maximum_logits=max(maximum_logits,float(np.max(np.abs(x['visibility_logits']-y['visibility_logits']))))
    if maximum_xy>1e-6 or maximum_logits>1e-5:
        raise ValueError('Normal intervention branch differs from ordinary point tracking')
    return reports,dict(normal_branch_max_xy=maximum_xy,normal_branch_max_visibility_logit=maximum_logits,
                        tolerances=dict(normalized_xy=1e-6,visibility_logit=1e-5),
                        baseline_report_sha256=file_hash(baseline),engineering_subset=subset)


def aggregate(rows):
    if not rows:raise ValueError('No paired videos')
    result={}
    for stratum in rows[0]['strata']:
        eligible=[r for r in rows if r['strata'][stratum]['full']['metrics'] is not None]
        for row in rows:
            values=row['strata'][stratum]
            if any(v['counts']!=values['full']['counts'] or (v['metrics'] is None)!=(values['full']['metrics'] is None)
                   for v in values.values()):raise ValueError('Paired stratum uses different target support')
        entry=dict(videos=len(eligible),unique_rgb_videos=len({r['video_sha256'] for r in eligible}),
                   excluded_without_visible_positions=len(rows)-len(eligible),
                   counts={key:sum(r['strata'][stratum]['full']['counts'][key] for r in rows)
                           for key in rows[0]['strata'][stratum]['full']['counts']},conditions={},full_minus_reset={})
        if eligible:
            for condition in ('full','reset_1','reset_24'):
                entry['conditions'][condition]={metric:float(np.mean([r['strata'][stratum][condition]['metrics'][metric] for r in eligible])) for metric in METRICS}
            for condition in ('reset_1','reset_24'):
                entry['full_minus_reset'][condition]={}
                for metric in METRICS:
                    a=[r['strata'][stratum]['full']['metrics'][metric] for r in eligible]
                    b=[r['strata'][stratum][condition]['metrics'][metric] for r in eligible]
                    groups=[r['video_sha256'] for r in eligible]
                    interval=paired_mean_interval(a,b,groups) if len(set(groups))>=2 else None
                    entry['full_minus_reset'][condition][metric]=dict(difference=float(np.mean(a)-np.mean(b)),interval=interval)
        result[stratum]=entry
    return result


def analyze(directory,baseline,root,allow_subset=False):
    directory,root=Path(directory),Path(root)
    reports,verification=verify_reports(directory,baseline,allow_subset)
    extraction=file_hash(root/'extraction.json')
    if reports['full']['dataset_extraction_sha256']!=extraction:
        raise ValueError('Raw dataset differs from reports')
    upstream=official_metric_matches()
    indexed={name:{row['id']:row for row in report['rows']} for name,report in reports.items()}
    pending=set(indexed['full']);rows=[]
    no_time=reports['full']['checkpoint']['encoder_architecture']['kind']=='none'
    for shard,name,example in iter_videos(root):
        if name not in pending:continue
        pending.remove(name)
        indices=np.flatnonzero((~example['occluded']).any(axis=1))
        target,occluded=example['points'][indices],example['occluded'][indices]
        first=(~occluded).argmax(axis=1);masks=strata_masks(first,occluded)
        row=dict(id=name,video_sha256=indexed['full'][name]['video_sha256'],strata={key:{} for key in masks},raw_label_audits={})
        normal=None
        for condition in ('full','reset_1','reset_24'):
            record=indexed[condition][name]
            if record['shard']!=shard:raise ValueError('Video shard changed')
            audit=audit_example(record,example,directory/condition)
            row['raw_label_audits'][condition]=audit
            with np.load(directory/condition/record['predictions'],allow_pickle=False) as prediction:
                xy,logits=prediction['predicted_xy'],prediction['visibility_logits']
            if condition=='full':normal=(xy.copy(),logits.copy())
            elif no_time and (not np.array_equal(xy,normal[0]) or not np.array_equal(logits,normal[1])):
                raise ValueError('History-free encoder changed under memory reset')
            for key,mask in masks.items():row['strata'][key][condition]=masked_metrics(target,occluded,xy,logits,mask)
            overall=row['strata']['all_post_query'][condition]['metrics']
            if overall is None or any(abs(overall[key]-record['metrics'][key])>1e-9 for key in METRICS):
                raise ValueError('Unrestricted masked scores differ from the official metric')
        rows.append(row)
        if not pending:break
    if pending:raise ValueError('Missing videos in raw dataset')
    for condition,report in reports.items():
        for metric in METRICS:
            average=float(np.mean([r['raw_label_audits'][condition]['metrics'][metric] for r in rows]))
            if abs(average-report['metrics'][metric])>1e-9:raise ValueError('Official report mean differs from original labels')
    return dict(scope='Engineering subset' if verification['engineering_subset'] else 'Full RoboTAP paired encoder-state intervention',
        videos=len(rows),verification=verification,official_metric=upstream,
        checkpoint_sha256=reports['full']['checkpoint']['checkpoint_sha256'],dataset_extraction_sha256=extraction,
        primary='All-post-query average Jaccard: normal state minus reset every frame',
        secondary='Reset every 24 frames; all age and reappearance strata are exploratory',
        age_bins=AGE_BINS,reappearance_gap_bins=GAP_BINS,
        aggregation='Unweighted video means within each GT-only eligible stratum; no comparison with fewer than one visible target',
        intervals='Paired duplicate-RGB cluster bootstrap, 10000 draws, seed 17; none with fewer than two clusters',
        limits='Shared full-history query anchors remain. Reset shifts state distributions. Strata differ in target support; across-age changes do not isolate forgetting. Units are frames, not seconds. No multiple-testing or training-seed adjustment.',
        strata=aggregate(rows),rows=rows,
        source_sha256={name:file_hash(name) for name in ('analyze_point_history.py','audit_robotap_predictions.py',
            'point_history.py','evaluate_point_history.py','summarize_points.py','paired_metrics.py','robotap_data.py','tapvid_metrics.py')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interventions',required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--root',default='data/robotap')
    parser.add_argument('--output',required=True)
    parser.add_argument('--allow-subset',action='store_true')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError('Choose a new analysis output')
    report=analyze(args.interventions,args.baseline,args.root,args.allow_subset)
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,report)
    print(json.dumps(dict(scope=report['scope'],videos=report['videos'],strata={k:v['videos'] for k,v in report['strata'].items()})),flush=True)


if __name__=='__main__':main()
