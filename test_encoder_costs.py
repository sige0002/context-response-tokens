import copy
import json
from pathlib import Path
import shutil

import pytest

from encoder_costs import select_videos,verify_measurement,paired_costs
from point_checkpoint import file_hash


def test_video_selection_uses_unique_rgb_lengths_and_ignores_input_order():
    rows=[dict(id=name,frames=frames,video_sha256=digest,shard='split')
          for name,frames,digest in [('a',52,'x'),('b',245,'y'),('c',1337,'z'),('d',245,'y')]]
    selected=select_videos(rows)
    assert [r['id'] for r in selected]==['a','b','c']
    assert select_videos(list(reversed(rows)))==selected
    with pytest.raises(ValueError):select_videos(rows[:2])


def fixture():
    directory=Path(__file__).parent/'results/real_video_cost_integration_rgbfix/mamba'
    plan=json.loads((directory/'plan.json').read_text())
    completion=json.loads(Path('results/video_training_integration_rgbfix_fp32/evaluation_verification.json').read_text())
    checkpoint=dict(step=3,checkpoint_sha256=completion['checkpoint_sha256'])
    video=next(row for row in json.loads(Path('results/robotap_data_audit.json').read_text())['rows']
               if row['id']==plan['video']['id'])
    return directory,plan,checkpoint,video


def test_real_gpu_cost_records_match_training_checkpoint_and_audited_rgb():
    directory,plan,checkpoint,video=fixture()
    actual=verify_measurement(directory,plan['checkpoint']['config'],checkpoint,video,3,1,
                              file_hash('data/robotap/extraction.json'))
    assert actual['frames_per_pass']==296
    assert actual['max_state_bytes']==79027200
    assert actual['torch_peak_allocated_bytes']>actual['parameter_bytes']+actual['max_state_bytes']


@pytest.mark.parametrize('fault',['checkpoint','rgb','prefix','metric','missing_frame','changing_state'])
def test_cost_verification_rejects_misleading_or_mismatched_measurements(tmp_path,fault):
    source,plan,checkpoint,video=fixture()
    directory=tmp_path/'measurement';shutil.copytree(source,directory)
    summary=json.loads((directory/'summary.json').read_text())
    if fault=='checkpoint':checkpoint=dict(checkpoint,checkpoint_sha256='other')
    elif fault=='rgb':plan['video']['input_rgb_sha256']='other'
    elif fault=='prefix':plan['video']['frames_used']=10
    elif fault=='metric':summary['bins'][0]['gpu_ms']['p50']*=.01
    else:
        rows=[json.loads(line) for line in (directory/'timings.jsonl').read_text().splitlines()]
        if fault=='missing_frame':rows.pop()
        else:
            rows[-1]['state_bytes']+=4
            from benchmark_real_video import summarize_rows
            summary.update(summarize_rows(rows,296,3))
        (directory/'timings.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (directory/'plan.json').write_text(json.dumps(plan))
    summary['plan_sha256']=file_hash(directory/'plan.json')
    summary['timings_sha256']=file_hash(directory/'timings.jsonl')
    (directory/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        verify_measurement(directory,plan['checkpoint']['config'],checkpoint,video,3,1,
                           file_hash('data/robotap/extraction.json'))


def test_paired_costs_require_same_videos_bins_and_hardware():
    directory,plan,checkpoint,video=fixture()
    measurement=verify_measurement(directory,plan['checkpoint']['config'],checkpoint,video,3,1,
                                   file_hash('data/robotap/extraction.json'))
    records=[dict(model=name,video=copy.deepcopy(video),measurement=copy.deepcopy(measurement))
             for name in ('mamba_interleaved','attention_interleaved')]
    result=paired_costs(records)
    assert all(b['mamba_over_other_gpu_median']==1 for b in result['comparisons'][0]['bins'])
    for fault in ('video','hardware','bin','duplicate'):
        changed=copy.deepcopy(records)
        if fault=='video':changed[1]['video']['video_sha256']='other'
        elif fault=='hardware':changed[1]['measurement']['hardware']['gpu']='different GPU'
        elif fault=='bin':changed[1]['measurement']['bins'][0]['history_max']=1
        else:changed.append(changed[0])
        with pytest.raises(ValueError):paired_costs(changed)
