"""Guard against rescheduling changing the scientific comparison or task."""
import copy

import pytest

from temporal_priority import relocate_command, validate_video_row, verify_plan


def row():
    config = dict(kind='mamba', steps=20000, seed=43, frames=8, classes=174, batch_size=4, accumulation=8)
    command = ['uv','run','--locked','python','train_video.py','--output','/study/run']
    for key,value in config.items():
        if key!='kind': command += ['--'+key.replace('_','-'),str(value)]
    evaluations = [['uv','run','--locked','python','evaluate_video.py','--checkpoint','/study/run/last.pt',
                    '--output','/study/run/'+name] for name in ('final_clip_stream.json','final_history_ablations.json')]
    return dict(config=config,run='/study/run',command=command,evaluations=evaluations)


def test_relocation_does_not_change_similarly_named_inputs():
    original=['--checkpoint','/study/run/last.pt','--manifest','/study/runaway/data.jsonl','--output','/study/run']
    actual=relocate_command(original,'/study/run','/priority/run')
    assert actual==['--checkpoint','/priority/run/last.pt','--manifest','/study/runaway/data.jsonl','--output','/priority/run']
    assert original[1]=='/study/run/last.pt'


def test_video_protocol_survives_relocation():
    value=row(); old=copy.deepcopy(value)
    value['command']=relocate_command(value['command'],value['run'],'/priority/new')
    value['evaluations']=[relocate_command(c,value['run'],'/priority/new') for c in value['evaluations']]
    value['run']='/priority/new'; validate_video_row(value)
    assert value['config']==old['config']


def test_changed_source_is_rejected_before_loading_jobs(tmp_path):
    source=tmp_path/'source.py'; source.write_text('changed\n')
    with pytest.raises(ValueError,match='Priority source changed'):
        verify_plan(dict(source_sha256={str(source):'not_the_current_digest'}))


@pytest.mark.parametrize('change',('point_training','fewer_updates','wrong_checkpoint','wrong_output','missing_history'))
def test_rejects_changed_task_budget_or_evaluation(change):
    value=row()
    if change=='point_training': value['command'][4]='train_points.py'
    elif change=='fewer_updates': value['command'][value['command'].index('--steps')+1]='1024'
    elif change=='wrong_checkpoint': value['evaluations'][0][-3]='/other/last.pt'
    elif change=='wrong_output': value['evaluations'][0][-1]='/other/final_clip_stream.json'
    elif change=='missing_history': value['evaluations'].pop()
    with pytest.raises(ValueError): validate_video_row(value)
