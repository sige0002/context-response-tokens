"""Final diagnostic grid and change-of-effect estimands, using synthetic counts."""
from copy import deepcopy
from pathlib import Path

import pytest

from exploration.final_memory_layers import aggregate,CONDITIONS,MODELS
from exploration.run_final_memory_layers import build_plan,verify_plan,ROOT


def fixture_records():
    records={}
    for m in MODELS:
        for seed in (42,43,44):
            full=(200 if m.startswith('mamba') else 210)+(seed-42)
            loss=1 if m.startswith('mamba') else 5
            scores={c:dict(correct=full if c=='full_stream' else full-loss,
                total=348,original_label_top1=(full if c=='full_stream' else full-loss)/348) for c in CONDITIONS}
            records[f'{m}_seed{seed}']=dict(scores=scores)
    return records


def test_effect_difference_is_not_raw_architecture_accuracy():
    records=fixture_records();result=aggregate(records)
    values=result['effect_differences']['clear_4']
    assert values['mean_pp']==pytest.approx(400/348)
    assert values['sample_sd_pp']==0
    # Mamba is less sensitive to erasure in this fixture but still less accurate.
    m=result['models'][MODELS[0]]['clear_4']['accuracy_percent']
    a=result['models'][MODELS[1]]['clear_4']['accuracy_percent']
    assert all(x<y for x,y in zip(m,a))
    assert result['effect_differences']['full_stream']['mean_pp']==0


def test_incomplete_grid_or_changed_cohort_is_rejected():
    records=fixture_records()
    missing=deepcopy(records);missing.pop(next(iter(missing)))
    with pytest.raises(ValueError,match='six'):aggregate(missing)
    missing=deepcopy(records);next(iter(missing.values()))['scores'].pop('clear_4')
    with pytest.raises(ValueError,match='conditions'):aggregate(missing)
    for field,value in [('total',347),('original_label_top1',.99),('correct',-1)]:
        changed=deepcopy(records);next(iter(changed.values()))['scores']['full_stream'][field]=value
        with pytest.raises(ValueError):aggregate(changed)


def test_current_plan_keeps_all_seeds_and_positions_without_loading_weights():
    plan=build_plan(ROOT/'results/temporal_priority_seed42_rgbfix',
        ROOT/'results/buffered_attention_cost_control',ROOT/'results/video_history_intervention_seed17')
    verify_plan(plan)
    assert len(plan['runs'])==6 and len(plan['subset'])==348
    assert {(r['model'],r['seed']) for r in plan['runs']}=={(m,s) for m in MODELS for s in (42,43,44)}
    assert plan['conditions']['reset_all']==[]
    assert all(r['config']['steps']==20000 and r['config']['frames']==8 for r in plan['runs'])
    assert plan['subset']==__import__('json').loads((Path(plan['cohort'])/'plan.json').read_text())['subset']
