import copy

import pytest

from benchmark_real_video import summarize_rows


def observations():
    return [dict(repeat=repeat,history_frames=frame,absolute_frames_after=frame+1,
        gpu_ms=frame+1.,model_wall_ms=frame+2.,pipeline_wall_ms=frame+3.,
        state_bytes=(frame+1)*10,torch_allocated_before_bytes=100,
        torch_allocated_after_bytes=120,torch_peak_allocated_bytes=200)
        for repeat in range(2) for frame in range(10)]


def test_summaries_account_for_every_frame_and_cold_start_separately():
    report=summarize_rows(observations(),10,2)
    assert [r['samples'] for r in report['bins']]==[2,14,4]
    assert [r['unique_frames'] for r in report['bins']]==[1,7,2]
    assert [r['gpu_ms']['p50'] for r in report['bins']]==[1.,5.,9.5]
    assert report['torch_peak_allocated_bytes']==200
    assert report['max_state_bytes']==100


@pytest.mark.parametrize('fault',['missing','duplicate','state_leak','invalid_time','invalid_memory'])
def test_invalid_benchmark_records_rejected(fault):
    rows=copy.deepcopy(observations())
    if fault=='missing': rows.pop()
    elif fault=='duplicate': rows[-1]=rows[0]
    elif fault=='state_leak': rows[-1]['absolute_frames_after']+=10
    elif fault=='invalid_time': rows[-1]['gpu_ms']=float('nan')
    elif fault=='invalid_memory': rows[-1]['torch_peak_allocated_bytes']=0
    with pytest.raises(ValueError):
        summarize_rows(rows,10,2)
