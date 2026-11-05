"""Recount gradient diagnostics, including each projection and MIMO rank."""
import argparse
import json
from pathlib import Path

import torch

from exploration.check_mimo_gradients import gradient_metrics
from video_checkpoint import file_hash


def run(root):
    torch.set_num_threads(2)
    summary = json.loads((root/'summary.json').read_text())
    plan = json.loads((root/'plan.json').read_text())
    if (not summary['completed'] or summary['plan_sha256'] != file_hash(root/'plan.json')
            or summary['artifact_sha256'] != file_hash(root/'gradients.pt')):
        raise ValueError('Changed or incomplete gradient artifacts')
    if any(file_hash(p)!=h for p,h in plan['sources'].items()):
        raise ValueError('Gradient diagnostic source changed')
    saved = torch.load(root/'gradients.pt',map_location='cpu',weights_only=True)
    results = {}
    for name, case in saved['cases'].items():
        actual, expected = case['gradients_native'], case['gradients_reference']
        report = summary['results'][name]
        if actual.keys()!=expected.keys() or actual.keys()!=report['gradients'].keys():
            raise ValueError('Gradient key coverage differs')
        for key in actual:
            if gradient_metrics(actual[key],expected[key]) != report['gradients'][key]:
                raise ValueError('Saved gradient metric recount differs')
        blocks = {}
        offset = 0
        sizes = [('z',512),('x',512)]+[(f'B{r}',64) for r in range(4)]
        sizes += [(f'C{r}',64) for r in range(4)]+[('dt',8),('A',8),('trap',8),('angle',16)]
        for key, size in sizes:
            selection = slice(offset,offset+size)
            blocks['in_proj.'+key] = gradient_metrics(actual['in_proj.weight'][selection],
                                                       expected['in_proj.weight'][selection])
            offset += size
        if offset != actual['in_proj.weight'].shape[0]:
            raise ValueError('Projection rows not fully covered')
        for key in ('mimo_x','mimo_z','mimo_o','B_bias','C_bias'):
            for rank in range(4):
                blocks[f'{key}.rank{rank}'] = gradient_metrics(actual[key][:,rank],expected[key][:,rank])
        nonzero = {k:v for k,v in blocks.items() if v['reference_norm']>plan['alarm_reference_norm_floor']}
        results[name] = dict(blocks=blocks,
            worst_nonzero_block=max(nonzero,key=lambda k:nonzero[k]['relative_l2']),
            max_nonzero_block_relative_l2=max(v['relative_l2'] for v in nonzero.values()),
            block_alarms=[k for k,v in nonzero.items() if v['relative_l2']>plan['alarm_relative_l2']])
    result = dict(passed_recount=True,summary_sha256=file_hash(root/'summary.json'),
        source_sha256=file_hash(__file__),results=results,
        scope='Recount saved native/FP64 gradient arrays and report projection/rank sub-blocks to avoid masking weak branches in whole-parameter norms. No additional GPU replay or correctness theorem.')
    with (root/'verification.json').open('x') as output:
        json.dump(result,output,indent=2,allow_nan=False)
        output.write('\n')
    print(json.dumps({name:{k:v for k,v in case.items() if k!='blocks'}
                      for name,case in results.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
