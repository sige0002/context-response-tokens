"""Independent CPU recount of the saved history intervention predictions.

Checks source identities, official manifest labels, input receipts, checkpoint
identity and paired counts. Does not independently repeat the GPU forwards.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from video_checkpoint import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    plan = json.loads((run/'plan.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    assert file_hash(run/'plan.json') == summary['plan_sha256']
    for filename, key in [('logits.npz', 'logits_sha256'), ('inputs.json', 'inputs_sha256')]:
        assert file_hash(run/filename) == summary[key]
    assert file_hash(plan['checkpoint']) == plan['checkpoint_info']['checkpoint_sha256']
    for filename, expected in plan['sources'].items():
        assert file_hash(filename) == expected, filename
    config = plan['checkpoint_info']['config']
    assert file_hash(config['val']) == config['val_sha256']
    rows = [json.loads(line) for line in Path(config['val']).read_text().splitlines() if line.strip()]
    inputs = json.loads((run/'inputs.json').read_text())
    old_inputs_path = Path(plan['prior'])/'video_inputs.json'
    assert file_hash(old_inputs_path) == plan['prior_inputs_sha256']
    old_inputs = json.loads(old_inputs_path.read_text())
    old_logits_path = Path(plan['prior'])/'video_logits.npz'
    assert file_hash(old_logits_path) == plan['prior_logits_sha256']
    with np.load(run/'logits.npz') as raw, np.load(old_logits_path) as old:
        assert set(raw.files) == {'targets', 'ids', *plan['conditions']}
        ids, targets = raw['ids'], raw['targets']
        assert len(ids) == len(set(ids)) == len(plan['subset']) == len(inputs) == len(old_inputs) == 348
        assert np.array_equal(targets, old['targets'])
        for index, (item, record, previous) in enumerate(zip(plan['subset'], inputs, old_inputs)):
            official = rows[item['index']]
            assert official['id'] == item['id'] == record['id'] == ids[index]
            assert official['label'] == item['label'] == record['label'] == targets[index]
            assert {k: v for k, v in record.items() if k != 'current_pixels_sha256'} == previous
        full_pred = raw['full_stream'].argmax(1)
        full_correct = full_pred == targets
        recount = {}
        for name in plan['conditions']:
            logits = raw[name]
            assert logits.shape == (348, 174) and np.isfinite(logits).all()
            pred = logits.argmax(1)
            correct = pred == targets
            counts = dict(correct=int(np.count_nonzero(correct)), total=len(targets),
                changed_predictions=int(np.count_nonzero(pred != full_pred)),
                gained_vs_stream=int(np.count_nonzero(correct & ~full_correct)),
                lost_vs_stream=int(np.count_nonzero(~correct & full_correct)))
            for key, value in counts.items():
                assert summary['scores'][name][key] == value, (name, key)
            for row, i, y, p in zip(summary['scores'][name]['predictions'], ids, targets, pred):
                assert row == dict(id=str(i), target=int(y), prediction=int(p))
            assert summary['scores'][name]['original_label_top1'] == counts['correct']/len(ids)
            recount[name] = counts
        replay = dict(exact=bool(np.array_equal(raw['full_clip'], old['111'])),
            logit_max_abs=float(np.max(np.abs(raw['full_clip'].astype(float)-old['111']))),
            changed_predictions=int(np.count_nonzero(raw['full_clip'].argmax(1) != old['111'].argmax(1))))
        assert replay == summary['prior_full_clip_replay']
    record = dict(passed=True, scope='CPU recount and provenance; no independent GPU forward',
        summary_sha256=file_hash(run/'summary.json'), verifier_sha256=file_hash(__file__),
        videos=348, conditions=len(plan['conditions']), counts=recount, full_clip_replay=replay)
    (run/'verification.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
