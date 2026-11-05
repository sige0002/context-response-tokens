"""Audit a pinned OVO-Bench annotation file before adopting it for memory.

No videos, models or source code from upstream are executed. Query time is
reported as query time, never as the distance to supporting evidence.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


COMMIT = 'c34093f9a072d93f9e1c6b923446dfba71f4528e'
BLOB = '72a84caa67e077f381434bbc811fc96b445c51be'
GROUPS = {'backward': ('EPM', 'ASI', 'HLD'),
          'realtime': ('STU', 'OJR', 'ATR', 'ACR', 'OCR', 'FPD'),
          'forward': ('REC', 'SSR', 'CRR')}


def time_summary(values):
    return dict(n=len(values), minimum=float(min(values)),
        median=float(np.median(values)), mean=float(np.mean(values)),
        p90=float(np.quantile(values, .9)), maximum=float(max(values)))


def audit(path):
    payload = path.read_bytes()
    blob = hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
    if blob != BLOB:
        raise ValueError('Annotations differ from the pinned upstream version')
    rows = json.loads(payload)
    known = {task for group in GROUPS.values() for task in group}
    if len({r['id'] for r in rows}) != len(rows) or {r['task'] for r in rows} != known:
        raise ValueError('Duplicate ids or unsupported task types')
    tasks = {}
    all_times = []
    for task in sorted(known):
        records = [r for r in rows if r['task'] == task]
        times, answer_counts, option_counts, text_mismatches = [], Counter(), Counter(), []
        for row in records:
            if task in GROUPS['forward']:
                times.extend(item['realtime'] for item in row['test_info'])
            else:
                options, gt = row['options'], row['gt']
                if not isinstance(gt, int) or not 0 <= gt < len(options):
                    raise ValueError('Invalid ground-truth option index')
                if options[gt] != row['answer']:
                    # Upstream evaluation uses gt, not the free-text answer.
                    # Wording differences are not automatically label errors.
                    text_mismatches.append(row['id'])
                times.append(row['realtime'])
                answer_counts[gt] += 1
                option_counts[len(options)] += 1
        if any(isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) or t < 0 for t in times):
            raise ValueError('Invalid query timestamp')
        fields = sorted(set().union(*(r.keys() for r in records)))
        tasks[task] = dict(records=len(records), evaluations=len(times),
            unique_videos=len({r['video'] for r in records}), fields=fields,
            query_seconds=time_summary(times), option_counts=dict(option_counts),
            answer_index_counts=dict(answer_counts),
            answer_text_not_exactly_selected_option_ids=text_mismatches)
        all_times.extend(times)
    groups = {}
    for name, names in GROUPS.items():
        subset = [r for r in rows if r['task'] in names]
        groups[name] = dict(records=len(subset), evaluations=sum(tasks[t]['evaluations'] for t in names),
            unique_videos=len({r['video'] for r in subset}))
        if name != 'forward':
            # Reference baselines from labels alone, not model predictions.
            first = [float(r['gt'] == 0) for r in subset]
            random = [1 / len(r['options']) for r in subset]
            groups[name]['nonvisual_references'] = dict(
                always_first_option_micro=float(np.mean(first)),
                uniform_random_option_expected_micro=float(np.mean(random)),
                always_first_option_task_macro=float(np.mean([
                    np.mean([r['gt'] == 0 for r in subset if r['task'] == t]) for t in names])),
                uniform_random_option_expected_task_macro=float(np.mean([
                    np.mean([1/len(r['options']) for r in subset if r['task'] == t]) for t in names])))
    backward_fields = set().union(*(set(tasks[t]['fields']) for t in GROUPS['backward']))
    # Make the exact available schema reviewable rather than guessing intervals
    # from English questions, or treating realtime as evidence completion.
    expected_fields = {'id','task','video','question','answer','options','gt','realtime'}
    if backward_fields != expected_fields:
        raise ValueError('Backward schema changed; review temporal evidence fields')
    return dict(upstream_commit=COMMIT, upstream_blob_sha1=blob,
        source_url=f'https://github.com/JoeLeelyf/OVO-Bench/blob/{COMMIT}/data/ovo_bench_new.json',
        bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
        records=len(rows), evaluations=len(all_times), unique_videos=len({r['video'] for r in rows}),
        query_seconds=time_summary(all_times), groups=groups, tasks=tasks,
        answer_text_exact_mismatches=sum(len(t['answer_text_not_exactly_selected_option_ids']) for t in tasks.values()),
        scoring_field='gt, as used by upstream utils/OVOBench.py; free-text answer not substituted',
        backward_has_explicit_evidence_interval=False,
        backward_has_answer_validity_window=False,
        derived_recall_delay_seconds=None,
        interpretation='Suitable candidate for backward QA; this schema alone cannot measure evidence-to-query delay.',
        limits=['Annotations only; videos were not downloaded or checked',
                'No recognition or memory accuracy was evaluated',
                'Current SSv2 classifier cannot answer these language questions without a separate readout',
                'Text-answer differences can be punctuation, paraphrases or substantive discrepancies; no labels were corrected',
                'Do not infer absence of future/current-frame answer shortcuts from task names'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = audit(Path(args.annotations))
    result['audit_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('records','evaluations','unique_videos','groups','derived_recall_delay_seconds')}))


if __name__ == '__main__':
    main()
