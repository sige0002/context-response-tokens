"""Pinned RIVER Retro-Memory audit and strict paired-delay MCQ scoring.

No model, video download, or benchmark training is performed. The readout and
causal video integration remain separate work; scores alone do not verify them.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


REVISION = '0338ea79419e375b5760c11a1f9a3e4194af6add'
COMMIT = '8aab82325d0986824f8504e86669fa52025593d2'
SHA256 = '34079d42b2ea51fb47b2c5b88cab5ba6ec724983d5f6b28eb3dcf735509170b1'
BLOB = 'c09120f319e13a1d0777eaf81740e6129cb2f39a'
DELAYS = ('short', 'medium', 'long', 'very_long')
BOUNDS = dict(short=(15, 30), medium=(30, 60), long=(300, 900), very_long=(1800, 3600))
# Sub-microsecond representation differences, not permission to include a frame
# beyond question_time. Actual decoding must still enforce PTS <= query time.
TIME_TOLERANCE = 1e-6


def read_pinned(path):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError('Annotations differ from the pinned RIVER release')
    blob = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
    if blob != BLOB:
        raise ValueError('Annotations differ from the official GitHub blob')
    return json.loads(payload)


def finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value)


def key(row):
    return row['question_id'], row['question_type']


def inspect_rows(rows):
    """Validate identity/schema; retain and report temporal defects, never fix them."""
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty annotation list')
    groups, seen, issues, rounding = defaultdict(list), set(), [], []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Expected annotation objects')
        for field in ('question_id', 'question_type', 'question', 'video_id', 'video_source', 'video_path'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f'Invalid {field}')
        if key(row) in seen:
            raise ValueError('Duplicate question_id/question_type pair')
        seen.add(key(row))
        if row['question_type'] not in {f'Retro-Memory@{d}' for d in DELAYS}:
            raise ValueError('Unsupported delay category')
        choices = row.get('choices')
        if (not isinstance(choices, list) or len(choices) not in (4, 5)
                or any(not isinstance(c, str) or not c.startswith(f'({chr(65+i)}) ')
                       for i, c in enumerate(choices))
                or row.get('correct_answer') not in tuple(chr(65+i) for i in range(len(choices)))):
            raise ValueError('Invalid four- or five-option MCQ')
        ref = row.get('time_reference')
        if not isinstance(ref, list) or len(ref) != 2:
            raise ValueError('Missing evidence interval')
        values = [*ref, row.get('question_time'), row.get('duration_sec'), row.get('fps')]
        if not all(finite_number(v) for v in values):
            raise ValueError('Non-finite or missing timestamp/fps')
        start, end, query, duration, fps = values
        if min(start, end, query) < 0 or duration <= 0 or fps <= 0:
            raise ValueError('Negative timestamp or non-positive duration/fps')
        reasons = []
        if start > end:
            reasons.append('evidence_start_after_end')
        if end > query:
            reasons.append('evidence_after_query')
        if query > duration + TIME_TOLERANCE:
            reasons.append('query_after_video')
        elif query > duration:
            rounding.append(dict(question_id=row['question_id'], question_type=row['question_type'],
                                 excess_seconds=query-duration))
        lo, hi = BOUNDS[row['question_type'].split('@')[1]]
        if not lo-TIME_TOLERANCE <= query-end <= hi+TIME_TOLERANCE:
            reasons.append('end_to_query_gap_outside_named_delay')
        if reasons:
            issues.append(dict(question_id=row['question_id'], question_type=row['question_type'],
                               time_reference=ref, question_time=query, duration_sec=duration,
                               reasons=reasons))
        groups[row['question_id']].append(row)
    for records in groups.values():
        if {r['question_type'] for r in records} != {f'Retro-Memory@{d}' for d in DELAYS}:
            raise ValueError('Question missing one or more paired delays')
        fixed = lambda r: {k: v for k, v in r.items() if k not in ('question_time', 'question_type')}
        if any(fixed(r) != fixed(records[0]) for r in records[1:]):
            raise ValueError('Question/evidence/answer/source changes between delays')
    invalid = {r['question_id'] for r in issues}
    eligible = [r for r in rows if r['question_id'] not in invalid]
    if not eligible:
        raise ValueError('No complete temporally valid question groups')
    return eligible, issues, rounding


def distribution(values):
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), maximum=max(values))


def audit_rows(rows):
    eligible, issues, rounding = inspect_rows(rows)
    delays = {}
    for delay in DELAYS:
        records = [r for r in eligible if r['question_type'] == f'Retro-Memory@{delay}']
        counts = Counter(r['correct_answer'] for r in records)
        delays[delay] = dict(evaluations=len(records),
            evidence_end_to_query_seconds=distribution([r['question_time']-r['time_reference'][1] for r in records]),
            query_seconds=distribution([r['question_time'] for r in records]),
            answer_counts=dict(sorted(counts.items())),
            option_counts=dict(sorted(Counter(len(r['choices']) for r in records).items())),
            always_A_accuracy=counts['A']/len(records),
            uniform_random_expected_accuracy=statistics.mean(1/len(r['choices']) for r in records))
    return dict(records=len(rows), unique_questions=len({r['question_id'] for r in rows}),
        unique_videos=len({(r['video_source'], r['video_id']) for r in rows}),
        source_records=dict(sorted(Counter(r['video_source'] for r in rows).items())),
        eligible_records=len(eligible), eligible_questions=len(eligible)//len(DELAYS),
        eligible_videos=len({(r['video_source'], r['video_id']) for r in eligible}),
        excluded_question_ids=sorted({i['question_id'] for i in issues}),
        temporal_issues=issues, sub_microsecond_duration_excess=rounding,
        time_tolerance_seconds=TIME_TOLERANCE, delays=delays,
        annotation_sha256=SHA256, upstream_git_blob=BLOB, upstream_git_commit=COMMIT,
        huggingface_revision=REVISION,
        source_url=f'https://huggingface.co/datasets/nanamma/RIVER/blob/{REVISION}/annotations/Retro-Memory.json',
        interpretation='Paired MCQ candidate with annotated evidence-to-query gaps; not a model result.',
        limits=[
            'Derived eligible subset; not the unmodified official benchmark score.',
            'Do not split by row: each question repeats four times, and videos contain multiple questions.',
            'The dataset card train label is not permission to train on this published evaluation set.',
            'Actual video duration, evidence sufficiency, repeats and current-frame shortcuts are unchecked.',
            'Intervening content and current images change across delays; this is not a pure forgetting experiment.',
            'Sourced from LVBench, LongVideoBench and Vript-RR; not a robotics task-success evaluation.',
            'No common language readout has been trained or evaluated for this encoder.',
            'Scores require separate verification of causal sampling, model identity and training provenance.',
        ])


def score(rows, predictions):
    """Require one integer choice per eligible query, with complete paired coverage."""
    eligible, issues, _ = inspect_rows(rows)
    if not isinstance(predictions, dict) or predictions.get('annotation_sha256') != SHA256:
        raise ValueError('Prediction annotation identity is missing or differs')
    expected = {key(r): r for r in eligible}
    output = predictions.get('predictions')
    if not isinstance(output, list):
        raise ValueError('Expected prediction list')
    answers = {}
    for row in output:
        if not isinstance(row, dict) or set(row) != {'question_id', 'question_type', 'choice_index'}:
            raise ValueError('Predictions must contain identifiers and one choice_index')
        if not all(isinstance(row[x], str) for x in ('question_id', 'question_type')):
            raise ValueError('Prediction identifiers must be strings')
        k, choice = key(row), row['choice_index']
        if k not in expected or k in answers:
            raise ValueError('Unknown, excluded or duplicate prediction')
        if type(choice) is not int or not 0 <= choice < len(expected[k]['choices']):
            raise ValueError('choice_index must be an integer within this question options')
        answers[k] = choice
    if answers.keys() != expected.keys():
        raise ValueError('Incomplete paired population')
    result = {}
    for delay in DELAYS:
        records = [r for r in eligible if r['question_type'] == f'Retro-Memory@{delay}']
        correct = lambda r, d: answers[(r['question_id'], f'Retro-Memory@{d}')] == ord(r['correct_answer'])-65
        transitions = Counter(f'{int(correct(r, "short"))}->{int(correct(r, delay))}' for r in records)
        hits = sum(correct(r, delay) for r in records)
        result[delay] = dict(correct=hits, total=len(records), accuracy=hits/len(records),
                             correctness_transitions_from_short=dict(sorted(transitions.items())))
    return dict(annotation_sha256=SHA256, subset='complete_temporally_valid_question_groups',
                excluded_question_ids=sorted({r['question_id'] for r in issues}), delays=result,
                mean_over_delays=statistics.mean(r['accuracy'] for r in result.values()),
                protocol_and_model_provenance_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit', 'score'))
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.command == 'score') != (args.predictions is not None):
        parser.error('--predictions is required only for score')
    rows = read_pinned(args.annotations)
    prediction_bytes = args.predictions.read_bytes() if args.predictions else None
    result = audit_rows(rows) if args.command == 'audit' else score(rows, json.loads(prediction_bytes))
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.predictions:
        result['predictions_sha256'] = hashlib.sha256(prediction_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('eligible_records', 'eligible_questions', 'delays') if k in result}))


if __name__ == '__main__':
    main()
