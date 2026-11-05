"""Tests for paired identity, causal metadata and exact MCQ scoring."""
from copy import deepcopy

import pytest

from exploration.river_memory import DELAYS, SHA256, audit_rows, inspect_rows, read_pinned, score


def records():
    output = []
    for q, answer in [('first', 'A'), ('second', 'C')]:
        for delay, gap in zip(DELAYS, (20, 45, 600, 2400)):
            output.append(dict(question_id=q, video_id='same-video', video_source='LVBench',
                video_path='same-video.mp4', question_type=f'Retro-Memory@{delay}', question=q,
                time_reference=[10., 15.], question_time=15.+gap, duration_sec=3000., fps=30.,
                choices=['(A) alpha', '(B) beta', '(C) gamma', '(D) delta'], correct_answer=answer))
    return output


def choices(rows):
    return dict(annotation_sha256=SHA256, predictions=[dict(question_id=r['question_id'],
        question_type=r['question_type'], choice_index=ord(r['correct_answer'])-65) for r in rows])


def test_shared_ids_are_paired_not_overwritten_or_treated_as_independent():
    rows = records()
    audit = audit_rows(rows)
    assert audit['records'] == 8 and audit['unique_questions'] == 2 and audit['unique_videos'] == 1
    assert audit['delays']['long']['evidence_end_to_query_seconds']['median'] == 600
    pred = choices(rows)
    # Lose first question, then recover the second, across the same source video.
    pred['predictions'][3]['choice_index'] = 1
    pred['predictions'][4]['choice_index'] = 0
    result = score(rows, pred)
    assert result['delays']['short']['accuracy'] == .5
    assert result['delays']['very_long']['correctness_transitions_from_short'] == {'1->0': 1, '0->1': 1}
    assert result['mean_over_delays'] == .75
    assert result['protocol_and_model_provenance_verified'] is False


def test_inverted_evidence_excludes_entire_group_without_repair():
    rows = records()
    for row in rows[:4]:
        row['time_reference'] = [25., 15.]
    original = deepcopy(rows)
    audit = audit_rows(rows)
    assert rows == original
    assert audit['excluded_question_ids'] == ['first']
    assert audit['eligible_records'] == 4
    assert len(audit['temporal_issues']) == 4
    with pytest.raises(ValueError, match='excluded'):
        score(rows, choices(rows))
    assert score(rows, choices(rows[4:]))['mean_over_delays'] == 1.


def test_five_options_have_different_chance_and_accept_last_option():
    rows = records()
    for row in rows[:4]:
        row['choices'].append('(E) epsilon')
        row['correct_answer'] = 'E'
    assert audit_rows(rows)['delays']['short']['uniform_random_expected_accuracy'] == .225
    assert score(rows, choices(rows))['mean_over_delays'] == 1.


def test_query_boundaries_and_actual_future_evidence_are_distinguished():
    rows = records()
    for row in rows[:4]:
        row['duration_sec'] = 2414.9999999999995
    audit = audit_rows(rows)
    assert len(audit['sub_microsecond_duration_excess']) == 1
    assert audit['eligible_records'] == 8
    for row in rows[:4]:
        row['duration_sec'] = 2414.99
    assert audit_rows(rows)['excluded_question_ids'] == ['first']
    rows = records()
    rows[0]['question_time'] = 14.
    eligible, issues, _ = inspect_rows(rows)
    assert len(eligible) == 4
    assert 'evidence_after_query' in issues[0]['reasons']


def test_missing_duplicate_changed_or_invalid_answers_are_rejected(tmp_path):
    rows = records()
    bad = deepcopy(rows)
    bad[1]['correct_answer'] = 'D'
    with pytest.raises(ValueError, match='changes between delays'):
        inspect_rows(bad)
    with pytest.raises(ValueError, match='Duplicate'):
        inspect_rows(rows + rows[:1])
    with pytest.raises(ValueError, match='missing'):
        inspect_rows(rows[:-1])
    pred = choices(rows)
    with pytest.raises(ValueError, match='Incomplete'):
        score(rows, {**pred, 'predictions': pred['predictions'][:-1]})
    with pytest.raises(ValueError, match='duplicate'):
        score(rows, {**pred, 'predictions': pred['predictions'] + pred['predictions'][:1]})
    for value in (True, 1., -1, 4, 'A', 'ABCD', None):
        bad = deepcopy(pred)
        bad['predictions'][0]['choice_index'] = value
        with pytest.raises(ValueError, match='integer'):
            score(rows, bad)
    with pytest.raises(ValueError, match='identity'):
        score(rows, {**pred, 'annotation_sha256': 'wrong'})
    path = tmp_path/'modified.json'
    path.write_text('[]')
    with pytest.raises(ValueError, match='pinned'):
        read_pinned(path)
