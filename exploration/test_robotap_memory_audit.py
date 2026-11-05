import numpy as np

from exploration.audit_robotap_memory import track_events


def test_completed_gaps_exclude_leading_and_trailing_occlusion():
    assert track_events([1, 1, 0, 1, 1, 0, 0, 1]) == (2, [(5, 2)], 1)
    assert track_events([1, 1, 1]) == (None, [], 0)
    assert track_events([0, 0, 0]) == (0, [], 0)


def test_counted_gaps_partition_occlusions_after_query():
    # Exhaust all short visibility sequences, using an independent run counter.
    for bits in range(1 << 10):
        occ = np.array([(bits >> i) & 1 for i in range(10)], dtype=bool)
        first, events, trailing = track_events(occ)
        if first is None:
            assert occ.all()
            continue
        assert sum(g for _, g in events)+trailing == int(occ[first:].sum())
        expected_ends = [i for i in range(first+1, len(occ)) if not occ[i] and occ[i-1]]
        assert [i for i, _ in events] == expected_ends
        for end, gap in events:
            assert gap > 0 and occ[end-gap:end].all()
            assert not occ[end-gap-1] and not occ[end]
