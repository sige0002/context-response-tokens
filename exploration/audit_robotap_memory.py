"""Count RoboTAP annotation support for history evaluation, without a model.

Occlusion gaps count only completed runs after the first visible query. Leading
and trailing occlusions are not reappearance events. No seconds are inferred.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from robotap_data import iter_videos, video_info


THRESHOLDS = (1, 8, 24, 64, 128, 256, 512)


def track_events(occluded):
    """Return (first visible, [(reappearance frame, gap)], trailing gap)."""
    visible = np.flatnonzero(~np.asarray(occluded, dtype=bool))
    if not len(visible):
        return None, [], 0
    events = [(int(end), int(end-start-1))
              for start, end in zip(visible[:-1], visible[1:]) if end-start > 1]
    return int(visible[0]), events, int(len(occluded)-visible[-1]-1)


def distribution(values):
    a = np.asarray(values, dtype=np.int64)
    if not len(a):
        return dict(count=0, min=None, median=None, p90=None, max=None)
    return dict(count=len(a), min=int(a.min()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), max=int(a.max()))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/robotap')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new output')
    root = Path(args.root)
    started = time.monotonic()
    rows, all_gaps, all_trailing, query_ages = [], [], [], []
    for shard, name, example in iter_videos(root):
        info = video_info(example)
        occluded = example['occluded']
        gaps, trailing, ages, tracks_by_threshold = [], [], [], Counter()
        valid_tracks = 0
        for track in occluded:
            first, events, censored = track_events(track)
            if first is None:
                continue
            valid_tracks += 1
            local = [gap for _, gap in events]
            gaps.extend(local)
            ages.append(len(track)-1-first)
            if censored:
                trailing.append(censored)
            for threshold in THRESHOLDS:
                tracks_by_threshold[threshold] += any(g >= threshold for g in local)
        row = dict(id=name, shard=shard, frames=info['frames'], tracks=info['tracks'],
            query_tracks=valid_tracks, video_sha256=info['video_sha256'],
            keys=sorted(example), reappearance=distribution(gaps),
            trailing_occlusions=distribution(trailing), query_to_end=distribution(ages),
            thresholds={str(t):dict(events=sum(g >= t for g in gaps),
                                   tracks=tracks_by_threshold[t]) for t in THRESHOLDS})
        rows.append(row)
        all_gaps.extend(gaps); all_trailing.extend(trailing); query_ages.extend(ages)
        if len(rows) % 25 == 0:
            print(json.dumps(dict(event='annotation_audit', videos=len(rows))), flush=True)
    if len(rows) != 265:
        raise ValueError(f'Expected 265 videos, found {len(rows)}')
    extraction = json.loads((root/'extraction.json').read_text())
    report = dict(
        scope='Annotation coverage only; no model fitting, predictions, or proof that memory is necessary',
        dataset='RoboTAP', source='https://robotap.github.io/',
        definitions=dict(gap='Consecutive occluded frames strictly between two visible observations, after first-visible query',
            events='Only the first visible frame after each gap; counts are correlated across tracks and videos',
            trailing='Occlusion continuing to the video end, right-censored and excluded from reappearance counts',
            query_to_end='Potential history age, not a period during which a point is invisible',
            time_units='Frames only; no FPS or timestamps inferred from arrays',
            clusters='Exact RGB SHA256; different annotations for identical RGB remain separate official examples'),
        sources={str(Path(__file__)):digest(__file__), 'robotap_data.py':digest('robotap_data.py'),
                 str(root/'extraction.json'):digest(root/'extraction.json')},
        verified_shards={k:v for k,v in extraction.items() if k.endswith('.pkl')},
        summary=dict(videos=len(rows), rgb_clusters=len({r['video_sha256'] for r in rows}),
            tracks=sum(r['tracks'] for r in rows), query_tracks=sum(r['query_tracks'] for r in rows),
            frames=distribution([r['frames'] for r in rows]),
            reappearance_gap=distribution(all_gaps), query_to_end=distribution(query_ages),
            trailing_occlusions=distribution(all_trailing),
            example_key_sets=sorted({tuple(r['keys']) for r in rows}),
            gap_at_least={str(t):dict(events=sum(r['thresholds'][str(t)]['events'] for r in rows),
                tracks=sum(r['thresholds'][str(t)]['tracks'] for r in rows),
                videos=sum(r['thresholds'][str(t)]['events'] > 0 for r in rows),
                rgb_clusters=len({r['video_sha256'] for r in rows if r['thresholds'][str(t)]['events']}))
                for t in THRESHOLDS}),
        gap_histogram=dict(sorted(Counter(all_gaps).items())), rows=rows,
        elapsed_seconds=time.monotonic()-started)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
