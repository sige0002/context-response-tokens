"""Balanced delayed-action probes with class-independent suffix observations.

This constructs a controlled diagnostic, not a new natural-video benchmark.
Only IDs and source annotations are used to select examples; no model scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_blocks(rows, classes, blocks, suffix_clips, seed):
    if classes < 2 or blocks < 1 or suffix_clips < 1:
        raise ValueError('At least two classes and positive block sizes required')
    rng = random.Random(seed)
    groups = {label: sorted((r for r in rows if r['label'] == label), key=lambda r: r['id'])
              for label in range(classes)}
    if any(len(group) < blocks for group in groups.values()):
        raise ValueError('Insufficient distinct evidence clips for every class')
    evidence = {label: rng.sample(group, blocks) for label, group in groups.items()}
    used = {r['id'] for group in evidence.values() for r in group}
    pool = sorted((r for r in rows if r['id'] not in used), key=lambda r: r['id'])
    if len(pool) < blocks * suffix_clips:
        raise ValueError('Insufficient disjoint suffix clips')
    suffix = rng.sample(pool, blocks * suffix_clips)
    return [dict(block=b,
                 evidence=[evidence[label][b] for label in range(classes)],
                 suffix=suffix[b*suffix_clips:(b+1)*suffix_clips]) for b in range(blocks)]


def verify_plan(plan):
    classes, delays, frames = plan['classes'], plan['delays'], plan['frames_per_clip']
    if classes < 2 or frames != 8 or delays != sorted(set(delays)) or delays[0] != 0 or delays[1] < 1:
        raise ValueError('Invalid classes, frame count or ordered delay grid')
    identities = {}
    for split in ('train', 'validation'):
        blocks = plan['splits'][split]
        if not blocks or [b['block'] for b in blocks] != list(range(len(blocks))):
            raise ValueError('Missing or duplicated blocks')
        used = {key: set() for key in ('id', 'group', 'sha256')}
        for block in blocks:
            evidence = block['evidence']
            if len(evidence) != classes or {r['label'] for r in evidence} != set(range(classes)):
                raise ValueError('Each common suffix must accompany every class exactly once')
            if len(block['suffix'])*frames < max(delays):
                raise ValueError('Suffix shorter than requested delay')
            for row in evidence+block['suffix']:
                if not 0 <= row['label'] < classes:
                    raise ValueError('Unknown label')
                for key in used:
                    if row[key] in used[key]:
                        raise ValueError('Source reused in evidence or suffix: '+key)
                    used[key].add(row[key])
        identities[split] = used
    if any(identities['train'][k] & identities['validation'][k] for k in identities['train']):
        raise ValueError('Source leakage between probe train and validation')
    return dict(classes=classes, positive_delay_current_only_ceiling=1/classes,
                evidence_clips={s: len(plan['splits'][s])*classes for s in identities},
                suffix_clips={s: sum(len(b['suffix']) for b in plan['splits'][s]) for s in identities},
                counts_per_positive_delay={s: len(plan['splits'][s])*classes for s in identities})


def build_plan(train, validation, train_blocks=4, validation_blocks=2,
               delays=(0, 1, 8, 32, 128), seed=71):
    paths = {'train': Path(train).resolve(), 'validation': Path(validation).resolve()}
    rows = {s: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            for s, p in paths.items()}
    classes = 174
    # Original corpus split integrity, including sources not sampled below.
    for key in ('id', 'group', 'sha256'):
        if {r[key] for r in rows['train']} & {r[key] for r in rows['validation']}:
            raise ValueError('Original corpus split overlap: '+key)
    suffix_clips = (max(delays)+7)//8
    plan = dict(format_version=1, classes=classes, frames_per_clip=8, delays=list(delays), seed=seed,
        manifests={s: dict(path=str(p), sha256=sha(p)) for s, p in paths.items()},
        splits={s: select_blocks(rows[s], classes, n, suffix_clips, seed+i)
                for i,(s,n) in enumerate((('train',train_blocks),('validation',validation_blocks)))},
        target='Original SSv2 action of the first eight observations; all later observations are distractors',
        readout='Frozen encoder; mean-pooled current tokens; separate ridge probe per condition, pooled across delays',
        ridge_alpha=1., feature_scaling='Center using training features only; no validation selection',
        conditions=['full_memory', 'suffix_only', 'saved_evidence'],
        scope='Controlled delayed-action diagnostic, not natural continuous-video or arbitrary-question recall',
        limits=['Suffixes concatenate independent clips; delay is observations, not elapsed natural seconds',
                'Original encoder was trained on eight-observation clips; longer inputs test length extrapolation',
                'Task is known before observation; this does not test query-agnostic memory',
                'A linear probe can underestimate information usable by other readouts',
                'Validation is previously inspected SSv2 validation, not an untouched external test',
                'Small probe split and one encoder seed cannot establish architectural superiority'],
        source_sha256=sha(__file__))
    plan['structure'] = verify_plan(plan)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',default='data/ssv2/videos_trainval/train.jsonl')
    p.add_argument('--validation',default='data/ssv2/videos_trainval/val.jsonl')
    p.add_argument('--train-blocks',type=int,default=4)
    p.add_argument('--validation-blocks',type=int,default=2)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError('Use a fresh plan path')
    plan=build_plan(args.train,args.validation,args.train_blocks,args.validation_blocks)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan['structure'],indent=2))


if __name__ == '__main__': main()
