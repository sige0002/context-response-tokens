"""Recognition/cost protocol without point-training completion dependencies."""
from pathlib import Path
import statistics

from gmmix_experiment import UV, verify_completion as verify_gmmix
from multidepth_experiment import verify_completion as verify_multi, verify_qualification
from point_checkpoint import file_hash
from point_experiment import read_json
from run_video_comparisons import verify_completion as verify_temporal, verify_inputs
from summarize_full_video import paired_correctness, verify_shared_protocol
from summarize_video import read_run


ROOT = Path(__file__).resolve().parent
MODELS = ('mamba_interleaved', 'attention_interleaved', 'mamba_late', 'gmmix_multidepth_after_encoder')


def relocate_command(command, old_run, new_run):
    """Move only whole paths at/below the run; never replace substrings."""
    old, new = Path(old_run), Path(new_run)
    result = []
    for value in command:
        path = Path(value)
        try:
            suffix = path.relative_to(old)
        except ValueError:
            result.append(value)
        else:
            result.append(str(new/suffix))
    return result


def video_verifier(config):
    return {'gmmix':verify_gmmix, 'gmmix_multidepth':verify_multi}.get(config['kind'], verify_temporal)


def validate_video_row(row):
    config = row['config']; kind = config['kind']; run = Path(row['run'])
    trainer = 'train_multidepth_video.py' if kind == 'gmmix_multidepth' else 'train_video.py'
    if trainer not in row['command'] or row['command'][row['command'].index('--output')+1] != str(run):
        raise ValueError('Wrong video trainer or output path')
    for key in ('steps', 'seed', 'frames', 'classes', 'batch_size', 'accumulation'):
        flag = '--'+key.replace('_','-')
        if row['command'][row['command'].index(flag)+1] != str(config[key]):
            raise ValueError('Video command changed the fixed protocol: '+key)
    evaluator = 'evaluate_multidepth_video.py' if kind == 'gmmix_multidepth' else 'evaluate_video.py'
    destinations = []
    for command in row['evaluations']:
        if evaluator not in command or command[command.index('--checkpoint')+1] != str(run/'last.pt'):
            raise ValueError('Wrong video evaluator or checkpoint')
        destination = Path(command[command.index('--output')+1])
        if destination.parent != run:
            raise ValueError('Evaluation output escaped its planned run')
        destinations.append(destination.name)
    if destinations != ['final_clip_stream.json', 'final_history_ablations.json']:
        raise ValueError('Missing or duplicate recognition/history evaluation')


def build_plan(video, gmmix, costs, multidepth, replications, output):
    locations = dict(video=video, gmmix=gmmix, costs=costs, multidepth=multidepth, replications=replications)
    old = {key:read_json(path/'plan.json') for key,path in locations.items()}
    vp, gp, cp, mp, rp = (old[k] for k in locations)
    if gp['video_plan_sha256'] != file_hash(video/'plan.json') or mp['video_plan_sha256'] != file_hash(video/'plan.json'):
        raise ValueError('Video plans belong to different studies')
    reference = vp['reference_config']
    if reference['seed'] != 42 or reference['steps'] != 20000:
        raise ValueError('Expected the original fixed-budget seed42 study')
    encoders = [dict(name=reference['kind']+'_'+reference['placement'], run=vp['reference'], config=reference)]
    encoders += [dict(name=r['kind']+'_'+r['placement'], run=r['run'],
                     config=dict(reference, kind=r['kind'], placement=r['placement'])) for r in vp['controls']]
    encoders.append(dict(name='gmmix_after_encoder', run=gp['run'], config=gp['config']))
    if len(cp['encoders']) != len(encoders) or any(not any(r['run']==e['run'] and r['config']==e['config'] for e in encoders) for r in cp['encoders']):
        raise ValueError('Cost and recognition encoder lists differ')
    verify_shared_protocol([r['config'] for r in encoders]+[mp['config']])
    multi_run = output/'gmmix_multidepth_seed42'
    multi = dict(name='gmmix_multidepth_after_encoder', run=str(multi_run), config=mp['config'],
        command=relocate_command(mp['command'], mp['run'], multi_run),
        evaluations=[relocate_command(c,mp['run'],multi_run) for c in mp['evaluations']])
    if rp['models'] != list(MODELS) or rp['additional_seeds'] != [43,44]:
        raise ValueError('Replication grid changed')
    replicas = []
    for row in rp['runs']:
        run = output/'replications'/row['name']
        config = row['video_config']
        base = next(r['config'] for r in encoders+[multi] if r['name']==row['model'])
        if config != dict(base,seed=row['seed']):
            raise ValueError('Replication changed conditions beyond seed')
        replicas.append(dict(name=row['name'], model=row['model'], seed=row['seed'], run=str(run), config=config,
            command=relocate_command(row['train_video'],row['video_run'],run),
            evaluations=[relocate_command(c,row['video_run'],run) for c in row['evaluate_video']]))
    if len(replicas)!=8 or {(r['model'],r['seed']) for r in replicas}!={(m,s) for m in MODELS for s in (43,44)}:
        raise ValueError('Incomplete replication grid')
    for row in [multi]+replicas: validate_video_row(row)
    source_names = {'temporal_priority.py','run_temporal_priority.py','test_temporal_priority.py',
        'summarize_full_video.py','summarize_video.py','paired_metrics.py','benchmark_multidepth_video.py'}
    for plan in (cp,mp,rp): source_names.update(plan['source_sha256'])
    plan = dict(legacy_plans={key:dict(path=str(path/'plan.json'),sha256=file_hash(path/'plan.json')) for key,path in locations.items()},
        execution_dependencies=dict(video=str(video),gmmix=str(gmmix)), encoders=encoders, multidepth=multi,
        replications=replicas, multidepth_integration=mp['integration'],
        multidepth_integration_sha256=mp['integration_sha256'],
        cost=dict(data_root=cp['data_root'], extraction_sha256=cp['extraction_sha256'], videos=cp['videos'],
                  repeats=cp['repeats'],warmup_passes=cp['warmup_passes'], dataset_audit=cp['dataset_audit'],
                  dataset_audit_sha256=cp['dataset_audit_sha256']),
        source_sha256={name:file_hash(ROOT/name) for name in sorted(source_names)},
        order=['five_video_summary','five_encoder_costs','multidepth_video','multidepth_costs','six_video_summary','eight_video_replications','three_seed_summary'],
        selection='Existing data, model grid, budgets, final-checkpoint rule and seeds unchanged. No accuracy-based acceptance gate.',
        scope='Temporal recognition and history interventions first, then paired runtime/memory and recognition replication. No point training is executed or awaited.',
        limits='SSv2 uses eight sparse observations from short clips. Cost on longer real sequences does not establish useful long-term recognition memory. No SOTA or IROS-readiness assertion.')
    return plan


def verify_plan(plan):
    for name,expected in plan['source_sha256'].items():
        if file_hash(ROOT/name)!=expected: raise ValueError('Priority source changed: '+name)
    for record in plan['legacy_plans'].values():
        if file_hash(record['path'])!=record['sha256']: raise ValueError('Pinned original plan changed')
    for row in plan['encoders']+[plan['multidepth']]+plan['replications']: verify_inputs(row['config'])
    for row in [plan['multidepth']]+plan['replications']: validate_video_row(row)
    cost = plan['cost']
    if file_hash(Path(cost['data_root'])/'extraction.json')!=cost['extraction_sha256'] or file_hash(cost['dataset_audit'])!=cost['dataset_audit_sha256']:
        raise ValueError('Cost dataset changed')
    if file_hash(Path(plan['multidepth_integration'])/'verification.json')!=plan['multidepth_integration_sha256']:
        raise ValueError('Baseline qualification changed')
    verify_qualification(plan['multidepth_integration'])


def verify_video(row):
    completed = video_verifier(row['config'])(row['run'],row['config'])
    values = read_run(row['run'])
    if values['config']!=row['config'] or values['checkpoint_sha256']!=completed['checkpoint_sha256'] or len(values['ids'])!=24777:
        raise ValueError('Incomplete or mismatched recognition evaluation')
    return completed, values


def summarize_videos(rows):
    measured = {}; output = []
    for row in rows:
        completed, values = verify_video(row); measured[row['name']] = values
        output.append(dict(name=row['name'],run=row['run'],checkpoint_sha256=completed['checkpoint_sha256'],
            evaluation_sha256=values['evaluation_sha256'],
            stream_accuracy=values['scores']['full','stream']['accuracy'],
            clip_stream_disagreements=values['disagreements'],
            history={} if row['config']['kind']=='none' else {
                condition:paired_correctness(values,values,('full','clip'),(condition,'clip'))
                for condition in ('current','reverse_history')}))
    reference = measured['mamba_interleaved']
    return dict(primary='Full-history streaming top-1; Mamba interleaved minus temporal attention',runs=output,
        comparisons={name:paired_correctness(reference,value) for name,value in measured.items() if name!='mamba_interleaved'},
        scope='One seed; paired source-video intervals conditional on weights. History interventions are clip-mode distribution changes, not a causal proof of memory.')


def summarize_replications(plan):
    base = {r['name']:r for r in plan['encoders']+[plan['multidepth']]}
    rows = [dict(base[m],model=m,seed=42) for m in MODELS] + plan['replications']
    values = {}
    for row in rows:
        completed, observed = verify_video(row)
        values[row['model'],row['seed']] = dict(accuracy=observed['scores']['full','stream']['accuracy'],
            checkpoint_sha256=completed['checkpoint_sha256'],evaluation_sha256=observed['evaluation_sha256'])
    comparisons = []
    for model in MODELS[1:]:
        differences = [100*(values['mamba_interleaved',s]['accuracy']-values[model,s]['accuracy']) for s in (42,43,44)]
        comparisons.append(dict(other=model,seed_order=[42,43,44],differences_pp=differences,
            mean_pp=statistics.mean(differences),sample_sd_pp=statistics.stdev(differences)))
    return dict(models={m:{str(s):values[m,s] for s in (42,43,44)} for m in MODELS},comparisons=comparisons,
        scope='All predeclared recognition seeds; fixed final checkpoint. Three seeds are a limited uncertainty estimate; no long-term-memory or robot-success conclusion.')
