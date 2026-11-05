"""Parameter-free matching controls on the point-head diagnostic query set.

CPU only. Added after the short-budget point-head results; no new fitting or
validation-based model selection. Later point locations only enter scoring.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import SiglipVisionModel

from temporal_siglip import DEFAULT_MODEL, DEFAULT_REVISION
from exploration.appearance_routing import AppearanceMetric
from exploration.oracle_transport import digest, sample
from exploration.readout_diagnostic import select_queries
from exploration.train_correspondence import predict_xy


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--readout-run', required=True)
    parser.add_argument('--metric-run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run, metric_run, output = Path(args.readout_run), Path(args.metric_run), Path(args.output)
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    plan = json.loads((run/'plan.json').read_text())
    config = plan['config']
    cache = Path(plan['cache'])
    cache_summary = json.loads((cache/'summary.json').read_text())
    final = json.loads((metric_run/'final.json').read_text())
    if digest(metric_run/'metric.pt') != final['checkpoint_sha256']:
        raise ValueError('Metric checkpoint changed')
    metric_plan = json.loads((metric_run/'plan.json').read_text())
    if Path(metric_plan['cache']) != cache or metric_plan['config']['val'] != config['val']:
        raise ValueError('Control data mismatch')
    methods = ['depth4_cosine', 'depth12_cosine', 'depth12_postnorm_cosine',
               'depth12_postnorm_inputnorm_cosine', 'depth12_learned_metric']
    record = dict(scope='Post-result matching controls; not a temporal encoder or independent test',
        methods=methods, readout_plan_sha256=digest(run/'plan.json'),
        readout_result_sha256=digest(run/'summary.json'), metric_checkpoint_sha256=final['checkpoint_sha256'],
        sources={str(p):digest(p) for p in (Path(__file__), Path('exploration/readout_diagnostic.py'),
            Path('exploration/train_correspondence.py'), Path('exploration/oracle_transport.py'))})
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps(record, indent=2)+'\n')
    model = SiglipVisionModel.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION,
                                            local_files_only=True).eval()
    norm = model.vision_model.post_layernorm
    del model
    metric = AppearanceMetric()
    state = torch.load(metric_run/'metric.pt', map_location='cpu', weights_only=True)['models']
    metric.load_state_dict({k.removeprefix('12.'):v for k,v in state.items() if k.startswith('12.')})
    metric.eval()
    rows, prediction_files = [], {}
    for scene in config['val']:
        path = cache/(scene+'.pt')
        if digest(path) != cache_summary['scenes'][scene]['sha256']:
            raise ValueError('Cache changed')
        data = torch.load(path, map_location='cpu', weights_only=True)
        selected = select_queries(data, scene, config['evaluation_queries'], config['seed'])
        truth, visible = data['points'][selected], data['visible'][selected]
        xy = truth[:, 0]
        raw4, raw12 = data['features']['4'], data['features']['12']
        postnorm = norm(raw12)
        displacement = torch.linalg.vector_norm((truth[:, 1:]-xy[:, None])*256, dim=-1)
        arrays = dict(truth=truth.numpy(), visible=visible.numpy(), selected=selected.numpy())
        for method in methods:
            features = raw4 if method == 'depth4_cosine' else raw12 if method in ('depth12_cosine','depth12_learned_metric') else postnorm
            reference = sample(features[0], xy, 14)
            if method == 'depth12_postnorm_inputnorm_cosine':
                reference = F.layer_norm(reference, (768,))
                current = F.layer_norm(features, (768,))
            else:
                current = features
            predictions = []
            for frame in range(1, len(features)):
                logits = metric(reference, current[frame]) if method == 'depth12_learned_metric' else F.normalize(reference, dim=-1) @ F.normalize(current[frame], dim=-1).T
                predictions.append(predict_xy(logits))
            prediction = torch.stack(predictions, 1)
            arrays[method] = prediction.numpy()
            distance = torch.linalg.vector_norm((prediction-truth[:, 1:])*256, dim=-1)
            for stratum, mask in dict(all_visible=visible[:, 1:],
                    moving_gt2=visible[:, 1:] & (displacement > 2),
                    moving_gt16=visible[:, 1:] & (displacement > 16)).items():
                values = distance[mask]
                n = len(values)
                rows.append(dict(scene=scene, method=method, stratum=stratum, points=n,
                    mean_error=float(values.mean()) if n else None,
                    **{f'pck_{q}':float((values < q).float().mean()) if n else None for q in (4, 8, 16)}))
        prediction_path = output/(scene+'.npz')
        np.savez_compressed(prediction_path, **arrays)
        prediction_files[scene] = digest(prediction_path)
    summary = []
    for method in methods:
        for stratum in ('all_visible', 'moving_gt2', 'moving_gt16'):
            selected = [r for r in rows if r['method']==method and r['stratum']==stratum and r['points']]
            summary.append(dict(method=method, stratum=stratum, scenes=len(selected),
                points=sum(r['points'] for r in selected),
                **{k:sum(r[k] for r in selected)/len(selected) if selected else None
                   for k in ('mean_error', 'pck_4', 'pck_8', 'pck_16')}))
    result = dict(plan_sha256=digest(output/'plan.json'), predictions_sha256=prediction_files,
                  rows=rows, summary=summary)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
