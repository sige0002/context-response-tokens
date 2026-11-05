"""Locate chunk-dependent rounding with a recorded MIMO adapter version."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch

from exploration.check_mimo_visual import stream_values
from exploration.mimo_fp64_reference import metric
from train_video import configure_reproducibility
from video_checkpoint import file_hash


def run(fixture, adapter_source, output):
    if output.exists():
        raise FileExistsError('Use a new output file')
    plan = json.loads((fixture/'plan.json').read_text())
    summary = json.loads((fixture/'summary.json').read_text())
    if file_hash(adapter_source) != plan['sources']['exploration/mimo_siglip.py']:
        raise ValueError('Adapter source differs from recorded migration')
    if (file_hash(fixture/'fixture.pt') != summary['fixture_sha256'] or
            file_hash(fixture/'mimo_model.pt') != summary['model_sha256']):
        raise ValueError('Fixture weights/input changed')
    spec = importlib.util.spec_from_file_location('mimo_numeric_snapshot', adapter_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    configure_reproducibility(42)
    model = module.load_model(fixture/'mimo_model.pt', 'cuda')
    pixels = torch.load(fixture/'fixture.pt', weights_only=True, map_location='cpu')['pixels'].cuda()
    records, active = {}, []
    def hook(layer, args, value):
        active.append((args[0].detach().float().cpu(), value.detach().float().cpu()))
    handle = model.encoder.encoder.adapters['3'].down.register_forward_hook(hook)
    try:
        for name, chunks in [('single', [1]*8), ('chunks', [3,1,4])]:
            active.clear()
            stream_values(model, pixels, chunks)
            records[name] = (torch.cat([v[0] for v in active], 1),
                             torch.cat([v[1] for v in active], 1))
    finally:
        handle.remove()
    result = dict(first_adapter_norm_input=metric(records['chunks'][0], records['single'][0]),
        first_adapter_down_output=metric(records['chunks'][1], records['single'][1]),
        fixture_sha256=file_hash(fixture/'fixture.pt'),
        adapter_source=str(adapter_source), adapter_source_sha256=file_hash(adapter_source),
        diagnostic_source_sha256=file_hash(__file__),
        stream_helper_sha256=file_hash('exploration/check_mimo_visual.py'),
        scope='Same weights/video; record first adapter normalized input and BF16 down-projection output for two stream partitions. Numerical diagnosis only.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--adapter-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.fixture, a.adapter_source, a.output)
