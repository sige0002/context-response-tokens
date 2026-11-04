"""Localize trained streaming errors without changing experiment sources."""
import importlib.util
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import SiglipImageProcessor
from mamba_ssm.modules.mamba3 import heavy_tail_activation
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

from temporal_siglip import TemporalSiglip, DEFAULT_MODEL, DEFAULT_REVISION, siso_stream_step
from train_temporal import FrameClips


def error(a, b):
    a, b = a.float(), b.float()
    diff = a - b
    return dict(max_abs=float(diff.abs().max()), rmse=float(diff.square().mean().sqrt()),
                relative_l2=float(diff.norm() / b.norm().clamp_min(1e-12)),
                reference_rms=float(b.square().mean().sqrt()))


def kernel_inputs(m, u):
    b, t, _ = u.shape
    h, p, s, g = m.nheads, m.headdim, m.d_state, m.num_bc_heads
    z, x, B, C, dt, A, trap, angles = m.in_proj(u).split(
        [m.d_inner, m.d_inner, s*g, s*g, h, h, h, m.num_rope_angles], -1)
    dt = F.softplus(dt + m.dt_bias)
    adt = (-heavy_tail_activation(A.float())).clamp(max=-m.A_floor) * dt
    return dict(Q=m.C_norm(C.reshape(b, t, g, s)), K=m.B_norm(B.reshape(b, t, g, s)),
                V=x.reshape(b, t, h, p), ADT=adt.transpose(1, 2), DT=dt.transpose(1, 2),
                Trap=trap.transpose(1, 2), Q_bias=m.C_bias.squeeze(1), K_bias=m.B_bias.squeeze(1),
                Angles=angles.unsqueeze(2).expand(-1, -1, h, -1).float(),
                D=m.D, Z=z.reshape(b, t, h, p))


def main():
    torch.set_num_threads(4)
    model = TemporalSiglip.load_checkpoint(
        "results/ssv2_pilot_mamba_interleaved_seed42/best_encoder.pt", "cuda")
    processor = SiglipImageProcessor.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    ds = FrameClips("data/ssv2/pilot/train.jsonl", processor, 6, 174)
    video = torch.stack([ds[i][0] for i in (0, 1)]).cuda()
    spec = importlib.util.spec_from_file_location("reference", "vendor/mamba/tests/ops/triton/test_mamba3_siso.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    captured = {}
    handles = [a.register_forward_pre_hook(lambda module, args, key=k:
               captured.__setitem__(key, args[0].detach().clone())) for k, a in model.adapters.items()]
    report = dict(scope="Trained real-image numerical diagnosis, two six-frame clips", layers={})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = model(video)
        for handle in handles:
            handle.remove()
        state, outputs = None, []
        for frame in video.unbind(1):
            y, state = model.step(frame, state)
            outputs.append(y)
        report["encoder"] = error(torch.stack(outputs, 1), full)
        for k, a in model.adapters.items():
            x = captured[k]
            batch, time, patches, dim = x.shape
            whole, _ = a(x)
            via_helper, _ = a(x, streaming=True)
            cache, parts = None, []
            for t in range(time):
                y, cache = a(x[:, t:t+1], cache=cache, offset=t, streaming=True)
                parts.append(y)
            layer = dict(helper_vs_official=error(via_helper, whole),
                         isolated_step_vs_full=error(torch.cat(parts, 1), whole))
            u = a.down(a.norm(x)).permute(0, 2, 1, 3).reshape(batch*patches, time, -1).contiguous()
            # Same projected inputs: isolate recurrence arithmetic from GEMMs.
            inputs = kernel_inputs(a.mixer, u[:8])
            kernel_full = mamba3_siso_combined(**inputs, chunk_size=64)
            cache, pieces = None, []
            for t in range(time):
                sliced = {name: (v[:, :, t:t+1] if name in ("ADT", "DT", "Trap") else
                          v[:, t:t+1] if name not in ("Q_bias", "K_bias", "D") else v)
                          for name, v in inputs.items()}
                y, *cache = mamba3_siso_combined(**sliced, Input_States=cache,
                                                chunk_size=64, return_final_states=True)
                pieces.append(y)
            kernel_stream = torch.cat(pieces, 1)
            layer["official_kernel_step_vs_full"] = error(kernel_stream, kernel_full)
            # The released fast kernel quantizes these tensors to BF16 even
            # when passed FP32. Give the reference those same quantized inputs.
            quantized = {name: (v.to(torch.bfloat16).float() if name in
                         ("Q", "K", "V", "Trap", "Angles", "Z") else v.float())
                         for name, v in inputs.items()}
            with torch.autocast("cuda", enabled=False):
                reference, _ = ref.mamba3_siso_fwd_ref(**quantized, dtype=torch.float32)
                recurrent, _ = ref.mamba3_siso_step_ref(**quantized)
            layer["fp32_reference_step_vs_full"] = error(recurrent, reference)
            layer["kernel_full_vs_reference"] = error(kernel_full, reference)
            layer["kernel_step_vs_reference"] = error(kernel_stream, reference)
            report["layers"][k] = layer
            print(k, json.dumps(layer), flush=True)
    output = Path("results/ssv2_implementation_audit/streaming_numerics.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("encoder", json.dumps(report["encoder"]), flush=True)


if __name__ == "__main__":
    main()
