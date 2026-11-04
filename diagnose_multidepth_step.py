"""Capture the first native training microbatch to locate reproducibility drift.

Run in separate processes with the same output-independent seed. Tensor copies
are diagnostics, not timing measurements or additional optimizer updates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import SiglipImageProcessor

from gmmix_multidepth import MultiDepthGatedMambaMixSiglip
from temporal_readout import VideoRecognizer
from train_video import configure_reproducibility
from video_data import RawVideoClips


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    # Mirror the trainer's import order as well as its execution settings.
    from mamba_ssm.ops.selective_scan_interface import selective_scan_cuda
    configure_reproducibility(42)
    encoder = MultiDepthGatedMambaMixSiglip.from_pretrained()
    model = VideoRecognizer(encoder, 174, "attention", 43).cuda().train()
    processor = SiglipImageProcessor.from_pretrained("google/siglip2-base-patch16-224",
        revision=encoder.checkpoint_revision, local_files_only=True)
    data = RawVideoClips("data/ssv2/video_training_fixture/train.jsonl", processor, 8, 174,
                        training=True, seed=42)
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(42)).tolist()
    pairs = [data[index] for index in order[:2]]
    images = torch.stack([pair[0] for pair in pairs]).cuda()
    labels = torch.tensor([pair[1] for pair in pairs], device="cuda")
    initial = {name:digest(value) for name,value in model.state_dict().items()}
    activations = {"input":images.cpu()}
    names = [f"encoder.feature_mlps.{i}" for i in range(4)]
    names += [f"encoder.core.{group}.{i}" for i in range(4)
              for group in ("spatial_blocks", "gates")]
    names += ["encoder.core", "readout"]
    handles = []
    def capture(name):
        def hook(module, inputs, value):
            activations[name] = value.detach().cpu()
        return hook
    for name,module in model.named_modules():
        if name in names:
            handles.append(module.register_forward_hook(capture(name)))
    before_rng = digest(torch.cuda.get_rng_state())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
    loss.backward()
    gradients = {name:parameter.grad.detach().cpu() for name,parameter in model.named_parameters()
                 if parameter.grad is not None}
    metadata = dict(loss=float(loss.detach()), order=order[:2], labels=labels.tolist(),
                    cuda_rng_before=before_rng, cuda_rng_after=digest(torch.cuda.get_rng_state()),
                    initial=initial, activation_hashes={name:digest(value) for name,value in activations.items()},
                    gradient_hashes={name:digest(value) for name,value in gradients.items()})
    torch.save(dict(activations=activations, gradients=gradients), output / "tensors.pt")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(dict(loss=metadata["loss"], activations=len(activations), gradients=len(gradients))), flush=True)


if __name__ == "__main__":
    main()
