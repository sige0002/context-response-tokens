"""Trace the unchanged native trainer to locate the first reproducibility drift.

Observes all input/activation/gradient/weight hashes and retains a small tensor
subset for numerical comparisons. No optimizer, precision or seed is changed.
The full training/evaluation/checkpoint loop remains train_multidepth_video.main.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

import train_multidepth_video as trainer


def digest(value):
    array = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    trace = Path(os.environ["MULTIDEPTH_TRACE_OUTPUT"])
    trace.mkdir(parents=True, exist_ok=False)
    original_recognizer = trainer.VideoRecognizer
    original_adamw = torch.optim.AdamW
    records, model_ref = [], []
    tensors = {}
    watched = ("encoder.core.gates.0.weight", "encoder.core.gates.0.bias")

    def keep(name):
        return name in watched or name.endswith((".A_log", ".D", ".dt_proj.bias"))

    def emit(record):
        records.append(record)
        with (trace / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    class ObservedRecognizer(original_recognizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            model_ref.append(self)
            self.trace_calls = 0
            self.trace_current = None
            names = [f"encoder.feature_mlps.{i}" for i in range(4)]
            names += [f"encoder.core.{group}.{i}" for i in range(4)
                      for group in ("spatial_blocks", "gates")]
            names += ["encoder.core", "readout"]
            for name, module in self.named_modules():
                if name in names:
                    module.register_forward_hook(self.capture(name))

        def capture(self, name):
            def hook(module, inputs, value):
                self.trace_current["activation_hashes"][name] = digest(value)
            return hook

        def forward(self, images):
            self.trace_calls += 1
            index = self.trace_calls
            self.trace_current = dict(event="forward", call=index, training=self.training,
                input_sha256=digest(images), cuda_rng_before=digest(torch.cuda.get_rng_state()),
                activation_hashes={})
            logits = super().forward(images)
            self.trace_current["cuda_rng_after"] = digest(torch.cuda.get_rng_state())
            tensors[f"forward/{index}/logits"] = logits.detach().cpu().clone()
            emit(self.trace_current)
            return logits

    class ObservedAdamW(original_adamw):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace_steps = 0

        def step(self, *args, **kwargs):
            self.trace_steps += 1
            index, model = self.trace_steps, model_ref[0]
            gradients = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    gradients[name] = digest(parameter.grad)
                    if keep(name):
                        tensors[f"update/{index}/gradient/{name}"] = parameter.grad.detach().cpu().clone()
            emit(dict(event="before_update", update=index, gradient_hashes=gradients))
            result = super().step(*args, **kwargs)
            hashes = {}
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    hashes[name] = digest(parameter)
                    if keep(name):
                        tensors[f"update/{index}/weight/{name}"] = parameter.detach().cpu().clone()
            emit(dict(event="after_update", update=index, parameter_hashes=hashes))
            return result

    trainer.VideoRecognizer = ObservedRecognizer
    torch.optim.AdamW = ObservedAdamW
    try:
        trainer.main()
    finally:
        torch.save(tensors, trace / "selected_tensors.pt")
        (trace / "metadata.json").write_text(json.dumps(dict(
            scope="Diagnostic hooks on the production multi-depth trainer; not a timing measurement",
            source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                           for name in ("trace_multidepth_training.py", "train_multidepth_video.py",
                                        "gmmix_multidepth.py", "gmmix_siglip.py")},
            arguments=sys.argv[1:], records=len(records), selected_tensors=len(tensors)), indent=2)+"\n")
        trainer.VideoRecognizer = original_recognizer
        torch.optim.AdamW = original_adamw


if __name__ == "__main__":
    main()
