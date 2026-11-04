import copy
import json
from pathlib import Path

import pytest
import torch

from gmmix_experiment import SHARED_KEYS, matched_config, verify_completion, verify_evaluation


def test_matched_gmmix_config_preserves_exact_published_core_and_build(monkeypatch):
    import gmmix_experiment as experiment
    monkeypatch.setattr(experiment, "verify_inputs", lambda config: None)
    reference = {key:"shared-" + key for key in SHARED_KEYS}
    reference.update(kind="mamba", placement="interleaved", source_sha256={"ours":"a"})
    template = dict(reference, kind="gmmix", placement="after_encoder", steps=3, frames=4,
                    source_sha256={"prior":"b"}, encoder_architecture={"layers":4},
                    selective_scan_extension_sha256="compiled-build", causal_conv1d_available=False)
    actual = matched_config(reference, template)
    assert all(actual[key] == reference[key] for key in SHARED_KEYS)
    assert actual["source_sha256"] == {"prior":"b"}
    assert actual["encoder_architecture"] == {"layers":4}
    assert actual["selective_scan_extension_sha256"] == "compiled-build"
    for key in ("model", "revision", "spatial_precision", "sampling", "readout_sha256", "reproducibility"):
        with pytest.raises(ValueError, match="differs"):
            matched_config(reference, dict(template, **{key:"different"}))


def test_gmmix_completion_rejects_partial_wrong_format_and_invented_accuracy(tmp_path):
    validation = tmp_path / "val.jsonl"
    validation.write_text('{"label":0}\n{"label":1}\n')
    expected = dict(kind="gmmix", steps=3, evaluate_every=2, val=str(validation))
    (tmp_path / "config.json").write_text(json.dumps(expected))
    history = [dict(step=s, validation=dict(samples=2, predictions=[0, 0], accuracy=.5, loss=.7)) for s in (2, 3)]
    (tmp_path / "history.json").write_text(json.dumps(history))
    (tmp_path / "progress.jsonl").write_text('{"event":"budget_complete","step":3}\n')
    saved = dict(format_version=3, model_type="gmmix_siglip_video", config=expected, step=3, history=history)
    torch.save(saved, tmp_path / "last.pt")
    assert verify_completion(tmp_path, expected)["scheduled_validations_recounted"] == 2
    for change in (dict(format_version=2), dict(step=2), dict(history=history[:1])):
        torch.save(dict(saved, **change), tmp_path / "last.pt")
        with pytest.raises(ValueError):
            verify_completion(tmp_path, expected)
    incorrect = copy.deepcopy(saved)
    incorrect["history"][-1]["validation"]["accuracy"] = .6
    (tmp_path / "history.json").write_text(json.dumps(incorrect["history"]))
    torch.save(incorrect, tmp_path / "last.pt")
    with pytest.raises(ValueError, match="differs from saved predictions"):
        verify_completion(tmp_path, expected)
    torch.save(saved, tmp_path / "last.pt")
    (tmp_path / "history.json").write_text(json.dumps(history))
    (tmp_path / "progress.jsonl").write_text('{"event":"stopped","step":3}\n')
    with pytest.raises(ValueError, match="normal budget completion"):
        verify_completion(tmp_path, expected)


def test_gmmix_evaluation_checks_real_saved_integration_and_rejects_partial_metrics(tmp_path):
    # A recorded three-video GPU integration, not a model accuracy benchmark.
    fixture = Path(__file__).parent / "results/gmmix_training_integration_rgbfix_fp32/evaluation.json"
    report = json.loads(fixture.read_text())
    completed = dict(checkpoint_sha256=report["checkpoint_sha256"])
    ablations = ["full", "current", "reverse_history"]
    assert verify_evaluation(fixture, report["config"], completed, ablations, ["clip", "stream"])["samples"] == 3
    for mutation in ("partial", "wrong_metric", "wrong_build"):
        changed = copy.deepcopy(report)
        if mutation == "partial":
            changed["ablations"]["full"]["modes"]["clip"]["clips"].pop()
        elif mutation == "wrong_metric":
            changed["ablations"]["current"]["modes"]["clip"]["accuracy"] = .123
        else:
            changed["config"]["selective_scan_extension_sha256"] = "different"
        path = tmp_path / "evaluation.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            verify_evaluation(path, report["config"], completed, ablations, ["clip", "stream"])
