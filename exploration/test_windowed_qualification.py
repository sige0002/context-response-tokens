import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from exploration.check_windowed_attention import ChronologicalWindowAttention, require_main_complete
from exploration.test_windowed_attention import chronological_reference
from exploration.windowed_attention import WindowedAttention
from temporal_siglip import CausalAttention


@pytest.mark.parametrize("window", [1, 8, 128])
def test_gpu_reference_and_ring_match_independent_dense_reference_on_cpu(window):
    torch.set_num_threads(1)
    torch.manual_seed(193)
    source = CausalAttention(32).double().eval()
    x = torch.randn(2, 145, 32, dtype=torch.float64)
    with torch.inference_mode():
        expected = chronological_reference(source, x, window)
        for mixer in (ChronologicalWindowAttention(source, window), WindowedAttention(source, window)):
            state, outputs = None, []
            for offset in range(x.shape[1]):
                out, state = mixer(x[:, offset:offset + 1], state, offset, True)
                outputs.append(out)
            torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=1e-12, rtol=1e-12)
            assert int(state[2]) == 145


@pytest.mark.parametrize("events,complete", [
    ([], False),
    ([{"event": "train", "step": 16000}], False),
    ([{"event": "budget_complete", "step": 16000}], False),
    ([{"event": "budget_complete", "step": 20000}], True),
])
def test_gpu_guard_requires_completed_main_budget(tmp_path, events, complete):
    (tmp_path / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    if complete:
        require_main_complete(tmp_path)
    else:
        with pytest.raises(RuntimeError, match="20000-update"):
            require_main_complete(tmp_path)


def test_gpu_guard_rejects_live_matching_process_even_with_completed_log(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "progress.jsonl").write_text(json.dumps({"event": "budget_complete", "step": 20000}) + "\n")
    script = tmp_path / "train_video.py"
    script.write_text("import time\ntime.sleep(20)\n")
    child = subprocess.Popen([sys.executable, str(script), "--output", str(run)], cwd=tmp_path)
    try:
        deadline = time.monotonic() + 5
        while str(script).encode() not in Path(f"/proc/{child.pid}/cmdline").read_bytes():
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
        with pytest.raises(RuntimeError, match="still live"):
            require_main_complete(run)
    finally:
        child.terminate()
        child.wait(timeout=5)
