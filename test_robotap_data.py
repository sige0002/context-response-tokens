import io
import pickle
import sys
import types

import numpy as np
import pytest

from robotap_data import NumpyUnpickler, video_info


def test_released_mediapy_array_keeps_exact_pixels_without_dependency(monkeypatch):
    module = types.ModuleType("mediapy")
    module._VideoArray = type("_VideoArray", (np.ndarray,), {"__module__": "mediapy"})
    monkeypatch.setitem(sys.modules, "mediapy", module)
    pixels = np.arange(72, dtype=np.uint8).reshape(2, 3, 4, 3)
    wrapped = pixels.view(module._VideoArray)
    wrapped.metadata = dict(fps=30)
    encoded = pickle.dumps(wrapped, protocol=4)
    monkeypatch.delitem(sys.modules, "mediapy")
    restored = NumpyUnpickler(io.BytesIO(encoded)).load()
    assert type(restored) is np.ndarray
    np.testing.assert_array_equal(restored, pixels)


def test_native_video_and_first_visible_query_coverage():
    example = dict(video=np.zeros((4, 3, 5, 3), dtype=np.uint8),
                   points=np.full((2, 4, 2), .5, dtype=np.float32),
                   occluded=np.array([[False, False, True, False], [True, True, False, False]]))
    # Pickle protocols used by old/new NumPy releases must preserve arrays.
    for protocol in (2, 4, 5):
        restored = NumpyUnpickler(io.BytesIO(pickle.dumps(example, protocol=protocol))).load()
        info = video_info(restored)
        assert info["frames"] == 4 and info["first_visible_queries"] == 2
        assert info["query_first_frame"] == 0 and info["query_last_frame"] == 2
        assert info["visible_positions_after_query"] == 3
        assert info["evaluated_positions_including_occlusions"] == 4
    example["points"] = example["points"][:, :3]
    with pytest.raises(ValueError, match="matching frame count"):
        video_info(example)


def test_no_evaluation_targets_and_nonfinite_labels_rejected():
    example = dict(video=np.zeros((2, 3, 5, 3), dtype=np.uint8),
                   points=np.zeros((1, 2, 2), dtype=np.float32),
                   occluded=np.array([[True, False]]))
    with pytest.raises(ValueError, match="after first-visible"):
        video_info(example)
    example["occluded"][:] = False
    example["points"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite xy"):
        video_info(example)
