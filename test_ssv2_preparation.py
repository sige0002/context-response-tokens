from extract_ssv2 import JoinedParts, select_rows
from extract_ssv2 import rgb_conversion_protocol, validate_cached_frames
import pytest


def test_joined_archive_handles_boundaries_and_eof(tmp_path):
    paths = [tmp_path / str(i) for i in range(3)]
    for path, contents in zip(paths, (b"ab", b"", b"cdefg")):
        path.write_bytes(contents)
    stream = JoinedParts(paths)
    try:
        assert stream.read(0) == b""
        assert stream.read(4) == b"abcd"
        assert stream.read(10) == b"efg"
        assert stream.read(1) == b""
    finally:
        stream.close()


def test_selection_uses_annotations_not_archive_order():
    labels = {"Moving something up": "0", "Moving something down": "1"}
    rows = [dict(id=str(i), template="Moving [something] " + ("up" if i % 2 else "down"))
            for i in range(20)]
    selected = select_rows(rows, labels, 3, 42)
    assert selected == select_rows(rows[::-1], labels, 3, 42)
    assert len(selected) == 6
    assert sum(r["label"] == 0 for r in selected) == 3
    assert len(select_rows(rows, labels, 0, 42)) == 20


def test_old_rgb_cache_cannot_be_silently_reused(tmp_path):
    (tmp_path/'frame.jpg').write_bytes(b'fixture')
    conversion=rgb_conversion_protocol()
    old=dict(frames=['frame.jpg'])
    with pytest.raises(ValueError,match='RGB conversion'):
        validate_cached_frames(old,tmp_path,conversion)
    current=dict(old,rgb_conversion=conversion)
    validate_cached_frames(current,tmp_path,conversion)
    with pytest.raises(ValueError):
        validate_cached_frames(current,tmp_path,dict(conversion,version='another'))
