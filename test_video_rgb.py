"""Catch the actual odd-width RGB conversion failure with known pixel values."""
import av
import numpy as np
import pytest

from video_data import frame_to_rgb


def frame(width, height, yuv):
    value=av.VideoFrame(width,height,'yuv420p')
    value.colorspace=5  # SMPTE170M / BT.601 matrix.
    value.color_range=1  # MPEG limited range.
    for plane,level in zip(value.planes,yuv):
        plane.update(bytes([level])*plane.buffer_size)
    return value


def bt601(yuv):
    y,u,v=np.asarray(yuv,dtype=float)-[16.,128.,128.]
    return np.clip(np.rint([1.164383*y+1.596027*v,
        1.164383*y-.391762*u-.812968*v,1.164383*y+2.017232*u]),0,255)


@pytest.mark.parametrize('size',[(1,1),(7,5),(31,17),(32,17),(426,240),(427,240),(428,240)])
@pytest.mark.parametrize('yuv',[(16,128,128),(100,128,128),(235,128,128),(100,90,200)])
def test_constant_yuv_is_correct_everywhere_including_odd_edges(size,yuv):
    source=frame(*size,yuv)
    expected=bt601(yuv)
    previous=None
    for _ in range(3):
        rgb=np.asarray(frame_to_rgb(source))
        assert rgb.shape==(size[1],size[0],3)
        assert np.abs(rgb.astype(float)-expected).max()<=2
        assert (rgb==rgb[0,0]).all()
        if previous is not None:
            np.testing.assert_array_equal(rgb,previous)
        previous=rgb.copy()


@pytest.mark.parametrize('width',[7,31,427])
def test_real_last_column_is_preserved_instead_of_cropped_or_repeated(width):
    source=frame(width,5,(16,128,128))
    plane=source.planes[0]
    luma=np.frombuffer(plane,dtype=np.uint8).reshape(plane.height,plane.line_size).copy()
    luma[:,-1]=55  # Padding must not be substituted for the visible edge.
    luma[:,width-1]=235
    plane.update(luma.tobytes())
    rgb=np.asarray(frame_to_rgb(source))
    assert rgb[:,-1].min()>=253
    assert rgb[:,:-1].max()<=2
