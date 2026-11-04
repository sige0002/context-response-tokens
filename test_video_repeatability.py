import numpy as np
import pytest

from diagnose_video_repeatability import compare_logits


def test_changed_predictions_include_ties_and_margin():
    a=np.asarray([[2.,2.,1.],[1.,3.,2.],[3.,1.,0.]])
    b=np.asarray([[2.,2.1,1.],[1.,2.9,2.],[2.9,1.,0.]])
    result=compare_logits(a,b)
    assert result['changed_predictions']==1
    assert result['changed_indices']==[0]
    assert result['changed_reference_margins']==[0.]
    assert not result['bitwise_equal']
    assert compare_logits(a,a)['bitwise_equal']


def test_invalid_logits_rejected():
    with pytest.raises(ValueError):
        compare_logits(np.ones((2,3)),np.ones((3,2)))
    with pytest.raises(ValueError):
        compare_logits(np.ones((2,3)),np.full((2,3),np.nan))
