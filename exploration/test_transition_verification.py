import numpy as np
import pytest
from exploration.verify_transition_conditioning import check_score


def test_wrong_labels_or_reported_score_are_rejected():
    logits=np.zeros((4,174));logits[np.arange(4),[0,1,2,3]]=1
    targets=np.array([0,2,2,3]);report=dict(correct=3,total=4,accuracy=.75)
    check_score(logits,targets,report)
    with pytest.raises(ValueError): check_score(logits,targets,dict(correct=4,total=4,accuracy=1.))
    with pytest.raises(ValueError): check_score(logits,np.array([0,1,2,3]),report)
    bad=logits.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError): check_score(bad,targets,report)
    with pytest.raises(ValueError): check_score(logits[:,:173],targets,report)
