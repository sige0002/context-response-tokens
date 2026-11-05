from copy import deepcopy

import numpy as np
import pytest
from exploration.verify_transition_conditioning import check_matched_inputs, check_score


def test_wrong_labels_or_reported_score_are_rejected():
    logits=np.zeros((4,174));logits[np.arange(4),[0,1,2,3]]=1
    targets=np.array([0,2,2,3]);report=dict(correct=3,total=4,accuracy=.75)
    check_score(logits,targets,report)
    with pytest.raises(ValueError): check_score(logits,targets,dict(correct=4,total=4,accuracy=1.))
    with pytest.raises(ValueError): check_score(logits,np.array([0,1,2,3]),report)
    bad=logits.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError): check_score(bad,targets,report)
    with pytest.raises(ValueError): check_score(logits[:,:173],targets,report)


def test_equal_predictions_cannot_substitute_for_matching_evaluation_inputs():
    # A frozen/non-injective encoder can give identical predictions for two
    # different inputs. Each arm can also be internally consistent while the
    # arms saw different images; compare recorded pixels across arms explicitly.
    arm=dict(initial_model_sha256='same-weights',training_input_metadata_sha256='train-metadata',
        training_pixels_sha256='train-pixels',trainable_parameters=10)
    for mode in ('initial','final','initial_stream','final_stream'):
        arm[mode]=dict(input_metadata_sha256='eval-metadata',input_pixels_sha256='eval-pixels')
    matched={name:deepcopy(arm) for name in ('appearance','difference')}
    check_matched_inputs(matched)
    for field in ('input_metadata_sha256','input_pixels_sha256'):
        changed=deepcopy(matched)
        for mode in ('initial','final','initial_stream','final_stream'):
            changed['difference'][mode][field]='different-evaluation-input'
        with pytest.raises(ValueError,match='Evaluation inputs differ across arms'):
            check_matched_inputs(changed)
