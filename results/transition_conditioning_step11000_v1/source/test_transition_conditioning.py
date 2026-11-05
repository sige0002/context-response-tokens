import copy
import pytest
import torch
from torch import nn

from exploration.transition_conditioning import ConditionedProjection


@pytest.mark.parametrize('mode', ['difference', 'appearance'])
def test_zero_initialization_preserves_native_projection(mode):
    torch.manual_seed(3)
    base = nn.Linear(12, 28); p = ConditionedProjection(copy.deepcopy(base), mode, 8)
    u = torch.randn(2, 7, 12)
    with p.context():
        assert torch.equal(base(u), p(u))


@pytest.mark.parametrize('mode', ['difference', 'appearance'])
def test_nonzero_branch_is_causal_and_chunk_equivalent(mode):
    torch.manual_seed(4)
    p = ConditionedProjection(nn.Linear(12, 28), mode, 8)
    nn.init.normal_(p.branch[-1].weight)
    u = torch.randn(2, 9, 12)
    with p.context(): all_frames = p(u)
    previous = None; parts = []
    for a,b in ((0,2),(2,3),(3,7),(7,9)):
        with p.context(previous):
            parts.append(p(u[:,a:b])); previous = p._last
            assert previous.untyped_storage().nbytes() == previous.numel()*previous.element_size()
    torch.testing.assert_close(torch.cat(parts,1), all_frames, rtol=1e-6, atol=1e-6)
    changed = u.clone(); changed[:,5:] += 100
    with p.context(): later_changed = p(changed)
    torch.testing.assert_close(later_changed[:,:5], all_frames[:,:5], rtol=0, atol=0)
    if mode == 'difference':
        with p.context(): repeated = p(u[:,:1].expand(-1,9,-1))
        torch.testing.assert_close(repeated, p.base(u[:,:1].expand(-1,9,-1)), rtol=0, atol=0)


def test_difference_uses_previous_input_and_receives_training_gradients():
    torch.manual_seed(5)
    p = ConditionedProjection(nn.Linear(12, 28), 'difference', 8)
    u = torch.randn(2, 3, 12, requires_grad=True)
    with p.context(): result = p(u)
    result.square().mean().backward()
    assert p.branch[-1].weight.grad.abs().sum() > 0
    assert p.branch[0].weight.grad.abs().sum() == 0  # expected at zero final projection
    with torch.no_grad(): p.branch[-1].weight.add_(p.branch[-1].weight.grad, alpha=-.01)
    p.zero_grad()
    with p.context(): p(u).square().mean().backward()
    assert p.branch[0].weight.grad.abs().sum() > 0


def test_episode_context_rejects_misuse_and_clears_after_exception():
    p = ConditionedProjection(nn.Linear(12,28))
    with pytest.raises(RuntimeError): p(torch.ones(2,1,12))
    with pytest.raises(ValueError):
        with p.context(torch.ones(1,1,12)):
            p(torch.ones(2,1,12))
    assert not p._active and p._previous is p._last is None
    with p.context():
        with pytest.raises(RuntimeError):
            with p.context(): pass


def test_pilot_sampling_does_not_reuse_training_or_prior_diagnostic_videos():
    from exploration.train_transition_conditioning import schedule, validation_subset
    rows = [dict(id=f'{c}_{i}',label=c) for c in range(174) for i in range(6)]
    planned = schedule(rows,4,16,701)
    assert len(planned)==len({r['id'] for r in planned})==64
    assert planned==schedule(rows,4,16,701)
    excluded={f'{c}_0' for c in range(174)}
    selected=validation_subset(rows,excluded,4,701)
    assert len(selected)==696 and not ({r['id'] for r in selected}&excluded)
    assert all(sum(r['label']==c for r in selected)==4 for c in range(174))
    assert selected==validation_subset(rows,excluded,4,701)
    with pytest.raises(ValueError): schedule(rows,100,16,701)
