from pathlib import Path
import json
import torch
from torch import nn
from exploration.train_transition_conditioning import checked_plan,datasets
from exploration.transition_conditioning import attach,save_model
from train_video import configure_reproducibility,parameter_groups
from video_checkpoint import load_video_checkpoint
from exploration.temporal_density import read,write
from temporal_siglip import siso_stream_step

root=Path('results/transition_conditioning_step11000_v1')
plan=checked_plan(root);out=root/'numerical_diagnostic';out.mkdir(exist_ok=False)
torch.set_num_threads(2);configure_reproducibility(701)
model,info=load_video_checkpoint(plan['checkpoint'],'cpu');train,val=datasets(info)
model.cuda().eval();sample=val.read(plan['validation'][0]['index'])[0][None].cuda()

@torch.inference_mode()
def measure(model):
    model.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        clip=model.encoder(sample); chunks=[];state=None
        for a,b in [(0,3),(3,4),(4,8)]:
            y,state=model.encoder.encode_chunk(sample[:,a:b],state);chunks.append(y)
        streamed=torch.cat(chunks,1)
        lc=model.readout(clip[:,-1]).float();ls=model.readout(streamed[:,-1]).float()
        e=(clip-streamed).float()
    return dict(max_abs=float(e.abs().max()),rms=float(e.square().mean().sqrt()),
        preset_allclose=bool(torch.allclose(clip,streamed,atol=.125,rtol=.005)),
        logit_max_abs=float((lc-ls).abs().max()),clip_prediction=int(lc.argmax()),stream_prediction=int(ls.argmax()))

results={'original_native':measure(model)}
model.cpu();attach(model,'difference',**plan['conditioning']);model.cuda().eval()
results['zero_conditioner']=measure(model)
opt=torch.optim.AdamW(parameter_groups(model,plan['weight_decay']),lr=plan['learning_rate'])
rows=read(root/'training_schedule.json')[:4]
images=torch.stack([train.read(r['index'])[0] for r in rows]).cuda()
labels=torch.tensor([r['label'] for r in rows],device='cuda')
losses=[]
for _ in range(3):
    model.train();opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=nn.functional.cross_entropy(model(images).float(),labels)
    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    losses.append(float(loss.detach()))
results['losses']=losses;results['trained_conditioner']=measure(model)
save_model(model,out/'trained_model.pt',plan['readout'])
weights={k:p.branch[-1].weight.detach().clone() for k,p in model.encoder.projections.items()}
with torch.no_grad():
    for p in model.encoder.projections.values():p.branch[-1].weight.zero_()
results['same_trained_base_conditioner_disabled']=measure(model)
with torch.no_grad():
    for k,p in model.encoder.projections.items():p.branch[-1].weight.copy_(weights[k])

captured={}
def capture(k):
    def hook(module,args):captured[k]=args[0].detach().clone()
    return hook
handles=[p.register_forward_pre_hook(capture(k)) for k,p in model.encoder.projections.items()]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(sample)
for h in handles:h.remove()
results['isolated']={}
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for k,p in model.encoder.projections.items():
        u=captured[k];m=model.encoder.encoder.adapters[k].mixer
        with p.context(): projected=p(u);full=m(u)
        previous=None;parts=[];state=None;projected_parts=[]
        for a,b in [(0,3),(3,4),(4,8)]:
            with p.context(previous):
                projected_parts.append(p(u[:,a:b]));piece,state=siso_stream_step(m,u[:,a:b],state)
                previous=p._last
            parts.append(piece)
        results['isolated'][k]=dict(projection_max_abs=float((projected-torch.cat(projected_parts,1)).abs().max()),
            native_recurrence_max_abs=float((full-torch.cat(parts,1)).abs().max()))
write(out/'summary.json',results);print(json.dumps(results,indent=2),flush=True)
