"""Real-image execution and optional single-image masked fitting demonstration."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageOps
import torch
from torch import nn
from torch.nn import functional as F

from vision_mamba3 import VisionMamba3


class ReconstructionDemo(nn.Module):
    def __init__(self,width,depth,state):
        super().__init__()
        self.encoder=VisionMamba3(width,depth,state)
        self.head=nn.Conv2d(width,3,1)

    def forward(self,x):
        features=self.encoder(x*2-1)
        prediction=F.interpolate(self.head(features),x.shape[-2:],mode='bilinear',align_corners=False)
        return prediction.sigmoid(),features


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',type=Path)
    parser.add_argument('--size',type=int,default=128)
    parser.add_argument('--width',type=int,default=128)
    parser.add_argument('--depth',type=int,default=2)
    parser.add_argument('--state',type=int,default=64)
    parser.add_argument('--train-steps',type=int,default=20)
    parser.add_argument('--output',type=Path,default=Path('results/vision'))
    a=parser.parse_args()
    if a.size<16 or a.size%16:raise ValueError('--size must be a multiple of16')
    if a.train_steps<0:raise ValueError('--train-steps must be nonnegative')
    torch.set_num_threads(4);torch.manual_seed(42)
    a.output.mkdir(parents=True,exist_ok=True)
    image=a.image
    source_url=None
    if image is None:
        source_url='https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg'
        image=Path('data/demo_dog.jpg');image.parent.mkdir(exist_ok=True)
        if not image.exists():
            with urllib.request.urlopen(source_url,timeout=60) as response:
                raw=response.read()
            image.write_bytes(raw)
    pil=ImageOps.exif_transpose(Image.open(image)).convert('RGB')
    pil=pil.resize((a.size,a.size),Image.Resampling.BILINEAR)
    target=torch.tensor(np.array(pil),device='cuda').permute(2,0,1)[None].float()/255
    generator=torch.Generator(device='cuda').manual_seed(42)
    grid=a.size//16
    chosen=torch.randperm(grid*grid,device='cuda',generator=generator)[:max(1,grid*grid//4)]
    mask=torch.zeros(grid*grid,device='cuda');mask[chosen]=1
    mask=F.interpolate(mask.reshape(1,1,grid,grid),size=(a.size,a.size),mode='nearest')
    observed=target*(1-mask)+.5*mask
    model=ReconstructionDemo(a.width,a.depth,a.state).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    history=[];start=time.perf_counter()
    # One gradient check is performed even for inference-only (--train-steps0).
    for step in range(max(1,a.train_steps)):
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prediction,features=model(observed)
            loss=((prediction.float()-target).square()*mask).sum()/(3*mask.sum())
        assert torch.isfinite(features).all() and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if step==0:
            first_loss=float(loss.detach())
            mixer_gradient=sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters() if '.mixer.' in n)**.5
            assert mixer_gradient>0
            print('Vision forward/backward PASS',list(features.shape),'Mamba gradient norm',mixer_gradient,flush=True)
        if a.train_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        history.append(float(loss.detach()))
        if (step+1)%5==0:print('step',step+1,'masked MSE',history[-1],flush=True)
    torch.cuda.synchronize();train_seconds=time.perf_counter()-start
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        prediction,features=model(observed)
        final_loss=float(((prediction.float()-target).square()*mask).sum()/(3*mask.sum()))
        # Measure the encoder on a complete RGB image, excluding reconstruction.
        encoded_input=target*2-1
        for _ in range(10):model.encoder(encoded_input)
        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(30):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();model.encoder(encoded_input);end.record();end.synchronize()
            times.append(begin.elapsed_time(end))
        incremental_peak=torch.cuda.max_memory_allocated()-base
    def picture(x):
        return Image.fromarray((x[0].detach().float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8))
    panels=[('Original',target),('Masked input',observed),('Prediction',prediction),
            ('Masked-area fill',target*(1-mask)+prediction*mask)]
    preview=Image.new('RGB',(a.size*4,a.size+24),'white');draw=ImageDraw.Draw(preview)
    for index,(title,x) in enumerate(panels):
        preview.paste(picture(x),(index*a.size,24));draw.text((index*a.size+3,4),title,fill='black')
    preview.save(a.output/'preview.png')
    report=dict(scope='Real-image GPU and learning smoke test. One image and fixed mask, '
        'no held-out evaluation, not segmentation/recognition accuracy or a pretrained encoder.',
        source_url=source_url,image_path=str(image),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        config=dict(size=a.size,width=a.width,depth=a.depth,state=a.state,scan_directions=4,shared_direction_weights=True,
                    train_steps=a.train_steps,seed=42),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        demo_head_parameters=sum(p.numel() for p in model.head.parameters()),
        feature_shape=list(features.shape),feature_dtype=str(features.dtype),feature_bytes=features.numel()*features.element_size(),
        finite_gradients=True,initial_mamba_gradient_norm=mixer_gradient,
        first_masked_mse=first_loss,final_masked_mse=final_loss,loss_history=history,
        training_seconds_including_initial_compile=train_seconds,
        encoder_batch1_warm_median_ms=float(np.median(times)),incremental_peak_bytes=incremental_peak,
        precision='FP32 weights and state parameters, BF16 autocast',
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['vision_mamba3.py','try_vision.py']})
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(dict(config=report['config'],state_dict=model.state_dict()),a.output/'demo.pt')
    print(json.dumps({k:v for k,v in report.items() if k!='loss_history'},indent=2),flush=True)


if __name__=='__main__':main()
