"""Recount recurring SSv2 errors and show the actual eight observed images.

The gallery contains local dataset material and is not a public-release asset.
Examples: smallest numeric ID repeatedly assigned the same wrong prediction
at all four checkpoints, for each of the six most frequent final confusions.
"""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import textwrap

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import SiglipImageProcessor

from video_checkpoint import file_hash
from video_data import frame_to_rgb, temporal_indices


def run(output):
    if output.exists(): raise FileExistsError('Preserve existing analysis')
    root=Path('results/ssv2_full_mamba_interleaved_seed42_rgbfix')
    config=json.loads((root/'config.json').read_text())
    manifest=Path(config['val']);rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    if file_hash(manifest)!=config['val_sha256']: raise ValueError('Validation changed')
    annotations=Path('data/ssv2/labels')
    names={int(v):k for k,v in json.loads((annotations/'labels.json').read_text()).items()}
    original={r['id']:r for r in json.loads((annotations/'validation.json').read_text())}
    for row in rows:
        template=original[row['id']]['template'].translate(str.maketrans('','','[]'))
        if template!=names[row['label']]: raise ValueError('Official label disagreement')
    history=json.loads((root/'history.json').read_text());steps=[9000,10000,11000,12000]
    predictions={r['step']:r['validation']['predictions'] for r in history if r['step'] in steps}
    if sorted(predictions)!=steps or any(len(p)!=len(rows) for p in predictions.values()):
        raise ValueError('Missing checkpoint predictions')
    target=np.array([r['label'] for r in rows]);p={s:np.array(predictions[s]) for s in steps}
    before,after=p[11000]==target,p[12000]==target
    frequent=Counter((int(y),int(z)) for y,z in zip(target,p[12000]) if y!=z)
    supports=Counter(target.tolist());persistent=np.stack([p[s]!=target for s in steps]).all(0)
    confusions=[];examples=[]
    for rank,((y,z),count) in enumerate(frequent.most_common(20),1):
        stable=[i for i,r in enumerate(rows) if r['label']==y and all(p[s][i]==z for s in steps)]
        stable.sort(key=lambda i:int(rows[i]['id']))
        confusions.append(dict(rank=rank,label=y,prediction=z,correct_name=names[y],predicted_name=names[z],
            count=count,class_support=supports[y],stable_wrong_count=len(stable)))
        if rank<=6 and stable:
            i=stable[0];row=rows[i];path=(manifest.parent/row['video']).resolve()
            if file_hash(path)!=row['sha256']: raise ValueError('Original video changed')
            examples.append(dict(index=i,id=row['id'],video=str(path),video_sha256=row['sha256'],
                correct_name=names[y],predicted_name=names[z],annotation=original[row['id']]['label'],
                predictions={str(s):int(p[s][i]) for s in steps}))
    result=dict(steps=steps,samples=len(rows),latest_correct=int(after.sum()),latest_wrong=int((~after).sum()),
        gained=int((after&~before).sum()),lost=int((before&~after).sum()),net_gain=int(after.sum()-before.sum()),
        persistently_wrong=int(persistent.sum()),confusions=confusions,examples=examples,
        sources={str(manifest):file_hash(manifest),str(annotations/'labels.json'):file_hash(annotations/'labels.json'),
            str(annotations/'validation.json'):file_hash(annotations/'validation.json'),__file__:file_hash(__file__)},
        prediction_sha256={str(s):__import__('hashlib').sha256(json.dumps(predictions[s],separators=(',',':')).encode()).hexdigest() for s in steps},
        selection='First six final confusion pairs by count; smallest ID with the identical wrong prediction at all four checkpoints.',
        scope='Original-label recount and example visualization, not a causal diagnosis or hidden-test evaluation.')
    output.mkdir(parents=True);processor=SiglipImageProcessor.from_pretrained(config['model'],revision=config['revision'],local_files_only=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    cards=[];sheets=[]
    for number,e in enumerate(examples):
        with av.open(e['video']) as container:
            container.streams.video[0].thread_count=1;frames=list(container.decode(video=0))
        indices=temporal_indices(len(frames),8);rgb=[frame_to_rgb(frames[i]) for i in indices]
        pixels=processor(images=rgb,return_tensors='np')['pixel_values']
        recovered=np.clip((pixels*np.array(processor.image_std)[None,:,None,None]+np.array(processor.image_mean)[None,:,None,None])*255,0,255).round().astype('uint8')
        e['indices']=indices;e['timestamps']=[frames[i].time for i in indices]
        e['decoded_frames']=len(frames);e['last_frame_time']=frames[-1].time
        sheet=Image.new('RGB',(1280,244),'white');draw=ImageDraw.Draw(sheet)
        lines=[f"ID {e['id']} | GT: {e['correct_name']}",f"Predicted at 9k / 10k / 11k / 12k: {e['predicted_name']}"]
        for j,line in enumerate(lines): draw.text((5,3+20*j),line,font=font,fill='black')
        frame_links=[]
        for j,array in enumerate(recovered):
            image=Image.fromarray(array.transpose(1,2,0));name=f"{e['id']}_{j}.png";image.save(output/name)
            sheet.paste(image.resize((160,160)),(160*j,53))
            draw.text((160*j+3,216),f"{frames[indices[j]].time:.2f}s",font=font,fill='black')
            frame_links.append(f'<img src="{name}" title="{frames[indices[j]].time:.3f}s">')
        sheets.append(sheet)
        video=Path(e['video']).relative_to(Path.cwd())
        cards.append(f'<section><h2>Video {e["id"]}</h2><p>GT: {html.escape(e["correct_name"])}</p>'
            f'<p>Prediction: {html.escape(e["predicted_name"])}</p><p>Original annotation: {html.escape(e["annotation"])}</p>'
            f'<video controls preload="metadata" src="../../{html.escape(str(video))}"></video>'
            f'<div class="frames">{"".join(frame_links)}</div></section>')
    for page in range(0,len(sheets),3):
        canvas=Image.new('RGB',(1280,244*len(sheets[page:page+3])),'white')
        for i,sheet in enumerate(sheets[page:page+3]): canvas.paste(sheet,(0,244*i))
        canvas.save(output/f'contact_sheet_{page//3+1}.png')
    (output/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Recurring SSv2 errors at 12k</title>'
        '<style>body{font-family:sans-serif;margin:24px;max-width:1400px}section{padding:20px 0;border-bottom:1px solid #ccc}'
        'video{width:360px;max-height:300px}.frames{display:flex;flex-wrap:wrap;gap:4px}.frames img{width:140px}</style>'
        '<h1>Recurring SSv2 errors at 12k</h1><p>Actual eight model observations, with the original local video. '
        'All examples received the same wrong class at 9k, 10k, 11k, and 12k. No causal explanation is implied.</p>'+''.join(cards))
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('samples','latest_correct','latest_wrong','gained','lost','net_gain','persistently_wrong')}))
    print('examples',[e['id'] for e in examples])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(Path(parser.parse_args().output))
