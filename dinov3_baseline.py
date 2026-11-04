"""Official DINOv3 ViT-B/16, random initialization, unchanged backbone."""
from pathlib import Path
import subprocess
import sys
from torch import nn

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'vendor/dinov3'
COMMIT = '6876159a11b4df116f30f667f8c9888617df0751'


class DINOv3Base(nn.Module):
    def __init__(self):
        super().__init__()
        actual = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != COMMIT:
            raise RuntimeError(f'DINOv3 source must be pinned to {COMMIT}, got {actual}')
        if str(SOURCE) not in sys.path:
            sys.path.insert(0, str(SOURCE))
        from dinov3.hub.backbones import dinov3_vitb16
        # This invokes official init_weights; no pretrained weight download.
        self.backbone = dinov3_vitb16(pretrained=False)

    def forward(self, image):
        h, w = image.shape[-2:]
        if h % 16 or w % 16:
            raise ValueError('DINOv3 baseline expects image dimensions divisible by16')
        tokens = self.backbone.forward_features(image)['x_norm_patchtokens']
        return tokens.transpose(1, 2).reshape(image.shape[0], 768, h // 16, w // 16)
