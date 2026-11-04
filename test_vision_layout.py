"""Coordinate and gradient checks for the2D interface; runs on CPU."""
import torch
from vision_mamba3 import cross_scan,cross_merge


def test_layout_and_gradient():
    # Non-square and multi-image inputs detect accidental transposes/batch mixing.
    for h,w in [(3,5),(1,7),(7,1)]:
        x=torch.arange(2*h*w*3,dtype=torch.float64).reshape(2,h,w,3).requires_grad_()
        scan=cross_scan(x)
        restored=cross_merge(scan,h,w)
        torch.testing.assert_close(restored,x)
        restored.sum().backward()
        torch.testing.assert_close(x.grad,torch.ones_like(x))
        # Read one explicit coordinate per direction, rather than roundtrip alone.
        y=scan.reshape(2,4,h*w,3)
        for b in range(2):
            for row in range(h):
                for col in range(w):
                    torch.testing.assert_close(y[b,0,row*w+col],x[b,row,col])
                    torch.testing.assert_close(y[b,1,h*w-1-(row*w+col)],x[b,row,col])
                    torch.testing.assert_close(y[b,2,col*h+row],x[b,row,col])
                    torch.testing.assert_close(y[b,3,h*w-1-(col*h+row)],x[b,row,col])
