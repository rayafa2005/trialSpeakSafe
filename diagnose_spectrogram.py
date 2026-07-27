"""
Quick diagnostic — prints actual shapes through spectrogram pipeline.
Run: py -3.11 diagnose_spectrogram.py
"""
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(4, 48000).to(device)
print(f"Input: {x.shape}")

from features.spectrogram import MultiStreamSpectrogram
spec = MultiStreamSpectrogram().to(device)

# Hook every forward to print shapes
hooks = []
def make_hook(name):
    def hook(mod, inp, out):
        in_s  = inp[0].shape if isinstance(inp, tuple) else "?"
        out_s = out.shape if hasattr(out, "shape") else type(out)
        print(f"  {name:40s}  in={in_s}  out={out_s}")
    return hook

for name, mod in spec.named_modules():
    hooks.append(mod.register_forward_hook(make_hook(name or "root")))

with torch.no_grad():
    out = spec(x)

for h in hooks:
    h.remove()

print(f"\nFinal spectrogram output: {out.shape}  (expected [4, 3, 128, T])")
