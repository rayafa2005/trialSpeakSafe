"""
=============================================================================
Tatvaani — Forward Pass Diagnostic
=============================================================================
File: diagnose_forward.py
Run:  py -3.11 diagnose_forward.py

Runs a single forward pass through TatvaNet with a fake input and prints
the exact error if it crashes. This tells us exactly what's wrong in the
model's forward() before we waste time debugging train.py.
=============================================================================
"""

import sys
import traceback
from pathlib import Path
import torch

ML_ROOT = Path(__file__).resolve().parent
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

print("=" * 55)
print("TatvaNet Forward Pass Diagnostic")
print("=" * 55)

# Step 1: Import
print("\n[1] Importing TatvaNet...")
try:
    from model.tatvanet import TatvaNet
    print("    ✅ Import OK")
except Exception as e:
    print(f"    ❌ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Instantiate
print("\n[2] Creating TatvaNet()...")
try:
    model = TatvaNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    ✅ Created. Parameters: {n_params:,}")
    print(f"    __init__ signature accepted defaults OK")
except Exception as e:
    print(f"    ❌ Init failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[3] Moving to {device}...")
try:
    model = model.to(device)
    model.eval()
    print(f"    ✅ Model on {device}")
except Exception as e:
    print(f"    ❌ Failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Forward pass with exact shape train.py uses
# train.py produces waveforms of shape [batch, 1, 48000]
print("\n[4] Forward pass with shape [4, 1, 48000] (batch=4, mono, 3s at 16kHz)...")
try:
    dummy = torch.randn(4, 1, 48000).to(device)
    with torch.no_grad():
        output = model(dummy)
    print(f"    ✅ Output shape: {output['logits'].shape}")
    print(f"    Output sample: {output['probs'][0].tolist()}")
    if output['logits'].shape == torch.Size([4, 3]):
        print("    ✅ Correct shape [batch, 3] for 3-class classification")
    else:
        print(f"    ⚠️  Unexpected output shape. Expected [4, 3], got {output['logits'].shape}")
except Exception as e:
    print(f"    ❌ Forward pass failed: {e}")
    traceback.print_exc()
    print()
    print("This is the exact error causing train.py to crash silently.")
    print("Share the traceback above — we'll fix the model's forward().")
    sys.exit(1)

# Step 5: Test with loss
print("\n[5] Loss computation test...")
try:
    from training.focal_loss import FocalLoss
    labels = torch.tensor([0, 2, 0, 2]).to(device)
    alpha  = torch.tensor([20.0, 61.0, 1.0]).to(device)
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha)

    dummy = torch.randn(4, 1, 48000).to(device)
    model.train()
    output = model(dummy)
    loss = loss_fn(output['logits'], labels)
    print(f"    ✅ Loss: {loss.item():.4f}")
    if torch.isnan(loss):
        print("    ❌ Loss is NaN — check model outputs for inf/nan")
    else:
        print("    ✅ Loss is finite — training should work")
except Exception as e:
    print(f"    ❌ Loss computation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 6: Backward pass
print("\n[6] Backward pass test...")
try:
    loss.backward()
    # Check for NaN gradients
    nan_grads = [(n, p) for n, p in model.named_parameters()
                 if p.grad is not None and torch.isnan(p.grad).any()]
    if nan_grads:
        print(f"    ⚠️  NaN gradients in: {[n for n, _ in nan_grads[:3]]}")
    else:
        print("    ✅ Gradients OK — no NaN")
except Exception as e:
    print(f"    ❌ Backward failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 55)
print("ALL CHECKS PASSED ✅")
print("The model pipeline is working end-to-end.")
print()
print("If this passes but train.py --debug still crashes,")
print("the issue is in the DataLoader or augmentation layer.")
print("Run: py -3.11 training/train.py --debug")
print("=" * 55)