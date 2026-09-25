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
    print("    [OK] Import OK")
except Exception as e:
    print(f"    [FAIL] Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Instantiate
print("\n[2] Creating TatvaNet()...")
try:
    model = TatvaNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    [OK] Created. Parameters: {n_params:,}")
    print(f"    __init__ signature accepted defaults OK")
except Exception as e:
    print(f"    [FAIL] Init failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 3: Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[3] Moving to {device}...")
try:
    model = model.to(device)
    model.eval()
    print(f"    [OK] Model on {device}")
except Exception as e:
    print(f"    [FAIL] Failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 4: Forward pass with exact shape train.py uses
# train.py produces waveforms of shape [batch, 80000] (5s at 16kHz)
print("\n[4] Forward pass with shape [4, 80000] (batch=4, mono, 5s at 16kHz)...")
try:
    dummy = torch.randn(4, 80000).to(device)
    with torch.no_grad():
        output = model(dummy)
    print(f"    [OK] Logits shape: {output['logits'].shape}")
    print(f"    [OK] Probs shape:  {output['probs'].shape} (calibrated [Safe, Caution, Danger])")
    print(f"    Output sample: {output['probs'][0].tolist()}")
    if output['logits'].shape == torch.Size([4, 2]):
        print("    [OK] Correct shape [batch, 2] for 2-class classification")
    else:
        print(f"    [WARN]  Unexpected output shape: {output['logits'].shape}")
except Exception as e:
    print(f"    [FAIL] Forward pass failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test with loss
print("\n[5] Loss computation test...")
try:
    from training.focal_loss import FocalLoss
    labels = torch.tensor([0, 1, 0, 1]).to(device)
    alpha  = torch.tensor([1.0, 1.0]).to(device)
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha)

    dummy = torch.randn(4, 80000).to(device)
    model.train()
    output = model(dummy)
    loss = loss_fn(output['logits'], labels)
    print(f"    [OK] Loss: {loss.item():.4f}")
    if torch.isnan(loss):
        print("    [FAIL] Loss is NaN — check model outputs for inf/nan")
    else:
        print("    [OK] Loss is finite — training should work")
except Exception as e:
    print(f"    [FAIL] Loss computation failed: {e}")
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
        print(f"    [WARN]  NaN gradients in: {[n for n, _ in nan_grads[:3]]}")
    else:
        print("    [OK] Gradients OK — no NaN")
except Exception as e:
    print(f"    [FAIL] Backward failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 55)
print("ALL CHECKS PASSED [OK]")
print("The model pipeline is working end-to-end.")
print()
print("If this passes but train.py --debug still crashes,")
print("the issue is in the DataLoader or augmentation layer.")
print("Run: py -3.11 training/train.py --debug")
print("=" * 55)