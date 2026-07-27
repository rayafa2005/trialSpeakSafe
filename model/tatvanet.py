"""
model/tatvanet.py
=================
Tatvaani ML — TatvaNet: Dual-Path Deepfake Detection Architecture

ARCHITECTURE OVERVIEW:
═══════════════════════════════════════════════════════════════
  Raw PCM (batch, 1, samples)
       │
       ├──────────────────────────────────────────────┐
       │                                              │
  [PATH A]                                       [PATH B]
  SincConv filterbank                     MultiStreamSpectrogram
  (learnable band-pass)                   (Mel + STFT + CQT)
       │                                              │
  DepthwiseSepCNN_A                        DeltaExtractor
  (1D conv, temporal)                     (9-channel Δ/ΔΔ)
       │                                              │
  (batch, 256)                        DepthwiseSepCNN_B + SE
  embedding                           (2D conv, spatial+temporal)
       │                                    │
       └──────── CrossAttentionFusion ──────┘
                        │
                   (batch, 256)
                        │
                  GRU (128 hidden)
                        │
                  Linear(128 → 3)
                        │
                  Softmax → [Safe, Caution, Danger]
═══════════════════════════════════════════════════════════════

TARGET: ~700K parameters, <2MB as INT8 TFLite
LATENCY: <50ms on mid-range Android device

NOVELTY: The 9-channel delta-augmented dual-path fusion with
cross-attention is not published in this combination anywhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── Squeeze-Excitation Block ─────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention block.
    
    Learns to weight each feature channel by its global importance.
    Particularly useful here because different channels (Mel vs STFT vs CQT
    vs their deltas) have varying importance depending on the type of fake.
    
    Args:
        channels  : Number of input/output channels
        reduction : Bottleneck reduction ratio. Default 8.
    """

    def __init__(self, channels: int = 64, reduction: int = 8, num_classes: int = 3):
        super().__init__()
        mid = max(channels // reduction, 4)  # Minimum 4 units

        self.squeeze  = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.excite   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, H, W)
        Returns:
            (batch, channels, H, W) — channel-reweighted tensor
        """
        scale = self.squeeze(x)              # (batch, channels, 1, 1)
        scale = self.excite(scale)           # (batch, channels)
        scale = scale.view(*scale.shape, 1, 1)  # (batch, channels, 1, 1)
        return x * scale                     # Element-wise channel scaling


# ─── Depthwise Separable Conv Block ──────────────────────────────────────────

class DepthwiseSepConv2d(nn.Module):
    """
    Depthwise Separable Convolution block (2D).
    
    Replaces standard Conv2d to reduce parameters ~8x while maintaining
    representational capacity. Key to keeping model under 2MB.
    
    Structure: Depthwise Conv → BN → ReLU → Pointwise Conv → BN → ReLU
    
    Args:
        in_ch   : Input channels
        out_ch  : Output channels
        kernel  : Kernel size for depthwise conv
        stride  : Stride for depthwise conv
        padding : Padding for depthwise conv
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise: one filter per input channel
            nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                      padding=padding, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise: mix channels with 1×1 conv
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSepConv1d(nn.Module):
    """
    Depthwise Separable Convolution block (1D) — for Path A temporal CNN.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel, stride=stride,
                      padding=padding, groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ─── Path A: Sinc-Conv Temporal CNN ──────────────────────────────────────────

class PathA_SincCNN(nn.Module):
    """
    Path A: Raw waveform → SincConv filterbank → 1D depthwise-sep CNN → embedding.
    
    This path captures temporal artifacts that spectrograms may miss,
    such as phase discontinuities and codec artifacts in the raw signal.
    
    Input:  (batch, samples) — raw PCM
    Output: (batch, 256)     — embedding vector
    """

    def __init__(self, sample_rate: int = 16000):
        super().__init__()

        # Import SincConv from features module
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from features.sinc_conv import SincConv

        # SincConv filterbank: raw PCM → 128 filtered feature maps
        self.sinc = SincConv(
            n_filters=128,
            kernel_size=251,
            sample_rate=sample_rate
        )

        # 1D CNN stack: reduces time dimension, increases depth
        # Input: (batch, 128, T) — T is audio length in samples
        self.conv_stack = nn.Sequential(
            DepthwiseSepConv1d(128, 128, kernel=3, stride=2, padding=1),  # T/2
            DepthwiseSepConv1d(128, 128, kernel=3, stride=2, padding=1),  # T/4
            DepthwiseSepConv1d(128, 64,  kernel=3, stride=2, padding=1),  # T/8
            DepthwiseSepConv1d(64,  32,  kernel=3, stride=4, padding=1),  # T/32
        )

        # Global average pooling: collapses time dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Project to 256-dim embedding
        self.proj = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, samples)
        Returns:
            (batch, 256)
        """
        x = x.unsqueeze(1)          # (batch, 1, samples)
        x = self.sinc(x)             # (batch, 128, samples)
        x = self.conv_stack(x)       # (batch, 32, T_reduced)
        x = self.pool(x)             # (batch, 32, 1)
        x = x.squeeze(-1)            # (batch, 32)
        x = self.proj(x)             # (batch, 256)
        return x


# ─── Path B: 9-Channel Spectrogram CNN ───────────────────────────────────────

class PathB_SpecCNN(nn.Module):
    """
    Path B: 9-channel delta-augmented spectrogram → 2D depthwise-sep CNN
            with SE attention → embedding.
    
    The 9-channel input (Mel/STFT/CQT + their Δ/ΔΔ) is processed like
    a multi-channel image. SE blocks learn which channels (e.g., ΔΔ-CQT)
    are most discriminative for a given sample.
    
    Input:  (batch, 9, 128, T)  — 9-channel spectrogram
    Output: (batch, 256)        — embedding vector
    """

    def __init__(self):
        super().__init__()

        # Initial conv: 9 channels → 32 channels
        self.stem = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # CNN blocks with SE attention and progressive downsampling
        # Each block halves the spatial dimensions
        self.block1 = nn.Sequential(
            DepthwiseSepConv2d(32,  64,  stride=2),  # (64, 64, T/2)
            SEBlock(64)
        )
        self.block2 = nn.Sequential(
            DepthwiseSepConv2d(64,  128, stride=2),  # (128, 32, T/4)
            SEBlock(128)
        )
        self.block3 = nn.Sequential(
            DepthwiseSepConv2d(128, 128, stride=2),  # (128, 16, T/8)
            SEBlock(128)
        )
        self.block4 = nn.Sequential(
            DepthwiseSepConv2d(128, 64,  stride=2),  # (64, 8, T/16)
            SEBlock(64)
        )

        # Global average pooling: collapse spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Project to 256-dim embedding
        self.proj = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 9, 128, T)
        Returns:
            (batch, 256)
        """
        x = self.stem(x)     # (batch, 32, 128, T)
        x = self.block1(x)   # (batch, 64,  64, T/2)
        x = self.block2(x)   # (batch, 128, 32, T/4)
        x = self.block3(x)   # (batch, 128, 16, T/8)
        x = self.block4(x)   # (batch, 64,  8,  T/16)
        x = self.pool(x)     # (batch, 64,  1,  1)
        x = x.flatten(1)     # (batch, 64)
        x = self.proj(x)     # (batch, 256)
        return x


# ─── Cross-Attention Fusion ───────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Fuses Path A (sinc/raw) and Path B (spectrogram) embeddings using
    cross-attention, allowing the model to dynamically weight which path
    is more informative for each individual audio sample.
    
    For example:
      - A fake with vocoder artifacts → Path B (spectrogram) gets higher weight
      - A fake with temporal discontinuities → Path A (raw) gets higher weight
    
    Args:
        embed_dim: Embedding dimension for both paths. Default 256.
        n_heads  : Number of attention heads. Default 4.
    """

    def __init__(self, embed_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

        # Multi-head attention: Path A queries Path B (and vice versa)
        self.attn_a_to_b = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        self.attn_b_to_a = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )

        # Layer norm for each path
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)

        # Final fusion: concatenate attended embeddings and project to 256
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            emb_a: (batch, 256) — Path A embedding (sinc/raw)
            emb_b: (batch, 256) — Path B embedding (spectrogram)
        Returns:
            (batch, 256) — fused embedding
        """
        # Reshape to sequence format for MultiheadAttention: (batch, 1, 256)
        a = emb_a.unsqueeze(1)
        b = emb_b.unsqueeze(1)

        # Path A attends to Path B (A queries what B knows)
        attended_a, _ = self.attn_a_to_b(query=a, key=b, value=b)
        attended_a = self.norm_a(a + attended_a)  # Residual connection

        # Path B attends to Path A
        attended_b, _ = self.attn_b_to_a(query=b, key=a, value=a)
        attended_b = self.norm_b(b + attended_b)

        # Squeeze sequence dim and concatenate
        fused = torch.cat([
            attended_a.squeeze(1),
            attended_b.squeeze(1)
        ], dim=1)  # (batch, 512)

        return self.fusion(fused)  # (batch, 256)


# ─── Full TatvaNet Model ──────────────────────────────────────────────────────

class TatvaNet(nn.Module):
    """
    TatvaNet: Full dual-path deepfake detection model.
    
    This is the complete model combining all components:
      Path A (SincConv raw) + Path B (9-channel spectrogram)
      → CrossAttentionFusion → GRU → 3-class classifier
    
    Labels:
      0 = Safe    (bonafide human speech)
      1 = Caution (uncertain / borderline)
      2 = Danger  (synthetic / AI-generated)
    
    Args:
        sample_rate  : Audio sample rate. Must match training data. Default 16000.
        gru_hidden   : GRU hidden size. Default 128.
        n_classes    : Number of output classes. Default 3.
        dropout      : Dropout rate for regularization. Default 0.3.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        gru_hidden:  int = 128,
        n_classes:   int = 3,
        dropout:     float = 0.3
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_classes   = n_classes

        # ── Feature extractors ─────────────────────────────────────────
        # Path A: raw waveform
        self.path_a = PathA_SincCNN(sample_rate=sample_rate)

        # Path B: 9-channel spectrogram (import here to avoid circular)
        self.path_b = PathB_SpecCNN()

        # Path B feature pipeline (spectrogram + delta)
        # Imported lazily to avoid import issues during unit test
        self._feature_pipeline = None

        # ── Fusion ────────────────────────────────────────────────────
        self.fusion = CrossAttentionFusion(embed_dim=256, n_heads=4)

        # ── Temporal modeling ─────────────────────────────────────────
        # GRU captures temporal dependencies across the full audio buffer
        # Input: fused embedding (256) treated as single timestep
        # In practice, we can pass multiple overlapping windows to GRU
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0  # No dropout on single-layer GRU
        )

        # ── Classifier ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers, kaiming for conv."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def _get_feature_pipeline(self, device):
        """Lazy-load feature pipeline to avoid circular imports at module level."""
        if self._feature_pipeline is None:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from features.delta import FeaturePipeline
            self._feature_pipeline = FeaturePipeline(
                sample_rate=self.sample_rate
            ).to(device)
        return self._feature_pipeline

    def forward(
        self,
        x: torch.Tensor,
        spec_features: torch.Tensor = None
    ) -> dict:
        """
        Forward pass through TatvaNet.
        
        Args:
            x             : (batch, samples) — raw PCM waveform
            spec_features : (batch, 9, 128, T) — pre-computed spectrogram
                            features. If None, computed on the fly.
                            Pre-computing is faster when doing multiple
                            forward passes on the same audio.
        
        Returns:
            dict with keys:
              'logits'      : (batch, 3) — raw classifier output
              'probs'       : (batch, 3) — softmax probabilities
              'verdict'     : (batch,)   — predicted class index
              'confidence'  : (batch,)   — probability of predicted class
        """
        # ── Path A: raw waveform ───────────────────────────────────────
        x = x.view(x.shape[0], 1, -1)  # ensure [B, 1, N] for sinc_conv (fixes 4D input bug)
        emb_a = self.path_a(x)          # (batch, 256)

        # ── Path B: 9-channel spectrogram ──────────────────────────────
        if spec_features is None:
            pipeline = self._get_feature_pipeline(x.device)
            spec_features = pipeline(x)  # (batch, 9, 128, T)

        emb_b = self.path_b(spec_features)  # (batch, 256)

        # ── Cross-attention fusion ─────────────────────────────────────
        fused = self.fusion(emb_a, emb_b)   # (batch, 256)

        # ── GRU temporal modeling ──────────────────────────────────────
        # Treat fused embedding as single timestep sequence
        gru_out, _ = self.gru(fused.unsqueeze(1))  # (batch, 1, 128)
        gru_out = gru_out.squeeze(1)               # (batch, 128)

        # ── Classification ─────────────────────────────────────────────
        logits = self.classifier(gru_out)  # (batch, 3)
        probs  = F.softmax(logits, dim=-1) # (batch, 3)

        verdict    = probs.argmax(dim=-1)                         # (batch,)
        confidence = probs.gather(1, verdict.unsqueeze(1)).squeeze(1)  # (batch,)

        return {
            "logits":     logits,
            "probs":      probs,
            "verdict":    verdict,
            "confidence": confidence,
        }

    def predict(self, x: torch.Tensor) -> tuple:
        """
        Convenience method for inference.
        Returns (verdict_label, confidence, probs).
        verdict_label: 'safe', 'caution', or 'danger'
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)

        label_map = {0: "safe", 1: "caution", 2: "danger"}
        verdict    = out["verdict"].item()
        confidence = out["confidence"].item()
        probs      = out["probs"].squeeze().tolist()

        return label_map[verdict], confidence, probs

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Quick unit test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing TatvaNet...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    batch   = 4
    samples = 2 * 16000  # 2 seconds at 16kHz

    # Random audio input
    x = torch.randn(batch, samples).to(device)

    model = TatvaNet(sample_rate=16000).to(device)
    model.eval()

    print(f"  Total parameters: {model.count_parameters():,}")
    assert model.count_parameters() < 2_000_000, \
        f"Model too large: {model.count_parameters():,} params (target < 2M)"

    # Forward pass
    with torch.no_grad():
        out = model(x)

    print(f"  Input shape:      {x.shape}")
    print(f"  Logits shape:     {out['logits'].shape}")
    print(f"  Probs shape:      {out['probs'].shape}")
    print(f"  Verdict shape:    {out['verdict'].shape}")
    print(f"  Confidence:       {out['confidence'].tolist()}")

    # Shape checks
    assert out["logits"].shape     == (batch, 3), f"Logits wrong: {out['logits'].shape}"
    assert out["probs"].shape      == (batch, 3), f"Probs wrong:  {out['probs'].shape}"
    assert out["verdict"].shape    == (batch,),   f"Verdict wrong:{out['verdict'].shape}"
    assert out["confidence"].shape == (batch,),   f"Conf wrong:   {out['confidence'].shape}"

    # Probability sanity: must sum to 1
    prob_sums = out["probs"].sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(batch, device=device), atol=1e-5), \
        f"Probabilities don't sum to 1: {prob_sums}"

    # Verdict must be 0, 1, or 2
    assert out["verdict"].min() >= 0 and out["verdict"].max() <= 2, \
        f"Verdict out of range: {out['verdict']}"

    print("  Shape checks:     PASSED")
    print("  Probability sum:  PASSED")
    print("  Verdict range:    PASSED")

    # Test predict() convenience method
    x_single = torch.randn(1, samples).to(device)
    label, conf, probs = model.predict(x_single)
    print(f"\n  predict() test:")
    print(f"    Verdict: {label}, Confidence: {conf:.3f}")
    print(f"    Probs: safe={probs[0]:.3f}, caution={probs[1]:.3f}, danger={probs[2]:.3f}")
    assert label in ("safe", "caution", "danger"), f"Invalid label: {label}"

    print("\nTatvaNet test PASSED.")
    print(f"Parameter count: {model.count_parameters():,}")