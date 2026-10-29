"""
server/app.py
=============
Tatvaani — FastAPI Server Inference Endpoint

Receives base64-encoded float32 raw PCM audio from the Android app,
feeds the raw (1, 32000) audio directly into TatvaNet (ONNX / PyTorch)
with zero external spectrogram preprocessing, and returns verdict + latency.

USAGE:
  pip install fastapi uvicorn onnxruntime numpy
  python server/app.py
  OR
  uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

ENDPOINTS:
  POST /predict  — main inference endpoint
  GET  /health   — health check
  GET  /latency  — returns recent latency stats
"""

import time
import base64
import logging
from pathlib import Path
from collections import deque

import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("tatvaani_server")

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Tatvaani Inference Server",
    description="Real-time AI audio deepfake detection (In-Graph TatvaNet)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Global model state ────────────────────────────────────────────────────────

session = None
torch_model = None
loaded_model_source = None
recent_latencies = deque(maxlen=100)

# Match Android 2.0s baseline (32,000 samples @ 16kHz)
TARGET_SAMPLES = 32000  # 2.0 seconds at 16kHz


def verdict_from_prob(p_fake: float) -> tuple[str, float, list[float]]:
    """
    Compute calibrated continuous 3-tier verdict from deepfake probability.
      p_fake < 0.35           -> SAFE (Genuine Human)
      0.35 <= p_fake <= 0.65  -> CAUTION (Uncertain)
      p_fake > 0.65           -> DANGER (AI Generated)
    """
    if p_fake < 0.35:
        verdict = "safe"
        confidence = 1.0 - p_fake
    elif p_fake > 0.65:
        verdict = "danger"
        confidence = p_fake
    else:
        verdict = "caution"
        confidence = 1.0 - abs(p_fake - 0.5) * 2.0

    p_safe_raw = max(0.0, 1.0 - p_fake / 0.5)
    p_danger_raw = max(0.0, (p_fake - 0.5) / 0.5)
    p_caution_raw = max(0.0, 1.0 - 2.0 * abs(p_fake - 0.5))
    total = p_safe_raw + p_caution_raw + p_danger_raw + 1e-8
    probs = [p_safe_raw / total, p_caution_raw / total, p_danger_raw / total]

    return verdict, float(confidence), probs


def prepare_audio_tensor(audio: np.ndarray) -> np.ndarray:
    """
    Pad or trim audio to exactly TARGET_SAMPLES (32,000 samples = 2.0s @ 16kHz)
    and peak-normalize so max(|x|) == 1.0, identical to Android FeatureExtractor.toModelAudio.
    """
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
    elif len(audio) > TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    peak = float(np.abs(audio).max())
    if peak > 1e-6:
        audio = audio / peak
    return audio[np.newaxis].astype(np.float32)  # (1, 32000)


# ── Model loading ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_model():
    global session, torch_model, loaded_model_source

    # 1. Primary: Load ONNX Model (identical binary to Android OnnxEngine)
    onnx_candidates = [
        Path("tatvaani_android/app/src/main/assets/tatvanet.onnx"),
        Path("export/tatvanet.onnx"),
        Path("tatvanet.onnx"),
    ]
    onnx_path = next((p for p in onnx_candidates if p.exists() and p.stat().st_size > 1_000_000), None)
    if onnx_path is not None:
        try:
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            loaded_model_source = f"onnx:{onnx_path}"
            log.info(f"ONNX model loaded from {onnx_path} (inputs={[i.name for i in session.get_inputs()]})")
            return
        except Exception as e:
            log.warning(f"Failed to load ONNX model ({e}). Falling back to PyTorch checkpoint...")
            session = None

    # 2. Fallback: Load PyTorch Checkpoint (.pt) with strict=True
    pt_candidates = [
        Path("checkpoints/best_model.pt"),
        Path("checkpoints/best_tatvanet.pt"),
        Path("best_model.pt"),
    ]
    pt_path = next((p for p in pt_candidates if p.exists()), None)
    if pt_path is not None:
        try:
            import torch
            from model.tatvanet import TatvaNet
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m = TatvaNet(sample_rate=16000, n_classes=2).to(device)
            ckpt = torch.load(pt_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state", ckpt)
            m.load_state_dict(state, strict=True)
            m.eval()
            torch_model = m
            loaded_model_source = f"pytorch:{pt_path}"
            log.info(f"PyTorch model loaded (strict=True) from {pt_path} on {device}")
            return
        except Exception as e:
            log.error(f"Could not load PyTorch checkpoint: {e}")
            torch_model = None

    log.warning("No ONNX or PyTorch model found — /predict will return mock response")


# ── Request/Response schemas ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    audio_b64:   str
    sample_rate: int = 16000

class PredictResponse(BaseModel):
    verdict:    str
    confidence: float
    probs:      list[float]
    latency_ms: int

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": (session is not None) or (torch_model is not None),
        "model_source": loaded_model_source,
    }


@app.get("/latency")
async def latency_stats():
    if not recent_latencies:
        return {"message": "No requests yet"}
    lats = list(recent_latencies)
    return {
        "n_requests": len(lats),
        "mean_ms":    round(sum(lats) / len(lats), 1),
        "min_ms":     round(min(lats), 1),
        "max_ms":     round(max(lats), 1),
        "p95_ms":     round(sorted(lats)[int(len(lats) * 0.95)], 1),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start_time = time.perf_counter()

    if request.sample_rate != 16000:
        raise HTTPException(
            status_code=400,
            detail=f"Sample rate must be 16000, got {request.sample_rate}"
        )

    # Decode base64 → float32 numpy array
    try:
        audio_bytes = base64.b64decode(request.audio_b64)
        audio = np.frombuffer(audio_bytes, dtype=np.float32).copy()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio data: {e}")

    if len(audio) < 1600:
        raise HTTPException(
            status_code=400,
            detail=f"Audio too short: {len(audio)} samples (min 1600)"
        )

    log.info(f"Received {len(audio)} samples ({len(audio)/16000:.2f}s)")

    if session is None and torch_model is None:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        log.warning("Model not loaded — returning mock response")
        return PredictResponse(
            verdict    = "caution",
            confidence = 0.50,
            probs      = [0.25, 0.50, 0.25],
            latency_ms = elapsed_ms
        )

    # Run inference directly on raw audio tensor (1, 32000)
    try:
        audio_tensor = prepare_audio_tensor(audio)  # (1, 32000)

        if session is not None:
            outputs = session.run(None, {"audio": audio_tensor})
            logits = outputs[0][0]
        else:
            import torch
            with torch.no_grad():
                device = next(torch_model.parameters()).device
                t_in = torch.from_numpy(audio_tensor).to(device)
                out = torch_model(t_in)
                logits = out["logits"][0].cpu().numpy()

        # Calculate continuous probability of fake
        if len(logits) == 2:
            exp_logits = np.exp(logits - np.max(logits))
            p_fake = float(exp_logits[1] / np.sum(exp_logits))
        elif len(logits) == 3:
            exp_logits = np.exp(logits - np.max(logits))
            probs_3 = exp_logits / np.sum(exp_logits)
            p_fake = float(probs_3[2])
        else:
            p_fake = float(1.0 / (1.0 + np.exp(-logits[0])))

        verdict_str, confidence, probs = verdict_from_prob(p_fake)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        recent_latencies.append(elapsed_ms)

        log.info(
            f"Verdict: {verdict_str} ({confidence*100:.1f}%) "
            f"in {elapsed_ms}ms | "
            f"probs: safe={probs[0]:.3f} caution={probs[1]:.3f} danger={probs[2]:.3f} (p_fake={p_fake:.3f})"
        )

        return PredictResponse(
            verdict    = verdict_str,
            confidence = confidence,
            probs      = probs,
            latency_ms = elapsed_ms
        )

    except Exception as e:
        log.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")