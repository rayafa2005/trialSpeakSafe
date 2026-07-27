"""
APP.PY WAS UPDATED TO FIT THE PIPELINE BETTER. THE MODEL IS BEING LOADED IN ONNX FORM
server/app.py
=============
Tatvaani — FastAPI Server Inference Endpoint

Receives base64-encoded PCM audio from the Android app,
runs TatvaNet ONNX inference, returns verdict + latency.

USAGE:
  pip install fastapi uvicorn onnxruntime librosa numpy scipy
  py server/app.py
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
import librosa
import scipy.special
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
    description="Real-time AI audio deepfake detection",
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
recent_latencies = deque(maxlen=100)

LABEL_NAMES = {0: "safe", 1: "caution", 2: "danger"}

# Match Android VerdictResult.MIN_CONFIDENCE_FOR_DECISIVE_VERDICT (0.70f)
CONFIDENCE_THRESHOLD = 0.70


def verdict_from_probs(probs: list[float]) -> tuple[str, float]:
    """Argmax label; return caution if top probability is below threshold."""
    idx = int(np.argmax(probs))
    max_prob = float(probs[idx])
    if max_prob < CONFIDENCE_THRESHOLD:
        return "caution", max_prob
    return LABEL_NAMES[idx], max_prob

# ── Feature extraction (matches Python/Jupyter pipeline) ──────────────────────

def extract_features(audio: np.ndarray, sr: int = 16000):
    """
    Build spec_features tensor (1, 9, 128, 201) from raw audio.
    Matches the librosa pipeline used in training/testing.
    """
    # Pad or trim to exactly 32000 samples (2 seconds)
    if len(audio) < 32000:
        audio = np.pad(audio, (0, 32000 - len(audio)))
    else:
        audio = audio[:32000]

    specs = []
    for n_mels in [64, 128, 128]:
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=512, hop_length=160
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = librosa.util.fix_length(mel_db, size=201, axis=1)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        if mel_db.shape[0] != 128:
            mel_db = np.resize(mel_db, (128, 201))
        specs.append(mel_db)

    # Delta and delta-delta
    for i in range(3):
        specs.append(librosa.feature.delta(specs[i]))
    for i in range(3):
        specs.append(librosa.feature.delta(specs[i], order=2))

    spec_tensor = np.stack(specs)[np.newaxis].astype(np.float32)  # (1, 9, 128, 201)
    audio_tensor = audio[np.newaxis].astype(np.float32)            # (1, 32000)

    return audio_tensor, spec_tensor

# ── Model loading ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_model():
    global session

    # Look for ONNX model in common locations
    candidates = [
        Path("tatvaani_android/app/src/main/assets/tatvanet.onnx"),
        Path("export/tatvanet.onnx"),
        Path("tatvanet.onnx"),
    ]

    model_path = next((p for p in candidates if p.exists()), None)

    if model_path is None:
        log.warning("No ONNX model found — /predict will return mock response")
        return

    try:
        session = ort.InferenceSession(str(model_path))
        log.info(f"ONNX model loaded from {model_path}")
        log.info(f"Inputs: {[i.name for i in session.get_inputs()]}")
    except Exception as e:
        log.error(f"Failed to load ONNX model: {e}")
        session = None

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
        "model_loaded": session is not None,
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

    # Mock response if model not loaded
    if session is None:
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        log.warning("Model not loaded — returning mock response")
        return PredictResponse(
            verdict    = "caution",
            confidence = 0.60,
            probs      = [0.25, 0.60, 0.15],
            latency_ms = elapsed_ms
        )

    # Run inference
    try:
        audio_tensor, spec_tensor = extract_features(audio)

        logits = session.run(None, {
            "audio":         audio_tensor,
            "spec_features": spec_tensor
        })[0][0]

        probs = scipy.special.softmax(logits).tolist()
        verdict_str, confidence = verdict_from_probs(probs)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        recent_latencies.append(elapsed_ms)

        log.info(
            f"Verdict: {verdict_str} ({confidence*100:.1f}%) "
            f"in {elapsed_ms}ms | "
            f"probs: safe={probs[0]:.3f} caution={probs[1]:.3f} danger={probs[2]:.3f}"
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
    log.info("Starting Tatvaani server on http://0.0.0.0:8000")
    log.info("Docs: http://localhost:8000/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")