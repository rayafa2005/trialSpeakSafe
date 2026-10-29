package com.tatvaani.app.features

import kotlin.math.abs

/**
 * FeatureExtractor.kt
 * ===================
 * Tatvaani — Raw Audio Input Preparation
 *
 * All spectral feature extraction (ConvSTFT, Mel, STFT, CQT, and Δ/ΔΔ)
 * is embedded directly inside the TatvaNet model graph.
 *
 * This helper performs ZERO spectral filtering or external DSP.
 * It strictly converts raw 16-bit PCM microphone samples into a
 * 2.0-second (32,000-sample) peak-normalized FloatArray in [-1.0, 1.0],
 * matching the exact input format used on the server and during training.
 */
class FeatureExtractor {

    companion object {
        const val SAMPLE_RATE    = 16000
        const val TARGET_SAMPLES = 32000     // 2.0 seconds at 16kHz
    }

    /**
     * Prepares the raw `audio` input tensor (1, 32000) for both On-Device ONNX/TFLite
     * and the FastAPI Server so both pipelines receive the exact same waveform:
     *   1. Convert 16-bit PCM ShortArray to float32 in [-1.0, 1.0]
     *   2. Pad or trim to exactly TARGET_SAMPLES (32,000 samples)
     *   3. Peak-normalize so max(|x|) == 1.0 (if non-silent)
     */
    fun toModelAudio(pcm: ShortArray, targetLength: Int = TARGET_SAMPLES): FloatArray {
        val out = FloatArray(targetLength)
        val n = minOf(pcm.size, targetLength)
        var peak = 0f
        for (i in 0 until n) {
            val v = pcm[i] / 32768.0f
            out[i] = v
            val absV = abs(v)
            if (absV > peak) peak = absV
        }
        if (peak > 1e-6f) {
            for (i in 0 until n) {
                out[i] /= peak
            }
        }
        return out
    }
}