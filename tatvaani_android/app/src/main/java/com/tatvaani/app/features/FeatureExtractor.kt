package com.tatvaani.app.features

import kotlin.math.*

/**
 * FeatureExtractor.kt
 * ===================
 * Tatvaani — On-Device DSP Feature Extraction
 *
 * Replicates the Python/librosa feature pipeline in Kotlin.
 * MUST produce identical output to Python for correct inference.
 *
 * CONSTANTS LOCKED TO PYTHON VALUES:
 *   SAMPLE_RATE   = 16000
 *   N_FFT         = 512    (librosa default for short clips)
 *   HOP_LENGTH    = 160
 *   N_MELS        = 128
 *   TARGET_FRAMES = 201    (1 + 32000 / 160)
 *   DELTA_WINDOW  = 2
 *
 * PIPELINE (matches Python exactly):
 *   1. Peak-normalize PCM → float [-1, 1]
 *   2. Pad/trim to 32000 samples (2s @ 16kHz)
 *   3. Compute 3 mel spectrograms (n_mels: 64, 128, 128) → resize all to [128, 201]
 *   4. Z-score normalize each: (x - mean) / std
 *   5. Compute delta + delta-delta → 9 channels [9, 128, 201]
 *   6. Flatten row-major for ONNX input
 *
 * OUTPUT: FloatArray of shape [9 * 128 * 201] flattened
 */
class FeatureExtractor {

    companion object {
        const val SAMPLE_RATE    = 16000
        const val N_FFT          = 512       // matches Python pipeline
        const val HOP_LENGTH     = 160
        const val N_MELS_1       = 64        // first mel channel
        const val N_MELS_2       = 128       // second and third mel channels
        const val TARGET_HEIGHT  = 128
        const val TARGET_FRAMES  = 201       // 1 + 32000 / 160
        const val TARGET_SAMPLES = 32000     // 2 seconds at 16kHz
        const val DELTA_WINDOW   = 2

        // Mel filterbank bounds
        const val F_MIN          = 0.0f      // librosa default fmin
        const val F_MAX          = 8000.0f   // sr/2
    }

    // Pre-computed filterbanks (lazy, computed once)
    private val melFilterbank64:  Array<FloatArray> by lazy { buildMelFilterbank(N_MELS_1) }
    private val melFilterbank128: Array<FloatArray> by lazy { buildMelFilterbank(N_MELS_2) }
    private val hannWindow:       FloatArray        by lazy { buildHannWindow() }
    private val deltaKernel:      FloatArray        by lazy { buildDeltaKernel() }

    /**
     * Prepares the raw `audio` input for the ONNX model.
     * Peak-normalizes and pads/trims to exactly TARGET_SAMPLES.
     */
    fun toModelAudio(pcm: ShortArray, targetLength: Int = TARGET_SAMPLES): FloatArray {
        val normalized = normalizePcm(pcm)
        val out = FloatArray(targetLength)
        val n = minOf(normalized.size, targetLength)
        normalized.copyInto(out, endIndex = n)
        return out
    }

    /**
     * Main entry point. Converts raw PCM ShortArray → 9-channel feature tensor.
     * Returns FloatArray shaped [9 * 128 * 201] flattened row-major.
     * Returns null if pcm is too short or empty.
     */
    fun extract(pcm: ShortArray): FloatArray? {
        if (pcm.isEmpty()) return null

        // Step 1: Normalize PCM to float [-1, 1] (peak norm)
        val audio = normalizePcm(pcm)

        // Step 2: Pad or trim to exactly 32000 samples
        val paddedAudio = FloatArray(TARGET_SAMPLES)
        val n = minOf(audio.size, TARGET_SAMPLES)
        audio.copyInto(paddedAudio, endIndex = n)

        // Step 3: Compute STFT magnitude frames [N_FFT/2+1, TARGET_FRAMES]
        val stftFrames = computeSTFT(paddedAudio)
        val T = stftFrames[0].size

        if (T < 4) return null

        // Step 4: Compute 3 mel spectrograms matching Python:
        //   mel1: n_mels=64  → resize to [128, 201]
        //   mel2: n_mels=128 → fix to [128, 201]
        //   mel3: n_mels=128 → fix to [128, 201]
        val mel1Raw = computeMel(stftFrames, melFilterbank64, N_MELS_1)   // [64, T]
        val mel2Raw = computeMel(stftFrames, melFilterbank128, N_MELS_2)  // [128, T]
        val mel3Raw = computeMel(stftFrames, melFilterbank128, N_MELS_2)  // [128, T]

        // Resize height to TARGET_HEIGHT=128 and width to TARGET_FRAMES=201
        val mel1 = resizeSpec(mel1Raw, TARGET_HEIGHT, TARGET_FRAMES)
        val mel2 = resizeSpec(mel2Raw, TARGET_HEIGHT, TARGET_FRAMES)
        val mel3 = resizeSpec(mel3Raw, TARGET_HEIGHT, TARGET_FRAMES)

        // Step 5: Z-score normalize each channel (matches Python)
        zscoreNormalize(mel1)
        zscoreNormalize(mel2)
        zscoreNormalize(mel3)

        // Step 6: Stack 3 base channels
        val channels3 = arrayOf(mel1, mel2, mel3)

        // Step 7: Compute delta and delta-delta → [9, 128, 201]
        val channels9 = computeDeltaChannels(channels3)

        // Step 8: Flatten to FloatArray for ONNX input
        return flattenChannels(channels9)
    }

    // ── Step 1: Normalize PCM ─────────────────────────────────────────────────

    private fun normalizePcm(pcm: ShortArray): FloatArray {
        val audio = FloatArray(pcm.size) { pcm[it] / 32768.0f }
        val peak  = audio.maxOfOrNull { abs(it) } ?: 1.0f
        if (peak > 1e-6f) {
            for (i in audio.indices) audio[i] /= peak
        }
        return audio
    }

    // ── Step 2: STFT ──────────────────────────────────────────────────────────

    /**
     * Compute STFT magnitude spectrogram.
     * Uses Hann window, hop=HOP_LENGTH, fft=N_FFT.
     * center=True padding (reflect) matches librosa default.
     * Returns [N_FFT/2+1, TARGET_FRAMES].
     */
    private fun computeSTFT(audio: FloatArray): Array<FloatArray> {
        val nFreqs  = N_FFT / 2 + 1
        val nFrames = TARGET_FRAMES
        val pad     = N_FFT / 2
        val result  = Array(nFreqs) { FloatArray(nFrames) }

        val window  = hannWindow
        val fftReal = FloatArray(N_FFT)
        val fftImag = FloatArray(N_FFT)

        for (t in 0 until nFrames) {
            val start = t * HOP_LENGTH - pad

            for (i in 0 until N_FFT) {
                fftReal[i] = reflectSample(audio, start + i) * window[i]
                fftImag[i] = 0f
            }

            fft(fftReal, fftImag)

            for (f in 0 until nFreqs) {
                result[f][t] = sqrt(fftReal[f] * fftReal[f] + fftImag[f] * fftImag[f])
            }
        }

        return result
    }

    /** Reflect-pad sample access (librosa center=True boundary). */
    private fun reflectSample(audio: FloatArray, index: Int): Float {
        var i = index
        val n = audio.size
        if (n == 0) return 0f
        if (i < 0) {
            i = -i - 1
            if (i >= n) return reflectSample(audio, i)
        } else if (i >= n) {
            i = 2 * n - i - 1
            if (i < 0) return reflectSample(audio, i)
        }
        return audio[i]
    }

    /**
     * Cooley-Tukey radix-2 in-place FFT.
     * N_FFT=512 is power of 2. ✓
     */
    private fun fft(real: FloatArray, imag: FloatArray) {
        val n = real.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) {
                var t = real[i]; real[i] = real[j]; real[j] = t
                t = imag[i]; imag[i] = imag[j]; imag[j] = t
            }
        }
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wRe = cos(ang).toFloat()
            val wIm = sin(ang).toFloat()
            var i = 0
            while (i < n) {
                var uRe = 1f; var uIm = 0f
                for (k in 0 until len / 2) {
                    val tRe = uRe * real[i+k+len/2] - uIm * imag[i+k+len/2]
                    val tIm = uRe * imag[i+k+len/2] + uIm * real[i+k+len/2]
                    real[i+k+len/2] = real[i+k] - tRe
                    imag[i+k+len/2] = imag[i+k] - tIm
                    real[i+k] += tRe
                    imag[i+k] += tIm
                    val newURe = uRe * wRe - uIm * wIm
                    uIm = uRe * wIm + uIm * wRe
                    uRe = newURe
                }
                i += len
            }
            len = len shl 1
        }
    }

    // ── Step 3: Mel spectrogram ───────────────────────────────────────────────

    /**
     * Compute mel spectrogram in dB from STFT magnitude.
     * Matches librosa.feature.melspectrogram + power_to_db.
     */
    private fun computeMel(
        stft: Array<FloatArray>,
        filterbank: Array<FloatArray>,
        nMels: Int
    ): Array<FloatArray> {
        val T      = stft[0].size
        val result = Array(nMels) { FloatArray(T) }

        for (m in 0 until nMels) {
            for (t in 0 until T) {
                var power = 0f
                for (f in stft.indices) {
                    val mag = stft[f][t]
                    power += filterbank[m][f] * mag * mag  // power spectrum
                }
                // power_to_db: 10 * log10(power / ref) where ref = max power
                result[m][t] = 10f * log10(maxOf(power, 1e-10f))
            }
        }

        // Subtract max (matches librosa power_to_db ref=np.max)
        var maxVal = Float.NEGATIVE_INFINITY
        for (row in result) for (v in row) if (v > maxVal) maxVal = v
        for (row in result) for (i in row.indices) row[i] = maxOf(row[i] - maxVal, -80f)

        return result
    }

    /**
     * Build mel filterbank matching librosa.filters.mel exactly.
     * Uses HTK formula, fmin=0, fmax=sr/2.
     */
    private fun buildMelFilterbank(nMels: Int): Array<FloatArray> {
        val nFreqs = N_FFT / 2 + 1
        val fb     = Array(nMels) { FloatArray(nFreqs) }

        fun hzToMel(hz: Double) = 2595.0 * log10(1.0 + hz / 700.0)
        fun melToHz(mel: Double) = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)

        val melMin   = hzToMel(F_MIN.toDouble())
        val melMax   = hzToMel(F_MAX.toDouble())
        val melPoints = DoubleArray(nMels + 2) { i ->
            melToHz(melMin + i.toDouble() * (melMax - melMin) / (nMels + 1))
        }

        // Convert mel center frequencies to FFT bin indices
        val bins = DoubleArray(nMels + 2) { i ->
            floor((N_FFT + 1) * melPoints[i] / SAMPLE_RATE)
        }

        for (m in 0 until nMels) {
            for (f in 0 until nFreqs) {
                val fBin = f.toDouble()
                fb[m][f] = when {
                    fBin < bins[m]   -> 0f
                    fBin <= bins[m+1] -> ((fBin - bins[m]) / (bins[m+1] - bins[m])).toFloat()
                    fBin <= bins[m+2] -> ((bins[m+2] - fBin) / (bins[m+2] - bins[m+1])).toFloat()
                    else             -> 0f
                }
            }
        }
        return fb
    }

    // ── Step 4: Resize spectrogram ────────────────────────────────────────────

    /**
     * Resize spectrogram to [targetHeight, targetWidth] using bilinear interpolation.
     * Matches librosa.util.fix_length for width and resize for height.
     */
    private fun resizeSpec(
        spec: Array<FloatArray>,
        targetHeight: Int,
        targetWidth: Int
    ): Array<FloatArray> {
        val srcH = spec.size
        val srcW = spec[0].size
        val result = Array(targetHeight) { FloatArray(targetWidth) }

        for (h in 0 until targetHeight) {
            val srcRow = h.toFloat() * (srcH - 1) / maxOf(targetHeight - 1, 1)
            val r0 = srcRow.toInt().coerceIn(0, srcH - 1)
            val r1 = (r0 + 1).coerceIn(0, srcH - 1)
            val rf = srcRow - r0

            for (w in 0 until targetWidth) {
                if (w < srcW) {
                    // Interpolate height only, width already matches or is padded
                    result[h][w] = spec[r0][w] * (1 - rf) + spec[r1][w] * rf
                }
                // else: zero padding (already 0f from FloatArray init)
            }
        }
        return result
    }

    // ── Step 5: Z-score normalization ─────────────────────────────────────────

    /**
     * Z-score normalize in place: (x - mean) / (std + 1e-6)
     * Matches Python: mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
     */
    private fun zscoreNormalize(spec: Array<FloatArray>) {
        var sum = 0.0
        var count = 0
        for (row in spec) for (v in row) { sum += v; count++ }
        if (count == 0) return
        val mean = (sum / count).toFloat()

        var variance = 0.0
        for (row in spec) for (v in row) {
            val diff = (v - mean).toDouble()
            variance += diff * diff
        }
        val std = sqrt(variance / count).toFloat()

        for (row in spec) for (i in row.indices) {
            row[i] = (row[i] - mean) / (std + 1e-6f)
        }
    }

    // ── Step 6: Delta computation ─────────────────────────────────────────────

    /**
     * Compute delta and delta-delta for all 3 channels → 9 channels.
     * Uses HTK regression formula matching librosa.feature.delta.
     */
    private fun computeDeltaChannels(channels: Array<Array<FloatArray>>): Array<Array<FloatArray>> {
        val delta  = channels.map { computeDelta(it) }.toTypedArray()
        val delta2 = delta.map   { computeDelta(it) }.toTypedArray()
        return channels + delta + delta2
    }

    private fun computeDelta(spec: Array<FloatArray>): Array<FloatArray> {
        val H = spec.size
        val T = spec[0].size
        val k = deltaKernel
        val w = DELTA_WINDOW
        val result = Array(H) { FloatArray(T) }

        for (h in 0 until H) {
            for (t in 0 until T) {
                var v = 0f
                for (n in -w..w) {
                    val idx = (t + n).coerceIn(0, T - 1)  // replicate padding
                    v += k[n + w] * spec[h][idx]
                }
                result[h][t] = v
            }
        }
        return result
    }

    private fun buildDeltaKernel(): FloatArray {
        val w    = DELTA_WINDOW
        val norm = 2f * (1..w).sumOf { it * it }.toFloat()
        return FloatArray(2 * w + 1) { i -> (i - w).toFloat() / norm }
    }

    // ── Step 7: Flatten ───────────────────────────────────────────────────────

    private fun flattenChannels(channels: Array<Array<FloatArray>>): FloatArray {
        val C = channels.size
        val H = channels[0].size
        val T = channels[0][0].size
        val out = FloatArray(C * H * T)
        var idx = 0
        for (c in 0 until C)
            for (h in 0 until H)
                for (t in 0 until T)
                    out[idx++] = channels[c][h][t]
        return out
    }

    // ── Window ────────────────────────────────────────────────────────────────

    private fun buildHannWindow(): FloatArray {
        return FloatArray(N_FFT) { i ->
            (0.5 * (1.0 - cos(2.0 * PI * i / (N_FFT - 1)))).toFloat()
        }
    }

    /**
     * Returns the expected tensor shape for the fixed pipeline.
     * Always [9, 128, 201] for 2s audio at 16kHz.
     */
    fun getOutputShape(): Triple<Int, Int, Int> = Triple(9, TARGET_HEIGHT, TARGET_FRAMES)
}