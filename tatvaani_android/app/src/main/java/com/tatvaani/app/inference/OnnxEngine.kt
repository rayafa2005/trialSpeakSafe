package com.tatvaani.app.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.tatvaani.app.features.FeatureExtractor
import com.tatvaani.app.model.InferenceSource
import com.tatvaani.app.model.Verdict
import com.tatvaani.app.model.VerdictResult
import java.nio.FloatBuffer

/**
 * OnnxEngine.kt
 * ============
 * Tatvaani — On-device ONNX Runtime inference.
 *
 * This is a fallback path when TFLite export/conversion is blocked.
 * It runs the exported ONNX model with two inputs:
 *  - audio:         float32 (1, 32000)
 *  - spec_features: float32 (1, 9, 128, T)
 *
 * Output:
 *  - logits: float32 (1, 3)
 */
class OnnxEngine(private val context: Context) {

    companion object {
        private const val TAG = "OnnxEngine"
        private const val MODEL_FILE = "tatvanet.onnx"
        private const val N_CLASSES = 3
    }

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private val featureExtractor = FeatureExtractor()

    var loadSkipReason: String? = null
        private set

    val isReady: Boolean get() = session != null

    fun initialize(): Boolean {
        loadSkipReason = null
        return try {
            val bytes = context.assets.open(MODEL_FILE).use { it.readBytes() }
            if (bytes.size < 1024 * 1024) {
                loadSkipReason =
                    "On-device ONNX model not installed. " +
                            "Copy tatvaani_ml/export/tatvanet.onnx into app/src/main/assets/."
                Log.w(TAG, loadSkipReason!!)
                return false
            }
            val e = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions()
            val s = e.createSession(bytes, opts)
            env = e
            session = s
            Log.d(TAG, "ONNX model loaded: $MODEL_FILE")
            true
        } catch (e: Exception) {
            loadSkipReason = "Failed to load ONNX model: ${e.message}"
            Log.e(TAG, loadSkipReason!!)
            false
        }
    }

    fun runInference(pcm: ShortArray): VerdictResult? {
        val s = session ?: run {
            Log.e(TAG, "ONNX session not initialized")
            return null
        }
        val e = env ?: run {
            Log.e(TAG, "ORT env not initialized")
            return null
        }

        val start = System.currentTimeMillis()

        // Input 1: peak-normalized waveform (1, 32000) — must match spec_features pipeline
        val targetSamples = FeatureExtractor.SAMPLE_RATE * 2
        val audio = featureExtractor.toModelAudio(pcm, targetSamples)

        // Input 2: spec_features from Kotlin pipeline (flattened [9*128*T])
        val featsFlat = featureExtractor.extract(pcm) ?: run {
            Log.e(TAG, "Feature extraction failed")
            return null
        }
        val tFrames = featsFlat.size / (9 * 128)
        if (tFrames <= 0) {
            Log.e(TAG, "Invalid feature shape: ${featsFlat.size}")
            return null
        }

        val audioTensor = OnnxTensor.createTensor(
            e,
            FloatBuffer.wrap(audio),
            longArrayOf(1, targetSamples.toLong())
        )
        val featTensor = OnnxTensor.createTensor(
            e,
            FloatBuffer.wrap(featsFlat),
            longArrayOf(1, 9, 128, tFrames.toLong())
        )

        try {
            val inputs = mapOf(
                "audio" to audioTensor,
                "spec_features" to featTensor
            )
            s.run(inputs).use { results ->
                val logitsAny = results[0].value
                val logits = (logitsAny as Array<FloatArray>)[0] // (1,3) -> [3]
                val probs = softmax(logits)
                val verdict = Verdict.fromProbs(probs)
                val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
                val conf = probs[maxIdx]
                Log.d(
                    TAG,
                    "probs safe=${"%.3f".format(probs[0])} caution=${"%.3f".format(probs[1])} " +
                            "danger=${"%.3f".format(probs[2])} -> $verdict (${(conf * 100).toInt()}%)"
                )
                val latency = System.currentTimeMillis() - start
                return VerdictResult(
                    verdict = verdict,
                    confidence = conf,
                    probs = probs,
                    latencyMs = latency,
                    source = InferenceSource.ON_DEVICE
                )
            }
        } catch (ex: Exception) {
            Log.e(TAG, "ONNX inference failed: ${ex.message}")
            return null
        } finally {
            audioTensor.close()
            featTensor.close()
        }
    }

    fun close() {
        try { session?.close() } catch (_: Exception) {}
        session = null
        try { env?.close() } catch (_: Exception) {}
        env = null
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.maxOrNull() ?: 0f
        var sum = 0.0
        val exps = FloatArray(logits.size)
        for (i in logits.indices) {
            val v = kotlin.math.exp((logits[i] - max).toDouble()).toFloat()
            exps[i] = v
            sum += v.toDouble()
        }
        val out = FloatArray(logits.size)
        val denom = if (sum <= 0.0) 1.0 else sum
        for (i in exps.indices) out[i] = (exps[i] / denom.toFloat())
        return out
    }
}

