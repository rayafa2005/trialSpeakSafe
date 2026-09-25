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

        //val start = System.currentTimeMillis()

        val start = System.currentTimeMillis()

        // Input: peak-normalized waveform (1, 32000) for 2.0s @ 16kHz
        val targetSamples = FeatureExtractor.TARGET_SAMPLES
        val audio = featureExtractor.toModelAudio(pcm, targetSamples)

        val audioTensor = OnnxTensor.createTensor(
            e,
            FloatBuffer.wrap(audio),
            longArrayOf(1, targetSamples.toLong())
        )

        var featTensor: OnnxTensor? = null

        try {
            val inputNames = s.inputNames
            val inputs = mutableMapOf<String, OnnxTensor>("audio" to audioTensor)

            // Backward compatibility if ONNX expects legacy spec_features input
            if (inputNames.contains("spec_features")) {
                val feats = featureExtractor.extract(pcm)
                    ?: FloatArray(9 * 128 * FeatureExtractor.TARGET_FRAMES)
                featTensor = OnnxTensor.createTensor(
                    e,
                    FloatBuffer.wrap(feats),
                    longArrayOf(1, 9, 128, FeatureExtractor.TARGET_FRAMES.toLong())
                )
                inputs["spec_features"] = featTensor
            }

            s.run(inputs).use { results ->
                val logitsAny = results[0].value
                val logits = (logitsAny as Array<FloatArray>)[0] // (1, 2) or (1, 3)
                
                val pFake = if (logits.size == 2) {
                    val exp0 = kotlin.math.exp(logits[0].toDouble())
                    val exp1 = kotlin.math.exp(logits[1].toDouble())
                    (exp1 / (exp0 + exp1)).toFloat()
                } else if (logits.size == 3) {
                    val probs = softmax(logits)
                    probs[2]
                } else {
                    (1.0 / (1.0 + kotlin.math.exp(-logits[0].toDouble()))).toFloat()
                }

                val latency = System.currentTimeMillis() - start
                return Verdict.evaluateProbability(pFake, latency, InferenceSource.ON_DEVICE)
            }
        } catch (ex: Exception) {
            Log.e(TAG, "ONNX inference failed: ${ex.message}")
            return null
        } finally {
            audioTensor.close()
            featTensor?.close()
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

