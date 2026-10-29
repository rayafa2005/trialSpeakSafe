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
import kotlin.math.exp

/**
 * OnnxEngine.kt
 * =============
 * Tatvaani — Primary On-Device ONNX Runtime Inference Engine.
 *
 * Feeds the raw 2.0-second audio waveform (1, 32000) directly into
 * `tatvanet.onnx`. All spectral and delta feature extraction happens
 * inside the model graph (ConvSTFT + SincConv).
 */
class OnnxEngine(private val context: Context) {

    companion object {
        private const val TAG = "OnnxEngine"
        private const val MODEL_FILE = "tatvanet.onnx"
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
                            "Copy export/tatvanet.onnx into app/src/main/assets/."
                Log.w(TAG, loadSkipReason!!)
                return false
            }
            val e = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions()
            val s = e.createSession(bytes, opts)
            env = e
            session = s
            Log.d(TAG, "ONNX model loaded: $MODEL_FILE (inputs=${s.inputNames})")
            true
        } catch (e: Exception) {
            loadSkipReason = "Failed to load ONNX model: ${e.message}"
            Log.e(TAG, loadSkipReason!!)
            false
        }
    }

    fun runInference(pcm: ShortArray): VerdictResult? {
        val audio = featureExtractor.toModelAudio(pcm, FeatureExtractor.TARGET_SAMPLES)
        return runInferenceOnPreparedAudio(audio)
    }

    fun runInferenceOnPreparedAudio(audio: FloatArray): VerdictResult? {
        val s = session ?: run {
            Log.e(TAG, "ONNX session not initialized")
            return null
        }
        val e = env ?: run {
            Log.e(TAG, "ORT env not initialized")
            return null
        }

        val start = System.currentTimeMillis()
        val audioTensor = OnnxTensor.createTensor(
            e,
            FloatBuffer.wrap(audio),
            longArrayOf(1, audio.size.toLong())
        )

        try {
            val inputs = mapOf("audio" to audioTensor)
            s.run(inputs).use { results ->
                val logitsAny = results[0].value
                val logits = (logitsAny as Array<FloatArray>)[0] // (1, 2)

                val pFake = computeFakeProbability(logits)
                val latency = System.currentTimeMillis() - start
                return Verdict.evaluateProbability(pFake, latency, InferenceSource.ON_DEVICE)
            }
        } catch (ex: Exception) {
            Log.e(TAG, "ONNX inference failed: ${ex.message}")
            return null
        } finally {
            audioTensor.close()
        }
    }

    fun close() {
        try { session?.close() } catch (_: Exception) {}
        session = null
        try { env?.close() } catch (_: Exception) {}
        env = null
    }

    private fun computeFakeProbability(logits: FloatArray): Float {
        return when (logits.size) {
            2 -> {
                val maxLogit = maxOf(logits[0], logits[1]).toDouble()
                val exp0 = exp(logits[0].toDouble() - maxLogit)
                val exp1 = exp(logits[1].toDouble() - maxLogit)
                (exp1 / (exp0 + exp1)).toFloat()
            }
            3 -> {
                val maxLogit = (logits.maxOrNull() ?: 0f).toDouble()
                val exp0 = exp(logits[0].toDouble() - maxLogit)
                val exp1 = exp(logits[1].toDouble() - maxLogit)
                val exp2 = exp(logits[2].toDouble() - maxLogit)
                (exp2 / (exp0 + exp1 + exp2)).toFloat()
            }
            else -> {
                (1.0 / (1.0 + exp(-logits[0].toDouble()))).toFloat()
            }
        }
    }
}
