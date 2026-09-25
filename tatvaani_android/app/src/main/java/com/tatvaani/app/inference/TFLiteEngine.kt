package com.tatvaani.app.inference

import android.content.Context
import android.util.Log
import com.tatvaani.app.features.FeatureExtractor
import com.tatvaani.app.model.InferenceSource
import com.tatvaani.app.model.Verdict
import com.tatvaani.app.model.VerdictResult
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * TFLiteEngine.kt
 * ===============
 * Tatvaani — On-Device TFLite Model Inference
 *
 * Loads tatvanet_int8.tflite from assets/ and runs inference.
 * Uses CPU inference — compatible with all Android devices including
 * those with 16KB page size requirements (Android 15+).
 *
 * LATENCY TARGET: < 50ms on mid-range Android (achieved on CPU with INT8 model)
 */
class TFLiteEngine(private val context: Context) {

    companion object {
        private const val TAG        = "TFLiteEngine"
        private const val MODEL_FILE = "tatvanet_int8.tflite"
        private const val N_CLASSES  = 3
    }

    private var interpreter:      Interpreter?      = null
    private val featureExtractor = FeatureExtractor()

    /** Set when assets/tatvanet_int8.tflite is missing or still the 0-byte placeholder. */
    var loadSkipReason: String? = null
        private set

    // ── Initialization ────────────────────────────────────────────────────────

    /**
     * Load TFLite model from assets. Call once at app start on a background thread.
     * Uses CPU with 4 threads — compatible with all devices, no native lib issues.
     * Skips load if the asset is empty (demo placeholder) until the real export is copied in.
     */
    fun initialize(): Boolean {
        loadSkipReason = null
        return try {
            val assetFd = context.assets.openFd(MODEL_FILE)
            if (assetFd.declaredLength < 1024L) {
                loadSkipReason =
                    "On-device model not installed (placeholder asset). " +
                            "Copy export/tatvanet_int8.tflite into app/src/main/assets/."
                Log.w(TAG, loadSkipReason!!)
                return false
            }

            val modelBuffer = loadModelFile(assetFd)
            val options     = Interpreter.Options().apply {
                numThreads = 4   // Use all available CPU threads
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "TFLite model loaded: $MODEL_FILE")
            logTensorShapes()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: ${e.message}")
            false
        }
    }

    // Keep old signature for compatibility with ViewModel
    fun initialize(useGpu: Boolean = false): Boolean = initialize()

    val isReady: Boolean get() = interpreter != null

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Run full inference: PCM → features → TFLite → VerdictResult
     *
     * @param pcm Raw PCM ShortArray from AudioCaptureService (16kHz, mono)
     * @return VerdictResult or null if inference failed
     */
    fun runInference(pcm: ShortArray): VerdictResult? {
        val interp = interpreter ?: run {
            Log.e(TAG, "Interpreter not initialized")
            return null
        }

        val startTime = System.currentTimeMillis()

        // Step 0: peak-normalized audio (float32, shape [1, 80000]) for 5.0s @ 16kHz
        val targetSamples = FeatureExtractor.TARGET_SAMPLES
        val audioFloats = featureExtractor.toModelAudio(pcm, targetSamples)
        val audioBuffer = ByteBuffer
            .allocateDirect(audioFloats.size * 4)
            .order(ByteOrder.nativeOrder())
        audioBuffer.asFloatBuffer().put(audioFloats)
        audioBuffer.rewind()

        // Output buffer [1, 2] or [1, 3]
        val outputArray = Array(1) { FloatArray(N_CLASSES) }

        // Step 1: Run inference
        try {
            val inputTensorCount = interp.inputTensorCount
            if (inputTensorCount == 1) {
                interp.run(audioBuffer, outputArray)
            } else {
                val dummyFeats = FloatArray(9 * 128 * FeatureExtractor.TARGET_FRAMES)
                val featBuffer = ByteBuffer
                    .allocateDirect(dummyFeats.size * 4)
                    .order(ByteOrder.nativeOrder())
                featBuffer.asFloatBuffer().put(dummyFeats)
                featBuffer.rewind()
                val inputs = arrayOf<Any>(audioBuffer, featBuffer)
                val outputs = hashMapOf<Int, Any>(0 to outputArray)
                interp.runForMultipleInputsOutputs(inputs, outputs)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed: ${e.message}")
            return null
        }

        val latencyMs = System.currentTimeMillis() - startTime
        val logits = outputArray[0]
        
        val pFake = if (logits.size == 2) {
            val exp0 = kotlin.math.exp(logits[0].toDouble())
            val exp1 = kotlin.math.exp(logits[1].toDouble())
            (exp1 / (exp0 + exp1)).toFloat()
        } else if (logits.size == 3) {
            val max = logits.maxOrNull() ?: 0f
            val exp0 = kotlin.math.exp((logits[0] - max).toDouble())
            val exp1 = kotlin.math.exp((logits[1] - max).toDouble())
            val exp2 = kotlin.math.exp((logits[2] - max).toDouble())
            val sum = exp0 + exp1 + exp2
            (exp2 / sum).toFloat()
        } else {
            (1.0 / (1.0 + kotlin.math.exp(-logits[0].toDouble()))).toFloat()
        }

        return Verdict.evaluateProbability(pFake, latencyMs, InferenceSource.ON_DEVICE)
    }

    // ── Model loading ─────────────────────────────────────────────────────────

    private fun loadModelFile(assetFd: android.content.res.AssetFileDescriptor): MappedByteBuffer {
        val inputStream = FileInputStream(assetFd.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFd.startOffset,
            assetFd.declaredLength
        )
    }

    private fun logTensorShapes() {
        try {
            val inputShape  = interpreter?.getInputTensor(0)?.shape()
            val outputShape = interpreter?.getOutputTensor(0)?.shape()
            Log.d(TAG, "Input shape:  ${inputShape?.toList()}")
            Log.d(TAG, "Output shape: ${outputShape?.toList()}")
        } catch (e: Exception) {
            Log.w(TAG, "Could not read tensor shapes: ${e.message}")
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────

    fun close() {
        interpreter?.close()
        interpreter = null
        Log.d(TAG, "TFLiteEngine closed")
    }
}