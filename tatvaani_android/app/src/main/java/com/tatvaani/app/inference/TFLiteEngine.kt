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
import kotlin.math.exp

/**
 * TFLiteEngine.kt
 * ===============
 * Tatvaani — Secondary On-Device TFLite Model Inference
 *
 * Feeds the raw 2.0-second waveform (1, 32000) directly into `tatvanet_int8.tflite`.
 * Dynamically reads output tensor class count (2 logits).
 */
class TFLiteEngine(private val context: Context) {

    companion object {
        private const val TAG        = "TFLiteEngine"
        private const val MODEL_FILE = "tatvanet_int8.tflite"
    }

    private var interpreter:      Interpreter?      = null
    private val featureExtractor = FeatureExtractor()

    var loadSkipReason: String? = null
        private set

    fun initialize(): Boolean {
        loadSkipReason = null
        return try {
            val assetFd = context.assets.openFd(MODEL_FILE)
            if (assetFd.declaredLength < 1024L) {
                loadSkipReason =
                    "On-device TFLite model not installed (placeholder asset). " +
                            "Using ONNX primary engine."
                Log.w(TAG, loadSkipReason!!)
                return false
            }

            val modelBuffer = loadModelFile(assetFd)
            val options     = Interpreter.Options().apply {
                numThreads = 4
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "TFLite model loaded: $MODEL_FILE")
            logTensorShapes()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite model: ${e.message}")
            false
        }
    }

    fun initialize(useGpu: Boolean = false): Boolean = initialize()

    val isReady: Boolean get() = interpreter != null

    fun runInference(pcm: ShortArray): VerdictResult? {
        val audioFloats = featureExtractor.toModelAudio(pcm, FeatureExtractor.TARGET_SAMPLES)
        return runInferenceOnPreparedAudio(audioFloats)
    }

    fun runInferenceOnPreparedAudio(audioFloats: FloatArray): VerdictResult? {
        val interp = interpreter ?: run {
            Log.e(TAG, "Interpreter not initialized")
            return null
        }

        val startTime = System.currentTimeMillis()

        val audioBuffer = ByteBuffer
            .allocateDirect(audioFloats.size * 4)
            .order(ByteOrder.nativeOrder())
        audioBuffer.asFloatBuffer().put(audioFloats)
        audioBuffer.rewind()

        val outShape = try {
            interp.getOutputTensor(0).shape()
        } catch (_: Exception) {
            intArrayOf(1, 2)
        }
        val numClasses = if (outShape.size >= 2) outShape[1] else 2
        val outputArray = Array(1) { FloatArray(numClasses) }

        try {
            interp.run(audioBuffer, outputArray)
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed: ${e.message}")
            return null
        }

        val latencyMs = System.currentTimeMillis() - startTime
        val logits = outputArray[0]

        val pFake = when (logits.size) {
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

        return Verdict.evaluateProbability(pFake, latencyMs, InferenceSource.ON_DEVICE)
    }

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

    fun close() {
        interpreter?.close()
        interpreter = null
        Log.d(TAG, "TFLiteEngine closed")
    }
}