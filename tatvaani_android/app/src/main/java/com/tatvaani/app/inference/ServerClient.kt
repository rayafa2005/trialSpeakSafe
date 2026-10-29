package com.tatvaani.app.inference

import android.util.Base64
import android.util.Log
import com.tatvaani.app.features.FeatureExtractor
import com.tatvaani.app.model.InferenceSource
import com.tatvaani.app.model.Verdict
import com.tatvaani.app.model.VerdictResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.TimeUnit

/**
 * ServerClient.kt
 * ===============
 * Tatvaani — Server-Side Inference REST Client
 *
 * Sends the exact same 32,000-sample float32 waveform passed to OnnxEngine
 * over HTTP POST /predict for cloud latency and parity benchmarking.
 */
class ServerClient {

    companion object {
        private const val TAG = "ServerClient"

        // PC IPv4: 192.168.137.109 | Emulator: 10.0.2.2
        var SERVER_URL = "http://10.37.221.85:8000"

        private const val ENDPOINT    = "/predict"
        private const val TIMEOUT_SEC = 5L
    }

    private val featureExtractor = FeatureExtractor()

    private val client = OkHttpClient.Builder()
        .connectTimeout(TIMEOUT_SEC, TimeUnit.SECONDS)
        .readTimeout(TIMEOUT_SEC, TimeUnit.SECONDS)
        .writeTimeout(TIMEOUT_SEC, TimeUnit.SECONDS)
        .build()

    suspend fun predict(pcm: ShortArray): VerdictResult? {
        val preparedAudio = featureExtractor.toModelAudio(pcm, FeatureExtractor.TARGET_SAMPLES)
        return predictPreparedAudio(preparedAudio)
    }

    /**
     * Send the exact FloatArray(32000) prepared for on-device inference to the server
     * so both models receive bit-for-bit identical input audio.
     */
    suspend fun predictPreparedAudio(audioFloats: FloatArray): VerdictResult? = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()

        try {
            val audioB64 = encodeFloatAudio(audioFloats)

            val json = JSONObject().apply {
                put("audio_b64",   audioB64)
                put("sample_rate", FeatureExtractor.SAMPLE_RATE)
            }.toString()

            val requestBody = json.toRequestBody("application/json".toMediaType())
            val request = Request.Builder()
                .url("$SERVER_URL$ENDPOINT")
                .post(requestBody)
                .header("Content-Type", "application/json")
                .build()

            val response  = client.newCall(request).execute()
            val latencyMs = System.currentTimeMillis() - startTime

            if (!response.isSuccessful) {
                Log.w(TAG, "Server returned ${response.code}")
                return@withContext null
            }

            val body = response.body?.string() ?: return@withContext null
            parseResponse(body, latencyMs)

        } catch (e: Exception) {
            Log.w(TAG, "Server request failed: ${e.message}")
            null
        }
    }

    private fun encodeFloatAudio(audioFloats: FloatArray): String {
        val buffer = ByteBuffer
            .allocate(audioFloats.size * 4)
            .order(ByteOrder.LITTLE_ENDIAN)
        for (sample in audioFloats) {
            buffer.putFloat(sample)
        }
        return Base64.encodeToString(buffer.array(), Base64.NO_WRAP)
    }

    private fun parseResponse(json: String, clientLatencyMs: Long): VerdictResult? {
        return try {
            val obj        = JSONObject(json)
            val verdictStr = obj.getString("verdict")
            val confidence = obj.getDouble("confidence").toFloat()

            val probsJson = obj.optJSONArray("probs")
            val probs = if (probsJson != null && probsJson.length() == 3) {
                FloatArray(3) { probsJson.getDouble(it).toFloat() }
            } else {
                FloatArray(3).also { arr ->
                    arr[verdictToIndex(verdictStr)] = confidence
                }
            }

            val verdict = when (verdictStr.lowercase()) {
                "safe"    -> Verdict.SAFE
                "danger"  -> Verdict.DANGER
                else      -> Verdict.CAUTION
            }

            Log.d(
                TAG,
                "Server: api=$verdictStr resolved=$verdict (${(confidence * 100).toInt()}%) " +
                        "probs=[${probs.joinToString { "%.2f".format(it) }}] in ${clientLatencyMs}ms"
            )

            VerdictResult(
                verdict    = verdict,
                confidence = confidence,
                probs      = probs,
                latencyMs  = clientLatencyMs,
                source     = InferenceSource.SERVER
            )

        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse server response: ${e.message}")
            null
        }
    }

    private fun verdictToIndex(verdict: String): Int = when (verdict.lowercase()) {
        "safe"    -> 0
        "caution" -> 1
        "danger"  -> 2
        else      -> 1
    }
}
