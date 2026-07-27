package com.tatvaani.app.inference

import android.util.Base64
import android.util.Log
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
 * Sends PCM audio to the FastAPI server and receives a verdict.
 * Runs in parallel with TFLiteEngine for latency benchmarking.
 *
 * The LatencyCard UI shows:
 *   On-Device: ~45ms  |  Server: ~280ms
 *
 * SERVER ENDPOINT: POST /predict
 *
 * Request JSON:
 *   { "audio_b64": "<base64 float32 PCM>", "sample_rate": 16000 }
 *
 * Response JSON:
 *   { "verdict": "safe|caution|danger", "confidence": 0.87,
 *     "probs": [0.87, 0.08, 0.05], "latency_ms": 23 }
 *
 * Times out after 5 seconds — on-device result always shown regardless.
 */
class ServerClient {

    companion object {
        private const val TAG = "ServerClient"

        // Emulator → host PC:  "http://10.0.2.2:8000"
        // Physical device → PC on same Wi‑Fi (NOT 10.0.2.2):
        //   1. On PC: ipconfig (Windows) or ip addr — note IPv4 e.g. 192.168.1.42
        //   2. Start API: cd tatvaani_ml/server && uvicorn app:app --host 0.0.0.0 --port 8000
        //   3. Set: ServerClient.SERVER_URL = "http://192.168.1.42:8000"
        //      (or TatvaaniViewModel.setServerUrl(...) before testing)
        // See tatvaani_android/DEMO_SPRINT.md for OnePlus demo steps.
        var SERVER_URL = "http://10.12.50.85:8000"

        private const val ENDPOINT    = "/predict"
        private const val TIMEOUT_SEC = 5L
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(TIMEOUT_SEC, TimeUnit.SECONDS)
        .readTimeout(TIMEOUT_SEC, TimeUnit.SECONDS)
        .writeTimeout(TIMEOUT_SEC, TimeUnit.SECONDS)
        .build()

    // ── Main inference call ───────────────────────────────────────────────────

    /**
     * Send PCM audio to server and return verdict.
     * Suspending — call from a coroutine scope.
     *
     * @param pcm Raw PCM ShortArray from AudioCaptureService
     * @return VerdictResult with source=SERVER, or null if request failed
     */
    suspend fun predict(pcm: ShortArray): VerdictResult? = withContext(Dispatchers.IO) {
        val startTime = System.currentTimeMillis()

        try {
            val audioB64 = encodeAudio(pcm)

            val json = JSONObject().apply {
                put("audio_b64",   audioB64)
                put("sample_rate", 16000)
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
            // Silently fail — on-device result shown regardless
            Log.w(TAG, "Server request failed: ${e.message}")
            null
        }
    }

    // ── Audio encoding ────────────────────────────────────────────────────────

    /**
     * Convert PCM ShortArray → float32 bytes → Base64 string.
     * Server receives float32 normalized to [-1, 1].
     */
    private fun encodeAudio(pcm: ShortArray): String {
        val buffer = ByteBuffer
            .allocate(pcm.size * 4)
            .order(ByteOrder.LITTLE_ENDIAN)
        for (sample in pcm) {
            buffer.putFloat(sample / 32768.0f)
        }
        return Base64.encodeToString(buffer.array(), Base64.NO_WRAP)
    }

    // ── Response parsing ──────────────────────────────────────────────────────

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

            // Same caution rule as on-device (Verdict.fromProbs), not raw server label only
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val verdict = Verdict.fromProbs(probs)
            val resolvedConfidence = probs[maxIdx]

            Log.d(
                TAG,
                "Server: api=$verdictStr resolved=$verdict (${(resolvedConfidence * 100).toInt()}%) " +
                        "probs=[${probs.joinToString { "%.2f".format(it) }}] in ${clientLatencyMs}ms"
            )

            VerdictResult(
                verdict    = verdict,
                confidence = resolvedConfidence,
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
