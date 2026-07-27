package com.tatvaani.app.audio

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * AudioCaptureService.kt
 * ======================
 * Tatvaani — Production Audio Capture Service
 *
 * CONSENT MODEL:
 * ──────────────
 * - Requests RECORD_AUDIO permission (microphone) once.
 * - Foreground service notification visible while recording is active.
 * - No audio stored to disk — processed in memory only.
 *
 * NOTE:
 * This service captures from the microphone pipeline (not MediaProjection).
 * That means it cannot directly capture other apps' internal playback audio.
 *
 * AUDIO SPEC — locked to Python training values:
 *   Sample rate : 16000 Hz  (spectrogram.py SAMPLE_RATE)
 *   Channels    : Mono      (CHANNEL_IN_MONO)
 *   Format      : PCM 16-bit
 *   Duration    : 2 seconds (train.py clip_duration)
 */
class AudioCaptureService : Service() {

    companion object {
        private const val TAG = "AudioCaptureService"

        const val SAMPLE_RATE      = 16000
        const val CHANNEL_CONFIG   = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT     = AudioFormat.ENCODING_PCM_16BIT
        const val CAPTURE_DURATION = 2

        private const val NOTIFICATION_ID      = 1001
        private const val NOTIFICATION_CHANNEL = "tatvaani_capture"
    }

    inner class CaptureBinder : Binder() {
        fun getService(): AudioCaptureService = this@AudioCaptureService
    }

    private val binder       = CaptureBinder()
    private var audioRecord:     AudioRecord?     = null
    private var captureJob:      Job?             = null
    private val serviceScope   = CoroutineScope(Dispatchers.IO)

    var onCaptureComplete: ((ShortArray) -> Unit)? = null
    var onCaptureError:    ((String) -> Unit)?     = null

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        Log.d(TAG, "Service created (Android ${Build.VERSION.SDK_INT})")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Must call startForeground immediately on Android 10+
        startForeground(NOTIFICATION_ID, buildNotification())
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        stopCapture()
        super.onDestroy()
    }

    // ── Capture ───────────────────────────────────────────────────────────────

    /**
     * Start capturing microphone audio (16kHz mono PCM16).
     * Tries multiple audio sources to improve OEM compatibility.
     */
    fun startCapture(durationSec: Int = CAPTURE_DURATION) {
        captureJob = serviceScope.launch {
            val sourcesToTry = listOf(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                MediaRecorder.AudioSource.MIC
            )

            var record: AudioRecord? = null
            for (source in sourcesToTry) {
                record = tryBuildMicAudioRecord(source)
                if (record != null) {
                    Log.d(TAG, "AudioRecord ready (source=$source)")
                    break
                }
            }

            if (record == null) {
                withContext(Dispatchers.Main) {
                    onCaptureError?.invoke(
                        "Audio capture unavailable.\n\n" +
                                "Please allow Microphone permission and try again."
                    )
                }
                cleanup()
                return@launch
            }

            audioRecord = record

            // Capture
            val totalSamples = SAMPLE_RATE * durationSec
            val pcmBuffer    = ShortArray(totalSamples)
            var samplesRead  = 0

            try {
                val minBuffer = AudioRecord.getMinBufferSize(
                    SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT
                )
                val chunkSize = maxOf(minBuffer / 2, 1024)

                record.startRecording()
                Log.d(TAG, "Recording ${durationSec}s at ${SAMPLE_RATE}Hz")

                while (samplesRead < totalSamples) {
                    val toRead = minOf(chunkSize, totalSamples - samplesRead)
                    val read   = record.read(pcmBuffer, samplesRead, toRead)
                    when {
                        read > 0  -> samplesRead += read
                        read < 0  -> break
                    }
                }

                record.stop()
                Log.d(TAG, "Captured $samplesRead samples")

                withContext(Dispatchers.Main) {
                    if (samplesRead >= totalSamples / 4) {
                        onCaptureComplete?.invoke(
                            if (samplesRead >= totalSamples) pcmBuffer
                            else pcmBuffer.copyOf(samplesRead)
                        )
                    } else {
                        onCaptureError?.invoke(
                            "Not enough audio captured.\n\n" +
                                    "Make sure you are speaking near the microphone, then try again."
                        )
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Capture exception: ${e.message}")
                withContext(Dispatchers.Main) {
                    onCaptureError?.invoke("Capture error — please try again")
                }
            } finally {
                record.release()
                audioRecord = null
                cleanup()
            }
        }
    }

    private fun tryBuildMicAudioRecord(audioSource: Int): AudioRecord? {
        return try {
            val minBuffer  = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
            val bufferSize = maxOf(minBuffer * 4, SAMPLE_RATE * 2 * 2)

            val rec = AudioRecord.Builder()
                .setAudioSource(audioSource)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setSampleRate(SAMPLE_RATE)
                        .setChannelMask(CHANNEL_CONFIG)
                        .setEncoding(AUDIO_FORMAT)
                        .build()
                )
                .setBufferSizeInBytes(bufferSize)
                .build()

            if (rec.state == AudioRecord.STATE_INITIALIZED) rec
            else { rec.release(); null }

        } catch (e: Exception) {
            Log.w(TAG, "Config failed: ${e.message}")
            null
        }
    }

    fun stopCapture() {
        captureJob?.cancel()
        captureJob = null
        audioRecord?.let {
            try { it.stop() } catch (_: Exception) {}
            it.release()
        }
        audioRecord = null
    }

    private fun cleanup() {
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun postError(message: String) {
        android.os.Handler(mainLooper).post {
            onCaptureError?.invoke(message)
        }
    }

    // ── Notification ──────────────────────────────────────────────────────────

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            NOTIFICATION_CHANNEL,
            "Tatvaani Audio Analysis",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Shown while Tatvaani is analyzing audio"
            setShowBadge(false)
        }
        getSystemService(NotificationManager::class.java)
            .createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification =
        NotificationCompat.Builder(this, NOTIFICATION_CHANNEL)
            .setContentTitle("Tatvaani — Analyzing audio")
            .setContentText("Recording ${CAPTURE_DURATION} seconds of device audio...")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .setSilent(true)
            .build()
}