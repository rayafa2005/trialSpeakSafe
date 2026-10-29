package com.tatvaani.app.ui

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.tatvaani.app.audio.AudioCaptureService
import com.tatvaani.app.features.FeatureExtractor
import com.tatvaani.app.inference.OnnxEngine
import com.tatvaani.app.inference.ServerClient
import com.tatvaani.app.inference.TFLiteEngine
import com.tatvaani.app.model.VerdictResult
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * TatvaaniViewModel.kt
 * ====================
 * Tatvaani — Main ViewModel
 *
 * Dual-Pipeline Execution:
 *  1. Captures 2.0s (32,000 samples) of raw 16kHz mono PCM via AudioCaptureService.
 *  2. Converts PCM once into a single peak-normalized FloatArray(32000) with zero filtering.
 *  3. Dispatches the exact same FloatArray(32000) in parallel to:
 *     - Primary On-Device Engine (OnnxEngine -> TFLiteEngine fallback)
 *     - Secondary Cloud Benchmark Server (ServerClient)
 */
class TatvaaniViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "TatvaaniViewModel"
    }

    sealed class UiState {
        object Idle       : UiState()
        object Requesting : UiState()
        object Recording  : UiState()
        object Processing : UiState()
        data class Result(
            val onDevice: VerdictResult?,
            val server:   VerdictResult?
        ) : UiState()
        data class Error(val message: String) : UiState()
    }

    private val _uiState = MutableStateFlow<UiState>(UiState.Idle)
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    // ── Engines (ONNX is primary on-device engine) ────────────────────────────

    private val onnxEngine       = OnnxEngine(application)
    private val tfliteEngine     = TFLiteEngine(application)
    private val serverClient     = ServerClient()
    private val featureExtractor = FeatureExtractor()

    // ── Service ───────────────────────────────────────────────────────────────

    private var captureService: AudioCaptureService? = null
    private var serviceBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val b = binder as? AudioCaptureService.CaptureBinder ?: return
            captureService = b.getService()
            serviceBound   = true
            Log.d(TAG, "Service connected — starting capture")

            captureService?.onCaptureComplete = { pcm -> onAudioCaptured(pcm) }
            captureService?.onCaptureError    = { msg -> onCaptureError(msg) }

            captureService?.startCapture()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            captureService = null
            serviceBound   = false
        }
    }

    // ── Init ──────────────────────────────────────────────────────────────────

    init {
        viewModelScope.launch {
            val onnxOk = onnxEngine.initialize()
            Log.d(TAG, "ONNX primary ready: ${onnxEngine.isReady}")
            if (!onnxOk) {
                val tfliteOk = tfliteEngine.initialize()
                Log.d(TAG, "TFLite fallback ready: ${tfliteEngine.isReady}")
                if (!tfliteOk) {
                    Log.w(TAG, "On-device inference unavailable (ONNX+TFLite failed)")
                }
            }
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    fun startCapture() {
        _uiState.value = UiState.Recording
        val context = getApplication<Application>()

        val serviceIntent = android.content.Intent(context, AudioCaptureService::class.java)
        context.startForegroundService(serviceIntent)
        context.bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    fun reset() {
        _uiState.value = UiState.Idle
        unbindService()
    }

    fun setRequesting() {
        _uiState.value = UiState.Requesting
    }

    fun onPermissionDenied(missing: String) {
        _uiState.value = UiState.Error(
            "Required permission not granted:\n$missing\n\n" +
                    "Open Settings → Apps → Tatvaani → Permissions and allow access, then try again."
        )
    }

    fun setServerUrl(url: String) {
        ServerClient.SERVER_URL = url
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    private fun onAudioCaptured(pcm: ShortArray) {
        Log.d(TAG, "Audio captured: ${pcm.size} samples (${pcm.size / 16000.0}s)")
        _uiState.value = UiState.Processing

        viewModelScope.launch {
            // Prepare raw 32,000-sample float32 waveform ONCE so both On-Device and Server
            // receive bit-for-bit identical audio without any external filtering.
            val preparedAudio = featureExtractor.toModelAudio(pcm, FeatureExtractor.TARGET_SAMPLES)

            val onDeviceDeferred = async {
                when {
                    onnxEngine.isReady   -> onnxEngine.runInferenceOnPreparedAudio(preparedAudio)
                    tfliteEngine.isReady -> tfliteEngine.runInferenceOnPreparedAudio(preparedAudio)
                    else                 -> null
                }
            }
            val serverDeferred = async { serverClient.predictPreparedAudio(preparedAudio) }

            val onDeviceResult = onDeviceDeferred.await()
            val serverResult   = serverDeferred.await()

            _uiState.value = when {
                onDeviceResult != null || serverResult != null ->
                    UiState.Result(onDevice = onDeviceResult, server = serverResult)
                !onnxEngine.isReady && !tfliteEngine.isReady ->
                    UiState.Error(
                        "Capture succeeded, but on-device model is not loaded.\n\n" +
                                "Copy export/tatvanet.onnx into app/src/main/assets/.\n\n" +
                                "Server benchmark also failed — check SERVER_URL and that the API is running."
                    )
                else ->
                    UiState.Error("Analysis failed — please try again")
            }
        }
    }

    private fun onCaptureError(message: String) {
        Log.e(TAG, "Capture error: $message")
        _uiState.value = UiState.Error(message)
    }

    private fun unbindService() {
        if (serviceBound) {
            try {
                getApplication<Application>().unbindService(serviceConnection)
            } catch (e: Exception) {
                Log.w(TAG, "Unbind error: ${e.message}")
            }
            serviceBound = false
        }
        captureService = null
    }

    override fun onCleared() {
        super.onCleared()
        unbindService()
        onnxEngine.close()
        tfliteEngine.close()
    }
}