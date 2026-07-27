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
 * Fixed flow:
 *  1. startCapture() starts the foreground service (microphone capture)
 *  2. bindService() connects to get the Binder reference
 *  3. onServiceConnected() sets callbacks then calls startCapture() on service
 *  4. Service captures audio and calls onCaptureComplete callback
 *  5. ViewModel runs TFLite + Server inference in parallel
 *  6. Results pushed to UI via StateFlow
 */
class TatvaaniViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "TatvaaniViewModel"
    }

    // ── UI State ──────────────────────────────────────────────────────────────

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

    // ── Engines ───────────────────────────────────────────────────────────────

    private val tfliteEngine = TFLiteEngine(application)
    private val onnxEngine   = OnnxEngine(application)
    private val serverClient = ServerClient()

    // ── Service ───────────────────────────────────────────────────────────────

    private var captureService: AudioCaptureService? = null
    private var serviceBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val b = binder as? AudioCaptureService.CaptureBinder ?: return
            captureService = b.getService()
            serviceBound   = true
            Log.d(TAG, "Service connected — starting capture")

            // Set callbacks before starting capture
            captureService?.onCaptureComplete = { pcm -> onAudioCaptured(pcm) }
            captureService?.onCaptureError    = { msg -> onCaptureError(msg) }

            // Start capture immediately — projection token is already in the service
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
            val tfliteOk = tfliteEngine.initialize()
            Log.d(TAG, "TFLite ready: ${tfliteEngine.isReady}")
            if (!tfliteOk) {
                val onnxOk = onnxEngine.initialize()
                Log.d(TAG, "ONNX ready: ${onnxEngine.isReady}")
                if (!onnxOk) {
                    Log.w(TAG, "On-device inference unavailable (TFLite+ONNX failed)")
                }
            }
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Called from MainActivity after RECORD_AUDIO permission granted.
     * Starts foreground service, then binds to get callbacks.
     */
    fun startCapture() {
        _uiState.value = UiState.Recording
        val context = getApplication<Application>()

        val serviceIntent = android.content.Intent(context, AudioCaptureService::class.java)

        // Start as foreground service first (required before bind on Android 10+)
        context.startForegroundService(serviceIntent)

        // Bind to get the Binder for callbacks
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
            val onDeviceDeferred = async {
                when {
                    tfliteEngine.isReady -> tfliteEngine.runInference(pcm)
                    onnxEngine.isReady   -> onnxEngine.runInference(pcm)
                    else                 -> null
                }
            }
            val serverDeferred   = async { serverClient.predict(pcm) }

            val onDeviceResult = onDeviceDeferred.await()
            val serverResult   = serverDeferred.await()

            _uiState.value = when {
                onDeviceResult != null || serverResult != null ->
                    UiState.Result(onDevice = onDeviceResult, server = serverResult)
                !tfliteEngine.isReady && !onnxEngine.isReady ->
                    UiState.Error(
                        "Capture succeeded, but on-device model is not loaded.\n\n" +
                                "For TFLite: copy tatvanet_int8.tflite into app/src/main/assets/.\n" +
                                "For ONNX: copy tatvanet.onnx into app/src/main/assets/.\n\n" +
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
        tfliteEngine.close()
        onnxEngine.close()
    }
}