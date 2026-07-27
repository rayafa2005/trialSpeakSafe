package com.tatvaani.app.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.tatvaani.app.model.Verdict
import com.tatvaani.app.model.VerdictResult

/**
 * MainActivity.kt
 * ===============
 * Tatvaani — Main Activity + Full Compose UI
 *
 * USER FLOW (production):
 * ─────────────────────────────────────────────
 * 1. IDLE screen — explains exactly what to do before tapping
 *    "Allow microphone, then tap Analyze"
 *
 * 2. User taps → Android asks for RECORD_AUDIO permission (once)
 *
 * 3. RECORDING screen — foreground notification visible the entire time
 *    UI shows pulsing animation + "Listening... 2 seconds"
 *    User is never in the dark about what's happening
 *
 * 4. PROCESSING — brief, shows "Analyzing..."
 *
 * 5. RESULT — full screen color verdict + confidence + latency card
 *
 * ERROR states — every error shows a specific, helpful message
 *    Not "something went wrong" — tells user exactly what to fix
 */
class MainActivity : ComponentActivity() {

    private var viewModelRef: TatvaaniViewModel? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        val audioOk = grants[Manifest.permission.RECORD_AUDIO] == true
        val notifyOk = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            grants[Manifest.permission.POST_NOTIFICATIONS] == true
        } else {
            true
        }
        if (audioOk && notifyOk) {
            viewModelRef?.startCapture()
        } else {
            val missing = buildList {
                if (!audioOk) add("microphone (required for audio capture)")
                if (!notifyOk) add("notifications (required for capture status)")
            }
            viewModelRef?.onPermissionDenied(missing.joinToString("\n"))
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            TatvaaniTheme {
                val vm: TatvaaniViewModel = viewModel()
                viewModelRef = vm
                val uiState by vm.uiState.collectAsStateWithLifecycle()

                // Trigger haptic on DANGER verdict
                LaunchedEffect(uiState) {
                    if (uiState is TatvaaniViewModel.UiState.Result) {
                        val r = uiState as TatvaaniViewModel.UiState.Result
                        val primary = Verdict.pickPrimaryForDisplay(r.onDevice, r.server)
                        if (primary?.verdict == Verdict.DANGER) triggerHaptic()
                    }
                }

                TatvaaniApp(
                    uiState       = uiState,
                    onRecordClick = { beginAnalyze() },
                    onReset = { vm.reset() }
                )
            }
        }
    }

    private fun beginAnalyze() {
        viewModelRef?.setRequesting()
        val needed = mutableListOf<String>()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            needed.add(Manifest.permission.RECORD_AUDIO)
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            needed.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        if (needed.isEmpty()) {
            viewModelRef?.startCapture()
        } else {
            permissionLauncher.launch(needed.toTypedArray())
        }
    }

    private fun triggerHaptic() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (getSystemService(VIBRATOR_MANAGER_SERVICE) as VibratorManager)
                .defaultVibrator
                .vibrate(VibrationEffect.createWaveform(longArrayOf(0, 200, 100, 300), -1))
        } else {
            @Suppress("DEPRECATION")
            (getSystemService(VIBRATOR_SERVICE) as Vibrator)
                .vibrate(longArrayOf(0, 200, 100, 300), -1)
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ROOT
// ══════════════════════════════════════════════════════════════════════════════

@Composable
fun TatvaaniApp(
    uiState:       TatvaaniViewModel.UiState,
    onRecordClick: () -> Unit,
    onReset:       () -> Unit
) {
    Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        when (uiState) {
            is TatvaaniViewModel.UiState.Idle       -> IdleScreen(onRecordClick)
            is TatvaaniViewModel.UiState.Requesting -> StatusScreen("Requesting permission...")
            is TatvaaniViewModel.UiState.Recording  -> RecordingScreen()
            is TatvaaniViewModel.UiState.Processing -> StatusScreen("Analyzing audio...")
            is TatvaaniViewModel.UiState.Result     -> ResultScreen(uiState, onReset)
            is TatvaaniViewModel.UiState.Error      -> ErrorScreen(uiState.message, onReset)
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// IDLE SCREEN — full instructions before user taps
// ══════════════════════════════════════════════════════════════════════════════

@Composable
fun IdleScreen(onRecordClick: () -> Unit) {
    Column(
        modifier            = Modifier
            .fillMaxSize()
            .padding(horizontal = 28.dp),
        verticalArrangement = Arrangement.SpaceBetween,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.height(56.dp))

        // App name
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text          = "SpeakSafe",
                fontSize      = 38.sp,
                fontWeight    = FontWeight.Bold,
                color         = Color(0xFF1E3A5F),
                letterSpacing = 4.sp
            )
            Text(
                text      = "AI Voice Detector",
                fontSize  = 16.sp,
                color     = Color(0xFF2E86AB),
                textAlign = TextAlign.Center
            )
        }

        // Step-by-step instructions — users know exactly what to do
        Column(
            horizontalAlignment = Alignment.Start,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text       = "Before you tap:",
                fontSize   = 16.sp,
                fontWeight = FontWeight.SemiBold,
                color      = Color(0xFF333333)
            )
            Spacer(Modifier.height(12.dp))
            InstructionStep(
                number = "1",
                text   = "If on a phone call — put it on speaker so the caller's voice plays through your phone."
            )
            Spacer(Modifier.height(8.dp))
            InstructionStep(
                number = "2",
                text   = "If testing — play a voice audio clip or video on your phone."
            )
            Spacer(Modifier.height(8.dp))
            InstructionStep(
                number = "3",
                text   = "Tap Analyze. Android will ask for microphone permission — allow it."
            )
            Spacer(Modifier.height(8.dp))
            InstructionStep(
                number = "4",
                text   = "SpeakSafe records 2 seconds of microphone audio and tells you if it's real."
            )

            Spacer(Modifier.height(16.dp))

            // Transparency note
            Surface(
                shape = RoundedCornerShape(8.dp),
                color = Color(0xFFE3F2FD)
            ) {
                Row(
                    modifier = Modifier.padding(12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text("🔒", fontSize = 18.sp)
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text      = "Audio is analyzed on your device. Nothing is stored or sent without your knowledge.",
                        fontSize  = 13.sp,
                        color     = Color(0xFF1565C0),
                        lineHeight = 18.sp
                    )
                }
            }
        }

        // Analyze button
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            PulseButton(onClick = onRecordClick, label = "TAP TO\nANALYZE")
            Spacer(Modifier.height(12.dp))
            Text(
                text      = "Audio must be playing before you tap",
                fontSize  = 13.sp,
                color     = Color(0xFF888888),
                textAlign = TextAlign.Center
            )
        }

        Spacer(Modifier.height(32.dp))
    }
}

@Composable
fun InstructionStep(number: String, text: String) {
    Row(verticalAlignment = Alignment.Top) {
        Surface(
            shape = CircleShape,
            color = Color(0xFF1E3A5F),
            modifier = Modifier.size(24.dp)
        ) {
            Box(contentAlignment = Alignment.Center) {
                Text(number, fontSize = 12.sp, color = Color.White, fontWeight = FontWeight.Bold)
            }
        }
        Spacer(Modifier.width(10.dp))
        Text(
            text       = text,
            fontSize   = 14.sp,
            color      = Color(0xFF444444),
            lineHeight = 20.sp,
            modifier   = Modifier.weight(1f)
        )
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// RECORDING SCREEN — user sees exactly what's happening
// ══════════════════════════════════════════════════════════════════════════════

@Composable
fun RecordingScreen() {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val scale by infiniteTransition.animateFloat(
        initialValue  = 0.85f,
        targetValue   = 1.1f,
        animationSpec = infiniteRepeatable(
            animation  = tween(700, easing = EaseInOut),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale"
    )

    Column(
        modifier            = Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(modifier = Modifier.size(160.dp).scale(scale), contentAlignment = Alignment.Center) {
            Box(
                modifier = Modifier
                    .size(160.dp)
                    .background(Color(0xFF1E3A5F).copy(alpha = 0.12f), CircleShape)
            )
            Box(
                modifier = Modifier
                    .size(110.dp)
                    .background(Color(0xFF1E3A5F), CircleShape),
                contentAlignment = Alignment.Center
            ) {
                Text("◉", fontSize = 40.sp, color = Color.White)
            }
        }

        Spacer(Modifier.height(28.dp))

        Text(
            text       = "Listening...",
            fontSize   = 24.sp,
            fontWeight = FontWeight.Bold,
            color      = Color(0xFF1E3A5F)
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text      = "Recording 2 seconds of microphone audio",
            fontSize  = 14.sp,
            color     = Color(0xFF666666),
            textAlign = TextAlign.Center
        )
        Spacer(Modifier.height(24.dp))

        // Consent reminder
        Surface(shape = RoundedCornerShape(8.dp), color = Color(0xFFF3F4F6)) {
            Text(
                text     = "🔔  A notification is shown while recording is active",
                fontSize = 13.sp,
                color    = Color(0xFF555555),
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp)
            )
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// STATUS SCREEN — processing / requesting
// ══════════════════════════════════════════════════════════════════════════════

@Composable
fun StatusScreen(message: String) {
    Column(
        modifier            = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        CircularProgressIndicator(color = Color(0xFF1E3A5F), modifier = Modifier.size(48.dp))
        Spacer(Modifier.height(24.dp))
        Text(message, fontSize = 18.sp, color = Color(0xFF1E3A5F), fontWeight = FontWeight.Medium)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// RESULT SCREEN
// ══════════════════════════════════════════════════════════════════════════════

@Composable
fun ResultScreen(state: TatvaaniViewModel.UiState.Result, onReset: () -> Unit) {
    val primary = Verdict.pickPrimaryForDisplay(state.onDevice, state.server)
        ?: run { /* show fallback UI */ return@ResultScreen }
    val alternate = listOfNotNull(state.onDevice, state.server).firstOrNull { it !== primary }

    val bgColor by animateColorAsState(
        targetValue   = primary.verdict.color,
        animationSpec = tween(500),
        label         = "bg"
    )

    Box(modifier = Modifier.fillMaxSize().background(bgColor)) {
        Column(
            modifier            = Modifier.fillMaxSize().padding(28.dp),
            verticalArrangement = Arrangement.SpaceBetween,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(Modifier.height(32.dp))

            // Main verdict
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(primary.verdict.emoji, fontSize = 72.sp, color = Color.White)
                Spacer(Modifier.height(12.dp))
                Text(
                    text          = primary.verdict.label,
                    fontSize      = 44.sp,
                    fontWeight    = FontWeight.Bold,
                    color         = Color.White,
                    letterSpacing = 4.sp
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text      = primary.verdict.description,
                    fontSize  = 17.sp,
                    color     = Color.White.copy(alpha = 0.9f),
                    textAlign = TextAlign.Center,
                    lineHeight = 24.sp
                )
                Spacer(Modifier.height(20.dp))
                Surface(
                    shape = RoundedCornerShape(50),
                    color = Color.White.copy(alpha = 0.2f)
                ) {
                    Text(
                        text     = "Confidence: ${primary.confidencePct}",
                        modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp),
                        fontSize = 16.sp,
                        color    = Color.White,
                        fontWeight = FontWeight.Medium
                    )
                }
                if (alternate != null && alternate.verdict != primary.verdict) {
                    Spacer(Modifier.height(10.dp))
                    Text(
                        text      = "Showing ${primary.source.displayName} " +
                                "(score ${"%.0f".format(primary.decisionStrength() * 100)} vs " +
                                "${alternate.source.displayName} ${"%.0f".format(alternate.decisionStrength() * 100)})",
                        fontSize  = 13.sp,
                        color     = Color.White.copy(alpha = 0.85f),
                        textAlign = TextAlign.Center
                    )
                }
            }

            // Bottom section
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                LatencyCard(state.onDevice, state.server)
                Spacer(Modifier.height(20.dp))
                Button(
                    onClick = onReset,
                    modifier = Modifier.fillMaxWidth(),
                    shape    = RoundedCornerShape(12.dp),
                    colors   = ButtonDefaults.buttonColors(
                        containerColor = Color.White.copy(alpha = 0.2f),
                        contentColor   = Color.White
                    )
                ) {
                    Text(
                        "ANALYZE AGAIN",
                        letterSpacing = 2.sp,
                        fontWeight    = FontWeight.Bold,
                        fontSize      = 15.sp
                    )
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// LATENCY CARD
// ══════════════════════════════════════════════════════════════════════════════

@Composable
fun LatencyCard(onDevice: VerdictResult?, server: VerdictResult?) {
    if (onDevice == null && server == null) return
    Surface(
        shape = RoundedCornerShape(14.dp),
        color = Color.White.copy(alpha = 0.15f)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 14.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment     = Alignment.CenterVertically
        ) {
            LatencyItem("On-Device", onDevice)
            Divider(
                modifier  = Modifier.height(40.dp).width(1.dp),
                color     = Color.White.copy(alpha = 0.3f)
            )
            LatencyItem("Server", server)
        }
    }
}

@Composable
fun LatencyItem(label: String, result: VerdictResult?) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(label, fontSize = 11.sp, color = Color.White.copy(alpha = 0.7f), letterSpacing = 1.sp)
        Spacer(Modifier.height(4.dp))
        if (result != null) {
            Text(
                "${result.latencyMs}ms",
                fontSize   = 22.sp,
                fontWeight = FontWeight.Bold,
                color      = Color.White
            )
            Text(result.verdict.label, fontSize = 11.sp, color = Color.White.copy(alpha = 0.7f))
        } else {
            Text("—", fontSize = 22.sp, color = Color.White.copy(alpha = 0.4f))
            Text("unavailable", fontSize = 11.sp, color = Color.White.copy(alpha = 0.4f))
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ERROR SCREEN — specific, actionable messages
// ══════════════════════════════════════════════════════════════════════════════

@Composable
fun ErrorScreen(message: String, onRetry: () -> Unit) {
    Column(
        modifier            = Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("⚠", fontSize = 56.sp)
        Spacer(Modifier.height(20.dp))
        Text(
            text       = "Could not analyze",
            fontSize   = 20.sp,
            fontWeight = FontWeight.Bold,
            color      = Color(0xFF1E3A5F)
        )
        Spacer(Modifier.height(12.dp))
        Text(
            text      = message,
            fontSize  = 15.sp,
            color     = Color(0xFF555555),
            textAlign = TextAlign.Center,
            lineHeight = 22.sp
        )
        Spacer(Modifier.height(32.dp))

        // Contextual help based on error
        if (message.contains("permission", ignoreCase = true) ||
            message.contains("settings", ignoreCase = true)) {
            Surface(
                shape = RoundedCornerShape(10.dp),
                color = Color(0xFFE3F2FD)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        "What to do:",
                        fontWeight = FontWeight.SemiBold,
                        fontSize   = 14.sp,
                        color      = Color(0xFF1565C0)
                    )
                    Spacer(Modifier.height(6.dp))
                    Text("• Settings → Apps → SpeakSafe → Permissions", fontSize = 13.sp, color = Color(0xFF666666))
                    Text("• Enable Microphone and Notifications", fontSize = 13.sp, color = Color(0xFF666666))
                    Text("• Return and tap TRY AGAIN", fontSize = 13.sp, color = Color(0xFF666666))
                }
            }
            Spacer(Modifier.height(20.dp))
        } else if (message.contains("audio", ignoreCase = true) ||
            message.contains("playing", ignoreCase = true)) {
            Surface(
                shape = RoundedCornerShape(10.dp),
                color = Color(0xFFFFF8E1)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        "What to do:",
                        fontWeight = FontWeight.SemiBold,
                        fontSize   = 14.sp,
                        color      = Color(0xFFE65100)
                    )
                    Spacer(Modifier.height(6.dp))
                    Text("• On a phone call? Put it on speaker first", fontSize = 13.sp, color = Color(0xFF666666))
                    Text("• Testing? Play a voice clip or video", fontSize = 13.sp, color = Color(0xFF666666))
                    Text("• Make sure audio is audible before tapping", fontSize = 13.sp, color = Color(0xFF666666))
                }
            }
            Spacer(Modifier.height(20.dp))
        }

        Button(
            onClick = onRetry,
            shape   = RoundedCornerShape(12.dp),
            colors  = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E3A5F))
        ) {
            Text("TRY AGAIN", letterSpacing = 2.sp, fontWeight = FontWeight.Bold)
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// REUSABLE: PULSE BUTTON
// ══════════════════════════════════════════════════════════════════════════════

@Composable
fun PulseButton(onClick: () -> Unit, label: String) {
    val infinite = rememberInfiniteTransition(label = "pulse")
    val scale by infinite.animateFloat(
        initialValue  = 1.0f,
        targetValue   = 1.06f,
        animationSpec = infiniteRepeatable(
            animation  = tween(1400, easing = EaseInOut),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale"
    )

    Box(modifier = Modifier.size(200.dp).scale(scale), contentAlignment = Alignment.Center) {
        Box(
            modifier = Modifier
                .size(200.dp)
                .background(Color(0xFF1E3A5F).copy(alpha = 0.1f), CircleShape)
        )
        Button(
            onClick  = onClick,
            modifier = Modifier.size(160.dp),
            shape    = CircleShape,
            colors   = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E3A5F))
        ) {
            Text(
                text       = label,
                textAlign  = TextAlign.Center,
                fontSize   = 16.sp,
                fontWeight = FontWeight.Bold,
                color      = Color.White,
                lineHeight = 22.sp
            )
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// THEME
// ══════════════════════════════════════════════════════════════════════════════

@Composable
fun TatvaaniTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary    = Color(0xFF1E3A5F),
            secondary  = Color(0xFF2E86AB),
            background = Color(0xFFF8FAFC),
            surface    = Color.White,
            error      = Color(0xFFB71C1C)
        ),
        content = content
    )
}