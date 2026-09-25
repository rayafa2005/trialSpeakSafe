package com.tatvaani.app.model

import androidx.compose.ui.graphics.Color

/**
 * VerdictResult.kt
 * ================
 * Tatvaani — Core data model for inference output.
 *
 * Shared between on-device TFLite and server inference paths.
 * Both paths produce this same structure so the UI doesn't care
 * which inference path generated the result.
 */

// ── Verdict enum ──────────────────────────────────────────────────────────────

enum class Verdict(
    val label:       String,
    val description: String,
    val colorHex:    Long,
    val emoji:       String
) {
    SAFE(
        label       = "SAFE",
        description = "Human voice detected",
        colorHex    = 0xFF1B5E20L,
        emoji       = "✓"
    ),
    CAUTION(
        label       = "CAUTION",
        description = "Uncertain — verify before trusting",
        colorHex    = 0xFFE65100L,
        emoji       = "⚠"
    ),
    DANGER(
        label       = "DANGER",
        description = "AI-generated voice detected",
        colorHex    = 0xFFB71C1CL,
        emoji       = "✕"
    );

    val color: Color get() = Color(colorHex)

    companion object {
        /** Shared by on-device (ONNX/TFLite) and server client — keep in sync with server/app.py */
        const val MIN_CONFIDENCE_FOR_DECISIVE_VERDICT = 0.70f

        /** Map TatvaNet class index (0=Safe, 1=Caution, 2=Danger) to Verdict */
        fun fromIndex(index: Int): Verdict = when (index) {
            0    -> SAFE
            1    -> CAUTION
            2    -> DANGER
            else -> CAUTION
        }

        /** Continuous 3-tier calibration from deepfake probability */
        fun evaluateProbability(
            pFake: Float,
            latencyMs: Long,
            source: InferenceSource
        ): VerdictResult {
            val (verdict, confidence) = when {
                pFake < 0.35f -> SAFE to (1.0f - pFake)
                pFake > 0.65f -> DANGER to pFake
                else -> CAUTION to (1.0f - kotlin.math.abs(pFake - 0.50f) * 2.0f)
            }
            val pSafeRaw = maxOf(0.0f, 1.0f - pFake / 0.5f)
            val pDangerRaw = maxOf(0.0f, (pFake - 0.5f) / 0.5f)
            val pCautionRaw = maxOf(0.0f, 1.0f - 2.0f * kotlin.math.abs(pFake - 0.5f))
            val total = pSafeRaw + pCautionRaw + pDangerRaw + 1e-8f
            val probs = floatArrayOf(pSafeRaw / total, pCautionRaw / total, pDangerRaw / total)

            return VerdictResult(
                verdict = verdict,
                confidence = confidence,
                probs = probs,
                latencyMs = latencyMs,
                source = source
            )
        }

        /** Argmax class selection without threshold gating */
        fun fromProbs(probs: FloatArray): Verdict {
            require(probs.size == 3) { "Expected 3 class probabilities, got ${probs.size}" }
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            return fromIndex(maxIdx)
        }

        /**
         * Gap between top-1 and top-2 class probabilities (how decisive the distribution is).
         */
        fun probabilityMargin(probs: FloatArray): Float {
            if (probs.isEmpty()) return 0f
            val sorted = probs.sortedDescending()
            return if (sorted.size >= 2) sorted[0] - sorted[1] else sorted[0]
        }

        /**
         * Choose which pipeline drives the main result UI when both ran.
         * Prioritizes the SERVER verdict if available; falls back to on-device when server is null.
         */
        fun pickPrimaryForDisplay(
            onDevice: VerdictResult?,
            server: VerdictResult?
        ): VerdictResult? {
            return server ?: onDevice
        }
    }
}

// ── InferenceSource ───────────────────────────────────────────────────────────

enum class InferenceSource(val displayName: String) {
    ON_DEVICE("On-Device"),
    SERVER("Server")
}

// ── VerdictResult ─────────────────────────────────────────────────────────────

/**
 * Complete result from one inference run.
 *
 * @param verdict      Classified verdict (SAFE / CAUTION / DANGER)
 * @param confidence   Probability of the predicted class [0.0, 1.0]
 * @param probs        All three class probabilities [safe, caution, danger]
 * @param latencyMs    Total inference time in milliseconds
 * @param source       Which path produced this result (on-device or server)
 */
data class VerdictResult(
    val verdict:    Verdict,
    val confidence: Float,
    val probs:      FloatArray,
    val latencyMs:  Long,
    val source:     InferenceSource
) {
    /** Confidence as integer percentage string e.g. "87%" */
    val confidencePct: String
        get() = "${(confidence * 100).toInt()}%"

    /** True if confidence is high enough for strong color display */
    val isHighConfidence: Boolean
        get() = confidence >= 0.75f

    /**
     * Single score for comparing on-device vs server (higher = show this result on top).
     * - [confidence]: predicted-class probability
     * - [Verdict.probabilityMargin]: how far ahead the top class is vs runner-up
     */
    fun decisionStrength(): Float {
        val margin = Verdict.probabilityMargin(probs)
        return confidence * 0.65f + margin * 0.35f
    }

    // FloatArray requires manual equals/hashCode
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is VerdictResult) return false
        return verdict    == other.verdict    &&
               confidence == other.confidence &&
               probs.contentEquals(other.probs) &&
               latencyMs  == other.latencyMs  &&
               source     == other.source
    }

    override fun hashCode(): Int {
        var result = verdict.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + probs.contentHashCode()
        result = 31 * result + latencyMs.hashCode()
        result = 31 * result + source.hashCode()
        return result
    }
}
