# Tatvaani Android — 3-day demo sprint (OnePlus, Android 15)

Package: `com.tatvaani.app`  
Project: `tatvaani_android/`

## Build and install on a physical device

1. Open `tatvaani_android/` in **Android Studio** (Ladybug or newer recommended).
2. Connect OnePlus via USB; enable **Developer options → USB debugging**.
3. On the phone, accept the RSA fingerprint prompt.
4. **Run** ▶ `app` (debug). Studio builds, installs, and launches on the device.

CLI (after `JAVA_HOME` points to JDK 17+):

```powershell
cd c:\Users\reema\Tatvaani\tatvaani_android
.\gradlew.bat installDebug
adb shell am start -n com.tatvaani.app/.ui.MainActivity
```

## Placeholder vs real TFLite model

- `app/src/main/assets/tatvanet_int8.tflite` may be a **0-byte placeholder** so the repo builds without the trained weights.
- The app **skips** loading until the file is ≥ 1 KB (real export). Capture and UI still work; on-device inference shows as unavailable until you copy the model.
- After training/export:

```text
tatvaani_ml/export/tatvanet_int8.tflite
  → copy to →
tatvaani_android/app/src/main/assets/tatvanet_int8.tflite
```

Then rebuild and reinstall.

## Server URL on a physical device (not the emulator)

`10.0.2.2` is **emulator-only** (host loopback). On a real phone, use your PC’s **LAN IPv4**.

1. **PC IP** (same Wi‑Fi as the phone):
   - Windows: `ipconfig` → e.g. `192.168.1.42`
2. **Start the API** bound to all interfaces:

```powershell
cd c:\Users\reema\Tatvaani\tatvaani_ml\server
py -3.11 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

3. **Allow Windows Firewall** for Python on port 8000 (private network).
4. **Set URL in code** before the demo (pick one):

```kotlin
// ServerClient.kt — change the default:
var SERVER_URL = "http://192.168.1.42:8000"

// Or at runtime from MainActivity / ViewModel:
viewModel.setServerUrl("http://192.168.1.42:8000")
```

5. Confirm from the phone browser: `http://192.168.1.42:8000/docs` (optional).

Cleartext HTTP is enabled in the manifest for LAN demo (`usesCleartextTraffic="true"`).

## Confirmed demo settings

| Item | Status |
|------|--------|
| GPU delegate | **Off** — `tensorflow-lite-gpu` not in `app/build.gradle.kts`; `TFLiteEngine` uses CPU `numThreads = 4` only |
| `extractNativeLibs` | **`true`** in `AndroidManifest.xml` |
| Foreground notification | **`AudioCaptureService`** calls `startForeground()` before capture; channel `tatvaani_capture` |
| Multi-config capture fallback | **`AudioCaptureService`** tries MEDIA + VOICE_COMMUNICATION + GAME + UNKNOWN, then simpler lists |

---

## OnePlus test checklist (you run on device)

### A. First launch and permissions

1. Install debug build; open **Tatvaani**.
2. Tap **TAP TO ANALYZE**.
3. Allow **Microphone** and **Notifications** when prompted.
4. Expect brief **Requesting permission…** then **Listening…**.

### B. Idle → Recording → Processing → Result

1. On idle screen, read the four steps (speaker / media / permission / 2 s capture).
2. Speak near the microphone (or play a voice clip out loud near the phone) at normal volume.
4. **Recording**: pulsing UI + system notification *“Tatvaani — Analyzing audio”* for ~2 s.
5. **Processing**: *“Analyzing audio…”*
6. **Result**: colored verdict screen; latency card shows **On-Device** and/or **Server** (either may show “unavailable” if model or server is down).

### C. Speaker + media (capture fallback)

1. If you want to demo with a call/video: put it on speaker so the sound is audible in the room.
2. Run analyze again (this uses the microphone — it will “hear” what’s playing).
3. If capture fails: error should mention audio/speaking — follow tips on error screen → **TRY AGAIN**.
4. `adb logcat -s AudioCaptureService` — look for `AudioRecord ready (source=...)`.

### D. Placeholder model path

1. With **0-byte** placeholder only: expect **Result** if server works, or **Error** explaining model + server if both fail.
2. After copying real `tatvanet_int8.tflite` and reinstalling: **On-Device** latency should populate.

### E. Server benchmark

1. Set `SERVER_URL` to PC LAN IP; start `uvicorn` on PC.
2. Run analyze with media playing.
3. Result card: **Server** row should show ms + verdict (not “unavailable”).

### F. Error paths (actionable copy)

| Action | Expected |
|--------|----------|
| Deny microphone | Error lists missing permission + Settings path |
| No audio playing during capture | Error: speaker / play clip / try again |
| Cancel from idle | Stays idle |

### G. Regression checks

1. No crash when tapping Analyze repeatedly.
2. Notification disappears after capture ends.
3. **Analyze again** returns to idle and repeats the flow.

### H. Optional logcat

```text
adb logcat -s TatvaaniViewModel AudioCaptureService TFLiteEngine ServerClient
```

---

## TatvaNet

Do **not** replace `tatvaani_ml/model/tatvanet.py` or retrain for this sprint. Android uses the exported `tatvanet_int8.tflite` only.
