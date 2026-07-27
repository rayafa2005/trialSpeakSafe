// ============================================================
// app/build.gradle.kts
// Tatvaani App — uses version catalog (libs.versions.toml)
// ============================================================

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)   // Compose compiler plugin — fixes Kotlin 2.0 warning
}

android {
    namespace = "com.tatvaani.app"
    compileSdk = 34


    defaultConfig {
        applicationId = "com.tatvaani.app"
        minSdk = 29
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
        // ADD THIS LINE — 16KB page size support
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }


    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    buildFeatures {
        compose = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        jniLibs {
            // Required when AndroidManifest sets extractNativeLibs=true (TFLite .so on device)
            useLegacyPackaging = true
        }
    }

    // TFLite model must not be compressed — loaded as raw bytes by TFLiteEngine
    androidResources {
        noCompress += "tflite"
        // ONNX model must not be compressed — loaded as raw bytes by OnnxEngine
        noCompress += "onnx"
    }


    // Register kotlin/ as a source directory

}

dependencies {
    // ── Jetpack Compose ────────────────────────────────────────
    //implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar", "*.aar"))))
    implementation(platform("androidx.compose:compose-bom:2024.02.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.8.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.7.0")
    debugImplementation("androidx.compose.ui:ui-tooling")
    debugImplementation("androidx.compose.ui:ui-test-manifest")

    // ── Core Android ───────────────────────────────────────────
    implementation("androidx.core:core-ktx:1.12.0")

    // ── TensorFlow Lite (on-device inference) ─────────────────
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    //implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")

    // ── ONNX Runtime (fallback on-device inference) ───────────
    // Full package for operator coverage. CPU-only.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")

    // ── Networking (server latency benchmark) ──────────────────
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // ── JSON parsing ───────────────────────────────────────────
    implementation("org.json:json:20231013")

    // ── Testing ────────────────────────────────────────────────
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}