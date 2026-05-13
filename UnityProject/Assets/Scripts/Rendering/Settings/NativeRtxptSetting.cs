using PathTracing;
using UnityEngine;

namespace PathTracing
{
    public enum RtxptPathTracerMode
    {
        /// <summary>Realtime mode: builds stable planes, then denoises with DLSS-RR.</summary>
        BuildStablePlanes = 0,
        /// <summary>Reference mode: accumulates frames without denoising.</summary>
        Reference         = 1,
    }

    /// <summary>
    /// Inspector-facing settings for <see cref="NativeRtxptFeature"/>.
    /// Denoising is handled exclusively by DLSS Ray Reconstruction (DLSS-RR).
    /// </summary>
    [System.Serializable]
    public class NativeRtxptSetting
    {
        // ── Path tracer mode ─────────────────────────────────────────────────
        public RtxptPathTracerMode pathTracerMode = RtxptPathTracerMode.BuildStablePlanes;

        // ── DLSS-RR upscaler quality ──────────────────────────────────────────
        public UpscalerMode upscalerMode = UpscalerMode.QUALITY;

        // ── Camera ───────────────────────────────────────────────────────────
        public bool cameraJitter = true;

        // ── Path tracing quality ─────────────────────────────────────────────
        [Range(1, 4)]
        public int samplesPerPixel = 1;

        [Range(1, 16)]
        public int maxBounces = 4;

        // ── Accumulation (reference mode) ────────────────────────────────────
        [Range(1, 4096)]
        public int maxAccumulationFrames = 256;

        // ── Debug ────────────────────────────────────────────────────────────
        public bool  showValidation  = false;
        public float debug           = 0.0f;
        public bool  tmpDisableDlssRR = false;

        // ── Misc ─────────────────────────────────────────────────────────────
        public bool skipRightEyeInVR = true;
    }
}
