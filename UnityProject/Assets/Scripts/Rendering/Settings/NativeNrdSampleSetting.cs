using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Direct C# port of the C++ <c>Settings</c> struct in NRDSample.cpp.
    /// All fields match the original defaults.  Use this with
    /// <c>CameraFrameState.GetNrdConstants(renderingData, NrdSampleSetting)</c>
    /// which faithfully follows the C++ <c>UpdateConstantBuffer</c> logic.
    /// </summary>
    [System.Serializable]
    public class NativeNrdSampleSetting
    {
        public NativeNrdShowMode showMode;
        public bool              showValidation = false;
        public bool              showMV         = false;

        // ── Camera ───────────────────────────────────────────────────────
        public bool cameraJitter = true;

        // ── Sun ──────────────────────────────────────────────────────────
        public float sunAngularDiameter = 0.533f; // degrees

        // ── Rendering ────────────────────────────────────────────────────
        public float exposure                = 80.0f;
        public float roughnessOverride       = 0.0f;
        public float metalnessOverride       = 0.0f;
        public float emissionIntensityLights = 1.0f;
        public float emissionIntensityCubes  = 1.0f;
        public float debug                   = 0.0f;
        public float meterToUnitsMultiplier  = 1.0f;

        [Range(0.0f, 1.0f)]
        public float separator = 0.0f;

        public float hitDistScale    = 3.0f;
        public float resolutionScale = 1.0f;

        [Range(1f, 120f)]
        public uint maxAccumulatedFrameNum = 31;

        [Range(1f, 20f)]
        public uint maxFastAccumulatedFrameNum = 7;

        public OnScreen     onScreen             = 0;
        public int          forcedMaterial       = 0;
        public DenoiserType denoiser             = 0; // 0 = DENOISER_REBLUR
        public int          rpp                  = 1;
        public int          bounceNum            = 1;
        public RESOLUTION   tracingMode          = RESOLUTION.RESOLUTION_HALF;
        public bool         SHARC                = true;
        public bool         PSR                  = false;
        public bool         normalMap            = true;
        public bool         TAA                  = true;
        public bool         emission             = true;
        public bool         importanceSampling   = true;
        public bool         specularLobeTrimming = true;
        public bool         adaptiveAccumulation = true;
        public bool         usePrevFrame         = true;
        public bool         boost                = false;
        public bool         SR                   = false;
        public bool         RR                   = false;
        public bool         tmpDisableRR         = false;
        public UpscalerMode upscalerMode         = UpscalerMode.NATIVE;
        public bool         confidence           = true;
        public float        denoisingRange       = 1000f;

        [Range(0.0f, 1.0f)]
        public float nisSharpness = 0.2f;

        public bool skipRightEyeInVR;
    }
}
