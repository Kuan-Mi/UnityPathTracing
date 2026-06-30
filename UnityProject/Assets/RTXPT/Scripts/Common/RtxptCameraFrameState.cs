using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using static PathTracing.PathTracingUtils;

namespace PathTracing
{
    /// <summary>
    /// RTXPT camera state.
    /// Stores effective RTXPT camera jitter, matching Sample::ComputeCameraJitter gating.
    /// </summary>
    public class RtxptCameraFrameState
    {
        public Matrix4x4 worldToView;
        public Matrix4x4 worldToClip;
        public Matrix4x4 viewToClip;
        public float3    camPos;
        public float2    viewportJitter;
        public float     resolutionScale;
        public int2      renderResolution;
        public uint      frameIndex;

        public Matrix4x4 prevWorldToView;
        public Matrix4x4 prevWorldToClip;
        public Matrix4x4 prevViewToClip;
        public float3    prevCamPos;
        public float2    prevViewportJitter;
        public float     prevResolutionScale;

        public float2 rawViewportJitter;
        public float2 prevRawViewportJitter;
        public float  perPixelJitterAAScale;
        public int    convergenceStep;

        /// <summary>
        /// Reference-mode accumulation sample index, mirroring C++ m_accumulationSampleIndex
        /// (Sample.cpp:1411-1449). Reset every realtime frame and on camera/settings changes;
        /// starts at -32 when accumulationPreWarmRealtimeCaches is on (the negative window runs
        /// the path tracer without accumulating, letting NEE-AT feedback and other temporal
        /// caches converge), then counts 0..accumulationTarget (clamped).
        /// </summary>
        public int accumulationSampleIndex;

        public RtxptCameraFrameState(float initialResolutionScale)
        {
            resolutionScale     = initialResolutionScale;
            prevResolutionScale = initialResolutionScale;
        }

        public void Update(RenderingData renderingData, bool settingsChanged, float currentResolutionScale, RtxptSetting setting)
        {
            prevWorldToView       = worldToView;
            prevWorldToClip       = worldToClip;
            prevViewToClip        = viewToClip;
            prevCamPos            = camPos;
            prevViewportJitter    = viewportJitter;
            prevRawViewportJitter = rawViewportJitter;
            prevResolutionScale   = resolutionScale;

            var cameraData = renderingData.cameraData;
            var xrPass     = cameraData.xr;
            if (xrPass.enabled)
            {
                worldToView = xrPass.GetViewMatrix();
                var proj = GL.GetGPUProjectionMatrix(xrPass.GetProjMatrix(), false);
                worldToClip = proj * worldToView;
                viewToClip  = proj;
                var invView = worldToView.inverse;
                camPos = new float3(invView.m03, invView.m13, invView.m23);
            }
            else
            {
                var cam = cameraData.camera;
                camPos      = new float3(cam.transform.position.x, cam.transform.position.y, cam.transform.position.z);
                worldToView = cam.worldToCameraMatrix;
                worldToClip = GetWorldToClipMatrix(cam);
                viewToClip  = GL.GetGPUProjectionMatrix(cam.projectionMatrix, false);
            }

            resolutionScale = currentResolutionScale;

            rawViewportJitter = R2Jitter(frameIndex + 1u);
            viewportJitter    = UseCameraJitter(setting) ? rawViewportJitter : float2.zero;

            perPixelJitterAAScale = ComputePerPixelJitterAAScale(setting);

            frameIndex++;

            bool hasCameraMoved = worldToView != prevWorldToView || worldToClip != prevWorldToClip;
            if (hasCameraMoved || settingsChanged)
                convergenceStep = 0;
            else
                convergenceStep++;

            // Original (Sample.cpp:1413-1418, 1449): resetAccum |= ResetAccumulation | RealtimeMode;
            // on reset start at -32 (pre-warm) or 0, otherwise advance and clamp at the target
            // (PostUpdatePathTracing increments after rendering ≡ incrementing before the next frame).
            bool resetAccum = setting.realtimeMode || hasCameraMoved || settingsChanged;
            if (resetAccum)
                accumulationSampleIndex = setting.accumulationPreWarmRealtimeCaches ? -32 : 0;
            else
                accumulationSampleIndex = math.min(accumulationSampleIndex + 1, setting.accumulationTarget);
        }

        private static bool UseCameraJitter(RtxptSetting setting)
        {
            return setting.cameraJitter && setting.realtimeMode && setting.realtimeAA != 0;
        }

        private static float ComputePerPixelJitterAAScale(RtxptSetting setting)
        {
            if (!setting.realtimeMode && setting.accumulationAA)
                return 1.0f;

            bool isDlssRR = setting.realtimeMode && setting.realtimeAA == 3 && !setting.tmpDisableDlssRR;
            return isDlssRR ? setting.dlssrrMicroJitter : 0.0f;
        }

        private static float2 R2Jitter(uint index)
        {
            const float a1 = 0.7548776662466927f;
            const float a2 = 0.5698402909980532f;
            return math.frac(new float2(0.5f, 0.5f) + (float)index * new float2(a1, a2)) - new float2(0.5f, 0.5f);
        }
    }
}