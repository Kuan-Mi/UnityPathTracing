using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using static PathTracing.PathTracingUtils;

namespace PathTracing
{
    /// <summary>
    /// Per-camera temporal state: current and previous frame matrices, jitter, resolution scale.
    /// PathTracingFeature owns one instance per camera key and calls Update() once per frame
    /// before passing the data to NRD/DLRR via their own input structs.
    /// </summary>
    public class CameraFrameState
    {
        // ── current frame ──────────────────────────────────────────────
        public Matrix4x4 worldToView;
        public Matrix4x4 worldToClip;
        public Matrix4x4 viewToClip;
        public float3    camPos;
        public float2    ViewportJitter;
        public float     resolutionScale;
        public int2      renderResolution;
        public uint      FrameIndex;

        // ── previous frame ─────────────────────────────────────────────
        public Matrix4x4 prevWorldToView;
        public Matrix4x4 prevWorldToClip;
        public Matrix4x4 prevViewToClip;
        public float3    prevCamPos;
        public float2    PrevViewportJitter;
        public float     prevResolutionScale;

        // ── other state ───────────────────────────────────────────────
        public int convergenceStep;
        
        private bool lastIsAutoExposureEnabled;
        private float lastExposure;
        private int lastBounceNum;
        

        public CameraFrameState(float initialResolutionScale)
        {
            resolutionScale     = initialResolutionScale;
            prevResolutionScale = initialResolutionScale;
        }
        
        /// <summary>
        /// Must be called once per frame before GetInteropDataPtr on NRD / DLRR.
        /// Saves current values to prev*, then refreshes from the camera.
        /// </summary>
        public void Update(RenderingData renderingData, PathTracingSetting setting)
        {
            
            
            
            
            // 1. save prev
            prevWorldToView     = worldToView;
            prevWorldToClip     = worldToClip;
            prevViewToClip      = viewToClip;
            prevCamPos          = camPos;
            PrevViewportJitter  = ViewportJitter;
            prevResolutionScale = resolutionScale;

            // 2. refresh from camera
            var cameraData = renderingData.cameraData;
            var xrPass     = cameraData.xr;
            if (xrPass.enabled)
            {
                worldToView = xrPass.GetViewMatrix();
                var proj    = GL.GetGPUProjectionMatrix(xrPass.GetProjMatrix(), false);
                worldToClip = proj * worldToView;
                viewToClip  = proj;
                var invView = worldToView.inverse;
                camPos      = new float3(invView.m03, invView.m13, invView.m23);
            }
            else
            {
                var cam     = cameraData.camera;
                camPos      = new float3(cam.transform.position.x, cam.transform.position.y, cam.transform.position.z);
                worldToView = cam.worldToCameraMatrix;
                worldToClip = GetWorldToClipMatrix(cam);
                viewToClip  = GL.GetGPUProjectionMatrix(cam.projectionMatrix, false);
            }

            // 3. resolution scale (RR forces 1.0)
            resolutionScale = setting.RR ? 1.0f : setting.resolutionScale;

            // 4. jitter
            PrevViewportJitter = ViewportJitter;
            ViewportJitter     = Halton2D(FrameIndex + 1) - new float2(0.5f, 0.5f);

            // 5. advance frame counter
            FrameIndex++;
            
            bool hasCameraMoved = worldToView != prevWorldToView || worldToClip != prevWorldToClip;
            bool settingsChanged =  setting.enableAutoExposure != lastIsAutoExposureEnabled 
                                    || !Mathf.Approximately(setting.exposure, lastExposure)
                                     || setting.referenceBounceNum != lastBounceNum;
            
            lastIsAutoExposureEnabled = setting.enableAutoExposure;
            lastExposure = setting.exposure;
            lastBounceNum = setting.referenceBounceNum;
            
            if (hasCameraMoved || settingsChanged || !setting.accumulateReference)
            {
                convergenceStep = 0;
            }
            else
            {
                convergenceStep++;
            }
            
        }
 

        // ── Halton helpers (moved from NRDDenoiser) ────────────────────

        public static float Halton(uint n, uint @base)
        {
            float a       = 1.0f;
            float b       = 0.0f;
            float baseInv = 1.0f / @base;
            while (n != 0)
            {
                a    *= baseInv;
                b    += a * (n % @base);
                n     = (uint)(n * baseInv);
            }
            return b;
        }

        public static uint ReverseBits32(uint v)
        {
            v = ((v & 0x55555555u) << 1)  | ((v >> 1)  & 0x55555555u);
            v = ((v & 0x33333333u) << 2)  | ((v >> 2)  & 0x33333333u);
            v = ((v & 0x0F0F0F0Fu) << 4)  | ((v >> 4)  & 0x0F0F0F0Fu);
            v = ((v & 0x00FF00FFu) << 8)  | ((v >> 8)  & 0x00FF00FFu);
            v = (v << 16) | (v >> 16);
            return v;
        }

        public static float Halton2(uint n)   => ReverseBits32(n) * 2.3283064365386963e-10f;
        public static float Halton1D(uint n)  => Halton2(n);

        public static float2 Halton2D(uint n) => new float2(Halton2(n), Halton(n, 3));
        

        uint GetMaxAccumulatedFrameNum(float accumulationTime, float fps)
        {
            return (uint)(accumulationTime * fps + 0.5f);
        }
        
        
        private static int2 ComputeOutputResolution(CameraData cameraData)
        {
            var xrPass = cameraData.xr;
            if (xrPass.enabled)
                return new int2(xrPass.renderTargetDesc.width, xrPass.renderTargetDesc.height);
            return new int2(
                (int)(cameraData.camera.pixelWidth * cameraData.renderScale),
                (int)(cameraData.camera.pixelHeight * cameraData.renderScale));
        }
        
        
        public GlobalConstants GetConstants(RenderingData renderingData, PathTracingSetting settings,LightCollector lightCollector)
        {
            var cameraData = renderingData.cameraData;

            var lightData = renderingData.lightData;
            var mainLight = lightData.mainLightIndex >= 0 ? lightData.visibleLights[lightData.mainLightIndex] : default;
            var mat = mainLight.localToWorldMatrix;
            Vector3 lightForward = mat.GetColumn(2);

            var gSunDirection = -lightForward;
            var up = new Vector3(0, 1, 0);
            var gSunBasisX = math.normalize(math.cross(new float3(up.x, up.y, up.z), new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z)));
            var gSunBasisY = math.normalize(math.cross(new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z), gSunBasisX));


            var outputResolution = ComputeOutputResolution(cameraData);

            var xrPass = cameraData.xr;
            var isXr = xrPass.enabled;

            var proj = isXr ? xrPass.GetProjMatrix() : cameraData.camera.projectionMatrix;

            var m11 = proj.m11;


            var rectW = (uint)(renderResolution.x * resolutionScale + 0.5f);
            var rectH = (uint)(renderResolution.y * resolutionScale + 0.5f);

            var rectWprev = (uint)(renderResolution.x * prevResolutionScale + 0.5f);
            var rectHprev = (uint)(renderResolution.y * prevResolutionScale + 0.5f);


            var renderSize = new float2((renderResolution.x), (renderResolution.y));
            var outputSize = new float2((outputResolution.x), (outputResolution.y));
            var rectSize = new float2(rectW, rectH);


            var rectSizePrev = new float2((rectWprev), (rectHprev));
            var jitter = (settings.cameraJitter ? ViewportJitter : 0f) / rectSize;


            var fovXRad = math.atan(1.0f / proj.m00) * 2.0f;
            var horizontalFieldOfView = fovXRad * Mathf.Rad2Deg;

            var nearZ = proj.m23 / (proj.m22 - 1.0f);

            var emissionIntensity = settings.emissionIntensity * (settings.emission ? 1.0f : 0.0f);

            var accumulationTime = 0.5f;
            var maxHistoryFrameNum = 60;

            var fps = 1000.0f / Mathf.Max(Time.deltaTime * 1000.0f, 0.0001f);
            fps = math.min(fps, 121.0f);

            // Debug.Log(fps);

            var resetHistoryFactor = 1.0f;


            float otherMaxAccumulatedFrameNum = GetMaxAccumulatedFrameNum(accumulationTime, fps);
            otherMaxAccumulatedFrameNum = math.min(otherMaxAccumulatedFrameNum, (maxHistoryFrameNum));
            otherMaxAccumulatedFrameNum *= resetHistoryFactor;


            var sharcMaxAccumulatedFrameNum = (uint)(otherMaxAccumulatedFrameNum * (settings.boost ? settings.boostFactor : 1.0f) + 0.5f);
            // Debug.Log($"sharcMaxAccumulatedFrameNum: {sharcMaxAccumulatedFrameNum}");
            var taaMaxAccumulatedFrameNum = otherMaxAccumulatedFrameNum * 0.5f;
            var prevFrameMaxAccumulatedFrameNum = otherMaxAccumulatedFrameNum * 0.3f;


            var minProbability = 0.0f;
            // if (settings.tracingMode == RESOLUTION.RESOLUTION_FULL_PROBABILISTIC)
            // {
            //     var mode = HitDistanceReconstructionMode.OFF;
            //     if (settings.denoiser == DenoiserType.DENOISER_REBLUR)
            //         mode = HitDistanceReconstructionMode.OFF;
            //     //     mode = m_ReblurSettings.hitDistanceReconstructionMode;
            //     // else if (m_Settings.denoiser == DenoiserType.DENOISER_RELAX)
            //     //     mode = m_RelaxSettings.hitDistanceReconstructionMode;
            //
            //     // Min / max allowed probability to guarantee a sample in 3x3 or 5x5 area - https://godbolt.org/z/YGYo1rjnM
            //     if (mode == HitDistanceReconstructionMode.AREA_3X3)
            //         minProbability = 1.0f / 4.0f;
            //     else if (mode == HitDistanceReconstructionMode.AREA_5X5)
            //         minProbability = 1.0f / 16.0f;
            // }


            var globalConstants = new GlobalConstants
            {
                gViewToWorld = worldToView.inverse,
                gViewToWorldPrev = prevWorldToView.inverse,
                gViewToClip = viewToClip,
                gWorldToView = worldToView,
                gWorldToViewPrev = prevWorldToView,
                gWorldToClip = worldToClip,
                gWorldToClipPrev = prevWorldToClip,

                gHitDistParams = new float4(3, 0.1f, 20, -25),
                gCameraFrustum = GetNrdFrustum(cameraData),
                gSunBasisX = new float4(gSunBasisX.x, gSunBasisX.y, gSunBasisX.z, 0),
                gSunBasisY = new float4(gSunBasisY.x, gSunBasisY.y, gSunBasisY.z, 0),
                gSunDirection = new float4(gSunDirection.x, gSunDirection.y, gSunDirection.z, 0),
                gCameraGlobalPos = new float4(camPos, 0),
                gCameraGlobalPosPrev = new float4(prevCamPos, 0),
                gViewDirection = new float4(cameraData.camera.transform.forward, 0),
                gHairBaseColor = new float4(0.1f, 0.1f, 0.1f, 1.0f),

                gHairBetas = new float2(0.25f, 0.3f),
                gOutputSize = outputSize,
                gRenderSize = renderSize,
                gRectSize = rectSize,
                gInvOutputSize = new float2(1.0f, 1.0f) / outputSize,
                gInvRenderSize = new float2(1.0f, 1.0f) / renderSize,
                gInvRectSize = new float2(1.0f, 1.0f) / rectSize,
                gRectSizePrev = rectSizePrev,
                gJitter = jitter,

                gEmissionIntensity = emissionIntensity,
                gNearZ = -nearZ,
                gSeparator = settings.splitScreen,
                gRoughnessOverride = 0,
                gMetalnessOverride = 0,
                gUnitToMetersMultiplier = 1.0f,
                gTanSunAngularRadius = math.tan(math.radians(settings.sunAngularDiameter * 0.5f)),
                gTanPixelAngularRadius = math.tan(0.5f * math.radians(horizontalFieldOfView) / rectSize.x),
                gDebug = 0,
                gPrevFrameConfidence = (settings.usePrevFrame && !settings.RR) ? prevFrameMaxAccumulatedFrameNum / (1.0f + prevFrameMaxAccumulatedFrameNum) : 0.0f,
                gUnproject = 1.0f / (0.5f * rectH * m11),
                gAperture = settings.dofAperture * 0.01f,
                gFocalDistance = settings.dofFocalDistance,
                gFocalLength = (0.5f * (35.0f * 0.001f)) / math.tan(math.radians(horizontalFieldOfView * 0.5f)),
                gTAA = (settings.denoiser != DenoiserType.DENOISER_REFERENCE && settings.TAA) ? 1.0f / (1.0f + taaMaxAccumulatedFrameNum) : 1.0f,
                gHdrScale = 1.0f,
                gExposure = settings.exposure,
                gMipBias = settings.mipBias,
                gOrthoMode = cameraData.camera.orthographic ? 1.0f : 0f,
                gIndirectDiffuse = settings.indirectDiffuse ? 1.0f : 0.0f,
                gIndirectSpecular = settings.indirectSpecular ? 1.0f : 0.0f,
                gMinProbability = minProbability,

                gSharcMaxAccumulatedFrameNum = sharcMaxAccumulatedFrameNum,
                gDenoiserType = (uint)settings.denoiser,
                gDisableShadowsAndEnableImportanceSampling = settings.importanceSampling ? 1u : 0u,
                gFrameIndex = (uint)Time.frameCount,
                gForcedMaterial = 0,
                gUseNormalMap = 1,
                gBounceNum = settings.bounceNum,
                gResolve = 1,
                gValidation = 1,
                gSR = (settings.SR && !settings.RR) ? 1u : 0u,
                gRR = settings.RR ? 1u : 0,
                gIsSrgb = 0,
                gOnScreen = 0,
                gTracingMode = settings.RR ? (uint)RESOLUTION.RESOLUTION_FULL_PROBABILISTIC : (uint)settings.tracingMode,
                gSampleNum = settings.rpp,
                gPSR = settings.psr ? (uint)1 : 0,
                gSHARC = settings.SHARC ? (uint)1 : 0,
                gTrimLobe = settings.specularLobeTrimming ? 1u : 0,
                gSpotLightCount = (uint)lightCollector.SpotCount,
                gAreaLightCount = (uint)lightCollector.AreaCount,
                gPointLightCount = (uint)lightCollector.PointCount,
                gSssScatteringColor = new float3(settings.sssScatteringColor.r, settings.sssScatteringColor.g, settings.sssScatteringColor.b),
                gSssMinThreshold = settings.sssMinThreshold,
                gSssTransmissionBsdfSampleCount = settings.sssTransmissionBsdfSampleCount,
                gSssTransmissionPerBsdfScatteringSampleCount = settings.sssTransmissionPerBsdfScatteringSampleCount,
                gSssScale = settings.sssScale,
                gSssAnisotropy = settings.sssAnisotropy,
                gSssMaxSampleRadius = settings.sssMaxSampleRadius,
                gIsEditor = cameraData.camera.cameraType == CameraType.SceneView ? 1u : 0u,
                gShowLight = settings.gShowLight ? 1u : 0u,
                gSharcDownscale =  settings.sharcDownscale,
                gSharcSceneScale =  settings.sharcSceneScale,
                sharcDebug =  settings.sharcDebug? 1u : 0u,
            };

            return globalConstants;
        }


    }
}
