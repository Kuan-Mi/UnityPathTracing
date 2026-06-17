using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using static PathTracing.PathTracingUtils;

namespace PathTracing
{
    /// <summary>NRD-Sample pipeline constant builders extracted from CameraFrameState (Core).</summary>
    public static class CameraFrameStateNrdExtensions
    {
        public static GlobalConstants GetConstants(this CameraFrameState cfs, RenderingData renderingData, NrdSampleSetting settings, LightCollector lightCollector)
        {
            var worldToView = cfs.worldToView;
            var worldToClip = cfs.worldToClip;
            var viewToClip = cfs.viewToClip;
            var camPos = cfs.camPos;
            var viewportJitter = cfs.viewportJitter;
            var resolutionScale = cfs.resolutionScale;
            var renderResolution = cfs.renderResolution;
            var frameIndex = cfs.frameIndex;
            var prevWorldToView = cfs.prevWorldToView;
            var prevWorldToClip = cfs.prevWorldToClip;
            var prevCamPos = cfs.prevCamPos;
            var prevViewportJitter = cfs.prevViewportJitter;
            var prevResolutionScale = cfs.prevResolutionScale;

            var cameraData = renderingData.cameraData;

            var     lightData    = renderingData.lightData;
            var     mainLight    = lightData.mainLightIndex >= 0 ? lightData.visibleLights[lightData.mainLightIndex] : default;
            var     mat          = mainLight.localToWorldMatrix;
            Vector3 lightForward = mat.GetColumn(2);

            var gSunDirection = -lightForward;
            var up            = new Vector3(0, 1, 0);
            var gSunBasisX    = math.normalize(math.cross(new float3(up.x, up.y, up.z), new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z)));
            var gSunBasisY    = math.normalize(math.cross(new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z), gSunBasisX));


            var outputResolution = CameraFrameState.ComputeOutputResolution(cameraData);

            var xrPass = cameraData.xr;
            var isXr   = xrPass.enabled;

            var proj = isXr ? xrPass.GetProjMatrix() : cameraData.camera.projectionMatrix;

            var m11 = proj.m11;


            var rectW = (uint)(renderResolution.x * resolutionScale + 0.5f);
            var rectH = (uint)(renderResolution.y * resolutionScale + 0.5f);

            var rectWprev = (uint)(renderResolution.x * prevResolutionScale + 0.5f);
            var rectHprev = (uint)(renderResolution.y * prevResolutionScale + 0.5f);


            var renderSize = new float2((renderResolution.x), (renderResolution.y));
            var outputSize = new float2((outputResolution.x), (outputResolution.y));
            var rectSize   = new float2(rectW, rectH);


            var rectSizePrev = new float2((rectWprev), (rectHprev));
            var jitter       = (settings.cameraJitter ? viewportJitter : 0f) / rectSize;


            var fovXRad               = math.atan(1.0f / proj.m00) * 2.0f;
            var horizontalFieldOfView = fovXRad * Mathf.Rad2Deg;

            var nearZ = proj.m23 / (proj.m22 - 1.0f);

            var emissionIntensity = settings.emissionIntensity * (settings.emission ? 1.0f : 0.0f);

            var accumulationTime   = 0.5f;
            var maxHistoryFrameNum = 60;

            var fps = 1000.0f / Mathf.Max(Time.deltaTime * 1000.0f, 0.0001f);
            fps = math.min(fps, 121.0f);

            // Debug.Log(fps);

            var resetHistoryFactor = 1.0f;


            float otherMaxAccumulatedFrameNum = CameraFrameState.GetMaxAccumulatedFrameNum(accumulationTime, fps);
            otherMaxAccumulatedFrameNum =  math.min(otherMaxAccumulatedFrameNum, (maxHistoryFrameNum));
            otherMaxAccumulatedFrameNum *= resetHistoryFactor;


            var sharcMaxAccumulatedFrameNum = (uint)(otherMaxAccumulatedFrameNum * (settings.boost ? settings.boostFactor : 1.0f) + 0.5f);
            // Debug.Log($"sharcMaxAccumulatedFrameNum: {sharcMaxAccumulatedFrameNum}");
            var taaMaxAccumulatedFrameNum       = otherMaxAccumulatedFrameNum * 0.5f;
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
                gViewToWorld     = worldToView.inverse,
                gViewToWorldPrev = prevWorldToView.inverse,
                gViewToClip      = viewToClip,
                gWorldToView     = worldToView,
                gWorldToViewPrev = prevWorldToView,
                gWorldToClip     = worldToClip,
                gWorldToClipPrev = prevWorldToClip,

                gHitDistParams       = new float4(3, 0.1f, 20, -25),
                gCameraFrustum       = GetNrdFrustum(cameraData),
                gSunBasisX           = new float4(gSunBasisX.x, gSunBasisX.y, gSunBasisX.z, 0),
                gSunBasisY           = new float4(gSunBasisY.x, gSunBasisY.y, gSunBasisY.z, 0),
                gSunDirection        = new float4(gSunDirection.x, gSunDirection.y, gSunDirection.z, 0),
                gCameraGlobalPos     = new float4(camPos, 0),
                gCameraGlobalPosPrev = new float4(prevCamPos, 0),
                gViewDirection       = new float4(cameraData.camera.transform.forward, 0),
                gHairBaseColor       = new float4(0.1f, 0.1f, 0.1f, 1.0f),

                gHairBetas     = new float2(0.25f, 0.3f),
                gOutputSize    = outputSize,
                gRenderSize    = renderSize,
                gRectSize      = rectSize,
                gInvOutputSize = new float2(1.0f, 1.0f) / outputSize,
                gInvRenderSize = new float2(1.0f, 1.0f) / renderSize,
                gInvRectSize   = new float2(1.0f, 1.0f) / rectSize,
                gRectSizePrev  = rectSizePrev,
                gJitter        = jitter,

                gEmissionIntensity      = emissionIntensity,
                gNearZ                  = -nearZ,
                gSeparator              = settings.splitScreen,
                gRoughnessOverride      = 0,
                gMetalnessOverride      = 0,
                gUnitToMetersMultiplier = 1.0f,
                gTanSunAngularRadius    = math.tan(math.radians(settings.sunAngularDiameter * 0.5f)),
                gTanPixelAngularRadius  = math.tan(0.5f * math.radians(horizontalFieldOfView) / rectSize.x),
                gDebug                  = 0,
                gPrevFrameConfidence    = (settings.usePrevFrame && !settings.RR) ? prevFrameMaxAccumulatedFrameNum / (1.0f + prevFrameMaxAccumulatedFrameNum) : 0.0f,
                gUnproject              = 1.0f / (0.5f * rectH * m11),
                gAperture               = settings.dofAperture * 0.01f,
                gFocalDistance          = settings.dofFocalDistance,
                gFocalLength            = (0.5f * (35.0f * 0.001f)) / math.tan(math.radians(horizontalFieldOfView * 0.5f)),
                gTAA                    = (settings.denoiser != DenoiserType.DENOISER_REFERENCE && settings.TAA) ? 1.0f / (1.0f + taaMaxAccumulatedFrameNum) : 1.0f,
                gHdrScale               = 1.0f,
                gExposure               = settings.exposure,
                gMipBias                = settings.mipBias,
                gOrthoMode              = cameraData.camera.orthographic ? 1.0f : 0f,
                gIndirectDiffuse        = settings.indirectDiffuse ? 1.0f : 0.0f,
                gIndirectSpecular       = settings.indirectSpecular ? 1.0f : 0.0f,
                gMinProbability         = minProbability,

                gSharcMaxAccumulatedFrameNum                 = sharcMaxAccumulatedFrameNum,
                gDenoiserType                                = (uint)settings.denoiser,
                gDisableShadowsAndEnableImportanceSampling   = settings.importanceSampling ? 1u : 0u,
                gFrameIndex                                  = (uint)Time.frameCount,
                gForcedMaterial                              = 0,
                gUseNormalMap                                = 1,
                gBounceNum                                   = settings.bounceNum,
                gResolve                                     = 1,
                gValidation                                  = 1,
                gSR                                          = settings.SR ? 1u : 0u,
                gRR                                          = settings.RR ? 1u : 0,
                gIsSrgb                                      = 0,
                gOnScreen                                    = (uint)settings.gOnScreen,
                gTracingMode                                 = settings.RR ? (uint)RESOLUTION.RESOLUTION_FULL_PROBABILISTIC : (uint)settings.tracingMode,
                gSampleNum                                   = settings.rpp,
                gPSR                                         = settings.psr ? (uint)1 : 0,
                gSHARC                                       = settings.SHARC ? (uint)1 : 0,
                gTrimLobe                                    = settings.specularLobeTrimming ? 1u : 0,
                gSpotLightCount                              = (uint)lightCollector.SpotCount,
                gAreaLightCount                              = (uint)lightCollector.AreaCount,
                gPointLightCount                             = (uint)lightCollector.PointCount,
                gSssMinThreshold                             = settings.sssMinThreshold,
                gSssTransmissionBsdfSampleCount              = settings.sssTransmissionBsdfSampleCount,
                gSssTransmissionPerBsdfScatteringSampleCount = settings.sssTransmissionPerBsdfScatteringSampleCount,
                gSssScale                                    = settings.sssScale,
                gSssAnisotropy                               = settings.sssAnisotropy,
                gSssMaxSampleRadius                          = settings.sssMaxSampleRadius,
                gIsEditor                                    = cameraData.camera.cameraType == CameraType.SceneView ? 1u : 0u,
                gShowLight                                   = 0u,
                gSharcDownscale                              = settings.sharcDownscale,
                gSharcSceneScale                             = settings.sharcSceneScale,
                sharcDebug                                   = settings.sharcDebug ? 1u : 0u,
            };

            return globalConstants;
        }

        public static NRDGlobalConstants GetNrdConstants(this CameraFrameState cfs, RenderingData renderingData, NativeNrdSampleSetting settings)
        {
            var worldToView = cfs.worldToView;
            var worldToClip = cfs.worldToClip;
            var viewToClip = cfs.viewToClip;
            var camPos = cfs.camPos;
            var viewportJitter = cfs.viewportJitter;
            var resolutionScale = cfs.resolutionScale;
            var renderResolution = cfs.renderResolution;
            var frameIndex = cfs.frameIndex;
            var prevWorldToView = cfs.prevWorldToView;
            var prevWorldToClip = cfs.prevWorldToClip;
            var prevCamPos = cfs.prevCamPos;
            var prevViewportJitter = cfs.prevViewportJitter;
            var prevResolutionScale = cfs.prevResolutionScale;

            var cameraData = renderingData.cameraData;

            var     lightData    = renderingData.lightData;
            var     mainLight    = lightData.mainLightIndex >= 0 ? lightData.visibleLights[lightData.mainLightIndex] : default;
            var     mat          = mainLight.localToWorldMatrix;
            Vector3 lightForward = mat.GetColumn(2);

            var gSunDirection = -lightForward;
            var up            = new Vector3(0, 1, 0);
            var gSunBasisX    = math.normalize(math.cross(new float3(up.x, up.y, up.z), new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z)));
            var gSunBasisY    = math.normalize(math.cross(new float3(gSunDirection.x, gSunDirection.y, gSunDirection.z), gSunBasisX));


            // ── Camera / projection ───────────────────────────────────────────────
            var xrPass = cameraData.xr;
            var isXr   = xrPass.enabled;
            var proj   = isXr ? xrPass.GetProjMatrix() : cameraData.camera.projectionMatrix;

            // project[1] in C++ == proj.m11 (cotangent of half vertical FOV)
            float project1 = proj.m11;

            // ── Resolution ───────────────────────────────────────────────────────
            var outputResolution = CameraFrameState.ComputeOutputResolution(cameraData);

            uint rectW     = (uint)(renderResolution.x * settings.resolutionScale + 0.5f);
            uint rectH     = (uint)(renderResolution.y * settings.resolutionScale + 0.5f);
            uint rectWprev = (uint)(renderResolution.x * prevResolutionScale + 0.5f);
            uint rectHprev = (uint)(renderResolution.y * prevResolutionScale + 0.5f);

            float2 renderSize   = new float2(renderResolution.x, renderResolution.y);
            float2 outputSize   = new float2(outputResolution.x, outputResolution.y);
            float2 rectSize     = new float2(rectW, rectH);
            float2 rectSizePrev = new float2(rectWprev, rectHprev);

            // ── Jitter ───────────────────────────────────────────────────────────
            float2 jitter     = (settings.cameraJitter ? viewportJitter : float2.zero) / rectSize;
            float2 jitterPrev = (settings.cameraJitter ? prevViewportJitter : float2.zero) / rectSizePrev;

            // ── Near Z (extracted from projection matrix, negated for Unity convention) ──
            float nearZ = proj.m23 / (proj.m22 - 1.0f);

            // ── Mip bias (mirrors C++ baseMipBias + renderSize.x/outputSize.x term) ──
            bool  usesUpscaling = settings.TAA || settings.SR || settings.RR;
            float baseMipBias   = (usesUpscaling ? -0.5f : 0.0f) + math.log2(settings.resolutionScale);
            float mipBias       = baseMipBias + math.log2(renderSize.x / outputSize.x);

            // ── Accumulated frame counters ───────────────────────────────────────
            int   maxAccum          = (int)settings.maxAccumulatedFrameNum;
            float taaMaxAccum       = maxAccum * 0.5f;
            float prevFrameMaxAccum = maxAccum * 0.3f;

            // ── HitDist parameters (nrd::ReblurHitDistanceParameters) ───────────
            // C++: hitDistanceParameters.A = hitDistScale * meterToUnitsMultiplier; B/C/D are defaults (0.1, 20, -25)
            float4 hitDistSettings = new float4(
                settings.hitDistScale * settings.meterToUnitsMultiplier,
                0.1f, 20.0f, 0f);

            // ── minProbability ───────────────────────────────────────────────────
            // Mirrors C++ logic: RESOLUTION_FULL_PROBABILISTIC → read denoiser HitDistReconstructionMode.
            // We don't have NRD denoiser settings here, so default to OFF (0.0).
            float minProbability = 0.0f;
            // (For full fidelity, pass ReblurSettings/RelaxSettings.hitDistanceReconstructionMode in.)

            // ── onScreen offset (NRD_MODE < OCCLUSION in C++ → no offset needed) ──
            uint onScreen = (uint)settings.onScreen;


            var fovXRad               = math.atan(1.0f / proj.m00) * 2.0f;
            var horizontalFieldOfView = fovXRad * Mathf.Rad2Deg;


            // ── FOV-derived values using settings.camFov ─────────────────────────
            float tanPixelAngularRadius = math.tan(0.5f * math.radians(horizontalFieldOfView) / rectSize.x);
            float focalLength           = (0.5f * (35.0f * 0.001f)) / math.tan(math.radians(horizontalFieldOfView * 0.5f));

            uint sharcDownscale = 5;

            // ── gInvSharcRenderSize ───────────────────────────────────────────────
            // Mirrors GetSharcDims(): 16 * round_up16(renderRes / sharcDownscale)
            float sharcW = 16.0f * math.ceil(renderSize.x / (sharcDownscale * 16.0f)) * 16.0f / 16.0f;
            float sharcH = 16.0f * math.ceil(renderSize.y / (sharcDownscale * 16.0f)) * 16.0f / 16.0f;
            // Simpler: align to 16
            int    sharcDimX          = 16 * ((int)(renderResolution.x / sharcDownscale + 15) / 16);
            int    sharcDimY          = 16 * ((int)(renderResolution.y / sharcDownscale + 15) / 16);
            float2 invSharcRenderSize = new float2(1.0f / sharcDimX, 1.0f / sharcDimY);

            // ── gDisableShadowsAndEnableImportanceSampling ────────────────────────
            // C++: sunDirection.z < 0 && importanceSampling (NRD_MODE < OCCLUSION implied)
            uint disableShadowsAndImportanceSampling = (gSunDirection.y < 0.0f && settings.importanceSampling) ? 1u : 0u;

            // ── gPrevFrameConfidence ──────────────────────────────────────────────
            // C++: usePrevFrame && NRD_MODE < OCCLUSION && !RR && denoiser != REFERENCE
            float prevFrameConfidence = (settings.usePrevFrame && !settings.RR && settings.denoiser != DenoiserType.DENOISER_REFERENCE)
                ? prevFrameMaxAccum / (1.0f + prevFrameMaxAccum)
                : 0.0f;

            // ── gResolve ─────────────────────────────────────────────────────────
            // C++: (denoiser == REFERENCE || RR) ? false : m_Resolve
            uint resolve = (settings.denoiser == DenoiserType.DENOISER_REFERENCE || settings.RR) ? 0u : 1u;

            // ── gOrthoMode ───────────────────────────────────────────────────────
            // C++: (flags & PROJ_ORTHO) == 0 ? 0.0 : -1.0
            float orthoMode = cameraData.camera.orthographic ? 1.0f : 0f;

            // ── gTAA ─────────────────────────────────────────────────────────────
            float taa = (settings.denoiser != DenoiserType.DENOISER_REFERENCE && settings.TAA)
                ? 1.0f / (1.0f + taaMaxAccum)
                : 1.0f;

            // ── gSeparator ───────────────────────────────────────────────────────
            // C++: USE_SHARC_DEBUG == 0 ? m_Settings.separator : 1.0f  →  use separator directly
            float separator = settings.separator;

            // ── View direction ────────────────────────────────────────────────────
            // C++: float3(mViewToWorld[2].xyz) * (m_PositiveZ ? -1 : 1)  → camera forward (Unity = -Z)
            float3 viewDir = cameraData.camera.transform.forward; // Unity cameras look down -Z

            // ── Assemble constants ────────────────────────────────────────────────
            return new NRDGlobalConstants
            {
                gViewToWorld     = worldToView.inverse,
                gViewToClip      = viewToClip,
                gWorldToView     = worldToView,
                gWorldToClip     = worldToClip,
                gWorldToViewPrev = prevWorldToView,
                gWorldToClipPrev = prevWorldToClip,
                gViewToWorldPrev = prevWorldToView.inverse,

                gHitDistSettings     = hitDistSettings,
                gCameraFrustum       = GetNrdFrustum(cameraData),
                gSunBasisX           = new float4(gSunBasisX, 0.0f),
                gSunBasisY           = new float4(gSunBasisY, 0.0f),
                gSunDirection        = new float4(gSunDirection, 0.0f),
                gCameraGlobalPos     = new float4(camPos, 0.0f), // w=CAMERA_RELATIVE=true
                gCameraGlobalPosPrev = new float4(prevCamPos, 0.0f),
                gViewDirection       = new float4(viewDir, 0.0f),
                gHairBaseColor       = new float4(0.1f, 0.1f, 0.1f, 1.0f),
                gHairBetas           = new float2(0.25f, 0.3f),

                gOutputSize         = outputSize,
                gRenderSize         = renderSize,
                gRectSize           = rectSize,
                gInvOutputSize      = new float2(1.0f, 1.0f) / outputSize,
                gInvRenderSize      = new float2(1.0f, 1.0f) / renderSize,
                gInvRectSize        = new float2(1.0f, 1.0f) / rectSize,
                gRectSizePrev       = rectSizePrev,
                gInvSharcRenderSize = invSharcRenderSize,
                gJitter             = jitter,
                gJitterPrev         = jitterPrev,

                gEmissionIntensityLights = settings.emission ? settings.emissionIntensityLights : 0.0f,
                gEmissionIntensityCubes  = settings.emission ? 1.0f : 0.0f,
                gNearZ                   = -nearZ, // C++ uses signed NEAR_Z * meterToUnitsMultiplier; here we use proj-derived value
                gSeparator               = separator,
                gRoughnessOverride       = settings.roughnessOverride,
                gMetalnessOverride       = settings.metalnessOverride,
                gUnitToMetersMultiplier  = 1.0f / settings.meterToUnitsMultiplier,
                gTanSunAngularRadius     = math.tan(math.radians(settings.sunAngularDiameter * 0.5f)),
                gTanPixelAngularRadius   = tanPixelAngularRadius,
                gDebug                   = 0,
                gPrevFrameConfidence     = prevFrameConfidence,
                gUnproject               = 1.0f / (0.5f * rectH * project1),
                gAperture                = settings.dofAperture * 0.01f,
                gFocalDistance           = settings.dofFocalDistance,
                gFocalLength             = focalLength,
                gTAA                     = taa,
                gHdrScale                = 1.0f, // C++ reads from display descriptor; default to SDR
                gExposure                = settings.enableAutoExposure ? 1 : settings.fixExposure,
                gMipBias                 = mipBias,
                gOrthoMode               = orthoMode,
                gMinProbability          = minProbability,

                gMaxAccumulatedFrameNum                    = (uint)maxAccum,
                gDenoiserType                              = (uint)settings.denoiser,
                gDisableShadowsAndEnableImportanceSampling = disableShadowsAndImportanceSampling,
                gFrameIndex                                = frameIndex,
                gForcedMaterial                            = (uint)settings.forcedMaterial,
                gUseNormalMap                              = settings.normalMap ? 1u : 0u,
                gBounceNum                                 = (uint)settings.bounceNum,
                gResolve                                   = resolve,
                gValidation                                = settings.showValidation ? 1u : 0u,
                gSR                                        = settings.SR ? 1u : 0u,
                gRR                                        = settings.RR ? 1u : 0u,
                gIsSrgb                                    = 0u,
                gOnScreen                                  = onScreen,
                gTracingMode                               = settings.RR ? (uint)RESOLUTION.RESOLUTION_FULL_PROBABILISTIC : (uint)settings.tracingMode,
                gSampleNum                                 = (uint)settings.rpp,
                gPSR                                       = settings.PSR ? 1u : 0u,
                gSHARC                                     = settings.SHARC ? 1u : 0u,
                gTrimLobe                                  = 0u,
            };
        }
    }
}
