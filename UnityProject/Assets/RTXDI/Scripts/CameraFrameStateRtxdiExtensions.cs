using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using static PathTracing.PathTracingUtils;

namespace PathTracing
{
    /// <summary>RTXDI pipeline constant builder extracted from CameraFrameState (Core).</summary>
    public static class CameraFrameStateRtxdiExtensions
    {
        public static GlobalConstants GetConstants(this CameraFrameState cfs, RenderingData renderingData, RtxdiSetting settings)
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

            var emissionIntensity = 1;

            var minProbability     = 0.0f;

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
                gSeparator              = 0,
                gRoughnessOverride      = 0,
                gMetalnessOverride      = 0,
                gUnitToMetersMultiplier = 1.0f,
                gTanSunAngularRadius    = 0,
                gTanPixelAngularRadius  = math.tan(0.5f * math.radians(horizontalFieldOfView) / rectSize.x),
                gDebug                  = 0,
                gPrevFrameConfidence    = 0,
                gUnproject              = 1.0f / (0.5f * rectH * m11),
                gAperture               = 0,
                gFocalDistance          = 5,
                gFocalLength            = (0.5f * (35.0f * 0.001f)) / math.tan(math.radians(horizontalFieldOfView * 0.5f)),
                gTAA                    = 0,
                gHdrScale               = 1.0f,
                gExposure               = settings.exposure,
                gMipBias                = 0,
                gOrthoMode              = cameraData.camera.orthographic ? 1.0f : 0f,
                gIndirectDiffuse        = 0,
                gIndirectSpecular       = 0,
                gMinProbability         = minProbability,

                gSharcMaxAccumulatedFrameNum                 = 0,
                gDenoiserType                                = 0,
                gDisableShadowsAndEnableImportanceSampling   = 0,
                gFrameIndex                                  = (uint)Time.frameCount,
                gForcedMaterial                              = 0,
                gUseNormalMap                                = 1,
                gBounceNum                                   = 0,
                gResolve                                     = 1,
                gValidation                                  = 1,
                gSR                                          = 0,
                gRR                                          = 1,
                gIsSrgb                                      = 0,
                gOnScreen                                    = 0,
                gTracingMode                                 = 0,
                gSampleNum                                   = 0,
                gPSR                                         = 0,
                gSHARC                                       = 0,
                gTrimLobe                                    = 0,
                gSpotLightCount                              = 0,
                gAreaLightCount                              = 0,
                gPointLightCount                             = 0,
                gSssMinThreshold                             = 0,
                gSssTransmissionBsdfSampleCount              = 0,
                gSssTransmissionPerBsdfScatteringSampleCount = 0,
                gSssScale                                    = 0,
                gSssAnisotropy                               = 0,
                gSssMaxSampleRadius                          = 0,
                gIsEditor                                    = cameraData.camera.cameraType == CameraType.SceneView ? 1u : 0u,
                gShowLight                                   = 0,
                gSharcDownscale                              = 0,
                gSharcSceneScale                             = 0,
                sharcDebug                                   = 0,
            };

            return globalConstants;
        }
    }
}
