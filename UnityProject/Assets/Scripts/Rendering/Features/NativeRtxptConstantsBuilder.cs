using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Builds <see cref="SampleConstants"/> and <see cref="SimpleViewConstants"/>
    /// from Unity camera / rendering-data each frame.
    /// Separated from <see cref="NativeRtxptFeature"/> to keep the feature class concise.
    /// </summary>
    internal static class NativeRtxptConstantsBuilder
    {
        /// <summary>
        /// Fills a complete <see cref="SampleConstants"/> struct ready for GPU upload.
        /// </summary>
        public static SampleConstants Build(
            RenderingData renderingData,
            NativeRtxptSetting setting,
            int2 renderRes,
            int2 displayRes,
            CameraFrameState fs)
        {
            var  cam    = renderingData.cameraData.camera;
            var  xrPass = renderingData.cameraData.xr;
            bool isXR   = xrPass.enabled;

            var proj = GL.GetGPUProjectionMatrix(
                isXR ? xrPass.GetProjMatrix() : cam.projectionMatrix, false);

            // ── SimpleViewConstants ───────────────────────────────────────────
            var view     = BuildSimpleViewConstants(fs.worldToView, fs.viewToClip, fs.worldToClip,  renderRes, 1.0f, fs.viewportJitter);
            var prevView = BuildSimpleViewConstants(fs.prevWorldToView, fs.prevViewToClip, fs.prevWorldToClip, renderRes, 1.0f, fs.prevViewportJitter);

            // ── Camera geometry ───────────────────────────────────────────────
            float nearZ = proj.m23 / (proj.m22 - 1.0f);
            float farZ  = proj.m23 / (proj.m22 + 1.0f);

            // Falcor-style ray-gen orthonormal frame
            var   viewInv     = fs.worldToView.inverse;
            float tanHalfFovY = 1.0f / proj.m11;
            float tanHalfFovX = 1.0f / proj.m00;
            var   right       = new Vector3(viewInv.m00, viewInv.m10, viewInv.m20);
            var   up          = new Vector3(viewInv.m01, viewInv.m11, viewInv.m21);
            var   fwd         = new Vector3(-viewInv.m02, -viewInv.m12, -viewInv.m22);

            float focalDist   = math.max(setting.cameraFocalDistance, 1e-4f);
            float spreadAngle = 2.0f * math.atan(tanHalfFovY / renderRes.y);

            var camera = new PathTracerCameraData
            {
                PosW                 = new Vector3(fs.camPos.x, fs.camPos.y, fs.camPos.z),
                NearZ                = nearZ,
                DirectionW           = fwd,
                PixelConeSpreadAngle = spreadAngle,
                CameraU              = right * tanHalfFovX,
                FarZ                 = farZ,
                CameraV              = up * tanHalfFovY,
                FocalDistance        = focalDist,
                CameraW              = fwd,
                AspectRatio          = (float)renderRes.x / renderRes.y,
                ViewportSizeX        = (uint)renderRes.x,
                ViewportSizeY        = (uint)renderRes.y,
                ApertureRadius       = setting.cameraAperture,
                _padding0            = 0f,
                JitterX              = fs.viewportJitter.x,
                JitterY              = fs.viewportJitter.y,
                _padding1            = 0f,
                _padding2            = 0f,
            };

            // ── Previous-frame camera ─────────────────────────────────────────
            var prevViewInv = fs.prevWorldToView.inverse;
            var prevRight   = new Vector3(prevViewInv.m00, prevViewInv.m10, prevViewInv.m20);
            var prevUp      = new Vector3(prevViewInv.m01, prevViewInv.m11, prevViewInv.m21);
            var prevFwd     = new Vector3(-prevViewInv.m02, -prevViewInv.m12, -prevViewInv.m22);

            var prevCamera = new PathTracerCameraData
            {
                PosW                 = new Vector3(fs.prevCamPos.x, fs.prevCamPos.y, fs.prevCamPos.z),
                NearZ                = nearZ,
                DirectionW           = prevFwd,
                PixelConeSpreadAngle = spreadAngle,
                CameraU              = prevRight * tanHalfFovX,
                FarZ                 = farZ,
                CameraV              = prevUp * tanHalfFovY,
                FocalDistance        = focalDist,
                CameraW              = prevFwd,
                AspectRatio          = (float)renderRes.x / renderRes.y,
                ViewportSizeX        = (uint)renderRes.x,
                ViewportSizeY        = (uint)renderRes.y,
                ApertureRadius       = setting.cameraAperture,
                _padding0            = 0f,
                JitterX              = fs.prevViewportJitter.x,
                JitterY              = fs.prevViewportJitter.y,
                _padding1            = 0f,
                _padding2            = 0f,
            };

            bool isDlssRR = setting.realtimeAA == 3 && !setting.tmpDisableDlssRR;

            // ── PathTracerConstants ───────────────────────────────────────────
            var ptConsts = new PathTracerConstants
            {
                imageWidth                                   = (uint)renderRes.x,
                imageHeight                                  = (uint)renderRes.y,
                sampleBaseIndex                              = 0u,
                perPixelJitterAAScale                        = setting.realtimeAA != 0 ? 1.0f : 0.0f,
                bounceCount                                  = (uint)setting.bounceCount,
                diffuseBounceCount                           = (uint)setting.diffuseBounceCount,
                environmentMapDiffuseSampleMIPLevel          = 0f,
                texLODBias                                   = setting.texLODBias,
                invSubSampleCount                            = 1.0f / math.max(setting.realtimeSamplesPerPixel, 1),
                fireflyFilterThreshold                       = 0f,
                preExposedGrayLuminance                      = 1.0f,
                denoisingEnabled                             = isDlssRR ? 1u : 0u,
                frameIndex                                   = fs.frameIndex,
                useReSTIRDI                                  = 0u,
                useReSTIRGI                                  = 0u,
                _padding5                                    = 0u,
                stablePlanesSplitStopThreshold               = 0.05f,
                _padding3                                    = 0f,
                _padding4                                    = 0u,
                stablePlanesSuppressPrimaryIndirectSpecularK = 0f,
                denoiserRadianceClampK                       = isDlssRR ? setting.dlssrrBrightnessClampK : setting.denoiserRadianceClampK,
                dlssRRBrightnessClampK                       = setting.dlssrrBrightnessClampK,
                stablePlanesAntiAliasingFallthrough          = 0.04f,
                activeStablePlaneCount                       = (uint)setting.stablePlanesActiveCount,
                maxStablePlaneVertexDepth                    = 8u,
                allowPrimarySurfaceReplacement               = setting.allowPrimarySurfaceReplacement ? 1u : 0u,
                genericTSLineStride                          = (uint)renderRes.x,
                genericTSPlaneStride                         = (uint)(renderRes.x * renderRes.y),
                neeEnabled                                   = setting.useNEE ? 1u : 0u,
                neeType                                      = (uint)setting.neeType,
                neeCandidateSamples                          = (uint)setting.neeCandidateSamples,
                neeFullSamples                               = (uint)setting.neeFullSamples,
                _padding6                                    = 0u,
                stfMagnificationMethod                       = 0u,
                stfFilterMode                                = 0u,
                stfGaussianSigma                             = 0f,
                camera                                       = camera,
                prevCamera                                   = prevCamera,
            };

            // ── EnvMapSceneParams (identity — overridden by env-map baker pass) ─
            var envMapParams = new EnvMapSceneParams
            {
                TransformRow0    = new Vector4(1, 0, 0, 0),
                TransformRow1    = new Vector4(0, 1, 0, 0),
                TransformRow2    = new Vector4(0, 0, 1, 0),
                InvTransformRow0 = new Vector4(1, 0, 0, 0),
                InvTransformRow1 = new Vector4(0, 1, 0, 0),
                InvTransformRow2 = new Vector4(0, 0, 1, 0),
                colorMultiplier  = Vector3.one,
                enabled          = RenderSettings.skybox != null ? 1f : 0f,
            };

            var envMapIS = new EnvMapImportanceSamplingParams
            {
                importanceInvDimX = 0f,
                importanceInvDimY = 0f,
                importanceBaseMip = 0u,
                _padding0         = 0u,
            };

            // ── DebugConstants ────────────────────────────────────────────────
            var debug = new DebugConstants
            {
                pickX                     = -1,
                pickY                     = -1,
                pick                      = 0,
                debugLineScale            = 1f,
                showWireframe             = 0u,
                debugViewType             = 0,
                debugViewStablePlaneIndex = -1,
                exploreDeltaTree          = 0,
                imageWidth                = renderRes.x,
                imageHeight               = renderRes.y,
                mouseX                    = 0,
                mouseY                    = 0,
                cameraPosW                = new Vector3(fs.camPos.x, fs.camPos.y, fs.camPos.z),
                _padding0                 = 0f,
            };

            return new SampleConstants
            {
                view                           = view,
                previousView                   = prevView,
                envMapSceneParams              = envMapParams,
                envMapImportanceSamplingParams = envMapIS,
                ptConsts                       = ptConsts,
                debug                          = debug,
                denoisingHitParamConsts        = new Vector4(3f, 0.1f, 20f, -25f),
                materialCount                  = 0u,
                _padding0                      = 0u,
                _padding1                      = 0u,
                _padding2                      = 0u,
            };
        }

        // ─────────────────────────────────────────────────────────────────────
        private static SimpleViewConstants BuildSimpleViewConstants(
            Matrix4x4 worldToView,
            Matrix4x4 viewToClipNoOffset,
            Matrix4x4 worldToClipNoOffset,
            int2 renderResolution,
            float resolutionScale,
            float2 jitter)
        {
            var w = renderResolution.x * resolutionScale;
            var h = renderResolution.y * resolutionScale;
            var vSize = new float2(w, h);
            
            // 1. 计算偏移矩阵 (NDC 空间平移)
            float offsetX = 2f * jitter.x / w;
            float offsetY = -2f * jitter.y / h;
            
            // Unity Matrix4x4.Translate 创建的是列主序平移矩阵
            Matrix4x4 pixelOffsetMatrix    = Matrix4x4.Translate(new Vector3(offsetX, offsetY, 0));
            Matrix4x4 pixelOffsetMatrixInv = Matrix4x4.Translate(new Vector3(-offsetX, -offsetY, 0));

            // 2. 【关键修复】在 Unity 中，应用 NDC 偏移需要左乘 (Pre-multiply)
            // Clip_jittered = T_jitter * Clip_base
            var viewToClip  = pixelOffsetMatrix * viewToClipNoOffset;
            var worldToClip = pixelOffsetMatrix * worldToClipNoOffset;

            // 3. 计算逆矩阵
            // (T * P * V)^-1 = V^-1 * P^-1 * T^-1
            // 在 Unity 中 A * B 的逆是 B.inv * A.inv
            // var clipToViewNoOffset  = viewToClipNoOffset.inverse;
            var clipToWorldNoOffset = worldToClipNoOffset.inverse;
            
            var ctw_scale = new float2(0.5f * w, -0.5f * h);
            var ctw_bias  = new float2(0.5f * w, 0.5f * h);

            return new SimpleViewConstants
            {
                matWorldToView         = worldToView,
                matViewToClip          = viewToClip,
                matWorldToClip         = worldToClip,
                matWorldToClipNoOffset = worldToClipNoOffset,
                matClipToWorldNoOffset = clipToWorldNoOffset,
                viewportOrigin         = float2.zero,
                viewportSize           = vSize,
                viewportSizeInv        = math.rcp(vSize),
                pixelOffset            = jitter,
                clipToWindowScale      = ctw_scale,
                clipToWindowBias       = ctw_bias,
            };
        }
    }
}