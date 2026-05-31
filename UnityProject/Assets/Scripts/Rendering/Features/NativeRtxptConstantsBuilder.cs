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
            RtxptCameraFrameState fs,
            float preExposedGrayLuminance,
            uint materialCount)
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
            // ComputeRayPinhole (PathTracerHelpers.hlsli) consumes NearZ/FarZ purely as positive
            // eye-space distances along the optical axis: tMin = NearZ*invCos (ray pushed onto the
            // near plane), tMax = FarZ*invCos. They are NOT clip-space/NDC depth, so the old
            // proj.m23/(m22∓1) extraction (an OpenGL [-1,1] decode applied to the D3D reverse-Z
            // `proj`) was the wrong kind of value and yielded a negated near / lost far.
            // NearZ: positive near-plane distance — cam.nearClipPlane (handedness-independent).
            float nearZ = cam.nearClipPlane;
            // FarZ: primary-ray tMax only (no depth-precision role here). cam.farClipPlane exists
            // for raster depth precision and would clip distant geometry/env the reference still
            // traces; mirror RTXPT's effectively-unbounded far plane instead.
            float farZ  = 1e7f;

            // Falcor-style ray-gen orthonormal frame
            var   viewInv     = fs.worldToView.inverse;
            float tanHalfFovY = 1.0f / proj.m11;
            float tanHalfFovX = 1.0f / proj.m00;
            var   right       = new Vector3(viewInv.m00, viewInv.m10, viewInv.m20);
            var   up          = new Vector3(viewInv.m01, viewInv.m11, viewInv.m21);
            var   fwd         = new Vector3(-viewInv.m02, -viewInv.m12, -viewInv.m22);

            float focalDist   = math.max(setting.cameraFocalDistance, 1e-4f);
            // Matches BridgeCamera (PathTracerShared.h:133): atan(2 * tan(fovY/2) / viewportHeight).
            float spreadAngle = math.atan(2.0f * tanHalfFovY / renderRes.y);
            // Aspect ratio in BridgeCamera is the *display* (output) aspect, not render aspect.
            float displayAspect = (float)displayRes.x / displayRes.y;
            float ulen = focalDist * tanHalfFovY * displayAspect; // CameraU length
            float vlen = focalDist * tanHalfFovY;                 // CameraV length

            var camera = new PathTracerCameraData
            {
                PosW                 = new Vector3(fs.camPos.x, fs.camPos.y, fs.camPos.z),
                NearZ                = nearZ,
                DirectionW           = fwd,
                PixelConeSpreadAngle = spreadAngle,
                CameraU              = right * ulen,
                FarZ                 = farZ,
                CameraV              = up * vlen,
                FocalDistance        = focalDist,
                CameraW              = fwd * focalDist,
                AspectRatio          = displayAspect,
                ViewportSizeX        = (uint)renderRes.x,
                ViewportSizeY        = (uint)renderRes.y,
                ApertureRadius       = setting.cameraAperture,
                _padding0            = 0f,
                JitterX              = -fs.viewportJitter.x,
                JitterY              =  fs.viewportJitter.y,
                _padding1            = 0f,
                _padding2            = 0f,
            };

            // ── Previous-frame camera ─────────────────────────────────────────
            // Reference (Sample.cpp:2085) memsets the whole SampleConstants to 0, then
            // UpdatePathTracerConstants (Sample.cpp:1489) writes only ptConsts.camera —
            // ptConsts.prevCamera is never assigned, so it stays all-zero in the capture, and
            // no shader ever reads it (it's only declared in PathTracerShared.h). The fork
            // populated it, diverging from the source for a dead field; leave it default for parity.
            var prevCamera = default(PathTracerCameraData);

            // DLSS upscaling MIP bias (Sample.cpp:1496) — sharpens textures to compensate for upscale.
            float renderArea  = (float)renderRes.x   * renderRes.y;
            float displayArea = (float)displayRes.x  * displayRes.y;
            float dlssBias    = -math.log2(math.sqrt(displayArea / math.max(renderArea, 1f)));

            int spp = math.max(setting.realtimeSamplesPerPixel, 1);

            // Original (Sample.cpp:1444,1499): sampleIndex = frameIndex % 8192, sampleBaseIndex = sampleIndex * spp.
            uint sampleIndex     = setting.realtimeMode ? (fs.frameIndex % 8192u) : 0u;
            uint sampleBaseIndex = sampleIndex * (uint)spp;

            // preExposedGrayLuminance mirrors Sample.cpp:1508 (luminance(GetPreExposedGray(0)) when tone
            // mapping is on, else 1.0). Supplied by the caller from the tone-mapping pass's auto-exposure
            // read-back so the firefly/DLSS clamps below adapt to scene luminance like the original.

            // Original (Sample.cpp:1511-1514): scales with sqrt(preExposedGrayLuminance) * 1e3.
            float fireflyThreshold;
            if (setting.realtimeMode)
                fireflyThreshold = setting.realtimeFireflyFilterEnabled
                    ? setting.realtimeFireflyFilterThreshold * math.sqrt(preExposedGrayLuminance) * 1e3f
                    : 0f;
            else
                fireflyThreshold = setting.referenceFireflyFilterEnabled
                    ? setting.referenceFireflyFilterThreshold * math.sqrt(preExposedGrayLuminance) * 1e3f
                    : 0f;

            // Original (Sample.cpp:1518): DLSSRRBrightnessClampK *= preExposedGrayLuminance (else 0).
            float dlssRRClamp = setting.dlssrrBrightnessClampK > 0f
                ? setting.dlssrrBrightnessClampK * preExposedGrayLuminance
                : 0f;

            // ── PathTracerConstants ───────────────────────────────────────────
            var ptConsts = new PathTracerConstants
            {
                imageWidth                                   = (uint)renderRes.x,
                imageHeight                                  = (uint)renderRes.y,
                sampleBaseIndex                              = sampleBaseIndex,
                perPixelJitterAAScale                        = fs.perPixelJitterAAScale,
                bounceCount                                  = (uint)setting.bounceCount,
                diffuseBounceCount                           = (uint)setting.diffuseBounceCount,
                // Original (Sample.cpp:1539): sample a pre-filtered MIP for diffuse env lookups.
                // Hardcoding 0 forced full-res env fetches on every diffuse bounce — noisier and
                // higher bandwidth than the intended prefiltered level.
                environmentMapDiffuseSampleMIPLevel          = (float)setting.environmentMapDiffuseSampleMIPLevel,
                texLODBias                                   = setting.texLODBias + dlssBias,
                invSubSampleCount                            = 1.0f / spp,
                fireflyFilterThreshold                       = fireflyThreshold,
                preExposedGrayLuminance                      = preExposedGrayLuminance,
                // Original (Sample.cpp:1521): hardcoded 0 — this is the legacy stable-planes / NRD
                // guide flag and is unused by DLSS-RR (the entire RTXPT codebase never sets it non-zero).
                // The fork's `realtimeAA==DLSS-RR ? 1 : 0` diverged from the DLSS-RR reference, which captures 0.
                denoisingEnabled                             = 0u,
                frameIndex                                   = fs.frameIndex,
                useReSTIRDI                                  = 0u,
                useReSTIRGI                                  = 0u,
                _padding5                                    = 0u,
                // Original (Sample.cpp:1526) reads these straight from the UI; the fork hardcoded
                // them, silently ignoring the inspector and deviating from the original tuning.
                stablePlanesSplitStopThreshold               = setting.stablePlanesSplitStopThreshold,
                _padding3                                    = 0f,
                _padding4                                    = 0u,
                stablePlanesSuppressPrimaryIndirectSpecularK = setting.stablePlanesSuppressPrimaryIndirectSpecular
                                                                   ? setting.stablePlanesSuppressPrimaryIndirectSpecularK
                                                                   : 0f,
                denoiserRadianceClampK                       = setting.denoiserRadianceClampK,
                dlssRRBrightnessClampK                       = dlssRRClamp,
                stablePlanesAntiAliasingFallthrough          = setting.stablePlanesAntiAliasingFallthrough,
                activeStablePlaneCount                       = (uint)setting.stablePlanesActiveCount,
                // Original (Sample.cpp:1524): min(StablePlanesMaxVertexDepth, cStablePlaneMaxVertexIndex, BounceCount).
                // Hardcoding 8 both ignored the inspector and skipped the BounceCount clamp, so lowering
                // BounceCount no longer reduced stable-plane build depth as it does in the original.
                maxStablePlaneVertexDepth                    = (uint)math.min(
                                                                   math.min((uint)setting.stablePlanesMaxVertexDepth,
                                                                            PathTracerConfig.cStablePlaneMaxVertexIndex),
                                                                   (uint)setting.bounceCount),
                allowPrimarySurfaceReplacement               = setting.allowPrimarySurfaceReplacement ? 1u : 0u,
                // Tiled-swizzled addressing (TS_TILE_SIZE = 8 in Utils.hlsli).
                // Strides must be rounded up to the tile size, not raw image dims.
                genericTSLineStride                          = (uint)(((renderRes.x + 7) / 8) * 8),
                genericTSPlaneStride                         = (uint)((((renderRes.x + 7) / 8) * 8) * (((renderRes.y + 7) / 8) * 8)),
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

            // ── EnvMapSceneParams ─────────────────────────────────────────────
            // This is g_Const.envMapSceneParams, read by the path tracer (EnvMap.hlsli sample).
            // Nothing overrides it later, so ColorMultiplier must be set here just like the
            // original (Sample.cpp:1913): TintColor * (Intensity / c_envMapRadianceScale). The
            // divide cancels the constant compression scale baked into the cube
            // (NativeRtxptEnvMapBakerPass.EnvMapRadianceScale) → net radiance = source * tint * intensity.
            Color   envTintLin = setting.environmentMapTint.linear;
            float   envColMul  = setting.environmentMapIntensity / NativeRtxptEnvMapBakerPass.EnvMapRadianceScale;
            var envMapParams = new EnvMapSceneParams
            {
                TransformRow0    = new Vector4(1, 0, 0, 0),
                TransformRow1    = new Vector4(0, 1, 0, 0),
                TransformRow2    = new Vector4(0, 0, 1, 0),
                InvTransformRow0 = new Vector4(1, 0, 0, 0),
                InvTransformRow1 = new Vector4(0, 1, 0, 0),
                InvTransformRow2 = new Vector4(0, 0, 1, 0),
                colorMultiplier  = new Vector3(envTintLin.r, envTintLin.g, envTintLin.b) * envColMul,
                enabled          = 1f,  // always enabled; env cube is baked each frame (directional lights + optional skybox)
            };

            // ImportanceMapDim = 1024, mipLevels = 11 → ImportanceBaseMip = 10, InvDim = 1/1024
            const int importanceMapDim = 1024;
            var envMapIS = new EnvMapImportanceSamplingParams
            {
                importanceInvDimX = 1.0f / importanceMapDim,
                importanceInvDimY = 1.0f / importanceMapDim,
                importanceBaseMip = 10u,   // log2(1024) = 10, i.e. mip 10 is 1×1
                _padding0         = 0u,
            };

            // ── DebugConstants ────────────────────────────────────────────────
            var debug = new DebugConstants
            {
                pickX                     = -1,
                pickY                     = -1,
                pick                      = 0,
                // Reference (RTVersionGConst capture) leaves this 0 — it is the debug-line draw
                // scale and there is no setting plumbed for it, so 1f was a stray non-default that
                // diverged from the source. Keep debug drawing off for parity.
                debugLineScale            = 0f,
                showWireframe             = 0u,
                debugViewType             = (int)(setting.showMode == NativeRtxptShowMode.NEELightColor
                                                ? RtxptDebugViewType.NEELightColor
                                                : setting.debugViewType),
                debugViewStablePlaneIndex = setting.debugViewStablePlaneIndex,
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
                // Original (Sample.cpp:2127): zero-initialized. Read only by the NRD ReBLUR
                // denoiser front-end packing (PostProcess.hlsl:544, #else branch), which never
                // runs under DLSS-RR — so it's a dead value there. The fork hardcoded the NRD
                // ReBLUR default {3, 0.1, 20, -25}; matching the source keeps DLSS-RR parity.
                denoisingHitParamConsts        = Vector4.zero,
                materialCount                  = materialCount,
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
