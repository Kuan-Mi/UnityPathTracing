using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using NativeRender;
using PathTracing.NativeInterop.NRI;
using PathTracing.NativeInterop.Streamline;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// ScriptableRendererFeature for the RTXPT (Path Tracing with Stable Planes + DLSS-RR) pipeline.
    ///
    /// Pass execution order:
    ///   Phase 0 : RtxptBuildTlasPass              - TLAS rebuild
    ///   Phase 1 : LightsBaker passes                    - env map / emissive / proxies / feedback (TODO)
    ///   Phase 2a: RtxptBuildStablePlanesPass       - BuildStablePlanes RT (PathTracePrePass)
    ///   Phase 2b: RtxptExportVisibilityBufferPass  - depth + motion vectors export
    ///   Phase 2c: RtxptLightingUpdateEndPass       - NEE-AT feedback processing (stub)
    ///   Phase 2d: RtxptFillStablePlanesPass        - FillStablePlanes RT (PathTrace) / Reference
    ///   Phase 3 : RtxptDenoiseSpecHitTPass         - specular hit-distance bilateral filter x2
    ///   Phase 4 : RtxptDlssBeforePass              - prepare DLSS-RR guide buffers
    ///   Phase 5 : SLDlssrrPass                           - DLSS Ray Reconstruction (denoise + upscale, via Streamline)
    ///   Phase 7 : RtxptAccumulationPass            - multi-frame accumulation (reference mode only)
    ///
    /// PT_USE_RESTIR_DI = 0, PT_USE_RESTIR_GI = 0 (no RTXDI).
    /// cStablePlaneCount = 3.
    /// </summary>
    public class RtxptFeature : ScriptableRendererFeature
    {
        // ---- Inspector fields -----------------------------------------------
        public RtxptSetting    setting;
        public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        public SampleConstants sampleConstants;

        // Phase 2a/2d: PathTracer RT shaders
        public LazyLoadReference<RayTraceShader> buildStablePlanesShader;
        public LazyLoadReference<RayTraceShader> fillStablePlanesShader;
        public LazyLoadReference<RayTraceShader> referenceShader;

        // Phase 2a/2d: per-pipeline extra hit-group blobs
        public LazyLoadReference<HitGroupShader>[] buildHitGroups;
        public LazyLoadReference<HitGroupShader>[] fillHitGroups;
        public LazyLoadReference<HitGroupShader>[] referenceHitGroups;

        // Phase 0: skinned-repack compute (Unity skinned VB → per-instance donut SoA buffer),
        // recorded by the TLAS build pass before the BLAS refit each frame. A plain Unity
        // ComputeShader: both src/dst are Unity GraphicsBuffers, bound without GetNativeBufferPtr.
        public ComputeShader skinnedRepackCs;

        // Phase 3
        public LazyLoadReference<NativeComputeShader> exportVisibilityBufferCs;

        // Phase 4
        public LazyLoadReference<NativeComputeShader> denoiseSpecHitTCs;

        // Phase 5
        public LazyLoadReference<NativeComputeShader> dlssBeforeCs;

        // Phase 6: Bloom (donut BloomPass: downsample → separable blur → composite).
        // Verbatim raster wrappers: blur = fullscreen_vs + passes/bloom_ps.hlsl; downsample/composite =
        // fullscreen_vs + blit_ps.hlsl (composite uses the constant-color blend = donut's ConstantColor).
        public LazyLoadReference<NativeRasterShader> bloomDownsampleRasterShader;
        public LazyLoadReference<NativeRasterShader> bloomBlurRasterShader;
        public LazyLoadReference<NativeRasterShader> bloomCompositeRasterShader;

        // Phase 7: Tone mapping — faithful native replica of RTXPT ToneMappingPasses.cpp, built from the
        // original shaders as thin verbatim #include wrappers: luminance_ps (raster) → donut
        // mipmapgen_cs mip reduce (shared mipMapGenCs asset) → capture_cs → CPU read-back →
        // ToneMapping main_ps (raster).
        public LazyLoadReference<NativeRasterShader>  luminanceRasterShader; // Luminance.rastershader (= fullscreen_vs + luminance_ps)
        public LazyLoadReference<NativeComputeShader> captureLuminanceCs; // capture_cs.computeshader (= ToneMapping.hlsl capture_cs)
        public LazyLoadReference<NativeRasterShader>  toneMapApplyRasterShader; // ToneMapping.rastershader (= fullscreen_vs + main_ps)

        // Phase 8
        public LazyLoadReference<NativeComputeShader> accumulationCs;

        // Phase Debug: StablePlanesDebugViz
        public LazyLoadReference<NativeComputeShader> stablePlanesDebugVizCs;

        // Phase Debug: ShaderDebug (DebugPrint readback, DebugLine/DebugTriangle draw, viz overlay).
        // Ports of Libraries/ShaderDebug/ShaderDebug.hlsl draw shaders + the Sample.cpp debug-lines draw.
        public LazyLoadReference<NativeRasterShader> shaderDebugTrianglesRasterShader;
        public LazyLoadReference<NativeRasterShader> shaderDebugLinesRasterShader;
        public LazyLoadReference<NativeRasterShader> shaderDebugFeedbackLinesRasterShader;
        public LazyLoadReference<NativeRasterShader> shaderDebugBlendVizRasterShader;

        // EnvMapBaker + LightingUpdateBegin compute shaders
        public LazyLoadReference<NativeComputeShader> baseLayerCs;
        public LazyLoadReference<NativeComputeShader> mipReduceCs;
        public LazyLoadReference<NativeComputeShader> envMapImportanceBakerCs;
        public LazyLoadReference<NativeComputeShader> mipMapGenCs; // mipmapgen_cs.computeshader (donut MipMapGenPass MODE_COLOR)
        public LazyLoadReference<NativeComputeShader> bc6uCompressCs;
        public LazyLoadReference<NativeComputeShader> envLightsBackupPastCs;
        public LazyLoadReference<NativeComputeShader> envLightsSubdivideBaseCs;
        public LazyLoadReference<NativeComputeShader> envLightsSubdivideBoostCs;
        public LazyLoadReference<NativeComputeShader> envLightsFillLookupMapCs;
        public LazyLoadReference<NativeComputeShader> envLightsMapPastToCurrentCs;
        public LazyLoadReference<NativeComputeShader> resetLightProxyCountersCs;
        public LazyLoadReference<NativeComputeShader> resetPastToCurrentHistoryCs;
        public LazyLoadReference<NativeComputeShader> computeWeightsCs;
        public LazyLoadReference<NativeComputeShader> computeProxyCountsCs;
        public LazyLoadReference<NativeComputeShader> computeProxyBaselineOffsetsCs;
        public LazyLoadReference<NativeComputeShader> createProxyJobsCs;
        public LazyLoadReference<NativeComputeShader> executeProxyJobsCs;
        public LazyLoadReference<NativeComputeShader> bakeEmissiveTrianglesCs;

        // LightingUpdateBegin feedback passes
        public LazyLoadReference<NativeComputeShader> processFeedbackHistoryPreFilterCs;

        public LazyLoadReference<NativeComputeShader> processFeedbackHistoryP0Cs;

        // LightingUpdateEnd feedback passes
        public LazyLoadReference<NativeComputeShader> processFeedbackHistoryP1aCs;
        public LazyLoadReference<NativeComputeShader> processFeedbackHistoryP1bCs;
        public LazyLoadReference<NativeComputeShader> processFeedbackHistoryP2Cs;
        public LazyLoadReference<NativeComputeShader> processFeedbackHistoryP3Cs;
        public LazyLoadReference<NativeComputeShader> clearFeedbackHistoryCs;

        // LightsBaker debug passes (LightsBaker::DebugGUI — draw all lights / NEE-AT debug viz)
        public LazyLoadReference<NativeComputeShader> debugDrawLightsCs;
        public LazyLoadReference<NativeComputeShader> processFeedbackHistoryDebugVizCs;

        // Phase 9: Output blit (debug display)
        public Material outputBlitMaterial;

        // ---- Pass instances -------------------------------------------------
        private RtxptBuildTlasPass              _buildTlasPass;
        private RtxptEnvMapBakerPass            _envMapBakerPass;
        private RtxptLightingUpdateBeginPass    _lightingUpdateBeginPass;
        private RtxptBuildStablePlanesPass      _buildStablePlanesPass;
        private RtxptExportVisibilityBufferPass _exportVisibilityBufferPass;
        private RtxptLightingUpdateEndPass      _lightingUpdateEndPass;
        private RtxptFillStablePlanesPass       _fillStablePlanesPass;
        private RtxptDenoisingGuidesBakePass    _denoisingGuidesBakePass;
        private RtxptDlssRRPrepareInputsPass    _dlssRrPrepareInputsPass;
        private SLDlssrrPass                    _dlssRRPass;
        private SLDlssgInputsPass               _slDlssgInputsPass;
        private bool                            _slFgLastEnabled;
        private RtxptBloomPass                  _bloomPass;
        private RtxptToneMappingMipChainPass    _toneMappingMipChainPass;
        private RtxptAccumulationPass           _accumulationPass;
        private RtxptStablePlanesDebugVizPass   _stablePlanesDebugVizPass;
        private RtxptShaderDebugBeginPass       _shaderDebugBeginPass;
        private RtxptShaderDebugDrawPass        _shaderDebugDrawPass;
        private RtxptOutputBlitPass             _outputBlitPass;


        private DepthBarrierFixPass _depthBarrierFixPass;

        // ---- Shared scene resources -----------------------------------------
        private RtxptGPUScene _gpuScene;

        /// <summary>Editor access for the "Info and statistics:" block (LightsBaker::InfoGUI).</summary>
        public RtxptLightingUpdateBeginPass LightingUpdateBeginPass => _lightingUpdateBeginPass;

        /// <summary>
        /// Sets setting.debugPixelX/Y from a normalized viewport position (top-left origin, [0,1])
        /// for the given camera, converting to that camera's RENDER resolution (debug-pixel coords
        /// are pre-upscale, so the DLSS scale is applied here). Unity-side equivalent of the C++
        /// right-click DebugPixel pick. Returns false until the camera has rendered at least one
        /// RTXPT frame (the render resolution isn't known before that).
        /// </summary>
        public bool TrySetDebugPixelFromViewport(Camera cam, Vector2 uvTopLeft)
        {
            if (setting == null || cam == null) return false;
            if (!_cameraFrameStates.TryGetValue(cam.GetInstanceID(), out var fs) || fs.renderResolution.x <= 0)
                return false;
            var rr = fs.renderResolution;
            setting.debugPixelX = Mathf.Clamp((int)(uvTopLeft.x * rr.x), 0, rr.x - 1);
            setting.debugPixelY = Mathf.Clamp((int)(uvTopLeft.y * rr.y), 0, rr.y - 1);
            return true;
        }

        // ---- Per-camera resource pools (key = instanceID + eyeIndex*100000) -
        private readonly Dictionary<long, RtxptTextureResources>  _texturePools      = new();
        private readonly Dictionary<long, RtxptBufferResources>   _bufferPools       = new();
        private readonly Dictionary<long, VolatileConstantBuffer> _constantBuffers   = new();
        private readonly Dictionary<long, SLDlssrr>               _dlrrDenoisers     = new();
        private readonly Dictionary<long, RtxptCameraFrameState>  _cameraFrameStates = new();

        // ---- Lifecycle ------------------------------------------------------

        public IntPtr blackTexturePtr;

        public override void Create()
        {
            setting         ??= new RtxptSetting();
            blackTexturePtr =   Texture2D.blackTexture.GetNativeTexturePtr();
        }

        private void CreatePasses()
        {
            _buildTlasPass ??= new RtxptBuildTlasPass(skinnedRepackCs)
            {
                renderPassEvent = renderPassEvent,
            };

            _envMapBakerPass ??= new RtxptEnvMapBakerPass(baseLayerCs.asset, mipReduceCs.asset, envMapImportanceBakerCs.asset, mipMapGenCs.asset, bc6uCompressCs.asset)
            {
                renderPassEvent = renderPassEvent,
            };

            _lightingUpdateBeginPass ??= new RtxptLightingUpdateBeginPass(
                    envLightsBackupPastCs.asset, envLightsSubdivideBaseCs.asset, envLightsSubdivideBoostCs.asset,
                    envLightsFillLookupMapCs.asset, envLightsMapPastToCurrentCs.asset,
                    resetLightProxyCountersCs.asset, resetPastToCurrentHistoryCs.asset,
                    computeWeightsCs.asset, computeProxyCountsCs.asset, computeProxyBaselineOffsetsCs.asset,
                    createProxyJobsCs.asset, executeProxyJobsCs.asset,
                    bakeEmissiveTrianglesCs.asset,
                    processFeedbackHistoryPreFilterCs.asset, processFeedbackHistoryP0Cs.asset,
                    debugDrawLightsCs.asset)
                { renderPassEvent = renderPassEvent };

            _buildStablePlanesPass      ??= new RtxptBuildStablePlanesPass(buildStablePlanesShader.asset, ResolveHitGroups(buildHitGroups)) { renderPassEvent = renderPassEvent };
            _exportVisibilityBufferPass ??= new RtxptExportVisibilityBufferPass(exportVisibilityBufferCs.asset) { renderPassEvent                             = renderPassEvent };

            _lightingUpdateEndPass ??= new RtxptLightingUpdateEndPass(
                    processFeedbackHistoryP1aCs.asset, processFeedbackHistoryP1bCs.asset,
                    processFeedbackHistoryP2Cs.asset, processFeedbackHistoryP3Cs.asset,
                    clearFeedbackHistoryCs.asset,
                    processFeedbackHistoryDebugVizCs.asset)
                { renderPassEvent = renderPassEvent };

            _fillStablePlanesPass     ??= new RtxptFillStablePlanesPass(fillStablePlanesShader.asset, referenceShader.asset, ResolveHitGroups(fillHitGroups), ResolveHitGroups(referenceHitGroups)) { renderPassEvent = renderPassEvent };
            _denoisingGuidesBakePass  ??= new RtxptDenoisingGuidesBakePass(denoiseSpecHitTCs.asset) { renderPassEvent                                                                                                 = renderPassEvent };
            _dlssRrPrepareInputsPass  ??= new RtxptDlssRRPrepareInputsPass(dlssBeforeCs.asset) { renderPassEvent                                                                                                      = renderPassEvent };
            _dlssRRPass               ??= new SLDlssrrPass { renderPassEvent                                                                                                                                          = renderPassEvent };
            _slDlssgInputsPass        ??= new SLDlssgInputsPass { renderPassEvent                                                                                                                                     = renderPassEvent };
            _bloomPass                ??= new RtxptBloomPass(bloomDownsampleRasterShader.asset, bloomBlurRasterShader.asset, bloomCompositeRasterShader.asset) { renderPassEvent                                      = renderPassEvent };
            _toneMappingMipChainPass  ??= new RtxptToneMappingMipChainPass(luminanceRasterShader.asset, mipMapGenCs.asset, captureLuminanceCs.asset, toneMapApplyRasterShader.asset) { renderPassEvent                = renderPassEvent };
            _accumulationPass         ??= new RtxptAccumulationPass(accumulationCs.asset) { renderPassEvent                                                                                                           = renderPassEvent };
            _stablePlanesDebugVizPass ??= new RtxptStablePlanesDebugVizPass(stablePlanesDebugVizCs.asset) { renderPassEvent                                                                                           = renderPassEvent };
            // The begin pass also clears the debug-viz texture via the plugin's UAV-clear event
            // (ClearUnorderedAccessViewFloat, matching the original's nvrhi clearTextureFloat).
            _shaderDebugBeginPass ??= new RtxptShaderDebugBeginPass { renderPassEvent = renderPassEvent };
            if (_shaderDebugDrawPass == null
                && shaderDebugTrianglesRasterShader.asset != null && shaderDebugLinesRasterShader.asset != null
                && shaderDebugFeedbackLinesRasterShader.asset != null && shaderDebugBlendVizRasterShader.asset != null)
                _shaderDebugDrawPass = new RtxptShaderDebugDrawPass(
                    shaderDebugTrianglesRasterShader.asset, shaderDebugLinesRasterShader.asset,
                    shaderDebugFeedbackLinesRasterShader.asset, shaderDebugBlendVizRasterShader.asset) { renderPassEvent = renderPassEvent };
            _outputBlitPass      ??= new RtxptOutputBlitPass(outputBlitMaterial) { renderPassEvent = renderPassEvent };
            _depthBarrierFixPass ??= new DepthBarrierFixPass { renderPassEvent                     = RenderPassEvent.AfterRendering };
        }

        private RtxptPassContext passCtx;

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            var cam = renderingData.cameraData.camera;
            if (cam.cameraType is CameraType.Preview or CameraType.Reflection) return;
            if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView) return;

            CreatePasses();

            cam.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.MotionVectors;

            var eyeIndex = renderingData.cameraData.xr.enabled
                ? renderingData.cameraData.xr.multipassId
                : 0;

            if (eyeIndex == 1 && setting.skipRightEyeInVR) return;

            // ---- Shared scene resources -------------------------------------
            _gpuScene ??= new RtxptGPUScene();

            if (eyeIndex == 0)
            {
                _gpuScene.UpdateForFrame();
            }

            // ---- Per-camera resource lookup / creation ----------------------
            var  uniqueKey = cam.GetInstanceID() + (eyeIndex * 100_000L);
            bool isVR      = renderingData.cameraData.xrRendering;

            if (!_texturePools.TryGetValue(uniqueKey, out var texPool))
            {
                texPool = new RtxptTextureResources();
                _texturePools.Add(uniqueKey, texPool);
            }

            if (!_bufferPools.TryGetValue(uniqueKey, out var bufPool))
            {
                bufPool = new RtxptBufferResources();
                _bufferPools.Add(uniqueKey, bufPool);
            }

            if (!_dlrrDenoisers.TryGetValue(uniqueKey, out var dlrr))
            {
                dlrr = new SLDlssrr(isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name);
                _dlrrDenoisers.Add(uniqueKey, dlrr);
            }

            if (!_constantBuffers.TryGetValue(uniqueKey, out var constantBuffer))
            {
                // Name mirrors the original RTXPT main constants buffer (Sample.cpp "SampleConstants").
                constantBuffer = new VolatileConstantBuffer(Marshal.SizeOf<SampleConstants>(), "SampleConstants");
                _constantBuffers.Add(uniqueKey, constantBuffer);
            }

            // ---- Resolution -------------------------------------------------
            var displayResolution = ComputeOutputResolution(renderingData.cameraData);
            var dlssRrMode        = NormalizeDlssRrMode(setting.upscalerMode);
            var renderResolution  = ComputeRenderResolution(displayResolution, dlssRrMode);

            // DLSS / DLSS-RR require an output of at least 32x32 (NGX minimum). In a
            // packaged build the window can momentarily come up at a degenerate size
            // (e.g. 144x1 while it is being created/resized), which makes the native
            // nri CreateUpscaler fail. Skip the whole path-tracing pass list for this
            // camera until it has a usable resolution.
            if (math.cmin(displayResolution) < MinUpscalerResolution ||
                math.cmin(renderResolution) < MinUpscalerResolution)
                return;

            bool texturesChanged = texPool.EnsureResources(renderResolution, displayResolution);
            texPool.EnsureEnvMapResources();
            bufPool.EnsureResources(renderResolution);
            bufPool.EnsureLightBuffers();
            bufPool.EnsureEnvBakerBuffers();
            // ShaderDebug buffer (~100 MB, mirrors the original's fixed layout) only lives while
            // the master toggle is on; the original allocates it unconditionally.
            bufPool.EnsureShaderDebugBuffers(setting.enableShaderDebug);

            // ---- Per-camera temporal state ----------------------------------
            if (!_cameraFrameStates.TryGetValue(uniqueKey, out var frameState))
            {
                frameState = new RtxptCameraFrameState(1.0f);
                _cameraFrameStates.Add(uniqueKey, frameState);
            }

            if (texturesChanged)
            {
                frameState.renderResolution = renderResolution;
                frameState.frameIndex       = 0;
            }

            // ---- Gather scene lights once (shared by env baker + lighting pass) ----
            // Both consumers previously each ran their own full-scene FindObjectsByType<Light>;
            // gathering once here halves that per-camera cost.
            var sceneLights = UnityEngine.Object.FindObjectsByType<Light>(FindObjectsSortMode.None);

            // ---- Resolve URP Volume overrides -------------------------------
            // Apply optional per-scene / per-region Volume overrides for environment-map and exposure
            // settings. Clone-on-write: `effectiveSetting` stays the authored `setting` until an
            // override is actually active, so the serialized inspector object is never mutated. Done
            // before PrepareBake so env-map overrides drive this frame's bake gating.
            var effectiveSetting = setting;
            RtxptEnvironmentMapVolume.ApplyOverrides(ref effectiveSetting, setting);
            RtxptExposureVolume.ApplyOverrides(ref effectiveSetting, setting);

            // Env bake gating must be decided before frameState.Update so a re-bake resets
            // accumulation this frame — original: Sample.cpp:1383, if (m_envMapBaker->Update(...))
            // m_ui.ResetAccumulation = true.
            bool envMapChanged = _envMapBakerPass.PrepareBake(effectiveSetting, sceneLights, texPool);

            frameState.Update(renderingData, texturesChanged || envMapChanged, 1.0f, effectiveSetting);

            // ---- Build & upload SampleConstants -----------------------------
            // Pre-exposed gray luminance comes from the tone-mapping pass's auto-exposure read-back
            // (populated by the previous frame's GPU→CPU copy, mirroring the original's 1-2 frame lag).
            // The pass instance persists across frames, so its last captured value is available here even
            // though this frame's tone-mapping pass hasn't recorded yet.
            float preExposedGrayLuminance = _toneMappingMipChainPass.GetPreExposedGrayLuminance(effectiveSetting);
            // MaterialCount (Sample.cpp:2095 = GetMaterialDataCount): shaders bounds-check material
            // indices against it (Bridge::loadIoR / loadHomogeneousVolumeData) — 0 disables IoR/volumes.
            sampleConstants = RtxptConstantsBuilder.Build(renderingData, effectiveSetting, renderResolution, displayResolution, frameState, preExposedGrayLuminance, _gpuScene.MaterialDataCount);

            constantBuffer.Upload(renderer, sampleConstants);

            // ---- Build shared pass context ----------------------------------
            passCtx ??= new RtxptPassContext();

            passCtx.ConstantBuffer    = constantBuffer;
            passCtx.GpuScene          = _gpuScene;
            passCtx.Textures          = texPool;
            passCtx.Buffers           = bufPool;
            passCtx.RenderResolution  = renderResolution;
            passCtx.DisplayResolution = displayResolution;
            passCtx.FrameState        = frameState;
            passCtx.Setting           = effectiveSetting;
            passCtx.SceneLights       = sceneLights;
            passCtx.blackTexturePtr   = blackTexturePtr;

            passCtx.ResolveNativePtrs();

            // ---- Phase 0: TLAS ---------------------------------------------
            if (eyeIndex == 0)
            {
                _buildTlasPass.Setup(_gpuScene,
                    _buildStablePlanesPass?.BuildPipeline,
                    _fillStablePlanesPass?.FillPipeline,
                    _fillStablePlanesPass?.RefPipeline);
                renderer.EnqueuePass(_buildTlasPass);
            }

            // ---- ShaderDebug::BeginFrame — reset header (counters + worldToClip) -----
            if (setting.enableShaderDebug && bufPool.ShaderDebugBuffer != null)
            {
                _shaderDebugBeginPass.Setup(passCtx);
                renderer.EnqueuePass(_shaderDebugBeginPass);
            }

            // ---- LightingUpdateBegin -----------------------------------------
            // Unified pass: EnvMapBaker → EnvLightsBaker → ProxyBuild
            // Correct order mirrors original RTXPT LightsBaker::UpdateFrame front half.
            _envMapBakerPass.Setup(passCtx);
            renderer.EnqueuePass(_envMapBakerPass);

            _lightingUpdateBeginPass.Setup(passCtx);
            renderer.EnqueuePass(_lightingUpdateBeginPass);

            // ---- Phase 2a: BuildStablePlanes RT (PathTracePrePass) ----------
            _buildStablePlanesPass.Setup(passCtx);
            renderer.EnqueuePass(_buildStablePlanesPass);

            // ---- Phase 2b: ExportVisibilityBuffer ---------------------------
            // Outputs Depth + MotionVectors needed by LightingUpdateEnd.
            _exportVisibilityBufferPass.Setup(passCtx);
            renderer.EnqueuePass(_exportVisibilityBufferPass);

            // ---- Phase 2c: LightingUpdateEnd --------------------------------
            // NEE-AT feedback processing: builds per-tile LocalSamplingBuffer
            // for FillStablePlanes, then clears feedback for current frame.
            _lightingUpdateEndPass.Setup(passCtx);
            renderer.EnqueuePass(_lightingUpdateEndPass);

            // ---- Phase 2d: FillStablePlanes RT (PathTrace) / Reference ------
            _fillStablePlanesPass.Setup(passCtx);
            renderer.EnqueuePass(_fillStablePlanesPass);

            // ---- Realtime-only phases (3-5): DLSS-RR path -------------------
            if (setting.realtimeMode)
            {
                // Phase 3: DenoiseSpecHitT x2
                _denoisingGuidesBakePass.Setup(passCtx);
                renderer.EnqueuePass(_denoisingGuidesBakePass);

                // Phase 4: DlssBefore
                _dlssRrPrepareInputsPass.Setup(passCtx);
                renderer.EnqueuePass(_dlssRrPrepareInputsPass);

                // Phase 5: DLSS-RR
                {
                    var dlrrInput = new SLDlssrr.SLDlssrrFrameInput
                    {
                        worldToView             = frameState.worldToView,
                        viewToClip              = frameState.viewToClip,
                        worldToClip             = frameState.worldToClip,
                        prevWorldToClip         = frameState.prevWorldToClip,
                        camPos                  = frameState.camPos,
                        viewportJitter          = frameState.viewportJitter,
                        renderResolution        = renderResolution,
                        outputResolution        = displayResolution,
                        frameIndex              = frameState.frameIndex,
                        cameraNear              = cam.nearClipPlane,
                        cameraFar               = cam.farClipPlane,
                        cameraFOV               = cam.fieldOfView * Mathf.Deg2Rad,
                        cameraAspect            = cam.aspect,
                        useSpecularMotionVector = true, // RTXPT supplies specular motion vectors
                        reset                   = false
                    };
                    var dlrrRes = new SLDlssrr.DlssrrResources
                    {
                        input              = texPool.OutputColor,
                        output             = texPool.DlssRrOutput,
                        mv                 = texPool.ScreenMotionVectors,
                        depth              = texPool.Depth,
                        diffAlbedo         = texPool.DlssRrDiffAlbedo,
                        specAlbedo         = texPool.DlssRrSpecAlbedo,
                        normalRoughness    = texPool.DlssRrNormalRoughness,
                        specularMvOrHitTex = texPool.DlssRrSpecMotionVectors,
                    };
                    _dlssRRPass.Setup(
                        dlrr.GetInteropDataPtr(dlrrInput, dlrrRes, dlssRrMode, setting.dlssRRPreset),
                        new SLDlssrrPass.Settings { tmpDisableRR = setting.tmpDisableDlssRR });
                    renderer.EnqueuePass(_dlssRRPass);
                }

                // Phase 6: Bloom on the display-res HDR DLSS-RR output (composited in place, before tone mapping).
                if (setting.enableBloom && setting.bloomIntensity > 0f && setting.bloomRadius > 0f && _bloomPass != null)
                {
                    _bloomPass.Setup(passCtx, texPool.DlssRrOutput);
                    renderer.EnqueuePass(_bloomPass);
                }

                // Phase 7: Tone mapping + auto-exposure on the display-res HDR DLSS-RR output.
                // Writes the LDR result into ProcessedOutputColor (unused elsewhere in realtime mode).
                if (setting.enableToneMapping && _toneMappingMipChainPass != null)
                {
                    _toneMappingMipChainPass.Setup(passCtx, texPool.DlssRrOutput, texPool.ProcessedOutputColor);
                    renderer.EnqueuePass(_toneMappingMipChainPass);
                }
            }
            else
            {
                // Phase 7: Accumulation (reference mode)
                _accumulationPass.Setup(passCtx);
                renderer.EnqueuePass(_accumulationPass);
            }

            PushDlssgFrameInputs(renderer, cam, frameState, texPool, displayResolution, renderResolution, texturesChanged, effectiveSetting, eyeIndex);

            // ---- Phase Debug: StablePlanesDebugViz (only when a debug view is active) ----
            if (setting.debugViewType != RtxptDebugViewType.Disabled && stablePlanesDebugVizCs.asset != null)
            {
                _stablePlanesDebugVizPass.Setup(passCtx);
                renderer.EnqueuePass(_stablePlanesDebugVizPass);
            }

            // ---- Phase Debug: ShaderDebug draw + readback (C++ EndFrameAndOutput) ----
            // Drawn into the image the blit will display (= C++ drawing on LdrColor pre-blit).
            if (setting.enableShaderDebug && _shaderDebugDrawPass != null && bufPool.ShaderDebugBuffer != null)
            {
                bool ranToneMap = setting.realtimeMode && setting.enableToneMapping && _toneMappingMipChainPass != null;
                var debugTarget = !setting.realtimeMode || ranToneMap
                    ? texPool.ProcessedOutputColor
                    : texPool.DlssRrOutput;
                _shaderDebugDrawPass.Setup(passCtx, debugTarget, displayResolution);
                renderer.EnqueuePass(_shaderDebugDrawPass);
            }

            // ---- Phase 9: Output blit (debug display) ----------------------
            {
                // When tone mapping ran, the final image lives in ProcessedOutputColor; show it in
                // place of the raw HDR DLSS-RR output unless the user picked an explicit debug view.
                var  displayMode = setting.showMode;
                bool toneMapRan  = setting.realtimeMode && setting.enableToneMapping && _toneMappingMipChainPass != null;
                if (toneMapRan && displayMode == RtxptShowMode.DlssRrOutput)
                    displayMode = RtxptShowMode.ProcessedOutput;

                _outputBlitPass.Setup(texPool, displayMode, 1.0f, setting.debugViewType);
                renderer.EnqueuePass(_outputBlitPass);
            }

            renderer.EnqueuePass(_depthBarrierFixPass);
        }

        // ---- Helpers -------------------------------------------------------

        private void PushDlssgFrameInputs(ScriptableRenderer renderer, Camera cam, RtxptCameraFrameState frameState,
            RtxptTextureResources texPool, int2 displayResolution, int2 renderResolution,
            bool texturesChanged, RtxptSetting effectiveSetting, int eyeIndex)
        {
            if (eyeIndex != 0)
                return;

            bool enableFg = effectiveSetting.FG && effectiveSetting.realtimeMode;
            if (enableFg != _slFgLastEnabled)
            {
                SLDlssg.SetFrameGeneration(enableFg);
                _slFgLastEnabled = enableFg;
            }

            if (!enableFg)
                return;

            var fgEventFunc = SLDlssg.GetFrameInputsEventFunc();
            if (fgEventFunc == IntPtr.Zero)
                return;

            if (texPool?.Depth == null || texPool.ScreenMotionVectors == null)
                return;

            var depthPtr = texPool.Depth.NativePtr;
            var mvPtr    = texPool.ScreenMotionVectors.NativePtr;
            if (depthPtr == IntPtr.Zero || mvPtr == IntPtr.Zero)
                return;

            Matrix4x4 viewToWorld        = frameState.worldToView.inverse;
            Matrix4x4 clipToCameraView   = frameState.viewToClip.inverse;
            Matrix4x4 worldToClipInv     = frameState.worldToClip.inverse;
            Matrix4x4 prevWorldToClipInv = frameState.prevWorldToClip.inverse;
            Matrix4x4 clipToPrevClip     = frameState.prevWorldToClip * worldToClipInv;
            Matrix4x4 prevClipToClip     = frameState.worldToClip * prevWorldToClipInv;

            var camRight = new Vector3(viewToWorld.m00, viewToWorld.m10, viewToWorld.m20);
            var camUp    = new Vector3(viewToWorld.m01, viewToWorld.m11, viewToWorld.m21);
            var camFwd   = -new Vector3(viewToWorld.m02, viewToWorld.m12, viewToWorld.m22);

            float renderW = math.max(1, renderResolution.x);
            float renderH = math.max(1, renderResolution.y);

            var fg = new SLDlssg.FrameInputs
            {
                depth         = depthPtr,
                motionVectors = mvPtr,
                frameToken    = SLStreamlineFrameLoop.CurrentFrameTokenPtr,

                depthState = SLDlssg.D3D12_STATE_UNORDERED_ACCESS,
                mvecState  = SLDlssg.D3D12_STATE_UNORDERED_ACCESS,

                mvecDepthW = (uint)renderResolution.x,
                mvecDepthH = (uint)renderResolution.y,
                colorW     = (uint)displayResolution.x,
                colorH     = (uint)displayResolution.y,

                cameraViewToClip = frameState.viewToClip,
                clipToCameraView = clipToCameraView,
                clipToPrevClip   = clipToPrevClip,
                prevClipToClip   = prevClipToClip,

                cameraPos   = new Vector3(frameState.camPos.x, frameState.camPos.y, frameState.camPos.z),
                cameraUp    = camUp,
                cameraRight = camRight,
                cameraFwd   = camFwd,

                jitterX    = frameState.viewportJitter.x,
                jitterY    = frameState.viewportJitter.y,
                mvecScaleX = 1.0f / renderW,
                mvecScaleY = 1.0f / renderH,

                cameraNear   = cam.nearClipPlane,
                cameraFar    = cam.farClipPlane,
                cameraFOV    = cam.fieldOfView * Mathf.Deg2Rad,
                cameraAspect = cam.aspect,

                depthInverted        = 1,
                cameraMotionIncluded = 1,
                motionVectors3D      = 0,
                reset                = (texturesChanged || frameState.frameIndex < 2) ? 1 : 0,
            };

            var fgPtr = SLDlssg.GetInteropDataPtr(fg, frameState.frameIndex);
            _slDlssgInputsPass.Setup(fgEventFunc, fgPtr);
            renderer.EnqueuePass(_slDlssgInputsPass);
        }

        // Resolves a serialized array of lazy hit-group references to the concrete assets the
        // RT passes expect, loading each on demand. Returns null for an unset array.
        private static HitGroupShader[] ResolveHitGroups(LazyLoadReference<HitGroupShader>[] refs)
        {
            if (refs == null) return null;
            var result = new HitGroupShader[refs.Length];
            for (int i = 0; i < refs.Length; i++)
                result[i] = refs[i].asset;
            return result;
        }

        // Minimum render/output size DLSS (NGX) will accept; below this CreateUpscaler fails.
        private const int MinUpscalerResolution = 32;

        private static int2 ComputeOutputResolution(CameraData cameraData) =>
            new int2(cameraData.cameraTargetDescriptor.width,
                cameraData.cameraTargetDescriptor.height);

        private static UpscalerMode NormalizeDlssRrMode(UpscalerMode mode)
        {
            // Streamline's DLSS-RR/DLSSD path fails NGX feature creation for eUltraQuality in
            // the bundled SDK/runtime. Match RTXPT's supported RR mode set by using Quality.
            return mode == UpscalerMode.ULTRA_QUALITY ? UpscalerMode.QUALITY : mode;
        }

        private static int2 ComputeRenderResolution(int2 outputRes, UpscalerMode mode)
        {
            // Query NGX for the SDK-recommended render resolution, matching RTXPT's
            // QueryDLSSOptimalSettings path. Results are cached in the native plugin.
            if (SLDlssrr.TryGetOptimalRenderSize(outputRes, mode, out var nativeRes))
                return nativeRes;
            Debug.LogError($"[RtxptFeature] Failed to get optimal render size from native plugin for mode {mode}, falling back to hardcoded scale table.");

            // Fallback (native plugin unavailable): hardcoded scale table.
            float scale = mode switch
            {
                UpscalerMode.NATIVE => 1.0f,
                UpscalerMode.ULTRA_QUALITY => 1.3f,
                UpscalerMode.QUALITY => 1.5f,
                UpscalerMode.BALANCED => 1.7f,
                UpscalerMode.PERFORMANCE => 2.0f,
                UpscalerMode.ULTRA_PERFORMANCE => 3.0f,
                _ => 1.0f,
            };
            return new int2((int)(outputRes.x / scale + 0.5f),
                (int)(outputRes.y / scale + 0.5f));
        }

        // ---- Cleanup -------------------------------------------------------

        protected override void Dispose(bool disposing)
        {
            if (!disposing) return;

            _buildTlasPass?.Dispose();
            _buildTlasPass = null;
            _lightingUpdateBeginPass?.Dispose();
            _lightingUpdateBeginPass = null;
            _envMapBakerPass?.Dispose();
            _envMapBakerPass = null;
            _buildStablePlanesPass?.Dispose();
            _buildStablePlanesPass = null;
            _fillStablePlanesPass?.Dispose();
            _fillStablePlanesPass = null;
            _lightingUpdateEndPass?.Dispose();
            _lightingUpdateEndPass = null;
            _exportVisibilityBufferPass?.Dispose();
            _exportVisibilityBufferPass = null;
            _denoisingGuidesBakePass?.Dispose();
            _denoisingGuidesBakePass = null;
            _dlssRrPrepareInputsPass?.Dispose();
            _dlssRrPrepareInputsPass = null;
            _slDlssgInputsPass       = null;
            _bloomPass?.Dispose();
            _bloomPass = null;
            _toneMappingMipChainPass?.Dispose();
            _toneMappingMipChainPass = null;
            _accumulationPass?.Dispose();
            _accumulationPass = null;
            _stablePlanesDebugVizPass?.Dispose();
            _stablePlanesDebugVizPass = null;
            _shaderDebugDrawPass?.Dispose();
            _shaderDebugDrawPass = null;
            _shaderDebugBeginPass?.Dispose();
            _shaderDebugBeginPass = null;
            _outputBlitPass       = null;
            _depthBarrierFixPass  = null;


            foreach (var p in _texturePools.Values) p.Dispose();
            _texturePools.Clear();
            foreach (var p in _bufferPools.Values) p.Dispose();
            _bufferPools.Clear();
            foreach (var cb in _constantBuffers.Values) cb.Dispose();
            _constantBuffers.Clear();
            foreach (var d in _dlrrDenoisers.Values) d?.Dispose();
            _dlrrDenoisers.Clear();
            _cameraFrameStates.Clear();

            _gpuScene?.Dispose();
            _gpuScene = null;
        }

        // ---- Editor helpers ----------------------------------------------------

#if UNITY_EDITOR
        private void Reset()
        {
            setting = new RtxptSetting();
            AutoFillShaders();
        }

        /// <summary>
        /// Debug readback: logs all emissive triangle light entries from the GPU LightBuffer.
        /// Run the scene for at least one frame before clicking.
        /// </summary>
        public void TestEmissiveTriangles()
        {
            if (_lightingUpdateBeginPass == null)
            {
                Debug.LogWarning("[RtxptFeature] LightingUpdateBeginPass not created — run the scene first.");
                return;
            }

            RtxptBufferResources buf = null;
            foreach (var kv in _bufferPools)
            {
                buf = kv.Value;
                break;
            }

            if (buf?.LightBuffer == null)
            {
                Debug.LogWarning("[RtxptFeature] LightBuffer is null — run the scene first.");
                return;
            }

            uint triOffset = _lightingUpdateBeginPass.EmissiveLightOffset;
            uint triCount  = _lightingUpdateBeginPass.EmissiveTriangleCount;

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"[Rtxpt Emissive Triangles] offset={triOffset}  count={triCount}  analyticCount={triOffset - 5368}  MaxLights={RtxptBufferResources.MaxLights}");

            // ---- CPU-side: SubInstanceData EmissiveLightMappingOffset ----
            if (_gpuScene != null)
            {
                var emissive = _gpuScene.GetEmissiveGeometries();
                sb.AppendLine($"  Emissive geometries ({emissive.Count}):");
                foreach (var e in emissive)
                    sb.AppendLine($"    inst={e.InstanceIndex} geom={e.GeometrySubIndex}  triCount={e.TriangleCount}");
            }

            // ---- GPU readback: LightBuffer entries at emissive range ----
            if (triCount == 0)
            {
                sb.AppendLine("  No emissive triangles collected last frame.");
                Debug.Log(sb.ToString());
                return;
            }

            // GPU readback is async (fence-gated): record the copy on a one-shot command buffer,
            // then poll across editor frames until the GPU has produced the data, and visualize.
            int readCount = (int)System.Math.Min(triCount, 4096u);
            var data      = new RtxptPolymorphicLightInfo[readCount];

            var rbCmd = new CommandBuffer { name = "Rtxpt_LightBufferReadback" };
            buf.LightBuffer.RequestReadback(rbCmd, (int)triOffset, readCount);
            Graphics.ExecuteCommandBuffer(rbCmd);
            rbCmd.Release();

#if UNITY_EDITOR
            var                                            lightBuffer  = buf.LightBuffer;
            int                                            framesWaited = 0;
            const int                                      maxFrames    = 180;
            UnityEditor.EditorApplication.CallbackFunction poll         = null;
            poll = () =>
            {
                framesWaited++;
                if (lightBuffer.TryGetReadback(data, readCount))
                {
                    UnityEditor.EditorApplication.update -= poll;
                    LogEmissiveTriangleReadback(sb, data, readCount, triOffset, triCount);
                }
                else if (framesWaited > maxFrames)
                {
                    UnityEditor.EditorApplication.update -= poll;
                    sb.AppendLine($"  Readback timed out after {maxFrames} frames (is the scene rendering?).");
                    Debug.Log(sb.ToString());
                }
            };
            UnityEditor.EditorApplication.update += poll;
#else
            sb.AppendLine("  Readback polling is editor-only.");
            Debug.Log(sb.ToString());
#endif
        }

        // Decodes and visualizes the emissive-triangle light entries read back from LightBuffer.
        private static void LogEmissiveTriangleReadback(
            System.Text.StringBuilder sb, RtxptPolymorphicLightInfo[] data, int readCount, uint triOffset, uint triCount)
        {
            int   nonZero  = 0;
            float drawTime = 30f; // seconds visible in scene view
            var   segments = new List<(Vector3 a, Vector3 b, Color c)>();

            for (int i = 0; i < readCount; i++)
            {
                var  info   = data[i];
                uint logRad = info.LogRadiance & 0xFFFFu;
                float intensity = (logRad == 0)
                    ? 0f
                    : Unity.Mathematics.math.exp2(((logRad - 1) / 65534f) * 48f - 8f);
                if (intensity < 0.001f) continue;
                nonZero++;

                // Decode color (RGB8 in bits 0-23 of ColorTypeAndFlags)
                float normR = (info.ColorTypeAndFlags & 0xFFu) / 255f;
                float normG = ((info.ColorTypeAndFlags >> 8) & 0xFFu) / 255f;
                float normB = ((info.ColorTypeAndFlags >> 16) & 0xFFu) / 255f;
                var   col   = new Color(normR, normG, normB, 1f); // use hue only, not intensity (too large)

                // Direction1/Direction2/Scalars pack edge1.xyz and edge2.xyz as fp16 pairs:
                //   low  16 bits → edge1 component,  high 16 bits → edge2 component
                // Center = base + (edge1 + edge2) / 3  (triangle centroid)
                var center = new Vector3(info.CenterX, info.CenterY, info.CenterZ);
                var edge1 = new Vector3(
                    Mathf.HalfToFloat((ushort)(info.Direction1 & 0xFFFFu)),
                    Mathf.HalfToFloat((ushort)(info.Direction2 & 0xFFFFu)),
                    Mathf.HalfToFloat((ushort)(info.Scalars & 0xFFFFu)));
                var edge2 = new Vector3(
                    Mathf.HalfToFloat((ushort)((info.Direction1 >> 16) & 0xFFFFu)),
                    Mathf.HalfToFloat((ushort)((info.Direction2 >> 16) & 0xFFFFu)),
                    Mathf.HalfToFloat((ushort)((info.Scalars >> 16) & 0xFFFFu)));
                var normal  = Vector3.Cross(edge1, edge2).normalized;
                var triBase = center - (edge1 + edge2) / 3f; // recover base vertex

                // Triangle outline + normal
                float size = 0.05f;
                segments.Add((triBase, triBase + edge1, col));
                segments.Add((triBase, triBase + edge2, col));
                segments.Add((triBase + edge1, triBase + edge2, col));
                segments.Add((center, center + normal * (size * 3f), col));

                if (nonZero <= 16)
                {
                    uint typeCode = (info.ColorTypeAndFlags >> 24) & 0xFu;
                    sb.AppendLine($"    [{(int)triOffset + i}] type={typeCode}  intensity={intensity:F3}" +
                                  $"  center=({info.CenterX:F2},{info.CenterY:F2},{info.CenterZ:F2})" +
                                  $"  normal=({normal.x:F2},{normal.y:F2},{normal.z:F2})" +
                                  $"  e1=({edge1.x:F2},{edge1.y:F2},{edge1.z:F2})");
                }
            }

            sb.AppendLine($"  Non-zero entries in first {readCount}: {nonZero}  (draws visible for {drawTime}s in Scene view)");
            if (triCount > (uint)readCount)
                sb.AppendLine($"  (only first {readCount} of {triCount} entries read back)");

            Debug.Log(sb.ToString());

#if UNITY_EDITOR
            ShowDebugSegments(segments, drawTime);
#endif
        }

#if UNITY_EDITOR
        // Editor-safe debug line drawing. Debug.DrawLine is unreliable outside play mode, so we
        // draw the collected segments with Handles via SceneView.duringSceneGui, forcing repaints
        // until the expiry time, then unhook. Visible in the Scene view regardless of play state.
        private static List<(Vector3 a, Vector3 b, Color c)> s_debugSegments;
        private static double                                s_debugExpiry;
        private static bool                                  s_debugHooked;

        private static void ShowDebugSegments(List<(Vector3 a, Vector3 b, Color c)> segments, float seconds)
        {
            s_debugSegments = segments;
            s_debugExpiry   = UnityEditor.EditorApplication.timeSinceStartup + seconds;

            if (!s_debugHooked)
            {
                s_debugHooked                        =  true;
                UnityEditor.SceneView.duringSceneGui += OnSceneGuiDrawDebug;
                UnityEditor.EditorApplication.update += RepaintWhileDebugging;
            }

            UnityEditor.SceneView.RepaintAll();
        }

        private static void OnSceneGuiDrawDebug(UnityEditor.SceneView sv)
        {
            if (s_debugSegments == null) return;
            foreach (var (a, b, c) in s_debugSegments)
            {
                UnityEditor.Handles.color = c;
                UnityEditor.Handles.DrawLine(a, b);
            }
        }

        private static void RepaintWhileDebugging()
        {
            if (UnityEditor.EditorApplication.timeSinceStartup < s_debugExpiry)
            {
                UnityEditor.SceneView.RepaintAll();
                return;
            }

            // Expired — unhook and clear.
            UnityEditor.SceneView.duringSceneGui -= OnSceneGuiDrawDebug;
            UnityEditor.EditorApplication.update -= RepaintWhileDebugging;
            s_debugHooked                        =  false;
            s_debugSegments                      =  null;
        }
#endif

        /// <summary>
        /// Debug readback for the NEE-AT data path.
        /// Run with setting.neeType = NEEAT for a few frames before clicking.
        /// </summary>
        public void TestNeeAtReadback()
        {
            RtxptBufferResources  buf = null;
            RtxptTextureResources tex = null;

            foreach (var kv in _bufferPools)
            {
                buf = kv.Value;
                break;
            }

            foreach (var kv in _texturePools)
            {
                tex = kv.Value;
                break;
            }

            if (buf?.LightControlBuffer == null || tex == null)
            {
                Debug.LogWarning("[RtxptFeature] NEE-AT readback unavailable - run the scene for a few frames first.");
                return;
            }

            var  ctrl                 = buf.LastLightControlData;
            uint invalidFeedbackCount = ReadUIntAt(buf.LightProxyCounters, (int)ctrl.TotalLightCount);
            uint derivedValidFeedback = ctrl.TotalMaxFeedbackCount > invalidFeedbackCount
                ? ctrl.TotalMaxFeedbackCount - invalidFeedbackCount
                : 0u;

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("[Rtxpt NEE-AT Readback]");
            sb.AppendLine($"  Inspector: useNEE={setting?.useNEE} neeType={setting?.neeType} candidateSamples={setting?.neeCandidateSamples} fullSamples={setting?.neeFullSamples}");
            sb.AppendLine($"  Control: importanceType={ctrl.ImportanceSamplingType} temporalRequired={ctrl.TemporalFeedbackRequired} lastTemporal={ctrl.LastFrameTemporalFeedbackAvailable} lastLocal={ctrl.LastFrameLocalSamplesAvailable}");
            sb.AppendLine($"  Lights: total={ctrl.TotalLightCount} env={ctrl.EnvmapQuadNodeCount} analytic={ctrl.AnalyticLightCount} emissiveTriangles={ctrl.TriangleLightCount}");
            sb.AppendLine($"  Proxies: samplingProxyCount={ctrl.SamplingProxyCount} proxyBuildTaskCount={ctrl.ProxyBuildTaskCount} validFeedback={derivedValidFeedback}/{ctrl.TotalMaxFeedbackCount} invalidFeedback={invalidFeedbackCount}");
            sb.AppendLine($"  LocalSampling: resolution={ctrl.LocalSamplingResolutionX}x{ctrl.LocalSamplingResolutionY} tileBufferHeight={ctrl.TileBufferHeight} bufferCount={buf.LocalSamplingBuffer?.count ?? 0}");
            sb.AppendLine(
                $"  Feedback: resolution={ctrl.BakerConstants.FeedbackResolutionX}x{ctrl.BakerConstants.FeedbackResolutionY} blended={ctrl.BakerConstants.BlendedFeedbackResolutionX}x{ctrl.BakerConstants.BlendedFeedbackResolutionY}");
            sb.AppendLine(
                $"  Weights: currentOffset={ctrl.BakerConstants.CurrentWeightsBufferOffset} historicOffset={ctrl.BakerConstants.HistoricWeightsBufferOffset} globalFeedbackWeight={ctrl.GlobalFeedbackUseWeight:F3} localToGlobal={ctrl.LocalToGlobalSampleRatio:F3}");

            AppendFloatBufferStats(sb, "LightWeights.current", buf.LightWeightsBuffer, (int)ctrl.BakerConstants.CurrentWeightsBufferOffset, (int)Math.Min(ctrl.TotalLightCount + 1u, 4096u));
            AppendUIntBufferStats(sb, "LightProxyCounters", buf.LightProxyCounters, 0, (int)Math.Min(ctrl.TotalLightCount + 1u, 4096u));
            AppendUIntBufferStats(sb, "LightSamplingProxies", buf.LightSamplingProxies, 0, Mathf.Min(buf.LightSamplingProxies?.count ?? 0, 4096));
            AppendUIntBufferStats(sb, "LocalSamplingBuffer", buf.LocalSamplingBuffer, 0, Mathf.Min(buf.LocalSamplingBuffer?.count ?? 0, 4096));

            AppendFloatTextureStats(sb, "FeedbackTotalWeight", tex.LightFeedbackTotalWeight, 8192);
            AppendUIntTextureStats(sb, "FeedbackCandidates", tex.LightFeedbackCandidates, 8192);
            AppendFloatTextureStats(sb, "FeedbackTotalWeightBlended", tex.FeedbackTotalWeightBlended, 8192);
            AppendUIntTextureStats(sb, "FeedbackCandidatesBlended", tex.FeedbackCandidatesBlended, 8192);

            if (ctrl.ImportanceSamplingType != 2 || ctrl.TemporalFeedbackRequired == 0)
                sb.AppendLine("  RESULT: NEE-AT is not enabled in LightingControl. Set RtxptSetting.neeType = NEEAT and useNEE = true.");
            else if (ctrl.SamplingProxyCount == 0)
                sb.AppendLine("  RESULT: NEE-AT config is active, but proxy data is empty. Check light counts, env/emissive setup, and proxy build passes.");
            else if (ctrl.LastFrameTemporalFeedbackAvailable == 0)
                sb.AppendLine("  RESULT: First-frame warmup state. Wait a few frames, then read back again.");
            else
                sb.AppendLine("  RESULT: NEE-AT control/proxy/local-sampling path is producing data.");

            Debug.Log(sb.ToString());
        }

        private static void AppendFloatBufferStats(System.Text.StringBuilder sb, string label, GraphicsBuffer buffer, int start, int count)
        {
            if (buffer == null || count <= 0 || start < 0 || start >= buffer.count)
            {
                sb.AppendLine($"  {label}: unavailable");
                return;
            }

            count = Mathf.Min(count, buffer.count - start);
            var data = new float[count];
            buffer.GetData(data, 0, start, count);

            int    nonZero = 0;
            float  min     = float.PositiveInfinity;
            float  max     = float.NegativeInfinity;
            double sum     = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                float v = data[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) continue;
                if (Mathf.Abs(v) > 1e-8f) nonZero++;
                min =  Mathf.Min(min, v);
                max =  Mathf.Max(max, v);
                sum += v;
            }

            if (float.IsInfinity(min)) min = max = 0.0f;
            sb.AppendLine($"  {label}: read={count} nonZero={nonZero} min={min:G4} max={max:G4} sum={sum:G5}");
        }

        private static uint ReadUIntAt(GraphicsBuffer buffer, int index)
        {
            if (buffer == null || index < 0 || index >= buffer.count)
                return 0u;

            var data = new uint[1];
            buffer.GetData(data, 0, index, 1);
            return data[0];
        }

        private static void AppendUIntBufferStats(System.Text.StringBuilder sb, string label, GraphicsBuffer buffer, int start, int count)
        {
            if (buffer == null || count <= 0 || start < 0 || start >= buffer.count)
            {
                sb.AppendLine($"  {label}: unavailable");
                return;
            }

            count = Mathf.Min(count, buffer.count - start);
            var data = new uint[count];
            buffer.GetData(data, 0, start, count);
            AppendUIntStats(sb, label, data);
        }

        private static void AppendFloatTextureStats(System.Text.StringBuilder sb, string label, NriTextureResource tex, int maxSamples)
        {
            if (tex?.Handle == null)
            {
                sb.AppendLine($"  {label}: unavailable");
                return;
            }

            var req = AsyncGPUReadback.Request(tex.Handle);
            req.WaitForCompletion();
            if (req.hasError)
            {
                sb.AppendLine($"  {label}: readback error");
                return;
            }

            var    data    = req.GetData<float>();
            int    count   = Mathf.Min(data.Length, maxSamples);
            int    nonZero = 0;
            float  min     = float.PositiveInfinity;
            float  max     = float.NegativeInfinity;
            double sum     = 0.0;
            for (int i = 0; i < count; i++)
            {
                float v = data[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) continue;
                if (Mathf.Abs(v) > 1e-8f) nonZero++;
                min =  Mathf.Min(min, v);
                max =  Mathf.Max(max, v);
                sum += v;
            }

            if (float.IsInfinity(min)) min = max = 0.0f;
            sb.AppendLine($"  {label}: read={count}/{data.Length} nonZero={nonZero} min={min:G4} max={max:G4} sum={sum:G5}");
        }

        private static void AppendUIntTextureStats(System.Text.StringBuilder sb, string label, NriTextureResource tex, int maxSamples)
        {
            if (tex?.Handle == null)
            {
                sb.AppendLine($"  {label}: unavailable");
                return;
            }

            var req = AsyncGPUReadback.Request(tex.Handle);
            req.WaitForCompletion();
            if (req.hasError)
            {
                sb.AppendLine($"  {label}: readback error");
                return;
            }

            var data   = req.GetData<uint>();
            int count  = Mathf.Min(data.Length, maxSamples);
            var sample = new uint[count];
            for (int i = 0; i < count; i++)
                sample[i] = data[i];
            AppendUIntStats(sb, label, sample, data.Length);
        }

        private static void AppendUIntStats(System.Text.StringBuilder sb, string label, uint[] data, int totalCount = -1)
        {
            int   nonZero    = 0;
            int   invalid    = 0;
            uint  min        = uint.MaxValue;
            uint  max        = 0;
            ulong sumLowBits = 0;
            for (int i = 0; i < data.Length; i++)
            {
                uint v = data[i];
                if (v != 0) nonZero++;
                if (v == 0xffffffffu) invalid++;
                if (v < min) min = v;
                if (v > max) max = v;
                sumLowBits += v & 0xffffu;
            }

            if (data.Length == 0) min = 0;
            string suffix             = totalCount >= 0 ? $"/{totalCount}" : "";
            sb.AppendLine($"  {label}: read={data.Length}{suffix} nonZero={nonZero} invalid=0xFFFFFFFF:{invalid} min={min} max={max} sumLow16={sumLowBits}");
        }


        public void AutoFillShaders()
        {
            const string shaderRoot = "Assets/RTXPT/Shaders";

            // Phase 2a/2d: PathTracer RT shaders
            buildStablePlanesShader              = LoadRs($"{shaderRoot}/BuildStablePlanes");
            fillStablePlanesShader               = LoadRs($"{shaderRoot}/FillStablePlanes");
            referenceShader                      = LoadRs($"{shaderRoot}/Reference");
            buildHitGroups = new[]
            {
                Ref(LoadHg($"{shaderRoot}/BuildSP_HitNonEmissive")),
                Ref(LoadHg($"{shaderRoot}/BuildSP_HitOpaqueNonEmissive")),
            };
            fillHitGroups = new[]
            {
                Ref(LoadHg($"{shaderRoot}/FillSP_HitNonEmissiveNameExperiment")),
                Ref(LoadHg($"{shaderRoot}/FillSP_HitOpaqueNonEmissive")),
            };
            referenceHitGroups = new[]
            {
                Ref(LoadHg($"{shaderRoot}/Reference_HitNonEmissive")),
                Ref(LoadHg($"{shaderRoot}/Reference_HitOpaqueNonEmissive")),
            };
            exportVisibilityBufferCs             = LoadCs($"{shaderRoot}/ProcessingPasses/ExportVisibilityBuffer");
            denoiseSpecHitTCs                    = LoadCs($"{shaderRoot}/ProcessingPasses/DenoisingGuidesBaker_DenoiseSpecHitT");
            dlssBeforeCs                         = LoadCs($"{shaderRoot}/ProcessingPasses/PostProcess_DenoiserPrepareInputsDlssRR");
            bloomDownsampleRasterShader          = LoadRas($"{shaderRoot}/ToneMapper/BloomDownsample");
            bloomBlurRasterShader                = LoadRas($"{shaderRoot}/ToneMapper/BloomBlur");
            bloomCompositeRasterShader           = LoadRas($"{shaderRoot}/ToneMapper/BloomComposite");
            luminanceRasterShader                = LoadRas($"{shaderRoot}/ToneMapper/Luminance");
            captureLuminanceCs                   = LoadCs($"{shaderRoot}/ToneMapper/capture_cs");
            toneMapApplyRasterShader             = LoadRas($"{shaderRoot}/ToneMapper/ToneMapping");
            accumulationCs                       = LoadCs($"{shaderRoot}/ProcessingPasses/AccumulationPass");
            stablePlanesDebugVizCs               = LoadCs($"{shaderRoot}/ProcessingPasses/PostProcess_StablePlanesDebugViz");
            shaderDebugTrianglesRasterShader     = LoadRas($"{shaderRoot}/ShaderDebug/ShaderDebugTriangles");
            shaderDebugLinesRasterShader         = LoadRas($"{shaderRoot}/ShaderDebug/ShaderDebugLines");
            shaderDebugFeedbackLinesRasterShader = LoadRas($"{shaderRoot}/ShaderDebug/ShaderDebugFeedbackLines");
            shaderDebugBlendVizRasterShader      = LoadRas($"{shaderRoot}/ShaderDebug/ShaderDebugBlendViz");
            skinnedRepackCs                      = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>($"{shaderRoot}/Misc/SkinnedRepack.compute");
            if (skinnedRepackCs == null)
                Debug.LogWarning($"[RtxptFeature] Missing ComputeShader at: {shaderRoot}/Misc/SkinnedRepack.compute");

            string lightRoot   = $"{shaderRoot}/Lighting";
            string distantRoot = $"{lightRoot}/Distant";
            baseLayerCs             = LoadCs($"{distantRoot}/BaseLayerCS");
            mipReduceCs             = LoadCs($"{distantRoot}/MIPReduceCS");
            envMapImportanceBakerCs = LoadCs($"{distantRoot}/EnvMapImportanceSamplingBaker");
            mipMapGenCs             = LoadCs($"{distantRoot}/mipmapgen_cs");
            bc6uCompressCs          = LoadCs($"{distantRoot}/BC6UCompress");

            resetLightProxyCountersCs     = LoadCs($"{lightRoot}/ResetLightProxyCounters");
            resetPastToCurrentHistoryCs   = LoadCs($"{lightRoot}/ResetPastToCurrentHistory");
            computeWeightsCs              = LoadCs($"{lightRoot}/ComputeWeights");
            computeProxyCountsCs          = LoadCs($"{lightRoot}/ComputeProxyCounts");
            computeProxyBaselineOffsetsCs = LoadCs($"{lightRoot}/ComputeProxyBaselineOffsets");
            createProxyJobsCs             = LoadCs($"{lightRoot}/CreateProxyJobs");
            executeProxyJobsCs            = LoadCs($"{lightRoot}/ExecuteProxyJobs");
            bakeEmissiveTrianglesCs       = LoadCs($"{lightRoot}/BakeEmissiveTriangles");

            processFeedbackHistoryPreFilterCs = LoadCs($"{lightRoot}/ProcessFeedbackHistoryPreFilter");
            processFeedbackHistoryP0Cs        = LoadCs($"{lightRoot}/ProcessFeedbackHistoryP0");

            processFeedbackHistoryP1aCs      = LoadCs($"{lightRoot}/ProcessFeedbackHistoryP1a");
            processFeedbackHistoryP1bCs      = LoadCs($"{lightRoot}/ProcessFeedbackHistoryP1b");
            processFeedbackHistoryP2Cs       = LoadCs($"{lightRoot}/ProcessFeedbackHistoryP2");
            processFeedbackHistoryP3Cs       = LoadCs($"{lightRoot}/ProcessFeedbackHistoryP3");
            clearFeedbackHistoryCs           = LoadCs($"{lightRoot}/ClearFeedbackHistory");
            debugDrawLightsCs                = LoadCs($"{lightRoot}/DebugDrawLights");
            processFeedbackHistoryDebugVizCs = LoadCs($"{lightRoot}/ProcessFeedbackHistoryDebugViz");

            UnityEditor.EditorUtility.SetDirty(this);
            return;

            static NativeComputeShader LoadCs(string path)
            {
                var s = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>(path + ".computeshader");
                if (s == null)
                    Debug.LogWarning($"[RtxptFeature] Missing NativeComputeShader at: {path}");
                return s;
            }

            static RayTraceShader LoadRs(string path)
            {
                var s = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTraceShader>(path + ".rayshader");
                if (s == null)
                    Debug.LogWarning($"[RtxptFeature] Missing RayTraceShader at: {path}");
                return s;
            }

            static HitGroupShader LoadHg(string path)
            {
                var s = UnityEditor.AssetDatabase.LoadAssetAtPath<HitGroupShader>(path + ".hitgroupshader");
                if (s == null)
                    Debug.LogWarning($"[RtxptFeature] Missing HitGroupShader at: {path}");
                return s;
            }

            static LazyLoadReference<T> Ref<T>(T asset) where T : UnityEngine.Object
            {
                var r = new LazyLoadReference<T>();
                r.asset = asset;
                return r;
            }

            static NativeRasterShader LoadRas(string path)
            {
                var s = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeRasterShader>(path + ".rastershader");
                if (s == null)
                    Debug.LogWarning($"[RtxptFeature] Missing NativeRasterShader at: {path}");
                return s;
            }
        }
#endif
    }
}
