using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using DLRR;
using NativeRender;
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
    ///   Phase 0 : NativeRtxptBuildTlasPass              - TLAS rebuild
    ///   Phase 1 : LightsBaker passes                    - env map / emissive / proxies / feedback (TODO)
    ///   Phase 2a: NativeRtxptBuildStablePlanesPass       - BuildStablePlanes RT (PathTracePrePass)
    ///   Phase 2b: NativeRtxptExportVisibilityBufferPass  - depth + motion vectors export
    ///   Phase 2c: NativeRtxptLightingUpdateEndPass       - NEE-AT feedback processing (stub)
    ///   Phase 2d: NativeRtxptFillStablePlanesPass        - FillStablePlanes RT (PathTrace) / Reference
    ///   Phase 3 : NativeRtxptDenoiseSpecHitTPass         - specular hit-distance bilateral filter x2
    ///   Phase 4 : NativeRtxptDlssBeforePass              - prepare DLSS-RR guide buffers
    ///   Phase 5 : DlssRRPass                             - DLSS Ray Reconstruction (denoise + upscale)
    ///   Phase 7 : NativeRtxptAccumulationPass            - multi-frame accumulation (reference mode only)
    ///
    /// PT_USE_RESTIR_DI = 0, PT_USE_RESTIR_GI = 0 (no RTXDI).
    /// cStablePlaneCount = 3.
    /// </summary>
    public class NativeRtxptFeature : ScriptableRendererFeature
    {
        // ---- Inspector fields -----------------------------------------------
        public NativeRtxptSetting setting;
        public RenderPassEvent    renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        public SampleConstants sampleConstants;

        // Phase 2a/2d: PathTracer RT shaders
        public RayTraceShader buildStablePlanesShader;
        public RayTraceShader fillStablePlanesShader;
        public RayTraceShader referenceShader;

        // Phase 2a/2d: per-pipeline extra hit-group blobs
        public HitGroupShader[] buildHitGroups;
        public HitGroupShader[] fillHitGroups;
        public HitGroupShader[] referenceHitGroups;

        // Phase 3
        public NativeComputeShader exportVisibilityBufferCs;

        // Phase 4
        public NativeComputeShader denoiseSpecHitTCs;

        // Phase 5
        public NativeComputeShader dlssBeforeCs;

        // Phase 6: Bloom (donut BloomPass: downsample → separable blur → composite)
        public NativeComputeShader bloomDownsampleCs;
        public NativeComputeShader bloomBlurCs;
        public NativeComputeShader bloomCompositeCs;

        // Phase 7: Tone mapping — faithful native replica of RTXPT ToneMappingPasses.cpp, built from the
        // original shaders as thin verbatim #include wrappers: luminance_ps (raster) → native mip reduce
        // → capture_cs → CPU read-back → ToneMapping main_ps (raster).
        public NativeRasterShader  luminanceRasterShader;   // Luminance.rastershader (= fullscreen_vs + luminance_ps)
        public NativeComputeShader luminanceMipCs;          // LuminanceMip.computeshader
        public NativeComputeShader captureLuminanceCs;      // capture_cs.computeshader (= ToneMapping.hlsl capture_cs)
        public NativeRasterShader  toneMapApplyRasterShader;// ToneMapping.rastershader (= fullscreen_vs + main_ps)

        // Phase 8
        public NativeComputeShader accumulationCs;

        // Phase Debug: StablePlanesDebugViz
        public NativeComputeShader stablePlanesDebugVizCs;

        // EnvMapBaker + LightingUpdateBegin compute shaders
        public NativeComputeShader baseLayerCs;
        public NativeComputeShader envMapImportanceBakerCs;
        public NativeComputeShader envLightsBackupPastCs;
        public NativeComputeShader envLightsSubdivideBaseCs;
        public NativeComputeShader envLightsSubdivideBoostCs;
        public NativeComputeShader envLightsFillLookupMapCs;
        public NativeComputeShader envLightsMapPastToCurrentCs;
        public NativeComputeShader resetLightProxyCountersCs;
        public NativeComputeShader resetPastToCurrentHistoryCs;
        public NativeComputeShader computeWeightsCs;
        public NativeComputeShader computeProxyCountsCs;
        public NativeComputeShader computeProxyBaselineOffsetsCs;
        public NativeComputeShader createProxyJobsCs;
        public NativeComputeShader executeProxyJobsCs;
        public NativeComputeShader bakeEmissiveTrianglesCs;

        // LightingUpdateBegin feedback passes
        public NativeComputeShader processFeedbackHistoryPreFilterCs;

        public NativeComputeShader processFeedbackHistoryP0Cs;

        // LightingUpdateEnd feedback passes
        public NativeComputeShader processFeedbackHistoryP1aCs;
        public NativeComputeShader processFeedbackHistoryP1bCs;
        public NativeComputeShader processFeedbackHistoryP2Cs;
        public NativeComputeShader processFeedbackHistoryP3Cs;
        public NativeComputeShader clearFeedbackHistoryCs;

        // Phase 9: Output blit (debug display)
        public Material outputBlitMaterial;

        // ---- Pass instances -------------------------------------------------
        private NativeRtxptBuildTlasPass              _buildTlasPass;
        private NativeRtxptEnvMapBakerPass            _envMapBakerPass;
        private NativeRtxptLightingUpdateBeginPass    _lightingUpdateBeginPass;
        private NativeRtxptBuildStablePlanesPass      _buildStablePlanesPass;
        private NativeRtxptExportVisibilityBufferPass _exportVisibilityBufferPass;
        private NativeRtxptLightingUpdateEndPass      _lightingUpdateEndPass;
        private NativeRtxptFillStablePlanesPass       _fillStablePlanesPass;
        private NativeRtxptDenoisingGuidesBakePass    _denoisingGuidesBakePass;
        private NativeRtxptDlssRRPrepareInputsPass    _dlssRrPrepareInputsPass;
        private DlssRRPass                            _dlssRRPass;
        private NativeRtxptBloomPass                  _bloomPass;
        private NativeRtxptToneMappingMipChainPass    _toneMappingMipChainPass;
        private NativeRtxptAccumulationPass           _accumulationPass;
        private NativeRtxptStablePlanesDebugVizPass   _stablePlanesDebugVizPass;
        private NativeRtxptOutputBlitPass             _outputBlitPass;

        // ---- Shared scene resources -----------------------------------------
        private NativeRtxptGPUScene _gpuScene;

        // ---- Per-camera resource pools (key = instanceID + eyeIndex*100000) -
        private readonly Dictionary<long, NativeRtxptTextureResources> _texturePools      = new();
        private readonly Dictionary<long, NativeRtxptBufferResources>  _bufferPools       = new();
        private readonly Dictionary<long, VolatileConstantBuffer>                _constantBuffers   = new();
        private readonly Dictionary<long, DlrrDenoiser>                _dlrrDenoisers     = new();
        private readonly Dictionary<long, RtxptCameraFrameState>       _cameraFrameStates = new();

        // ---- Lifecycle ------------------------------------------------------

        public IntPtr blackTexturePtr;
        
        public override void Create()
        {
            setting         ??= new NativeRtxptSetting();
            blackTexturePtr =   Texture2D.blackTexture.GetNativeTexturePtr();
        }

        private void CreatePasses()
        {
            _buildTlasPass ??= new NativeRtxptBuildTlasPass
            {
                renderPassEvent = renderPassEvent,
            };

            _envMapBakerPass ??= new NativeRtxptEnvMapBakerPass(baseLayerCs, envMapImportanceBakerCs)
            {
                renderPassEvent = renderPassEvent,
            };

            if (_lightingUpdateBeginPass == null)
            {
                _lightingUpdateBeginPass = new NativeRtxptLightingUpdateBeginPass(
                        envLightsBackupPastCs, envLightsSubdivideBaseCs, envLightsSubdivideBoostCs,
                        envLightsFillLookupMapCs, envLightsMapPastToCurrentCs,
                        resetLightProxyCountersCs, resetPastToCurrentHistoryCs,
                        computeWeightsCs, computeProxyCountsCs, computeProxyBaselineOffsetsCs,
                        createProxyJobsCs, executeProxyJobsCs,
                        bakeEmissiveTrianglesCs,
                        processFeedbackHistoryPreFilterCs, processFeedbackHistoryP0Cs)
                    { renderPassEvent = renderPassEvent };
            }

            _buildStablePlanesPass ??= new NativeRtxptBuildStablePlanesPass(
                    buildStablePlanesShader, buildHitGroups)
                { renderPassEvent = renderPassEvent };

            _exportVisibilityBufferPass ??= new NativeRtxptExportVisibilityBufferPass(exportVisibilityBufferCs) { renderPassEvent = renderPassEvent };

            _lightingUpdateEndPass ??= new NativeRtxptLightingUpdateEndPass(
                    processFeedbackHistoryP1aCs, processFeedbackHistoryP1bCs,
                    processFeedbackHistoryP2Cs, processFeedbackHistoryP3Cs,
                    clearFeedbackHistoryCs)
                { renderPassEvent = renderPassEvent };

            _fillStablePlanesPass ??= new NativeRtxptFillStablePlanesPass(
                    fillStablePlanesShader, referenceShader,
                    fillHitGroups, referenceHitGroups)
                { renderPassEvent = renderPassEvent };
            _denoisingGuidesBakePass  ??= new NativeRtxptDenoisingGuidesBakePass(denoiseSpecHitTCs) { renderPassEvent       = renderPassEvent };
            _dlssRrPrepareInputsPass  ??= new NativeRtxptDlssRRPrepareInputsPass(dlssBeforeCs) { renderPassEvent            = renderPassEvent };
            _dlssRRPass               ??= new DlssRRPass { renderPassEvent                                                  = renderPassEvent };
            // Bloom is optional — only build once all three compute shaders are assigned/imported.
            if (_bloomPass == null && bloomDownsampleCs != null && bloomBlurCs != null && bloomCompositeCs != null)
                _bloomPass = new NativeRtxptBloomPass(bloomDownsampleCs, bloomBlurCs, bloomCompositeCs) { renderPassEvent = renderPassEvent };
            // Tone mapping — faithful RTXPT replica. Built once all four original-shader wrappers exist
            // (Native*Pipeline throws on a null shader); run AutoFillShaders if these are unset.
            if (_toneMappingMipChainPass == null && luminanceRasterShader != null && luminanceMipCs != null
                && captureLuminanceCs != null && toneMapApplyRasterShader != null)
                _toneMappingMipChainPass = new NativeRtxptToneMappingMipChainPass(
                    luminanceRasterShader, luminanceMipCs, captureLuminanceCs, toneMapApplyRasterShader) { renderPassEvent = renderPassEvent };
            _accumulationPass         ??= new NativeRtxptAccumulationPass(accumulationCs) { renderPassEvent                 = renderPassEvent };
            _stablePlanesDebugVizPass ??= new NativeRtxptStablePlanesDebugVizPass(stablePlanesDebugVizCs) { renderPassEvent = renderPassEvent };
            _outputBlitPass           ??= new NativeRtxptOutputBlitPass(outputBlitMaterial) { renderPassEvent               = renderPassEvent };
        }

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
            _gpuScene ??= new NativeRtxptGPUScene();

            if (eyeIndex == 0)
            {
                _gpuScene.UpdateForFrame();
            }

            // ---- Per-camera resource lookup / creation ----------------------
            var  uniqueKey = cam.GetInstanceID() + (eyeIndex * 100_000L);
            bool isVR      = renderingData.cameraData.xrRendering;

            if (!_texturePools.TryGetValue(uniqueKey, out var texPool))
            {
                texPool = new NativeRtxptTextureResources();
                _texturePools.Add(uniqueKey, texPool);
            }

            if (!_bufferPools.TryGetValue(uniqueKey, out var bufPool))
            {
                bufPool = new NativeRtxptBufferResources();
                _bufferPools.Add(uniqueKey, bufPool);
            }

            if (!_dlrrDenoisers.TryGetValue(uniqueKey, out var dlrr))
            {
                dlrr = new DlrrDenoiser(isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name);
                _dlrrDenoisers.Add(uniqueKey, dlrr);
            }

            if (!_constantBuffers.TryGetValue(uniqueKey, out var constantBuffer))
            {
                constantBuffer = new VolatileConstantBuffer(Marshal.SizeOf<SampleConstants>());
                _constantBuffers.Add(uniqueKey, constantBuffer);
            }

            // ---- Resolution -------------------------------------------------
            var displayResolution = ComputeOutputResolution(renderingData.cameraData);
            var renderResolution  = ComputeRenderResolution(displayResolution, setting.upscalerMode);

            bool texturesChanged = texPool.EnsureResources(renderResolution, displayResolution);
            texPool.EnsureEnvMapResources();
            bufPool.EnsureResources(renderResolution);
            bufPool.EnsureLightBuffers();
            bufPool.EnsureEnvBakerBuffers();

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

            frameState.Update(renderingData, texturesChanged, 1.0f, setting);

            // ---- Build & upload SampleConstants -----------------------------
            sampleConstants = NativeRtxptConstantsBuilder.Build(renderingData, setting, renderResolution, displayResolution, frameState);

            constantBuffer.Upload(renderer, sampleConstants);

            // ---- Gather scene lights once (shared by env baker + lighting pass) ----
            // Previously both passes each ran their own full-scene FindObjectsByType<Light>;
            // gathering once here halves that per-camera cost.
            var sceneLights = UnityEngine.Object.FindObjectsByType<Light>(FindObjectsSortMode.None);

            // ---- Build shared pass context ----------------------------------
            var passCtx = new NativeRtxptPassContext
            {
                ConstantBuffer    = constantBuffer,
                GpuScene          = _gpuScene,
                Textures          = texPool,
                Buffers           = bufPool,
                RenderResolution  = renderResolution,
                DisplayResolution = displayResolution,
                FrameState        = frameState,
                Setting           = setting,
                SceneLights       = sceneLights,
                blackTexturePtr =  blackTexturePtr,
            };
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
                    var dlrrInput = new DlrrDenoiser.DlrrFrameInput
                    {
                        worldToView      = frameState.worldToView,
                        viewToClip       = frameState.viewToClip,
                        viewportJitter   = frameState.viewportJitter,
                        renderResolution = renderResolution,
                        frameIndex       = frameState.frameIndex,
                        outputWidth      = (ushort)displayResolution.x,
                        outputHeight     = (ushort)displayResolution.y,
                        useMv            = true
                    };
                    var dlrrRes = new DlrrDenoiser.DlrrResources
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
                        dlrr.GetInteropDataPtr(dlrrInput, dlrrRes, 1.0f, setting.upscalerMode),
                        new DlssRRPass.Settings { tmpDisableRR = setting.tmpDisableDlssRR });
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

            // ---- Phase Debug: StablePlanesDebugViz (only when a debug view is active) ----
            if (setting.debugViewType != RtxptDebugViewType.Disabled && stablePlanesDebugVizCs != null)
            {
                _stablePlanesDebugVizPass.Setup(passCtx);
                renderer.EnqueuePass(_stablePlanesDebugVizPass);
            }

            // ---- Phase 9: Output blit (debug display) ----------------------
            {
                // When tone mapping ran, the final image lives in ProcessedOutputColor; show it in
                // place of the raw HDR DLSS-RR output unless the user picked an explicit debug view.
                var displayMode = setting.showMode;
                bool toneMapRan = setting.realtimeMode && setting.enableToneMapping && _toneMappingMipChainPass != null;
                if (toneMapRan && displayMode == NativeRtxptShowMode.DlssRrOutput)
                    displayMode = NativeRtxptShowMode.ProcessedOutput;

                _outputBlitPass.Setup(texPool, displayMode, 1.0f, setting.debugViewType);
                renderer.EnqueuePass(_outputBlitPass);
            }
        }

        // ---- Helpers -------------------------------------------------------

        private static int2 ComputeOutputResolution(CameraData cameraData) =>
            new int2(cameraData.cameraTargetDescriptor.width,
                cameraData.cameraTargetDescriptor.height);

        private static int2 ComputeRenderResolution(int2 outputRes, UpscalerMode mode)
        {
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
            _bloomPass?.Dispose();
            _bloomPass = null;
            _toneMappingMipChainPass?.Dispose();
            _toneMappingMipChainPass = null;
            _accumulationPass?.Dispose();
            _accumulationPass = null;
            _stablePlanesDebugVizPass?.Dispose();
            _stablePlanesDebugVizPass = null;
            _outputBlitPass           = null;

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
            setting = new NativeRtxptSetting();
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
                Debug.LogWarning("[NativeRtxptFeature] LightingUpdateBeginPass not created — run the scene first.");
                return;
            }

            NativeRtxptBufferResources buf = null;
            foreach (var kv in _bufferPools)
            {
                buf = kv.Value;
                break;
            }

            if (buf?.LightBuffer == null)
            {
                Debug.LogWarning("[NativeRtxptFeature] LightBuffer is null — run the scene first.");
                return;
            }

            uint triOffset = _lightingUpdateBeginPass.EmissiveLightOffset;
            uint triCount  = _lightingUpdateBeginPass.EmissiveTriangleCount;

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"[Rtxpt Emissive Triangles] offset={triOffset}  count={triCount}  analyticCount={triOffset - 5368}  MaxLights={NativeRtxptBufferResources.MaxLights}");

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
            var lightBuffer = buf.LightBuffer;
            int framesWaited = 0;
            const int maxFrames = 180;
            UnityEditor.EditorApplication.CallbackFunction poll = null;
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
        private static double s_debugExpiry;
        private static bool   s_debugHooked;

        private static void ShowDebugSegments(List<(Vector3 a, Vector3 b, Color c)> segments, float seconds)
        {
            s_debugSegments = segments;
            s_debugExpiry   = UnityEditor.EditorApplication.timeSinceStartup + seconds;

            if (!s_debugHooked)
            {
                s_debugHooked = true;
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
            s_debugHooked   = false;
            s_debugSegments = null;
        }
#endif

        /// <summary>
        /// Debug readback for the NEE-AT data path.
        /// Run with setting.neeType = NEEAT for a few frames before clicking.
        /// </summary>
        public void TestNeeAtReadback()
        {
            NativeRtxptBufferResources buf = null;
            NativeRtxptTextureResources tex = null;

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
                Debug.LogWarning("[NativeRtxptFeature] NEE-AT readback unavailable - run the scene for a few frames first.");
                return;
            }

            var ctrl = buf.LastLightControlData;
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
            sb.AppendLine($"  Feedback: resolution={ctrl.BakerConstants.FeedbackResolutionX}x{ctrl.BakerConstants.FeedbackResolutionY} blended={ctrl.BakerConstants.BlendedFeedbackResolutionX}x{ctrl.BakerConstants.BlendedFeedbackResolutionY}");
            sb.AppendLine($"  Weights: currentOffset={ctrl.BakerConstants.CurrentWeightsBufferOffset} historicOffset={ctrl.BakerConstants.HistoricWeightsBufferOffset} globalFeedbackWeight={ctrl.GlobalFeedbackUseWeight:F3} localToGlobal={ctrl.LocalToGlobalSampleRatio:F3}");

            AppendFloatBufferStats(sb, "LightWeights.current", buf.LightWeightsBuffer, (int)ctrl.BakerConstants.CurrentWeightsBufferOffset, (int)Math.Min(ctrl.TotalLightCount + 1u, 4096u));
            AppendUIntBufferStats(sb, "LightProxyCounters", buf.LightProxyCounters, 0, (int)Math.Min(ctrl.TotalLightCount + 1u, 4096u));
            AppendUIntBufferStats(sb, "LightSamplingProxies", buf.LightSamplingProxies, 0, Mathf.Min(buf.LightSamplingProxies?.count ?? 0, 4096));
            AppendUIntBufferStats(sb, "LocalSamplingBuffer", buf.LocalSamplingBuffer, 0, Mathf.Min(buf.LocalSamplingBuffer?.count ?? 0, 4096));

            AppendFloatTextureStats(sb, "FeedbackTotalWeight", tex.LightFeedbackTotalWeight, 8192);
            AppendUIntTextureStats(sb, "FeedbackCandidates", tex.LightFeedbackCandidates, 8192);
            AppendFloatTextureStats(sb, "FeedbackTotalWeightBlended", tex.FeedbackTotalWeightBlended, 8192);
            AppendUIntTextureStats(sb, "FeedbackCandidatesBlended", tex.FeedbackCandidatesBlended, 8192);

            if (ctrl.ImportanceSamplingType != 2 || ctrl.TemporalFeedbackRequired == 0)
                sb.AppendLine("  RESULT: NEE-AT is not enabled in LightingControl. Set NativeRtxptSetting.neeType = NEEAT and useNEE = true.");
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

            int nonZero = 0;
            float min = float.PositiveInfinity;
            float max = float.NegativeInfinity;
            double sum = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                float v = data[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) continue;
                if (Mathf.Abs(v) > 1e-8f) nonZero++;
                min = Mathf.Min(min, v);
                max = Mathf.Max(max, v);
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

        private static void AppendFloatTextureStats(System.Text.StringBuilder sb, string label, Nri.NriTextureResource tex, int maxSamples)
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

            var data = req.GetData<float>();
            int count = Mathf.Min(data.Length, maxSamples);
            int nonZero = 0;
            float min = float.PositiveInfinity;
            float max = float.NegativeInfinity;
            double sum = 0.0;
            for (int i = 0; i < count; i++)
            {
                float v = data[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) continue;
                if (Mathf.Abs(v) > 1e-8f) nonZero++;
                min = Mathf.Min(min, v);
                max = Mathf.Max(max, v);
                sum += v;
            }

            if (float.IsInfinity(min)) min = max = 0.0f;
            sb.AppendLine($"  {label}: read={count}/{data.Length} nonZero={nonZero} min={min:G4} max={max:G4} sum={sum:G5}");
        }

        private static void AppendUIntTextureStats(System.Text.StringBuilder sb, string label, Nri.NriTextureResource tex, int maxSamples)
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

            var data = req.GetData<uint>();
            int count = Mathf.Min(data.Length, maxSamples);
            var sample = new uint[count];
            for (int i = 0; i < count; i++)
                sample[i] = data[i];
            AppendUIntStats(sb, label, sample, data.Length);
        }

        private static void AppendUIntStats(System.Text.StringBuilder sb, string label, uint[] data, int totalCount = -1)
        {
            int nonZero = 0;
            int invalid = 0;
            uint min = uint.MaxValue;
            uint max = 0;
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
            string suffix = totalCount >= 0 ? $"/{totalCount}" : "";
            sb.AppendLine($"  {label}: read={data.Length}{suffix} nonZero={nonZero} invalid=0xFFFFFFFF:{invalid} min={min} max={max} sumLow16={sumLowBits}");
        }


        public void AutoFillShaders()
        {
            const string shaderRoot = "Assets/RTXPT/Shaders";

            // Phase 2a/2d: PathTracer RT shaders
            buildStablePlanesShader  = LoadRs($"{shaderRoot}/BuildStablePlanes");
            fillStablePlanesShader   = LoadRs($"{shaderRoot}/FillStablePlanes");
            referenceShader          = LoadRs($"{shaderRoot}/Reference");
            exportVisibilityBufferCs = LoadCs($"{shaderRoot}/ProcessingPasses/ExportVisibilityBuffer");
            denoiseSpecHitTCs        = LoadCs($"{shaderRoot}/ProcessingPasses/DenoisingGuidesBaker_DenoiseSpecHitT");
            dlssBeforeCs             = LoadCs($"{shaderRoot}/ProcessingPasses/PostProcess_DenoiserPrepareInputsDlssRR");
            bloomDownsampleCs        = LoadCs($"{shaderRoot}/ToneMapper/BloomDownsample");
            bloomBlurCs              = LoadCs($"{shaderRoot}/ToneMapper/BloomBlur");
            bloomCompositeCs         = LoadCs($"{shaderRoot}/ToneMapper/BloomComposite");
            luminanceRasterShader    = LoadRas($"{shaderRoot}/ToneMapper/Luminance");
            luminanceMipCs           = LoadCs($"{shaderRoot}/ToneMapper/LuminanceMip");
            captureLuminanceCs       = LoadCs($"{shaderRoot}/ToneMapper/capture_cs");
            toneMapApplyRasterShader = LoadRas($"{shaderRoot}/ToneMapper/ToneMapping");
            accumulationCs           = LoadCs($"{shaderRoot}/ProcessingPasses/AccumulationPass");
            stablePlanesDebugVizCs   = LoadCs($"{shaderRoot}/ProcessingPasses/PostProcess_StablePlanesDebugViz");

            string lightRoot   = $"{shaderRoot}/Lighting";
            string distantRoot = $"{lightRoot}/Distant";
            baseLayerCs             = LoadCs($"{distantRoot}/BaseLayerCS");
            envMapImportanceBakerCs = LoadCs($"{distantRoot}/EnvMapImportanceSamplingBaker");

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

            processFeedbackHistoryP1aCs = LoadCs($"{lightRoot}/ProcessFeedbackHistoryP1a");
            processFeedbackHistoryP1bCs = LoadCs($"{lightRoot}/ProcessFeedbackHistoryP1b");
            processFeedbackHistoryP2Cs  = LoadCs($"{lightRoot}/ProcessFeedbackHistoryP2");
            processFeedbackHistoryP3Cs  = LoadCs($"{lightRoot}/ProcessFeedbackHistoryP3");
            clearFeedbackHistoryCs      = LoadCs($"{lightRoot}/ClearFeedbackHistory");

            UnityEditor.EditorUtility.SetDirty(this);
            return;

            static NativeComputeShader LoadCs(string path)
            {
                var s = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeComputeShader>(path + ".computeshader");
                if (s == null)
                    Debug.LogWarning($"[NativeRtxptFeature] Missing NativeComputeShader at: {path}");
                return s;
            }

            static RayTraceShader LoadRs(string path)
            {
                var s = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTraceShader>(path + ".rayshader");
                if (s == null)
                    Debug.LogWarning($"[NativeRtxptFeature] Missing RayTraceShader at: {path}");
                return s;
            }

            static NativeRasterShader LoadRas(string path)
            {
                var s = UnityEditor.AssetDatabase.LoadAssetAtPath<NativeRasterShader>(path + ".rastershader");
                if (s == null)
                    Debug.LogWarning($"[NativeRtxptFeature] Missing NativeRasterShader at: {path}");
                return s;
            }
        }
#endif
    }
}
