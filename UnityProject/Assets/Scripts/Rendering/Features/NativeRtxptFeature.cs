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
        private NativeRtxptDlssRRPrepareInputsPass             _dlssRrPrepareInputsPass;
        private DlssRRPass                            _dlssRRPass;
        private NativeRtxptAccumulationPass           _accumulationPass;
        private NativeRtxptStablePlanesDebugVizPass   _stablePlanesDebugVizPass;
        private NativeRtxptOutputBlitPass             _outputBlitPass;
        private NativeFrameTick                       _nativeFrameTickPass;

        // ---- Shared scene resources -----------------------------------------
        private NativeRtxptGPUScene _gpuScene;

        // ---- Per-camera resource pools (key = instanceID + eyeIndex*100000) -
        private readonly Dictionary<long, NativeRtxptTextureResources> _texturePools      = new();
        private readonly Dictionary<long, NativeRtxptBufferResources>  _bufferPools       = new();
        private readonly Dictionary<long, GraphicsBuffer>              _constantBuffers   = new();
        private readonly Dictionary<long, DlrrDenoiser>                _dlrrDenoisers     = new();
        private readonly Dictionary<long, CameraFrameState>            _cameraFrameStates = new();

        private readonly SampleConstants[] _sampleConstantsArray = new SampleConstants[1];

        // ---- Lifecycle ------------------------------------------------------

        public override void Create()
        {
            setting ??= new NativeRtxptSetting();
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
            _dlssRrPrepareInputsPass           ??= new NativeRtxptDlssRRPrepareInputsPass(dlssBeforeCs) { renderPassEvent                     = renderPassEvent };
            _dlssRRPass               ??= new DlssRRPass { renderPassEvent                                                  = renderPassEvent };
            _accumulationPass         ??= new NativeRtxptAccumulationPass(accumulationCs) { renderPassEvent                 = renderPassEvent };
            _stablePlanesDebugVizPass ??= new NativeRtxptStablePlanesDebugVizPass(stablePlanesDebugVizCs) { renderPassEvent = renderPassEvent };
            _outputBlitPass           ??= new NativeRtxptOutputBlitPass(outputBlitMaterial) { renderPassEvent               = renderPassEvent };
            _nativeFrameTickPass      ??= new NativeFrameTick { renderPassEvent                                             = renderPassEvent };
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
                constantBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1,
                    Marshal.SizeOf<SampleConstants>());
                _constantBuffers.Add(uniqueKey, constantBuffer);
            }

            // ---- Resolution -------------------------------------------------
            var displayResolution = ComputeOutputResolution(renderingData.cameraData);
            var renderResolution  = ComputeRenderResolution(displayResolution, setting.upscalerMode);

            bool texturesChanged = texPool.EnsureResources(renderResolution, displayResolution);
            bufPool.EnsureResources(renderResolution);
            bufPool.EnsureLightBuffers();

            // ---- Per-camera temporal state ----------------------------------
            if (!_cameraFrameStates.TryGetValue(uniqueKey, out var frameState))
            {
                frameState = new CameraFrameState(1.0f);
                _cameraFrameStates.Add(uniqueKey, frameState);
            }

            if (texturesChanged)
            {
                frameState.renderResolution = renderResolution;
                frameState.frameIndex       = 0;
            }

            frameState.Update(renderingData, texturesChanged, 1.0f);

            // ---- Build & upload SampleConstants -----------------------------
            sampleConstants = NativeRtxptConstantsBuilder.Build(renderingData, setting, renderResolution, displayResolution, frameState);

            _sampleConstantsArray[0] = sampleConstants;
            constantBuffer.SetData(_sampleConstantsArray);

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
                _outputBlitPass.Setup(texPool, setting.showMode, 1.0f, setting.debugViewType);
                renderer.EnqueuePass(_outputBlitPass);
            }

            // ---- Frame tick ------------------------------------------------
            renderer.EnqueuePass(_nativeFrameTickPass);
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

            int readCount = (int)System.Math.Min(triCount, 4096u);
            var data      = new RtxptPolymorphicLightInfo[readCount];
            buf.LightBuffer.GetData(data, 0, (int)triOffset, readCount);

            int   nonZero  = 0;
            float drawTime = 30f; // seconds visible in scene view

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

                // Draw triangle outline + normal
                float size = 0.05f;
                Debug.DrawLine(triBase, triBase + edge1, col, drawTime);
                Debug.DrawLine(triBase, triBase + edge2, col, drawTime);
                Debug.DrawLine(triBase + edge1, triBase + edge2, col, drawTime);
                Debug.DrawLine(center, center + normal * (size * 3f), col, drawTime);

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
        }
#endif
    }
}