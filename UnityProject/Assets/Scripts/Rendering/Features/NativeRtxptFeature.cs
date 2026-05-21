 
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
    ///   Phase 2 : NativeRtxptPathTracerPass             - BuildStablePlanes + FillStablePlanes (realtime) / Reference
    ///   Phase 3 : NativeRtxptExportVisibilityBufferPass - depth + motion vectors export
    ///   Phase 4 : NativeRtxptDenoiseSpecHitTPass        - specular hit-distance bilateral filter x2
    ///   Phase 5 : NativeRtxptNoDenoiserFinalMergePass   - merge stable planes to OutputColor
    ///   Phase 6 : NativeRtxptDlssBeforePass             - prepare DLSS-RR guide buffers
    ///   Phase 7 : DlssRRPass                            - DLSS Ray Reconstruction (denoise + upscale)
    ///   Phase 8 : NativeRtxptAccumulationPass           - multi-frame accumulation (reference mode only)
    ///
    /// PT_USE_RESTIR_DI = 0, PT_USE_RESTIR_GI = 0 (no RTXDI).
    /// cStablePlaneCount = 3.
    /// </summary>
    public class NativeRtxptFeature : ScriptableRendererFeature
    {
        // ---- Inspector fields -----------------------------------------------
        public NativeRtxptSetting setting;
        public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        // Phase 2: PathTracer RT shaders
        public RayTraceShader buildStablePlanesShader;
        public RayTraceShader fillStablePlanesShader;
        public RayTraceShader referenceShader;

        // Phase 2: per-pipeline extra hit-group blobs
        public HitGroupShader[] buildHitGroups;
        public HitGroupShader[] fillHitGroups;
        public HitGroupShader[] referenceHitGroups;

        // Phase 3
        public NativeComputeShader exportVisibilityBufferCs;
        // Phase 4
        public NativeComputeShader denoiseSpecHitTCs;
        // Phase 5
        public NativeComputeShader noDenoiserFinalMergeCs;
        // Phase 6
        public NativeComputeShader dlssBeforeCs;
        // Phase 8
        public NativeComputeShader accumulationCs;

        // ---- Pass instances -------------------------------------------------
        private NativeRtxptBuildTlasPass              _buildTlasPass;
        private NativeRtxptPathTracerPass             _pathTracerPass;
        private NativeRtxptExportVisibilityBufferPass _exportVisibilityBufferPass;
        private NativeRtxptDenoiseSpecHitTPass        _denoiseSpecHitTPass;
        private NativeRtxptNoDenoiserFinalMergePass   _noDenoiserFinalMergePass;
        private NativeRtxptDlssBeforePass             _dlssBeforePass;
        private DlssRRPass                            _dlssRRPass;
        private NativeRtxptAccumulationPass           _accumulationPass;
        private NativeFrameTick                       _nativeFrameTickPass;

        // ---- Shared scene resources -----------------------------------------
        private NRDSampleResource   _nrdSampleResource;
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
                renderPassEvent           = renderPassEvent,
            };

            if (_pathTracerPass == null
                && buildStablePlanesShader != null
                && fillStablePlanesShader  != null
                && referenceShader         != null)
            {
                _pathTracerPass = new NativeRtxptPathTracerPass(
                    buildStablePlanesShader, fillStablePlanesShader, referenceShader,
                    buildHitGroups, fillHitGroups, referenceHitGroups)
                { renderPassEvent = renderPassEvent };
            }

            if (_exportVisibilityBufferPass == null && exportVisibilityBufferCs != null)
                _exportVisibilityBufferPass = new NativeRtxptExportVisibilityBufferPass(exportVisibilityBufferCs)
                    { renderPassEvent = renderPassEvent };

            if (_denoiseSpecHitTPass == null && denoiseSpecHitTCs != null)
                _denoiseSpecHitTPass = new NativeRtxptDenoiseSpecHitTPass(denoiseSpecHitTCs)
                    { renderPassEvent = renderPassEvent };

            if (_noDenoiserFinalMergePass == null && noDenoiserFinalMergeCs != null)
                _noDenoiserFinalMergePass = new NativeRtxptNoDenoiserFinalMergePass(noDenoiserFinalMergeCs)
                    { renderPassEvent = renderPassEvent };

            if (_dlssBeforePass == null && dlssBeforeCs != null)
                _dlssBeforePass = new NativeRtxptDlssBeforePass(dlssBeforeCs)
                    { renderPassEvent = renderPassEvent };

            _dlssRRPass ??= new DlssRRPass { renderPassEvent = renderPassEvent };

            if (_accumulationPass == null && accumulationCs != null)
                _accumulationPass = new NativeRtxptAccumulationPass(accumulationCs)
                    { renderPassEvent = renderPassEvent };

            _nativeFrameTickPass ??= new NativeFrameTick { renderPassEvent = renderPassEvent };
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            var cam = renderingData.cameraData.camera;
            if (cam.cameraType is CameraType.Preview or CameraType.Reflection) return;
            if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView) return;

            CreatePasses();

            cam.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.MotionVectors;

            var eyeIndex = renderingData.cameraData.xr.enabled
                ? renderingData.cameraData.xr.multipassId : 0;

            if (eyeIndex == 1 && setting.skipRightEyeInVR) return;

            // ---- Shared scene resources -------------------------------------
            _nrdSampleResource ??= new NRDSampleResource();
            _gpuScene          ??= new NativeRtxptGPUScene();

            if (eyeIndex == 0)
            {
                _nrdSampleResource.UpdateForFrame();
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
            _sampleConstantsArray[0] = NativeRtxptConstantsBuilder.Build(
                renderingData, setting, renderResolution, displayResolution, frameState);
            constantBuffer.SetData(_sampleConstantsArray);

            // ---- Build shared pass context ----------------------------------
            var passCtx = new NativeRtxptPassContext
            {
                ConstantBuffer    = constantBuffer,
                NrdSampleResource = _nrdSampleResource,
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
                _buildTlasPass.SetNRDSampleResource(_nrdSampleResource);
                renderer.EnqueuePass(_buildTlasPass);
            }

            // ---- Phase 1: LightsBaker (TODO) --------------------------------

            // ---- Phase 2: PathTracer RT Shader ------------------------------
            if (_pathTracerPass != null)
            {
                _pathTracerPass.Setup(passCtx);
                renderer.EnqueuePass(_pathTracerPass);
            }

            // ---- Phase 3: ExportVisibilityBuffer ----------------------------
            if (_exportVisibilityBufferPass != null)
            {
                _exportVisibilityBufferPass.Setup(passCtx);
                renderer.EnqueuePass(_exportVisibilityBufferPass);
            }

            // ---- Realtime-only phases (4-7) ---------------------------------
            if (setting.realtimeMode)
            {
                // Phase 4: DenoiseSpecHitT x2
                if (_denoiseSpecHitTPass != null)
                {
                    _denoiseSpecHitTPass.Setup(passCtx);
                    renderer.EnqueuePass(_denoiseSpecHitTPass);
                }

                // Phase 5: NoDenoiserFinalMerge
                if (_noDenoiserFinalMergePass != null)
                {
                    _noDenoiserFinalMergePass.Setup(passCtx);
                    renderer.EnqueuePass(_noDenoiserFinalMergePass);
                }

                // Phase 6: DlssBefore
                if (_dlssBeforePass != null)
                {
                    _dlssBeforePass.Setup(passCtx);
                    renderer.EnqueuePass(_dlssBeforePass);
                }

                // Phase 7: DLSS-RR
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
                    };
                    var dlrrRes = new DlrrDenoiser.DlrrResources
                    {
                        input           = texPool.OutputColor,
                        output          = texPool.DlssRrOutput,
                        mv              = texPool.ScreenMotionVectors,
                        depth           = texPool.Depth,
                        diffAlbedo      = texPool.DlssRrDiffAlbedo,
                        specAlbedo      = texPool.DlssRrSpecAlbedo,
                        normalRoughness = texPool.DlssRrNormalRoughness,
                        specHitDistance = texPool.DlssRrSpecHitDistance,
                    };
                    _dlssRRPass.Setup(
                        dlrr.GetInteropDataPtr(dlrrInput, dlrrRes, 1.0f, setting.upscalerMode),
                        new DlssRRPass.Settings { tmpDisableRR = setting.tmpDisableDlssRR });
                    renderer.EnqueuePass(_dlssRRPass);
                }
            }
            else
            {
                // Phase 8: Accumulation (reference mode)
                if (_accumulationPass != null)
                {
                    _accumulationPass.Setup(passCtx);
                    renderer.EnqueuePass(_accumulationPass);
                }
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
                UpscalerMode.NATIVE            => 1.0f,
                UpscalerMode.ULTRA_QUALITY     => 1.3f,
                UpscalerMode.QUALITY           => 1.5f,
                UpscalerMode.BALANCED          => 1.7f,
                UpscalerMode.PERFORMANCE       => 2.0f,
                UpscalerMode.ULTRA_PERFORMANCE => 3.0f,
                _                              => 1.0f,
            };
            return new int2((int)(outputRes.x / scale + 0.5f),
                            (int)(outputRes.y / scale + 0.5f));
        }

        // ---- Cleanup -------------------------------------------------------

        protected override void Dispose(bool disposing)
        {
            if (!disposing) return;

            _pathTracerPass?.Dispose();             _pathTracerPass            = null;
            _exportVisibilityBufferPass?.Dispose(); _exportVisibilityBufferPass = null;
            _denoiseSpecHitTPass?.Dispose();        _denoiseSpecHitTPass        = null;
            _noDenoiserFinalMergePass?.Dispose();   _noDenoiserFinalMergePass   = null;
            _dlssBeforePass?.Dispose();             _dlssBeforePass             = null;
            _accumulationPass?.Dispose();           _accumulationPass           = null;

            foreach (var p in _texturePools.Values)  p.Dispose();   _texturePools.Clear();
            foreach (var p in _bufferPools.Values)   p.Dispose();   _bufferPools.Clear();
            foreach (var cb in _constantBuffers.Values) cb.Dispose(); _constantBuffers.Clear();
            foreach (var d in _dlrrDenoisers.Values) d?.Dispose();  _dlrrDenoisers.Clear();
            _cameraFrameStates.Clear();

            _nrdSampleResource?.Dispose(); _nrdSampleResource = null;
            _gpuScene?.Dispose();          _gpuScene          = null;
        }

        // ---- Editor helpers ----------------------------------------------------

#if UNITY_EDITOR
        private void Reset()
        {
            setting = new NativeRtxptSetting();
            AutoFillShaders();
        }

        public void AutoFillShaders()
        {
            const string shaderRoot = "Assets/RTXPT/Shaders";

            // Phase 2: PathTracer RT shaders
            buildStablePlanesShader = LoadRs($"{shaderRoot}/BuildStablePlanes");
            fillStablePlanesShader  = LoadRs($"{shaderRoot}/FillStablePlanes");
            referenceShader         = LoadRs($"{shaderRoot}/Reference");

            // Phase 3
            exportVisibilityBufferCs = LoadCs($"{shaderRoot}/ProcessingPasses/ExportVisibilityBuffer");

            // Phase 4
            denoiseSpecHitTCs = LoadCs($"{shaderRoot}/ProcessingPasses/DenoisingGuidesBaker_DenoiseSpecHitT");

            // Phase 5
            noDenoiserFinalMergeCs = LoadCs($"{shaderRoot}/ProcessingPasses/PostProcess_NoDenoiserFinalMerge");

            // Phase 6
            dlssBeforeCs = LoadCs($"{shaderRoot}/ProcessingPasses/PostProcess_DenoiserPrepareInputsDlssRR");

            // Phase 8
            accumulationCs = LoadCs($"{shaderRoot}/ProcessingPasses/AccumulationPass");

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
