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
    /// ScriptableRendererFeature for the RTXPT (Path Tracing with Stable Planes) pipeline.
    ///
    /// Denoising is done by DLSS Ray Reconstruction (DLSS-RR) 鈥?no NRD.
    ///
    /// Pass execution order:
    ///   Phase 0 : NativeRtxptBuildTlasPass          鈥?TLAS rebuild
    ///   Phase 1 : LightsBaker passes                鈥?env map bake, emissive triangles, proxies, feedback
    ///   Phase 2 : PathTracer RT shader              鈥?primary path tracing (DXR lib_6_9)
    ///   Phase 3 : ExportVisibilityBuffer CS         鈥?depth + motion vectors export
    ///   Phase 4 : DenoiseSpecHitT CS (脳2)           鈥?specular hit-distance bilateral filter
    ///   Phase 5 : NoDenoiserFinalMerge CS           鈥?merge stable planes 鈫?OutputColor
    ///   Phase 6 : DlssBeforePass (CS)               鈥?prepare DLSS-RR guide buffers
    ///   Phase 7 : DlssRRPass                        鈥?DLSS Ray Reconstruction (denoise + upscale)
    ///   Phase 8 : AccumulationPass CS (ref. mode)  鈥?multi-frame accumulation (reference mode only)
    ///
    /// PT_USE_RESTIR_DI = 0, PT_USE_RESTIR_GI = 0 (no RTXDI).
    /// cStablePlaneCount = 3.
    /// </summary>
    public class NativeRtxptFeature : ScriptableRendererFeature
    {
        //  Inspector fields 
        public NativeRtxptSetting setting;

        public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        public ComputeShader updateSkinnedPrimitivesCS;

        //  Phase 5: NoDenoiserFinalMerge 
        // TODO: replace with NativeComputeShader once asset is wired in.
        // public NativeComputeShader noDenoiserFinalMergeCs;

        //  Phase 6: DlssBefore (guide buffer preparation) 
        // TODO: replace with NativeComputeShader once asset is wired in.
        // public NativeComputeShader dlssBeforeCs;

        //  Pass instances 
        private NativeRtxptBuildTlasPass _buildTlasPass;
        private DlssRRPass               _dlssRRPass;
        private NativeFrameTick          _nativeFrameTickPass;

        // TODO: add pass instances as they are implemented:
        // private NativeRtxptExportVisibilityBufferPass    _exportVisibilityBufferPass;
        // private NativeRtxptDenoiseSpecHitTPass           _denoiseSpecHitTPass;       // 脳2 ping-pong
        // private NativeRtxptNoDenoiserFinalMergePass      _noDenoiserFinalMergePass;
        // private NativeRtxptDlssBeforePass                _dlssBeforePass;
        // private NativeRtxptAccumulationPass              _accumulationPass;          // reference mode

        //  Shared scene resource 
        private NRDSampleResource _nrdSampleResource;

        //  Per-camera resource pools 
        // Key = camera.GetInstanceID() + eyeIndex 脳 100_000L
        private readonly Dictionary<long, NativeRtxptTextureResources> _texturePools    = new();
        private readonly Dictionary<long, NativeRtxptBufferResources>  _bufferPools     = new();
        private readonly Dictionary<long, GraphicsBuffer>             _constantBuffers = new();

        // DLSS-RR denoiser instance per camera.
        private readonly Dictionary<long, DlrrDenoiser>      _dlrrDenoisers      = new();
        private readonly Dictionary<long, CameraFrameState>  _cameraFrameStates  = new();

        private readonly SampleConstants[] _sampleConstantsArray = new SampleConstants[1];

        //  Lifecycle 

        public override void Create()
        {
            setting ??= new NativeRtxptSetting();
        }

        private void CreatePasses()
        {
            _buildTlasPass ??= new NativeRtxptBuildTlasPass
            {
                updateSkinnedPrimitivesCS = this.updateSkinnedPrimitivesCS,
                renderPassEvent           = renderPassEvent,
            };

            _dlssRRPass ??= new DlssRRPass
            {
                renderPassEvent = renderPassEvent,
            };

            _nativeFrameTickPass ??= new NativeFrameTick
            {
                renderPassEvent = renderPassEvent,
            };
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            var cam = renderingData.cameraData.camera;
            if (cam.cameraType is CameraType.Preview or CameraType.Reflection)
                return;
            if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView)
                return;

            CreatePasses();

            cam.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.MotionVectors;

            var eyeIndex = renderingData.cameraData.xr.enabled
                ? renderingData.cameraData.xr.multipassId
                : 0;

            if (eyeIndex == 1 && setting.skipRightEyeInVR)
                return;

            //  Shared scene resource 
            if (_nrdSampleResource == null)
                _nrdSampleResource = new NRDSampleResource();

            if (eyeIndex == 0)
                _nrdSampleResource.UpdateForFrame();

            //  Per-camera resource lookup / creation 
            var uniqueKey = cam.GetInstanceID() + (eyeIndex * 100_000L);
            bool isVR     = renderingData.cameraData.xrRendering;

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

            //  DLSS-RR denoiser instance 
            if (!_dlrrDenoisers.TryGetValue(uniqueKey, out var dlrr))
            {
                var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                dlrr = new DlrrDenoiser(camName);
                _dlrrDenoisers.Add(uniqueKey, dlrr);
            }

            //  Constant buffer 
            if (!_constantBuffers.TryGetValue(uniqueKey, out var constantBuffer))
            {
                constantBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, Marshal.SizeOf<SampleConstants>());
                _constantBuffers.Add(uniqueKey, constantBuffer);
            }

            //  Resolution 
            var displayResolution = ComputeOutputResolution(renderingData.cameraData);
            var renderResolution  = ComputeRenderResolution(displayResolution, setting.upscalerMode);

            bool texturesChanged = texPool.EnsureResources(renderResolution, displayResolution);
            bufPool.EnsureResources(renderResolution);
            bufPool.EnsureLightBuffers();

            // ── Per-camera temporal state ──────────────────────────────────────
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

            // ── Build & upload SampleConstants ────────────────────────────────
            _sampleConstantsArray[0] = NativeRtxptConstantsBuilder.Build(
                renderingData, setting, renderResolution, displayResolution, frameState);
            constantBuffer.SetData(_sampleConstantsArray);

            //  Phase 0: TLAS 
            if (eyeIndex == 0)
            {
                _buildTlasPass.SetNRDSampleResource(_nrdSampleResource);
                renderer.EnqueuePass(_buildTlasPass);
            }

            //  Phase 1: LightsBaker 
            // TODO: enqueue LightsBaker passes once implemented.

            //  Phase 2: PathTracer RT Shader 
            // TODO: enqueue RT pass once NativeRayTraceShader asset is wired.

            //  Phase 3: ExportVisibilityBuffer 
            // TODO: enqueue once NativeComputeShader asset is wired.

            //  Phase 4: DenoiseSpecHitT (脳2 ping-pong) 
            // TODO: enqueue once NativeComputeShader asset is wired.

            //  Phase 5: NoDenoiserFinalMerge 
            // Merges stable planes 鈫?OutputColor (no NRD denoising).
            // TODO: enqueue once NativeComputeShader asset is wired.

            //  Phase 6: DLSS-RR guide buffers (DlssBefore) 
            // TODO: enqueue once NativeComputeShader asset is wired.

            //  Phase 7: DLSS Ray Reconstruction 
            if (texPool.DlssRrOutput.IsCreated)
            {
                // TODO: fill DlrrFrameInput from CameraFrameState once PT is wired.
                // var dlrrInput = new DlrrDenoiser.DlrrFrameInput { ... };
                // var dlrrRes   = new DlrrDenoiser.DlrrResources
                // {
                //     input           = texPool.OutputColor,
                //     output          = texPool.DlssRrOutput,
                //     mv              = texPool.ScreenMotionVectors,
                //     depth           = texPool.Depth,
                //     diffAlbedo      = texPool.DlssRrDiffAlbedo,
                //     specAlbedo      = texPool.DlssRrSpecAlbedo,
                //     normalRoughness = texPool.DlssRrNormalRoughness,
                //     specHitDistance = texPool.DlssRrSpecHitDistance,
                // };
                // _dlssRRPass.Setup(dlrr.GetInteropDataPtr(dlrrInput, dlrrRes, 1.0f, setting.upscalerMode),
                //                   new DlssRRPass.Settings { tmpDisableRR = setting.tmpDisableDlssRR });
                // renderer.EnqueuePass(_dlssRRPass);
            }

            //  Phase 8: AccumulationPass (reference mode only) 
            // if (setting.pathTracerMode == RtxptPathTracerMode.Reference)
            //     TODO: enqueue AccumulationPass.

            //  Frame tick 
            renderer.EnqueuePass(_nativeFrameTickPass);
        }

        // ── Helpers ─────────────────────────────────────────────────────────
        // (SampleConstants building is in NativeRtxptConstantsBuilder.cs)

        private static int2 ComputeOutputResolution(CameraData cameraData)
        {
            return new int2(cameraData.cameraTargetDescriptor.width,
                            cameraData.cameraTargetDescriptor.height);
        }

        private static int2 ComputeRenderResolution(int2 outputRes, UpscalerMode mode)
        {
            // Match NativeNrdTextureResources.GetUpscaledResolution scale factors.
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
            return new int2((int)(outputRes.x / scale + 0.5f), (int)(outputRes.y / scale + 0.5f));
        }

        //  Cleanup 

        protected override void Dispose(bool disposing)
        {
            if (!disposing) return;

            foreach (var pool in _texturePools.Values) pool.Dispose();
            _texturePools.Clear();

            foreach (var pool in _bufferPools.Values) pool.Dispose();
            _bufferPools.Clear();

            foreach (var cb in _constantBuffers.Values) cb.Dispose();
            _constantBuffers.Clear();

            foreach (var dlrr in _dlrrDenoisers.Values) dlrr?.Dispose();
            _dlrrDenoisers.Clear();

            _cameraFrameStates.Clear();

            _nrdSampleResource?.Dispose();
            _nrdSampleResource = null;
        }
    }
}
