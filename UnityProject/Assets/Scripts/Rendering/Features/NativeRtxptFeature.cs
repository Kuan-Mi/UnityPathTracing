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
        // 鈹€鈹€ Inspector fields 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        public NativeRtxptSetting setting;

        public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        public ComputeShader updateSkinnedPrimitivesCS;

        // 鈹€鈹€ Phase 5: NoDenoiserFinalMerge 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        // TODO: replace with NativeComputeShader once asset is wired in.
        // public NativeComputeShader noDenoiserFinalMergeCs;

        // 鈹€鈹€ Phase 6: DlssBefore (guide buffer preparation) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        // TODO: replace with NativeComputeShader once asset is wired in.
        // public NativeComputeShader dlssBeforeCs;

        // 鈹€鈹€ Pass instances 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        private NativeRtxptBuildTlasPass _buildTlasPass;
        private DlssRRPass               _dlssRRPass;
        private NativeFrameTick          _nativeFrameTickPass;

        // TODO: add pass instances as they are implemented:
        // private NativeRtxptExportVisibilityBufferPass    _exportVisibilityBufferPass;
        // private NativeRtxptDenoiseSpecHitTPass           _denoiseSpecHitTPass;       // 脳2 ping-pong
        // private NativeRtxptNoDenoiserFinalMergePass      _noDenoiserFinalMergePass;
        // private NativeRtxptDlssBeforePass                _dlssBeforePass;
        // private NativeRtxptAccumulationPass              _accumulationPass;          // reference mode

        // 鈹€鈹€ Shared scene resource 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        private NRDSampleResource _nrdSampleResource;

        // 鈹€鈹€ Per-camera resource pools 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        // Key = camera.GetInstanceID() + eyeIndex 脳 100_000L
        private readonly Dictionary<long, NativeRtxptTextureResources> _texturePools    = new();
        private readonly Dictionary<long, NativeRtxptBufferResources>  _bufferPools     = new();
        private readonly Dictionary<long, NativeBuffer>                _constantBuffers = new();

        // DLSS-RR denoiser instance per camera.
        private readonly Dictionary<long, DlrrDenoiser> _dlrrDenoisers = new();

        // 鈹€鈹€ Lifecycle 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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

            // 鈹€鈹€ Shared scene resource 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            if (_nrdSampleResource == null)
                _nrdSampleResource = new NRDSampleResource();

            if (eyeIndex == 0)
                _nrdSampleResource.UpdateForFrame();

            // 鈹€鈹€ Per-camera resource lookup / creation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
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

            // 鈹€鈹€ DLSS-RR denoiser instance 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            if (!_dlrrDenoisers.TryGetValue(uniqueKey, out var dlrr))
            {
                var camName = isVR ? $"{cam.name}_Eye{eyeIndex}" : cam.name;
                dlrr = new DlrrDenoiser(camName);
                _dlrrDenoisers.Add(uniqueKey, dlrr);
            }

            // 鈹€鈹€ Constant buffer 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            if (!_constantBuffers.TryGetValue(uniqueKey, out var constantBuffer))
            {
                // TODO: replace 256 placeholder with actual SampleConstants size.
                constantBuffer = new NativeBuffer(256);
                _constantBuffers.Add(uniqueKey, constantBuffer);
            }

            // 鈹€鈹€ Resolution 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            var displayResolution = ComputeOutputResolution(renderingData.cameraData);
            var renderResolution  = ComputeRenderResolution(displayResolution, setting.upscalerMode);

            bool texturesChanged = texPool.EnsureResources(renderResolution, displayResolution);
            bufPool.EnsureResources(renderResolution);
            bufPool.EnsureLightBuffers();

            // 鈹€鈹€ Phase 0: TLAS 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            if (eyeIndex == 0)
            {
                _buildTlasPass.SetNRDSampleResource(_nrdSampleResource);
                renderer.EnqueuePass(_buildTlasPass);
            }

            // 鈹€鈹€ Phase 1: LightsBaker 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            // TODO: enqueue LightsBaker passes once implemented.

            // 鈹€鈹€ Phase 2: PathTracer RT Shader 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            // TODO: enqueue RT pass once NativeRayTraceShader asset is wired.

            // 鈹€鈹€ Phase 3: ExportVisibilityBuffer 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            // TODO: enqueue once NativeComputeShader asset is wired.

            // 鈹€鈹€ Phase 4: DenoiseSpecHitT (脳2 ping-pong) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            // TODO: enqueue once NativeComputeShader asset is wired.

            // 鈹€鈹€ Phase 5: NoDenoiserFinalMerge 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            // Merges stable planes 鈫?OutputColor (no NRD denoising).
            // TODO: enqueue once NativeComputeShader asset is wired.

            // 鈹€鈹€ Phase 6: DLSS-RR guide buffers (DlssBefore) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            // TODO: enqueue once NativeComputeShader asset is wired.

            // 鈹€鈹€ Phase 7: DLSS Ray Reconstruction 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            if (setting.pathTracerMode == RtxptPathTracerMode.BuildStablePlanes
                && texPool.DlssRrOutput.IsCreated)
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

            // 鈹€鈹€ Phase 8: AccumulationPass (reference mode only) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            // if (setting.pathTracerMode == RtxptPathTracerMode.Reference)
            //     TODO: enqueue AccumulationPass.

            // 鈹€鈹€ Frame tick 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            renderer.EnqueuePass(_nativeFrameTickPass);
        }

        // 鈹€鈹€ Helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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

        // 鈹€鈹€ Cleanup 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

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

            _nrdSampleResource?.Dispose();
            _nrdSampleResource = null;
        }
    }
}
