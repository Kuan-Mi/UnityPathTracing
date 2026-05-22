using System;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Phase 1 (env map): bakes directional lights + sky into a cubemap env map every frame,
    /// then builds the importance-sampling map for the path tracer.
    ///
    /// GPU pipeline:
    ///   1. BaseLayerCS          – writes mip0 + mip1 of the 256×256 cubemap
    ///   2. ImportanceBakerCS    – builds 1024×1024 flat importance map from mip0
    ///
    /// The results are exposed as IntPtrs on <see cref="NativeRtxptPassContext"/>:
    ///   <c>BakedEnvCubePtr</c>      → t_EnvironmentMap  (TextureCube mip0)
    ///   <c>EnvImportanceMapPtr</c>  → t_EnvLookupMap    (Texture2D R32F)
    /// </summary>
    public class NativeRtxptEnvMapBakerPass : ScriptableRenderPass, IDisposable
    {
        // ── Shader / pipeline constants ─────────────────────────────────────
        private const int CubeDim            = 256; // environment cube face dimension
        private const int CubeDimLowRes      = 32; // low-res pre-pass dim (unused; ProcSkyEnabled=0)
        private const int ImportanceMapDim   = 1024; // EMISB_IMPORTANCE_MAP_DIM
        private const int ImportanceSamples  = 16; // EMISB_IMPORTANCE_SAMPLES_PER_PIXEL
        private const int ImportanceSamplesX = 4; // sqrt(16)
        private const int ImportanceSamplesY = 4;

        // BaseLayerCS dispatches over mip1 coords: (CubeDim/2)/8 = 16 groups per axis
        private const int BaseLayerGroupsXY = (CubeDim / 2 + 7) / 8; // 16

        // ImportanceBakerCS dispatches over (ImportanceMapDim)/16 = 64 groups per axis
        private const int ImportanceBakerGroupsXY = (ImportanceMapDim + 15) / 16; // 64

        // ── GPU pipelines ───────────────────────────────────────────────────
        private readonly NativeComputePipeline      _baseLayerCs;
        private readonly NativeComputeDescriptorSet _baseLayerDs;
        private readonly NativeComputePipeline      _importanceBakerCs;
        private readonly NativeComputeDescriptorSet _importanceBakerDs;

        // ── Owned render textures ───────────────────────────────────────────
        private RenderTexture _envCubeMip0Rt; // 256×256 Cube RGBA16F  UAV+SRV
        private RenderTexture _envCubeMip1Rt; // 128×128 Cube RGBA16F  UAV       (BaseLayerCS dst1)
        private RenderTexture _importanceMapRt; // 1024×1024 2D RFloat   UAV
        private RenderTexture _radianceMapRt; // 1024×1024 2D RGBA16F  UAV       (importance baker side output)
        private RenderTexture _dummyCubeRt; // 4×4 Cube RGBA8        dummy SRV for unused cube slots

        // ── Constant buffers ────────────────────────────────────────────────
        private GraphicsBuffer _envBakerCb; // EnvMapBakerConstants            (704 bytes)
        private GraphicsBuffer _importanceBakerCb; // EnvMapImportanceSamplingBakerConstants (48 bytes)

        // CPU-side staging
        private static readonly byte[] s_envBakerBytes   = new byte[704];
        private static readonly byte[] s_importanceBytes = new byte[48];

        // ── Per-frame state ─────────────────────────────────────────────────
        private NativeRtxptPassContext _ctx;
        private bool                   _shadersReady;

        // ====================================================================
        // Constructor / Dispose
        // ====================================================================

        public NativeRtxptEnvMapBakerPass(
            NativeComputeShader baseLayerCs,
            NativeComputeShader importanceBakerCs)
        {
            if (baseLayerCs != null)
            {
                _baseLayerCs = new NativeComputePipeline(baseLayerCs);
                _baseLayerDs = new NativeComputeDescriptorSet(_baseLayerCs);
            }

            if (importanceBakerCs != null)
            {
                _importanceBakerCs = new NativeComputePipeline(importanceBakerCs);
                _importanceBakerDs = new NativeComputeDescriptorSet(_importanceBakerCs);
            }

            _shadersReady = _baseLayerCs != null && _importanceBakerCs != null;

            EnsureRenderTextures();
            EnsureConstantBuffers();
        }

        public void Dispose()
        {
            DestroyRT(ref _envCubeMip0Rt);
            DestroyRT(ref _envCubeMip1Rt);
            DestroyRT(ref _importanceMapRt);
            DestroyRT(ref _radianceMapRt);
            DestroyRT(ref _dummyCubeRt);
            _envBakerCb?.Dispose();
            _envBakerCb = null;
            _importanceBakerCb?.Dispose();
            _importanceBakerCb = null;
        }

        // ====================================================================
        // Setup (called from Feature on the main thread each frame)
        // ====================================================================

        public void Setup(NativeRtxptPassContext ctx)
        {
            _ctx = ctx;
            EnsureRenderTextures();
            EnsureConstantBuffers();

            // Collect directional lights and upload to GPU
            FillEnvBakerConstants(ctx.Setting);
            _envBakerCb.SetData(s_envBakerBytes);

            FillImportanceBakerConstants();
            _importanceBakerCb.SetData(s_importanceBytes);

            // Expose baked pointers to downstream passes
            ctx.BakedEnvCubePtr     = _envCubeMip0Rt.IsCreated() ? _envCubeMip0Rt.GetNativeTexturePtr() : IntPtr.Zero;
            ctx.EnvImportanceMapPtr = _importanceMapRt.IsCreated() ? _importanceMapRt.GetNativeTexturePtr() : IntPtr.Zero;
        }

        // ====================================================================
        // RecordRenderGraph
        // ====================================================================

        private class PassData
        {
            // Pipelines
            public NativeComputePipeline      BaseLayerCs;
            public NativeComputeDescriptorSet BaseLayerDs;
            public NativeComputePipeline      ImportanceBakerCs;
            public NativeComputeDescriptorSet ImportanceBakerDs;

            // Constant buffers (native ptrs)
            public IntPtr EnvBakerCbPtr;
            public IntPtr ImportanceBakerCbPtr;

            // Texture ptrs
            public IntPtr SkyTexturePtr; // SRV equirect input (may be blackTexture)
            public IntPtr EnvCubeMip0Ptr; // UAV cube mip0 output + SRV for importance baker
            public IntPtr EnvCubeMip1Ptr; // UAV cube mip1 output
            public IntPtr ImportanceMapPtr; // UAV importance map output
            public IntPtr RadianceMapPtr; // UAV radiance map output
            public IntPtr DummyCubePtr; // dummy SRV for unused TextureCube slots
            public IntPtr DummyTex2DPtr; // dummy SRV for unused Texture2D slots
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (!_shadersReady)
            {
                Debug.LogWarning("[NativeRtxptEnvMapBakerPass] Shaders not assigned — env map baker skipped.");
                return;
            }

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.EnvMapBaker", out var passData);

            passData.BaseLayerCs       = _baseLayerCs;
            passData.BaseLayerDs       = _baseLayerDs;
            passData.ImportanceBakerCs = _importanceBakerCs;
            passData.ImportanceBakerDs = _importanceBakerDs;

            passData.EnvBakerCbPtr        = _envBakerCb.GetNativeBufferPtr();
            passData.ImportanceBakerCbPtr = _importanceBakerCb.GetNativeBufferPtr();

            var skyTex = _ctx.Setting?.environmentMap;
            passData.SkyTexturePtr = skyTex != null
                ? skyTex.GetNativeTexturePtr()
                : Texture2D.blackTexture.GetNativeTexturePtr();
            passData.EnvCubeMip0Ptr   = _envCubeMip0Rt.GetNativeTexturePtr();
            passData.EnvCubeMip1Ptr   = _envCubeMip1Rt.GetNativeTexturePtr();
            passData.ImportanceMapPtr = _importanceMapRt.GetNativeTexturePtr();
            passData.RadianceMapPtr   = _radianceMapRt.GetNativeTexturePtr();
            passData.DummyCubePtr     = _dummyCubeRt.GetNativeTexturePtr();
            passData.DummyTex2DPtr    = Texture2D.blackTexture.GetNativeTexturePtr();

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        // ====================================================================
        // ExecutePass – GPU dispatch
        // ====================================================================

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            cmd.BeginSample("Rtxpt.EnvMapBaker");

            // ── 1. BaseLayerCS ─────────────────────────────────────────────
            // Writes mip0 (256×256×6) to EnvCubeMip0 and mip1 (128×128×6) to EnvCubeMip1.
            // Dispatch: (16, 16, 6) groups  [numthreads(8,8,1) over mip1 = 128×128×6]
            {
                var ds = data.BaseLayerDs;
                ds.SetConstantBuffer("g_Const", data.EnvBakerCbPtr);
                ds.SetTexture("t_SrcEquirectangularEnvMap", data.SkyTexturePtr);
                ds.SetTexture("t_SrcCubemapEnvMap", data.DummyCubePtr);
                ds.SetTexture("t_LowResPrePassCube", data.DummyCubePtr);
                ds.SetTexture("t_ProcSkyTransmittance", data.DummyTex2DPtr);
                ds.SetTexture("t_ProcSkyScatter", data.DummyTex2DPtr);
                ds.SetRWTexture("u_EnvMapCubeFacesDst0", data.EnvCubeMip0Ptr);
                ds.SetRWTexture("u_EnvMapCubeFacesDst1", data.EnvCubeMip1Ptr);
                data.BaseLayerCs.Dispatch(cmd, ds,
                    BaseLayerGroupsXY, BaseLayerGroupsXY, 6);
            }

            // ── 2. ImportanceBakerCS ───────────────────────────────────────
            // Reads the baked cubemap mip0 as TextureCube and writes the 1024×1024
            // flat importance map (R32F) and radiance map (RGBA16F).
            // Dispatch: (64, 64, 1) groups  [numthreads(16,16,1) over 1024×1024]
            {
                var ds = data.ImportanceBakerDs;
                ds.SetConstantBuffer("g_BuilderConsts", data.ImportanceBakerCbPtr);
                ds.SetTexture("t_EnvMapCube", data.EnvCubeMip0Ptr);
                ds.SetRWTexture("u_ImportanceMap", data.ImportanceMapPtr);
                ds.SetRWTexture("u_RadianceMap", data.RadianceMapPtr);
                data.ImportanceBakerCs.Dispatch(cmd, ds,
                    ImportanceBakerGroupsXY, ImportanceBakerGroupsXY, 1);
            }

            cmd.EndSample("Rtxpt.EnvMapBaker");
        }

        // ====================================================================
        // Constant buffer helpers
        // ====================================================================

        private void FillEnvBakerConstants(NativeRtxptSetting setting)
        {
            Array.Clear(s_envBakerBytes, 0, s_envBakerBytes.Length);

            int lightCount = 0;
            var lights     = UnityEngine.Object.FindObjectsByType<Light>(FindObjectsSortMode.None);
            foreach (var light in lights)
            {
                if (!light.enabled || !light.gameObject.activeInHierarchy) continue;
                if (light.type != LightType.Directional) continue;
                if (lightCount >= 16) break;

                Color linear    = light.color.linear;
                float intensity = light.intensity;

                int offset = lightCount * 32;

                // ColorIntensity: float4(r, g, b, intensity)
                WriteF32(s_envBakerBytes, offset + 0, linear.r * intensity);
                WriteF32(s_envBakerBytes, offset + 4, linear.g * intensity);
                WriteF32(s_envBakerBytes, offset + 8, linear.b * intensity);
                WriteF32(s_envBakerBytes, offset + 12, 1.0f);

                // Direction: incoming direction = -light.transform.forward
                Vector3 fwd = light.transform.forward;
                WriteF32(s_envBakerBytes, offset + 16, -fwd.x);
                WriteF32(s_envBakerBytes, offset + 20, -fwd.y);
                WriteF32(s_envBakerBytes, offset + 24, -fwd.z);

                // AngularSize: Sun's angular diameter ~0.53° = 0.009273 rad
                WriteF32(s_envBakerBytes, offset + 28, 0.009273f);

                lightCount++;
            }

            // ProceduralSkyConstants occupies bytes 512–671 (160 bytes); leave zeroed.

            // ScaleColor + DirectionalLightCount at byte 672:
            float envIntensity = setting?.environmentMapIntensity ?? 1.0f;
            Color tint         = setting?.environmentMapTint ?? Color.white;
            int   o            = 672;
            WriteF32(s_envBakerBytes, o + 0, tint.linear.r * envIntensity);
            WriteF32(s_envBakerBytes, o + 4, tint.linear.g * envIntensity);
            WriteF32(s_envBakerBytes, o + 8, tint.linear.b * envIntensity);
            WriteU32(s_envBakerBytes, o + 12, (uint)lightCount);

            // CubeDim, CubeDimLowRes, ProcSkyEnabled, BackgroundSourceType at byte 688:
            bool hasSky = setting?.environmentMap != null;
            WriteU32(s_envBakerBytes, o + 16, (uint)CubeDim);
            WriteU32(s_envBakerBytes, o + 20, (uint)CubeDimLowRes);
            WriteU32(s_envBakerBytes, o + 24, 0u); // ProcSkyEnabled = 0
            WriteU32(s_envBakerBytes, o + 28, hasSky ? 1u : 0u); // BackgroundSourceType
        }

        private static void FillImportanceBakerConstants()
        {
            Array.Clear(s_importanceBytes, 0, s_importanceBytes.Length);

            // SourceCubeDim, SourceCubeMIPCount, SampleIndex, Padding1
            WriteU32(s_importanceBytes, 0, (uint)CubeDim); // SourceCubeDim
            WriteU32(s_importanceBytes, 4, 1u); // SourceCubeMIPCount = 1 (we only have mip0)
            WriteU32(s_importanceBytes, 8, 0u); // SampleIndex = 0
            WriteU32(s_importanceBytes, 12, 0u); // Padding1

            // ImportanceMapDim: uint2(1024, 1024)
            WriteU32(s_importanceBytes, 16, (uint)ImportanceMapDim);
            WriteU32(s_importanceBytes, 20, (uint)ImportanceMapDim);

            // ImportanceMapDimInSamples: uint2(1024*4, 1024*4) = (4096, 4096)
            WriteU32(s_importanceBytes, 24, (uint)(ImportanceMapDim * ImportanceSamplesX));
            WriteU32(s_importanceBytes, 28, (uint)(ImportanceMapDim * ImportanceSamplesY));

            // ImportanceMapNumSamples: uint2(4, 4)
            WriteU32(s_importanceBytes, 32, (uint)ImportanceSamplesX);
            WriteU32(s_importanceBytes, 36, (uint)ImportanceSamplesY);

            // ImportanceMapInvSamples: 1/16 = 0.0625
            WriteF32(s_importanceBytes, 40, 1.0f / ImportanceSamples);

            // ImportanceMapBaseMip = 0 (flat map, no mip hierarchy)
            WriteU32(s_importanceBytes, 44, 0u);
        }

        // ── Bit writers ─────────────────────────────────────────────────────

        private static unsafe void WriteF32(byte[] buf, int offset, float v)
        {
            fixed (byte* p = &buf[offset])
                *(float*)p = v;
        }

        private static unsafe void WriteU32(byte[] buf, int offset, uint v)
        {
            fixed (byte* p = &buf[offset])
                *(uint*)p = v;
        }

        // ====================================================================
        // Resource management
        // ====================================================================

        private void EnsureRenderTextures()
        {
            _envCubeMip0Rt   = EnsureCubeRT(ref _envCubeMip0Rt, CubeDim, RenderTextureFormat.ARGBHalf, true);
            _envCubeMip1Rt   = EnsureCubeRT(ref _envCubeMip1Rt, CubeDim / 2, RenderTextureFormat.ARGBHalf, true);
            _importanceMapRt = Ensure2DRT(ref _importanceMapRt, ImportanceMapDim, RenderTextureFormat.RFloat, true);
            _radianceMapRt   = Ensure2DRT(ref _radianceMapRt, ImportanceMapDim, RenderTextureFormat.ARGBHalf, true);
            _dummyCubeRt     = EnsureCubeRT(ref _dummyCubeRt, 4, RenderTextureFormat.ARGB32, false);
        }

        private static RenderTexture EnsureCubeRT(ref RenderTexture rt, int size,
            RenderTextureFormat fmt, bool randomWrite)
        {
            if (rt != null && rt.IsCreated()) return rt;
            rt?.Release();
            rt = new RenderTexture(size, size, 0, fmt)
            {
                dimension         = TextureDimension.Cube,
                useMipMap         = false,
                autoGenerateMips  = false,
                enableRandomWrite = randomWrite,
                hideFlags         = HideFlags.HideAndDontSave,
            };
            rt.Create();
            return rt;
        }

        private static RenderTexture Ensure2DRT(ref RenderTexture rt, int size,
            RenderTextureFormat fmt, bool randomWrite)
        {
            if (rt != null && rt.IsCreated()) return rt;
            rt?.Release();
            rt = new RenderTexture(size, size, 0, fmt)
            {
                dimension         = TextureDimension.Tex2D,
                useMipMap         = false,
                autoGenerateMips  = false,
                enableRandomWrite = randomWrite,
                hideFlags         = HideFlags.HideAndDontSave,
            };
            rt.Create();
            return rt;
        }

        private void EnsureConstantBuffers()
        {
            if (_envBakerCb == null || !_envBakerCb.IsValid())
            {
                _envBakerCb?.Dispose();
                _envBakerCb = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, 704);
            }

            if (_importanceBakerCb == null || !_importanceBakerCb.IsValid())
            {
                _importanceBakerCb?.Dispose();
                _importanceBakerCb = new GraphicsBuffer(GraphicsBuffer.Target.Constant, 1, 48);
            }
        }

        private static void DestroyRT(ref RenderTexture rt)
        {
            if (rt == null) return;
            rt.Release();
            UnityEngine.Object.DestroyImmediate(rt);
            rt = null;
        }
    }
}