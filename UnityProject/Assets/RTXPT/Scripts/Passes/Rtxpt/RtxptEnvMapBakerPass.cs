using System;
using System.Runtime.InteropServices;
using NativeRender;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using Object = UnityEngine.Object;

namespace PathTracing
{
    public class RtxptEnvMapBakerPass : ScriptableRenderPass, IDisposable
    {
        // Baked radiance cube resolution. Matches the original EnvMapBaker target for an
        // image/HDR background: 2048 (EnvMapBaker.cpp:375, non-procedural). The importance &
        // radiance maps (1024) are built by sampling this cube's mip0, so this drives the
        // fidelity of all environment lighting. CubeDimLowRes mirrors m_cubeDimLowResDim =
        // cubeDim/2 (EnvMapBaker.cpp:318). Public so the texture allocator stays in sync.
        public const  int CubeDim       = 2048;
        private const int CubeDimLowRes = CubeDim / 2;

        // Number of cubemap mips, mirroring EnvMapBaker.cpp InitBuffers:
        //   mipLevels = log2(cubeDim / c_BlockCompressionBlockSize), c_BlockCompressionBlockSize = 4.
        // 2048 → 9 (mips 0..8). BaseLayerCS writes mip 0+1; MIPReduceCS fills mips 2..N-1, so the
        // reduce loop issues (CubeMipCount-2) dispatches — must stay ≤ the descriptor-set ring depth (8).
        public static readonly int CubeMipCount            = (int)(System.Math.Log(CubeDim / 4.0, 2.0) + 0.5);
        private const          int ImportanceMapDim        = 1024;
        private const          int ImportanceSamples       = 16;
        private const          int ImportanceSamplesX      = 4;
        private const          int ImportanceSamplesY      = 4;
        private const          int BaseLayerGroupsXY       = (CubeDim / 2 + 7) / 8;
        private const          int ImportanceBakerGroupsXY = (ImportanceMapDim + 15) / 16;

        // donut MipMapGenPass constants (mipmapgen_cb.h): GROUP_SIZE / NUM_LODS / MAX_PASSES. The
        // importance & radiance map mip chains are built with the ORIGINAL donut mipmapgen_cs shader
        // (mipmapgen_cs.computeshader) — SRV input mip + bounded UAV output-mip array — instead of
        // Unity's cmd.GenerateMips, so the chains are bit-identical to RTXPT (which uses this exact pass).
        private const int kGroupSize = 16;
        private const int kNumLods   = 4;
        private const int kMaxPasses = 4;

        // Mirrors the original EnvMapBaker defaults (EnvMapBaker.h): m_BC6UCompressionEnabled = true
        // and m_compressionQuality = 1 (Fast / P1-only). When on, the baked RGBA16F cube is
        // BC6H-compressed (BC6UCompress CS → RGBA32_UINT scratch → reinterpret-copy into a BC6H
        // cube) and the path tracer samples that BC6H cube as t_EnvironmentMap — matching the
        // original GetEnvMapCube() returning m_cubemapBC6H when m_outputIsCompressed. The existing
        // BC6UCompress.computeshader asset has no QUALITY macro, so ENCODE_P2 is false ⇒ exactly the
        // "Fast" (m_BC6UCompressLowPSO) variant. Set false to sample the uncompressed cube instead.
        public const bool EnableBC6UCompression = false;

        // DXGI_FORMAT_R16G16B16A16_FLOAT — the env cube's format, used to bind it as a typed
        // Texture2DArray SRV for the BC6 compressor (the cube's faces, not a TextureCube view).
        private const uint DXGI_FORMAT_R16G16B16A16_FLOAT = 10;

        // BC6UCompress is [numthreads(8,8,1)] over 4×4-block tiles. The original dispatches the
        // scratch base block count for every mip and relies on the in-shader bounds guard to clip
        // smaller mips (EnvMapBaker.cpp:614-619), so we match it: (CubeDim/4 + 7) / 8.
        private const int BC6UGroupsXY = (CubeDim / 4 + 7) / 8;

        // Constant radiance-compression factor applied at bake time so 32-bit HDR source
        // radiance fits the 16-bit-float / BC6U range the baker uses (Sample.cpp:89,
        // c_envMapRadianceScale = 1/4). Tint & intensity are NOT applied here — they are
        // applied at sample time via EnvMapSceneParams.ColorMultiplier, which also divides
        // by this factor to cancel it (Sample.cpp:1912). Lowering below 1/4 breaks BC6U.
        public const float EnvMapRadianceScale = 0.25f;

        // Mirrors HLSL EnvMapBakerConstants (EnvMapBaker.hlsl):
        //   EMB_DirectionalLight DirectionalLights[16]  — 512 bytes (16 × float4 ColorIntensity + float3 Direction + float AngularSize)
        //   ProceduralSkyConstants ProcSkyConsts        — 160 bytes (always zeroed in Unity)
        //   float3 ScaleColor; uint DirectionalLightCount;
        //   uint CubeDim, CubeDimLowRes, ProcSkyEnabled, BackgroundSourceType;
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private unsafe struct EnvMapBakerCB
        {
            public fixed float DirectionalLights[16 * 8]; // EMB_DirectionalLight[16]
            public fixed uint  ProcSkyConstsPad[160 / 4]; // ProceduralSkyConstants (zeroed)
            public       float ScaleColorR, ScaleColorG, ScaleColorB;
            public       uint  DirectionalLightCount;
            public       uint  CubeDim, CubeDimLowRes, ProcSkyEnabled, BackgroundSourceType;
        }

        // donut MipmmapGenConstants (mipmapgen_cb.h): { uint dispatch; uint numLODs; uint padding[2]; }.
        // Bound as root constants for mipmapgen_cs.computeshader (varies per pass).
        [StructLayout(LayoutKind.Sequential)]
        private struct MipMapGenCB
        {
            public uint dispatch; // pass index (unused by MODE_COLOR; kept for donut layout parity)
            public uint numLODs; // output mips this pass (1..NUM_LODS)
            public uint pad0, pad1;
        }

        // Mirrors HLSL EnvMapImportanceSamplingBakerConstants (EnvMapImportanceSamplingBaker.hlsl)
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct ImportanceBakerCB
        {
            public uint  SourceCubeDim,              SourceCubeMIPCount, SampleIndex, Padding1;
            public uint  ImportanceMapDimX,          ImportanceMapDimY;
            public uint  ImportanceMapDimInSamplesX, ImportanceMapDimInSamplesY;
            public uint  ImportanceMapNumSamplesX,   ImportanceMapNumSamplesY;
            public float ImportanceMapInvSamples;
            public uint  ImportanceMapBaseMip;
        }

        internal static unsafe int EnvBakerCbSize        => sizeof(EnvMapBakerCB);
        internal static        int ImportanceBakerCbSize => System.Runtime.InteropServices.Marshal.SizeOf<ImportanceBakerCB>();

        private readonly NativeComputePipeline      _baseLayerCs;
        private readonly NativeComputeDescriptorSet _baseLayerDs;
        private readonly NativeComputePipeline      _mipReduceCs;
        private readonly NativeComputeDescriptorSet _mipReduceDs;
        private readonly NativeComputePipeline      _importanceBakerCs;
        private readonly NativeComputeDescriptorSet _importanceBakerDs;
        private readonly NativeComputePipeline      _mipMapGenCs; // donut mipmapgen_cs (importance/radiance mip chains)
        private readonly NativeComputeDescriptorSet _mipMapGenDs; // reused over ≤3 passes × 2 maps = 6 dispatches (< ring depth 8)
        private readonly NativeComputePipeline      _bc6uCompressCs; // null when compression disabled / shader missing
        private readonly NativeComputeDescriptorSet _bc6uCompressDs;

        // Pinned ring for the BC6H copy plugin-event payload. The data must stay alive until the
        // render thread consumes the event, so it lives in persistent storage (mirrors the dispatch
        // header ring in NativeComputeDescriptorSet). The bake runs at most once per frame, so a
        // few slots is ample.
        private const int                                                        CopyRingSize = 3;
        private       NativeArray<NativeRenderPlugin.BC6HCopy_RenderEventData>[] _copyRing;
        private       int                                                        _copyRingIdx;

        private static EnvMapBakerCB     s_envBakerCb;
        private static ImportanceBakerCB s_importanceCb;

        private RtxptPassContext _ctx;

        // When true the env contents are identical to what is already baked into this camera's
        // cube/importance maps, so the whole bake (base layer + importance map + mip chains) is
        // skipped this frame. Mirrors the original EnvMapBaker::Update contentsChanged early-out.
        private bool _skipBake;

        // Debug escape hatch mirroring the original m_dbgForceDynamic ("Force dynamic" checkbox,
        // EnvMapBaker.cpp:429,712): when true the cache is bypassed and the bake runs every frame.
        public static bool DebugForceDynamic = false;

        public RtxptEnvMapBakerPass(NativeComputeShader baseLayerCs, NativeComputeShader mipReduceCs, NativeComputeShader importanceBakerCs, NativeComputeShader mipMapGenCs, NativeComputeShader bc6uCompressCs = null)
        {
            _baseLayerCs       = new NativeComputePipeline(baseLayerCs);
            _baseLayerDs       = new NativeComputeDescriptorSet(_baseLayerCs);
            _mipReduceCs       = new NativeComputePipeline(mipReduceCs);
            _mipReduceDs       = new NativeComputeDescriptorSet(_mipReduceCs);
            _importanceBakerCs = new NativeComputePipeline(importanceBakerCs);
            _importanceBakerDs = new NativeComputeDescriptorSet(_importanceBakerCs);
            _mipMapGenCs       = new NativeComputePipeline(mipMapGenCs);
            _mipMapGenDs       = new NativeComputeDescriptorSet(_mipMapGenCs);

            if (EnableBC6UCompression && bc6uCompressCs != null)
            {
                _bc6uCompressCs = new NativeComputePipeline(bc6uCompressCs);
                _bc6uCompressDs = new NativeComputeDescriptorSet(_bc6uCompressCs);

                _copyRing = new NativeArray<NativeRenderPlugin.BC6HCopy_RenderEventData>[CopyRingSize];
                for (int i = 0; i < CopyRingSize; i++)
                    _copyRing[i] = new NativeArray<NativeRenderPlugin.BC6HCopy_RenderEventData>(1, Allocator.Persistent);
            }
        }

        public void Dispose()
        {
            _baseLayerDs?.Dispose();
            _baseLayerCs?.Dispose();
            _mipReduceDs?.Dispose();
            _mipReduceCs?.Dispose();
            _importanceBakerDs?.Dispose();
            _importanceBakerCs?.Dispose();
            _mipMapGenDs?.Dispose();
            _mipMapGenCs?.Dispose();
            _bc6uCompressDs?.Dispose();
            _bc6uCompressCs?.Dispose();
            if (_copyRing != null)
            {
                for (int i = 0; i < CopyRingSize; i++)
                    if (_copyRing[i].IsCreated)
                        _copyRing[i].Dispose();
                _copyRing = null;
            }
        }

        // Snapshots the BC6H copy payload into the next pinned ring slot and returns a stable
        // pointer for CommandBuffer.IssuePluginEventAndData. Returns IntPtr.Zero if unavailable.
        private unsafe IntPtr StageCopyEvent(IntPtr scratch, IntPtr bc6h)
        {
            if (_copyRing == null || scratch == IntPtr.Zero || bc6h == IntPtr.Zero) return IntPtr.Zero;
            int ring = _copyRingIdx % CopyRingSize;
            _copyRingIdx++;
            _copyRing[ring][0] = new NativeRenderPlugin.BC6HCopy_RenderEventData
            {
                srcScratch = (ulong)scratch,
                dstBC6H    = (ulong)bc6h,
                mipLevels  = (uint)CubeMipCount,
                arraySize  = 6,
            };
            return (IntPtr)_copyRing[ring].GetUnsafePtr();
        }

        /// <summary>
        /// Decides whether anything that affects the baked cube/importance maps changed since this
        /// camera's last bake — the C# equivalent of EnvMapBaker::Update's contentsChanged gating
        /// (EnvMapBaker.cpp:425-459): renderPassesDirty (fresh/resized textures, source change) is
        /// covered by <c>tex.EnvBaked</c>, the per-light isnear comparison and source-path change by
        /// the constants/identity signature. Must run before RtxptCameraFrameState.Update so a
        /// re-bake resets accumulation the same frame, mirroring Sample.cpp:1383-1384
        /// (<c>if (m_envMapBaker->Update(...)) m_ui.ResetAccumulation = true;</c>).
        /// Returns true when a bake will be recorded this frame.
        /// </summary>
        public bool PrepareBake(RtxptSetting setting, Light[] sceneLights, RtxptTextureResources tex)
        {
            FillEnvBakerConstants(setting, sceneLights);

            Texture sky      = setting?.environmentMap;
            int     skyId    = sky != null ? sky.GetInstanceID() : 0;
            uint    skyEdits = sky != null ? sky.updateCount : 0u; // bumps on CPU uploads (Apply etc.)
            ulong   signature = ComputeEnvSignature(skyId, skyEdits);

            // A RenderTexture source can be rewritten on the GPU without any CPU-observable change
            // (updateCount does not track compute/UAV writes), so treat it as dynamic and re-bake
            // every frame — the role the (animated) procedural sky plays in the original.
            bool dynamicSource = sky is RenderTexture;

            _skipBake = !DebugForceDynamic && !dynamicSource
                        && tex.EnvBaked && tex.EnvBakeSignature == signature;
            if (!_skipBake)
            {
                // The bake is guaranteed to be recorded this frame, so mark it done now.
                tex.EnvBaked         = true;
                tex.EnvBakeSignature = signature;
            }

            return !_skipBake;
        }

        public void Setup(RtxptPassContext ctx)
        {
            _ctx = ctx;
            FillImportanceBakerConstants();
        }

        // FNV-1a hash over the env baker constants (directional lights, scale color, counts,
        // background type) plus the sky texture identity & update count — the same inputs the
        // original baker compares to detect changes (exact bytes instead of dm::isnear, which is
        // strictly more conservative: tiny changes re-bake instead of being missed).
        private static unsafe ulong ComputeEnvSignature(int skyTextureId, uint skyUpdateCount)
        {
            ulong h = 14695981039346656037UL;
            fixed (EnvMapBakerCB* p = &s_envBakerCb)
            {
                byte* b = (byte*)p;
                int   n = sizeof(EnvMapBakerCB);
                for (int i = 0; i < n; i++)
                {
                    h ^= b[i];
                    h *= 1099511628211UL;
                }
            }

            h ^= (uint)skyTextureId;
            h *= 1099511628211UL;
            h ^= skyUpdateCount;
            h *= 1099511628211UL;
            return h;
        }

        private class PassData
        {
            internal NativeComputePipeline      BaseLayerCs;
            internal NativeComputeDescriptorSet BaseLayerDs;
            internal NativeComputePipeline      MipReduceCs;
            internal NativeComputeDescriptorSet MipReduceDs;
            internal NativeComputePipeline      ImportanceBakerCs;
            internal NativeComputeDescriptorSet ImportanceBakerDs;
            internal NativeComputePipeline      MipMapGenCs;
            internal NativeComputeDescriptorSet MipMapGenDs;
            internal int                        ImportanceMapMips;
            internal int                        RadianceMapMips;
            internal VolatileConstantBuffer     EnvBakerCb;
            internal VolatileConstantBuffer     ImportanceBakerCb;
            internal EnvMapBakerCB              EnvBakerCbData;
            internal ImportanceBakerCB          ImportanceCbData;
            internal IntPtr                     SkyTexturePtr;
            // True when SkyTexturePtr is a cubemap (DepthOrArraySize==6) source, so it must bind to
            // t_SrcCubemapEnvMap (BackgroundSourceType==2) rather than the equirectangular 2D slot.
            internal bool                       SourceIsCube;
            internal IntPtr                     EnvCubePtr;
            internal IntPtr                     ImportanceMapPtr;
            internal IntPtr                     RadianceMapPtr;
            internal RenderTexture              ImportanceMapRt;
            internal RenderTexture              RadianceMapRt;
            internal IntPtr                     DummyCubePtr;
            internal IntPtr                     DummyTex2DPtr;

            // BC6H compression (null pipeline ⇒ skipped).
            internal NativeComputePipeline      Bc6uCompressCs;
            internal NativeComputeDescriptorSet Bc6uCompressDs;
            internal IntPtr                     Bc6uScratchPtr;
            internal IntPtr                     Bc6uCubePtr;
            internal IntPtr                     Bc6uCopyEventFunc;
            internal IntPtr                     Bc6uCopyEventData;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            // Env contents unchanged since last bake — the cube/importance maps already hold valid
            // data, so don't enqueue any GPU work this frame (the expensive part of the regression).
            if (_skipBake) return;

            using var builder = renderGraph.AddUnsafePass<PassData>("EnvMapBaker", out var pd);

            pd.BaseLayerCs       = _baseLayerCs;
            pd.BaseLayerDs       = _baseLayerDs;
            pd.MipReduceCs       = _mipReduceCs;
            pd.MipReduceDs       = _mipReduceDs;
            pd.ImportanceBakerCs = _importanceBakerCs;
            pd.ImportanceBakerDs = _importanceBakerDs;
            pd.MipMapGenCs       = _mipMapGenCs;
            pd.MipMapGenDs       = _mipMapGenDs;
            pd.EnvBakerCb        = _ctx.Buffers.EnvBakerCb;
            pd.ImportanceBakerCb = _ctx.Buffers.ImportanceBakerCb;
            pd.EnvBakerCbData    = s_envBakerCb;
            pd.ImportanceCbData  = s_importanceCb; 
            pd.SkyTexturePtr     = _ctx.environmentMapPtrPtr != IntPtr.Zero ? _ctx.environmentMapPtrPtr : _ctx.blackTexturePtr;
            // A real cube source only counts when its native pointer is bound; otherwise SkyTexturePtr
            // fell back to the 2D black dummy and must use the equirectangular path.
            pd.SourceIsCube      = _ctx.environmentMapPtrPtr != IntPtr.Zero && IsCubeTexture(_ctx.Setting?.environmentMap);
            pd.EnvCubePtr        = _ctx.Textures.EnvCubemap.NativePtr;
            pd.ImportanceMapPtr  = _ctx.Textures.EnvImportanceMap.NativePtr;
            pd.RadianceMapPtr    = _ctx.Textures.EnvRadianceMap.NativePtr;
            pd.ImportanceMapRt   = _ctx.Textures.EnvImportanceMap.rt;
            pd.RadianceMapRt     = _ctx.Textures.EnvRadianceMap.rt;
            pd.ImportanceMapMips = pd.ImportanceMapRt != null ? pd.ImportanceMapRt.mipmapCount : 0;
            pd.RadianceMapMips   = pd.RadianceMapRt != null ? pd.RadianceMapRt.mipmapCount : 0;
            pd.DummyCubePtr      = _ctx.Textures.EnvDummyCube.NativePtr;
            pd.DummyTex2DPtr     = _ctx.blackTexturePtr;

            // BC6H compression: wire the compressor + reinterpret-copy event when enabled and the
            // BC6H cube is available. Staged here (record time) because all inputs are known then.
            if (_bc6uCompressCs != null && _bc6uCompressCs.IsValid &&
                _ctx.Textures.EnvCubemapBC6H != IntPtr.Zero)
            {
                pd.Bc6uCompressCs    = _bc6uCompressCs;
                pd.Bc6uCompressDs    = _bc6uCompressDs;
                pd.Bc6uScratchPtr    = _ctx.Textures.EnvCubemapBC6HScratch.NativePtr;
                pd.Bc6uCubePtr       = _ctx.Textures.EnvCubemapBC6H;
                pd.Bc6uCopyEventFunc = NativeRenderPlugin.NR_GetBC6HCopyEventFunc();
                pd.Bc6uCopyEventData = StageCopyEvent(pd.Bc6uScratchPtr, pd.Bc6uCubePtr);
            }

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.RtxptEnvMapBaker);

            // ── Base layer: BaseLayerCS writes cube mip 0 (Dst0) and the first solid-angle
            //    downsample into mip 1 (Dst1) of the same cubemap (EnvMapBaker.cpp:537-550). The
            //    original wraps this dispatch in the "ProcSkyBaseBake" marker. ──
            {
                var ds = data.BaseLayerDs;
                data.EnvBakerCb.UploadDirect(context.cmd, data.EnvBakerCbData); // single-value overload
                ds.SetConstantBuffer("g_Const", data.EnvBakerCb);
                // The source background is sampled as either an equirectangular Texture2D
                // (BackgroundSourceType==1) or a TextureCube (==2). Bind the real source to its
                // matching slot and a dummy of the right dimension to the other so neither SRV is
                // null. The native plugin auto-selects a TEXTURECUBE SRV for a DepthOrArraySize==6
                // resource, so a Unity Cubemap pointer binds correctly to t_SrcCubemapEnvMap.
                if (data.SourceIsCube)
                {
                    ds.SetTexture("t_SrcEquirectangularEnvMap", data.DummyTex2DPtr);
                    ds.SetTexture("t_SrcCubemapEnvMap", data.SkyTexturePtr);
                }
                else
                {
                    ds.SetTexture("t_SrcEquirectangularEnvMap", data.SkyTexturePtr);
                    ds.SetTexture("t_SrcCubemapEnvMap", data.DummyCubePtr);
                }
                ds.SetTexture("t_LowResPrePassCube", data.DummyCubePtr);
                ds.SetTexture("t_ProcSkyTransmittance", data.DummyTex2DPtr);
                ds.SetTexture("t_ProcSkyScatter", data.DummyTex2DPtr);
                ds.SetRWTexture("u_EnvMapCubeFacesDst0", data.EnvCubePtr, 0); // mip 0
                ds.SetRWTexture("u_EnvMapCubeFacesDst1", data.EnvCubePtr, 1); // mip 1
                cmd.BeginSample(RenderPassMarkers.RtxptEnvMapProcSkyBaseBake);
                data.BaseLayerCs.Dispatch(cmd, ds, BaseLayerGroupsXY, BaseLayerGroupsXY, 6);
                cmd.EndSample(RenderPassMarkers.RtxptEnvMapProcSkyBaseBake);
            }

            // ── MIP reduce: fill mips 2..N-1 from the previous mip with the solid-angle weighted
            //    downsample (MIPReduceCS), mirroring EnvMapBaker.cpp:555-591 ("EnvMapBakerMIPs").
            //    Each iteration reads mip i-1 and writes mip i of the same cube; the plugin's
            //    per-resource UAV barrier (NotifyResourceStates uavAccess=true) serializes the
            //    read-after-write chain. ──
            {
                cmd.BeginSample(RenderPassMarkers.RtxptEnvMapBakerMIPs);
                var ds = data.MipReduceDs;
                for (int i = 2; i < CubeMipCount; i++)
                {
                    int destRes = CubeDim >> i;
                    var groups  = (uint)((destRes + 7) / 8); // EMB_NUM_COMPUTE_THREADS_PER_DIM = 8
                    ds.SetRWTexture("u_EnvMapCubeFacesDst", data.EnvCubePtr, i); // dest mip i
                    ds.SetRWTexture("u_EnvMapCubeFacesSrc", data.EnvCubePtr, i - 1); // src  mip i-1
                    data.MipReduceCs.Dispatch(cmd, ds, groups, groups, 6);
                }

                cmd.EndSample(RenderPassMarkers.RtxptEnvMapBakerMIPs);
            }

            // ── BC6H compression (EnvMapBaker.cpp:593-631, "BC6UCompression"): compress every cube
            //    mip into the RGBA32_UINT scratch (one packed 128-bit block per texel), then
            //    reinterpret-copy the scratch into the BC6H cube. The path tracer samples that BC6H
            //    cube as t_EnvironmentMap. The importance/radiance baker below still reads the
            //    uncompressed RGBA16F cube (m_cubemap), matching the original. ──
            if (data.Bc6uCompressCs != null)
            {
                cmd.BeginSample(RenderPassMarkers.RtxptEnvMapBC6UCompression);
                var ds = data.Bc6uCompressDs;
                for (int i = 0; i < CubeMipCount; i++)
                {
                    // SrcTexture (t0): cube mip i as a 6-slice Texture2DArray (not a TextureCube).
                    ds.SetTextureArraySlice("SrcTexture", data.EnvCubePtr, i, DXGI_FORMAT_R16G16B16A16_FLOAT);
                    // OutputTexture (u0): scratch mip i (RWTexture2DArray<uint4>).
                    ds.SetRWTexture("OutputTexture", data.Bc6uScratchPtr, i);
                    data.Bc6uCompressCs.Dispatch(cmd, ds, BC6UGroupsXY, BC6UGroupsXY, 6);
                }

                // Reinterpret-copy scratch (RGBA32_UINT) → BC6H cube, per subresource.
                if (data.Bc6uCopyEventFunc != IntPtr.Zero && data.Bc6uCopyEventData != IntPtr.Zero)
                    cmd.IssuePluginEventAndData(data.Bc6uCopyEventFunc, 1, data.Bc6uCopyEventData);

                cmd.EndSample(RenderPassMarkers.RtxptEnvMapBC6UCompression);
            }

            // ── Importance/radiance map build from the baked cube's mip 0 (samples SampleLevel 0).
            //    The original wraps this in "ISBake", with the build dispatch under "GenIM" and each
            //    mip-chain generation under "MipMapGen::Dispatch" (donut MipMapGenPass). ──
            {
                cmd.BeginSample(RenderPassMarkers.RtxptEnvMapISBake);

                var ds = data.ImportanceBakerDs;
                data.ImportanceBakerCb.UploadDirect(context.cmd, data.ImportanceCbData); // single-value overload
                ds.SetConstantBuffer("g_BuilderConsts", data.ImportanceBakerCb);
                ds.SetTexture("t_EnvMapCube", data.EnvCubePtr);
                ds.SetRWTexture("u_ImportanceMap", data.ImportanceMapPtr);
                ds.SetRWTexture("u_RadianceMap", data.RadianceMapPtr);
                cmd.BeginSample(RenderPassMarkers.RtxptEnvMapGenIM);
                data.ImportanceBakerCs.Dispatch(cmd, ds, ImportanceBakerGroupsXY, ImportanceBakerGroupsXY, 1);
                cmd.EndSample(RenderPassMarkers.RtxptEnvMapGenIM);

                // ── Importance & radiance map mip chains via the ORIGINAL donut mipmapgen_cs shader
                //    (SRV input mip + bounded UAV output-mip array), not Unity's cmd.GenerateMips, so the
                //    chains are bit-identical to RTXPT's donut MipMapGenPass(MODE_COLOR). Mip 0 is the
                //    GenIM output above; this fills mips 1..N-1. The native plugin brackets each dispatch
                //    with balanced per-subresource barriers (mip N read / mips N+1.. UAV on one resource). ──
                cmd.BeginSample(RenderPassMarkers.RtxptEnvMapMipMapGen);
                GenerateDonutMipChain(cmd, data.MipMapGenCs, data.MipMapGenDs, data.ImportanceMapPtr, ImportanceMapDim, data.ImportanceMapMips);
                cmd.EndSample(RenderPassMarkers.RtxptEnvMapMipMapGen);

                cmd.BeginSample(RenderPassMarkers.RtxptEnvMapMipMapGen);
                GenerateDonutMipChain(cmd, data.MipMapGenCs, data.MipMapGenDs, data.RadianceMapPtr, ImportanceMapDim, data.RadianceMapMips);
                cmd.EndSample(RenderPassMarkers.RtxptEnvMapMipMapGen);

                cmd.EndSample(RenderPassMarkers.RtxptEnvMapISBake);
            }

            cmd.EndSample(RenderPassMarkers.RtxptEnvMapBaker);
        }

        // donut MipMapGenPass::Dispatch replicated exactly so the generated chain is bit-identical to
        // RTXPT. Group count = ceil(baseDim / GROUP_SIZE) for EVERY pass (donut over-dispatches; the
        // out-of-range UAV writes are dropped by D3D12), numLODs = min(mipCount - i*NUM_LODS - 1,
        // NUM_LODS), at most MAX_PASSES. Pass i reads mip i*NUM_LODS via the t_input SRV and writes mips
        // i*NUM_LODS+1.. via the u_output UAV array (same resource). mip 0 (the GenIM output) is the
        // chain's source and is never rewritten here. The native per-subresource barriers (see
        // ComputeDescriptorSet::Dispatch) make the SRV-read-mip / UAV-write-mips split legal under
        // Unity's per-resource state tracker.
        private static unsafe void GenerateDonutMipChain(
            CommandBuffer cmd, NativeComputePipeline cs, NativeComputeDescriptorSet ds,
            IntPtr mapPtr, int dim, int mipCount)
        {
            if (cs == null || ds == null || mapPtr == IntPtr.Zero || mipCount <= 1) return;

            uint groups = (uint)((dim + kGroupSize - 1) / kGroupSize);
            for (int i = 0; i < kMaxPasses; i++)
            {
                int inputMip = i * kNumLods;
                if (inputMip >= mipCount) break;
                int numLODs = Mathf.Min(mipCount - inputMip - 1, kNumLods);
                if (numLODs <= 0) break;

                var mc = new MipMapGenCB { dispatch = (uint)i, numLODs = (uint)numLODs };
                ds.SetTexture("t_input", mapPtr, inputMip, 1); // SRV: MostDetailedMip = inputMip, 1 mip
                ds.SetRWTextureMipArray("u_output", mapPtr, inputMip + 1); // UAV array: mips inputMip+1 .. +NUM_LODS
                ds.SetRootConstants("c_MipMapgen", &mc, 4);
                cs.Dispatch(cmd, ds, groups, groups, 1);
            }
        }

        private static unsafe void FillEnvBakerConstants(RtxptSetting setting, Light[] sceneLights)
        {
            s_envBakerCb = default;
            ref var cb = ref s_envBakerCb;

            int lightCount = 0;
            foreach (var light in sceneLights)
            {
                if (light == null || !light.enabled || !light.gameObject.activeInHierarchy) continue;
                if (light.type != LightType.Directional) continue;
                if (lightCount >= 16) break;

                var lightAngularSize = light.GetComponent<LightAngularSize>();
                
                Color   linear = light.color.linear;
                Vector3 fwd    = light.transform.forward;
                int     f      = lightCount * 8; // 8 floats per EMB_DirectionalLight
                cb.DirectionalLights[f + 0] = linear.r;
                cb.DirectionalLights[f + 1] = linear.g;
                cb.DirectionalLights[f + 2] = linear.b;
                cb.DirectionalLights[f + 3] = light.intensity;
                cb.DirectionalLights[f + 4] = fwd.x;
                cb.DirectionalLights[f + 5] = fwd.y;
                cb.DirectionalLights[f + 6] = fwd.z;
                cb.DirectionalLights[f + 7] = lightAngularSize != null ? lightAngularSize.GetAngularSize() : 0.1f;
                lightCount++;
            }

            Texture sky = setting?.environmentMap;
            // 0 - disabled; 1 - equirectangular Texture2D; 2 - TextureCube (EnvMapBaker.hlsl).
            uint backgroundSourceType = sky == null ? 0u : (IsCubeTexture(sky) ? 2u : 1u);

            // Original (EnvMapBaker.cpp:480): ScaleColor is the constant radiance-compression
            // factor only. Tint/intensity are applied at sample time via ColorMultiplier
            // (RtxptLightingUpdateBeginPass), NOT baked in here. The fork baked tint*intensity
            // here AND multiplied again at sample time, double-applying it (and dropping the BC6U
            // compression scale), so any non-1 intensity diverged from the original.
            cb.ScaleColorR           = EnvMapRadianceScale;
            cb.ScaleColorG           = EnvMapRadianceScale;
            cb.ScaleColorB           = EnvMapRadianceScale;
            cb.DirectionalLightCount = (uint)lightCount;
            cb.CubeDim               = CubeDim;
            cb.CubeDimLowRes         = CubeDimLowRes;
            cb.ProcSkyEnabled        = 0;
            cb.BackgroundSourceType  = backgroundSourceType;
        }

        // A Cubemap, or a Cube-dimension RenderTexture, is baked via the TextureCube source path
        // (BackgroundSourceType==2); everything else is treated as an equirectangular Texture2D.
        private static bool IsCubeTexture(Texture tex)
        {
            return tex is Cubemap
                || (tex is RenderTexture rt && rt.dimension == TextureDimension.Cube);
        }

        private static void FillImportanceBakerConstants()
        {
            s_importanceCb = default;
            ref var cb = ref s_importanceCb;
            cb.SourceCubeDim              = (uint)CubeDim;
            cb.SourceCubeMIPCount         = (uint)CubeMipCount; // EnvMapBaker.cpp: sourceCubemap->getDesc().mipLevels
            cb.SampleIndex                = 0;
            cb.Padding1                   = 0;
            cb.ImportanceMapDimX          = (uint)ImportanceMapDim;
            cb.ImportanceMapDimY          = (uint)ImportanceMapDim;
            cb.ImportanceMapDimInSamplesX = (uint)(ImportanceMapDim * ImportanceSamplesX);
            cb.ImportanceMapDimInSamplesY = (uint)(ImportanceMapDim * ImportanceSamplesY);
            cb.ImportanceMapNumSamplesX   = (uint)ImportanceSamplesX;
            cb.ImportanceMapNumSamplesY   = (uint)ImportanceSamplesY;
            cb.ImportanceMapInvSamples    = 1.0f / ImportanceSamples;
            cb.ImportanceMapBaseMip       = 10;
        }
    }
}