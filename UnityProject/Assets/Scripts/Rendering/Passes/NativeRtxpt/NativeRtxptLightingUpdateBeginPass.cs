using System;
using System.Runtime.InteropServices;
using NativeRender;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using Object = UnityEngine.Object;

namespace PathTracing
{
    /// <summary>
    /// LightingUpdateBegin — single unified pass that mirrors LightsBaker::UpdateFrame
    /// (the front half) in original RTXPT.
    ///
    /// Absorbs NativeRtxptEnvMapBakerPass, NativeRtxptEnvLightsBakerPass, and
    /// NativeRtxptLightingPass into one ScriptableRenderPass with the correct dispatch order:
    ///
    ///   CPU (Setup)
    ///     ControlDataSetup          — pack LightingControlData → LightControlBuffer
    ///     EnvmapAndAnalyticLights   — pack point/spot lights   → LightBuffer / LightExBuffer
    ///
    ///   GPU (ExecutePass)
    ///     1.  EnvMapBaker BaseLayerCS          — bake env cubemap (mip0 + mip1)
    ///     2.  EnvMapBaker ImportanceBakerCS    — build importance + radiance maps + GenerateMips
    ///     3.  ResetLightProxyCounters
    ///     4.  ResetPastToCurrentHistory
    ///     5.  EnvLightsBackupPast
    ///     6.  EnvLightsSubdivideBase
    ///     7.  EnvLightsSubdivideBoost
    ///     8.  BakeEmissiveTriangles            (TODO — stub)
    ///     9.  EnvLightFillLookupMap
    ///     10. EnvLightsMapPastToCurrent
    ///     11. ProcessFeedbackHistoryPreFilter  (TODO — stub)
    ///     12. ProcessFeedbackHistoryP0         (TODO — stub)
    ///     13. ComputeWeights
    ///     14. ComputeProxyCounts
    ///     15. ComputeProxyBaselineOffsets
    ///     16. CreateProxyJobs
    ///     17. ExecuteProxyJobs
    /// </summary>
    public class NativeRtxptLightingUpdateBeginPass : ScriptableRenderPass, IDisposable
    {
        // ====================================================================
        // Constants
        // ====================================================================

        // PolymorphicLight.h
        private const uint  kTypeShift        = 24;
        private const uint  kShapingEnableBit = 1u << 28;
        private const float kMinLog2Radiance  = -8f;
        private const float kMaxLog2Radiance  = 40f;

        // Env quad-tree (LightingConfig.h)
        private const uint EnvQtBaseResolution  = 4;
        private const uint EnvQtSubdivisions    = 24;
        private const uint EnvQtAdditionalNodes = 3 * EnvQtSubdivisions; // 72

        private const uint EnvQtUnboostedCount = EnvQtBaseResolution * EnvQtBaseResolution // 16
                                                 + EnvQtAdditionalNodes; // = 88

        private const uint EnvQtBoostSubdivision = 20;
        private const uint EnvQtBoostNodesMult   = EnvQtBoostSubdivision * 3 + 1; // 61
        private const uint EnvQtTotalNodeCount   = EnvQtUnboostedCount * EnvQtBoostNodesMult; // 5368

        // NEEATBaker.hlsli
        private const uint LLB_NUM_COMPUTE_THREADS     = 128;
        private const uint LLB_LOCAL_BLOCK_SIZE        = 32;
        private const uint LLB_WEIGHTS_ITEMS_PER_GROUP = LLB_LOCAL_BLOCK_SIZE * LLB_NUM_COMPUTE_THREADS;
        private const uint LLB_MAX_PROXIES_PER_TASK    = 32;

        private static readonly uint LLB_MAX_PROXY_PROC_TASKS =
            (uint)NativeRtxptBufferResources.MaxLights +
            ((uint)NativeRtxptBufferResources.ProxySamplingCount + LLB_MAX_PROXIES_PER_TASK - 1) / LLB_MAX_PROXIES_PER_TASK;

        private const uint WeightsCountHalf = NativeRtxptBufferResources.MaxLights + 1;

        // DXGI format constants
        private const uint DXGI_FORMAT_R32_FLOAT = 41u;
        private const uint DXGI_FORMAT_R32_UINT  = 42u;

        // Struct strides
        private static readonly int StrideCtrl     = Marshal.SizeOf<RtxptLightingControlData>();
        private static readonly int StrideLights   = Marshal.SizeOf<RtxptPolymorphicLightInfo>();
        private static readonly int StrideLightsEx = Marshal.SizeOf<RtxptPolymorphicLightInfoEx>();

        // EnvMapBaker dimensions
        private const int CubeDim                 = 256;
        private const int CubeDimLowRes           = 32;
        private const int ImportanceMapDim        = 1024;
        private const int ImportanceSamples       = 16;
        private const int ImportanceSamplesX      = 4;
        private const int ImportanceSamplesY      = 4;
        private const int BaseLayerGroupsXY       = (CubeDim / 2 + 7) / 8; // 16
        private const int ImportanceBakerGroupsXY = (ImportanceMapDim + 15) / 16; // 64

        // EnvLightLookupMap dimension
        private const int EnvLookupMapDim = 1024;

        // ====================================================================
        // GPU pipelines — EnvMapBaker
        // ====================================================================

        private readonly NativeComputePipeline      _baseLayerCs;
        private readonly NativeComputeDescriptorSet _baseLayerDs;
        private readonly NativeComputePipeline      _importanceBakerCs;
        private readonly NativeComputeDescriptorSet _importanceBakerDs;

        // ====================================================================
        // GPU pipelines — EnvLightsBaker
        // ====================================================================

        private readonly NativeComputePipeline      _backupPastCs;
        private readonly NativeComputeDescriptorSet _backupPastDs;
        private readonly NativeComputePipeline      _subdivideBaseCs;
        private readonly NativeComputeDescriptorSet _subdivideBaseDs;
        private readonly NativeComputePipeline      _subdivideBoostCs;
        private readonly NativeComputeDescriptorSet _subdivideBoostDs;
        private readonly NativeComputePipeline      _fillLookupMapCs;
        private readonly NativeComputeDescriptorSet _fillLookupMapDs;
        private readonly NativeComputePipeline      _mapPastToCurrentCs;
        private readonly NativeComputeDescriptorSet _mapPastToCurrentDs;

        // ====================================================================
        // GPU pipelines — LightingPass (proxy build)
        // ====================================================================

        private readonly NativeComputePipeline      _resetLightProxyCountersCs;
        private readonly NativeComputeDescriptorSet _resetLightProxyCountersDs;
        private readonly NativeComputePipeline      _resetPastToCurrentHistoryCs;
        private readonly NativeComputeDescriptorSet _resetPastToCurrentHistoryDs;
        private readonly NativeComputePipeline      _computeWeightsCs;
        private readonly NativeComputeDescriptorSet _computeWeightsDs;
        private readonly NativeComputePipeline      _computeProxyCountsCs;
        private readonly NativeComputeDescriptorSet _computeProxyCountsDs;
        private readonly NativeComputePipeline      _computeProxyBaselineOffsetsCs;
        private readonly NativeComputeDescriptorSet _computeProxyBaselineOffsetsDs;
        private readonly NativeComputePipeline      _createProxyJobsCs;
        private readonly NativeComputeDescriptorSet _createProxyJobsDs;
        private readonly NativeComputePipeline      _executeProxyJobsCs;
        private readonly NativeComputeDescriptorSet _executeProxyJobsDs;

        // ====================================================================
        // GPU pipeline — BakeEmissiveTriangles
        // ====================================================================

        private readonly NativeComputePipeline      _bakeEmissiveTrianglesCs;
        private readonly NativeComputeDescriptorSet _bakeEmissiveTrianglesDs;

        // ====================================================================
        // Owned render textures
        // ====================================================================

        private RenderTexture _envCubeMip0Rt; // 256×256 Cube RGBA16F UAV
        private RenderTexture _envCubeMip1Rt; // 128×128 Cube RGBA16F UAV
        private RenderTexture _importanceMapRt; // 1024×1024 2D RFloat UAV + mips
        private RenderTexture _radianceMapRt; // 1024×1024 2D RGBA16F UAV + mips
        private RenderTexture _dummyCubeRt; // 4×4 Cube dummy SRV
        private RenderTexture _envLightLookupMapRt; // 1024×1024 2D R32_UINT UAV

        // ====================================================================
        // GPU constant buffers
        // ====================================================================

        private GraphicsBuffer _envBakerCb; // EnvMapBakerConstants (704 bytes)
        private GraphicsBuffer _importanceBakerCb; // EnvMapImportanceSamplingBakerConstants (48 bytes)

        // ====================================================================
        // CPU staging
        // ====================================================================

        private static readonly byte[]                        s_envBakerBytes   = new byte[704];
        private static readonly byte[]                        s_importanceBytes = new byte[48];
        private static readonly RtxptLightingControlData[]    s_controlStaging  = new RtxptLightingControlData[1];
        private static          RtxptPolymorphicLightInfo[]   s_lightsStaging   = new RtxptPolymorphicLightInfo[NativeRtxptBufferResources.MaxLights];
        private static          RtxptPolymorphicLightInfoEx[] s_lightsExStaging = new RtxptPolymorphicLightInfoEx[NativeRtxptBufferResources.MaxLights];

        // ====================================================================
        // Per-frame state
        // ====================================================================

        private NativeRtxptPassContext _ctx;
        private int                    _analyticLightCount;
        private int                    _emissiveTaskCount;
        private uint                   _emissiveTotalTriCount;
        private bool                   _ping = true; // ping-pong for weights buffer
        private int                    _dbgFrameCounter;

        /// <summary>lightsBuffer index where emissive triangles start (= EnvQtTotalNodeCount + analyticLightCount).</summary>
        public uint EmissiveLightOffset  => EnvQtTotalNodeCount + (uint)_analyticLightCount;
        /// <summary>Total emissive triangle-light count produced last frame.</summary>
        public uint EmissiveTriangleCount => _emissiveTotalTriCount;

        // ====================================================================
        // Constructor
        // ====================================================================

        public NativeRtxptLightingUpdateBeginPass(
            // EnvMapBaker
            NativeComputeShader baseLayerCs,
            NativeComputeShader importanceBakerCs,
            // EnvLightsBaker
            NativeComputeShader envLightsBackupPastCs,
            NativeComputeShader envLightsSubdivideBaseCs,
            NativeComputeShader envLightsSubdivideBoostCs,
            NativeComputeShader envLightsFillLookupMapCs,
            NativeComputeShader envLightsMapPastToCurrentCs,
            // Proxy build
            NativeComputeShader resetLightProxyCountersCs,
            NativeComputeShader resetPastToCurrentHistoryCs,
            NativeComputeShader computeWeightsCs,
            NativeComputeShader computeProxyCountsCs,
            NativeComputeShader computeProxyBaselineOffsetsCs,
            NativeComputeShader createProxyJobsCs,
            NativeComputeShader executeProxyJobsCs,
            // Emissive triangles
            NativeComputeShader bakeEmissiveTrianglesCs)
        {
            // EnvMapBaker
            _baseLayerCs       = new NativeComputePipeline(baseLayerCs);
            _baseLayerDs       = new NativeComputeDescriptorSet(_baseLayerCs);
            _importanceBakerCs = new NativeComputePipeline(importanceBakerCs);
            _importanceBakerDs = new NativeComputeDescriptorSet(_importanceBakerCs);

            // EnvLightsBaker
            _backupPastCs       = new NativeComputePipeline(envLightsBackupPastCs);
            _backupPastDs       = new NativeComputeDescriptorSet(_backupPastCs);
            _subdivideBaseCs    = new NativeComputePipeline(envLightsSubdivideBaseCs);
            _subdivideBaseDs    = new NativeComputeDescriptorSet(_subdivideBaseCs);
            _subdivideBoostCs   = new NativeComputePipeline(envLightsSubdivideBoostCs);
            _subdivideBoostDs   = new NativeComputeDescriptorSet(_subdivideBoostCs);
            _fillLookupMapCs    = new NativeComputePipeline(envLightsFillLookupMapCs);
            _fillLookupMapDs    = new NativeComputeDescriptorSet(_fillLookupMapCs);
            _mapPastToCurrentCs = new NativeComputePipeline(envLightsMapPastToCurrentCs);
            _mapPastToCurrentDs = new NativeComputeDescriptorSet(_mapPastToCurrentCs);

            // Proxy build
            _resetLightProxyCountersCs     = new NativeComputePipeline(resetLightProxyCountersCs);
            _resetLightProxyCountersDs     = new NativeComputeDescriptorSet(_resetLightProxyCountersCs);
            _resetPastToCurrentHistoryCs   = new NativeComputePipeline(resetPastToCurrentHistoryCs);
            _resetPastToCurrentHistoryDs   = new NativeComputeDescriptorSet(_resetPastToCurrentHistoryCs);
            _computeWeightsCs              = new NativeComputePipeline(computeWeightsCs);
            _computeWeightsDs              = new NativeComputeDescriptorSet(_computeWeightsCs);
            _computeProxyCountsCs          = new NativeComputePipeline(computeProxyCountsCs);
            _computeProxyCountsDs          = new NativeComputeDescriptorSet(_computeProxyCountsCs);
            _computeProxyBaselineOffsetsCs = new NativeComputePipeline(computeProxyBaselineOffsetsCs);
            _computeProxyBaselineOffsetsDs = new NativeComputeDescriptorSet(_computeProxyBaselineOffsetsCs);
            _createProxyJobsCs             = new NativeComputePipeline(createProxyJobsCs);
            _createProxyJobsDs             = new NativeComputeDescriptorSet(_createProxyJobsCs);
            _executeProxyJobsCs            = new NativeComputePipeline(executeProxyJobsCs);
            _executeProxyJobsDs            = new NativeComputeDescriptorSet(_executeProxyJobsCs);
            _bakeEmissiveTrianglesCs       = new NativeComputePipeline(bakeEmissiveTrianglesCs);
            _bakeEmissiveTrianglesDs       = new NativeComputeDescriptorSet(_bakeEmissiveTrianglesCs);

            EnsureRenderTextures();
            EnsureConstantBuffers();
        }

        // ====================================================================
        // Dispose
        // ====================================================================

        public void Dispose()
        {
            // EnvMapBaker pipelines
            _baseLayerDs?.Dispose();
            _baseLayerCs?.Dispose();
            _importanceBakerDs?.Dispose();
            _importanceBakerCs?.Dispose();

            // EnvLightsBaker pipelines
            _backupPastDs?.Dispose();
            _backupPastCs?.Dispose();
            _subdivideBaseDs?.Dispose();
            _subdivideBaseCs?.Dispose();
            _subdivideBoostDs?.Dispose();
            _subdivideBoostCs?.Dispose();
            _fillLookupMapDs?.Dispose();
            _fillLookupMapCs?.Dispose();
            _mapPastToCurrentDs?.Dispose();
            _mapPastToCurrentCs?.Dispose();

            // Proxy build pipelines
            _resetLightProxyCountersDs?.Dispose();
            _resetLightProxyCountersCs?.Dispose();
            _resetPastToCurrentHistoryDs?.Dispose();
            _resetPastToCurrentHistoryCs?.Dispose();
            _computeWeightsDs?.Dispose();
            _computeWeightsCs?.Dispose();
            _computeProxyCountsDs?.Dispose();
            _computeProxyCountsCs?.Dispose();
            _computeProxyBaselineOffsetsDs?.Dispose();
            _computeProxyBaselineOffsetsCs?.Dispose();
            _createProxyJobsDs?.Dispose();
            _createProxyJobsCs?.Dispose();
            _executeProxyJobsDs?.Dispose();
            _executeProxyJobsCs?.Dispose();
            _bakeEmissiveTrianglesDs?.Dispose();
            _bakeEmissiveTrianglesCs?.Dispose();

            // Render textures
            DestroyRT(ref _envCubeMip0Rt);
            DestroyRT(ref _envCubeMip1Rt);
            DestroyRT(ref _importanceMapRt);
            DestroyRT(ref _radianceMapRt);
            DestroyRT(ref _dummyCubeRt);
            DestroyRT(ref _envLightLookupMapRt);

            // Constant buffers
            _envBakerCb?.Dispose();
            _envBakerCb = null;
            _importanceBakerCb?.Dispose();
            _importanceBakerCb = null;
        }

        // ====================================================================
        // Setup — main thread, called once per frame before RecordRenderGraph
        // ====================================================================

        public void Setup(NativeRtxptPassContext ctx)
        {
            _dbgFrameCounter++;
            _ctx = ctx;
            EnsureRenderTextures();
            EnsureConstantBuffers();

            // --- EnvMapBaker CPU work ---
            FillEnvBakerConstants(ctx.Setting);
            _envBakerCb.SetData(s_envBakerBytes);
            FillImportanceBakerConstants();
            _importanceBakerCb.SetData(s_importanceBytes);

            // Expose baked env pointers for downstream passes (BuildStablePlanes / FillStablePlanes)
            ctx.BakedEnvCubePtr                = _envCubeMip0Rt.IsCreated() ? _envCubeMip0Rt.GetNativeTexturePtr() : IntPtr.Zero;
            ctx.EnvImportanceMapPtr            = _importanceMapRt.IsCreated() ? _importanceMapRt.GetNativeTexturePtr() : IntPtr.Zero;
            ctx.EnvRadianceAndImportanceMapPtr = _radianceMapRt.IsCreated() ? _radianceMapRt.GetNativeTexturePtr() : IntPtr.Zero;

            // Expose env-light lookup map pointer
            ctx.EnvLightLookupMapPtr = _envLightLookupMapRt != null && _envLightLookupMapRt.IsCreated()
                ? _envLightLookupMapRt.GetNativeTexturePtr()
                : IntPtr.Zero;

            // --- LightsBaker CPU work (ControlDataSetup + EnvmapAndAnalyticLightBuffers) ---
            _analyticLightCount = CollectAndPackLights();
            UploadLightData();

            // --- Emissive triangles: build tasks, upload to scratch, update SubInstance buffer ---
            var gpuScene = ctx.GpuScene;


            uint emissiveLightOffset = EnvQtTotalNodeCount + (uint)_analyticLightCount;
            gpuScene.PrepareEmissiveTriangleTasks(emissiveLightOffset, ctx.Buffers.LightScratchBuffer);
            _emissiveTaskCount     = gpuScene.LastEmissiveTaskCount;
            _emissiveTotalTriCount = gpuScene.LastEmissiveTriangleCount;
        }

        // ====================================================================
        // RecordRenderGraph
        // ====================================================================

        private class PassData
        {
            // --- EnvMapBaker ---
            internal NativeComputePipeline      BaseLayerCs;
            internal NativeComputeDescriptorSet BaseLayerDs;
            internal NativeComputePipeline      ImportanceBakerCs;
            internal NativeComputeDescriptorSet ImportanceBakerDs;
            internal IntPtr                     EnvBakerCbPtr;
            internal IntPtr                     ImportanceBakerCbPtr;
            internal IntPtr                     SkyTexturePtr;
            internal IntPtr                     EnvCubeMip0Ptr;
            internal IntPtr                     EnvCubeMip1Ptr;
            internal IntPtr                     ImportanceMapPtr;
            internal IntPtr                     RadianceMapPtr;
            internal RenderTexture              ImportanceMapRt;
            internal RenderTexture              RadianceMapRt;
            internal IntPtr                     DummyCubePtr;
            internal IntPtr                     DummyTex2DPtr;

            // --- EnvLightsBaker ---
            internal NativeComputePipeline      BackupPastCs;
            internal NativeComputeDescriptorSet BackupPastDs;
            internal NativeComputePipeline      SubdivideBaseCs;
            internal NativeComputeDescriptorSet SubdivideBaseDs;
            internal NativeComputePipeline      SubdivideBoostCs;
            internal NativeComputeDescriptorSet SubdivideBoostDs;
            internal NativeComputePipeline      FillLookupMapCs;
            internal NativeComputeDescriptorSet FillLookupMapDs;
            internal NativeComputePipeline      MapPastToCurrentCs;
            internal NativeComputeDescriptorSet MapPastToCurrentDs;
            internal IntPtr                     EnvLightLookupMapPtr;

            // --- BakeEmissiveTriangles ---
            internal NativeComputePipeline      BakeEmissiveTrianglesCs;
            internal NativeComputeDescriptorSet BakeEmissiveTrianglesDs;
            internal IntPtr                     SubInstanceDataPtr;
            internal int                        SubInstanceDataCount;
            internal int                        SubInstanceDataStride;
            internal IntPtr                     InstanceDataPtr;
            internal int                        InstanceDataCount;
            internal int                        InstanceDataStride;
            internal IntPtr                     GeometryDataPtr;
            internal int                        GeometryDataCount;
            internal int                        GeometryDataStride;
            internal IntPtr                     PTMaterialDataPtr;
            internal int                        PTMaterialDataCount;
            internal int                        PTMaterialDataStride;
            internal int                        EmissiveTaskCount;

            // --- Proxy build ---
            internal NativeComputePipeline      ResetProxyCountersCs;
            internal NativeComputeDescriptorSet ResetProxyCountersDs;
            internal NativeComputePipeline      ResetPastToCurrentCs;
            internal NativeComputeDescriptorSet ResetPastToCurrentDs;
            internal NativeComputePipeline      ComputeWeightsCs;
            internal NativeComputeDescriptorSet ComputeWeightsDs;
            internal NativeComputePipeline      ComputeProxyCountsCs;
            internal NativeComputeDescriptorSet ComputeProxyCountsDs;
            internal NativeComputePipeline      ComputeProxyBaselineOffsetsCs;
            internal NativeComputeDescriptorSet ComputeProxyBaselineOffsetsDs;
            internal NativeComputePipeline      CreateProxyJobsCs;
            internal NativeComputeDescriptorSet CreateProxyJobsDs;
            internal NativeComputePipeline      ExecuteProxyJobsCs;
            internal NativeComputeDescriptorSet ExecuteProxyJobsDs;
            internal uint                       TotalLightCount;
            internal uint                       HistoricTotalLightCount;

            // --- Shared ---
            internal NativeRtxptPassContext Ctx;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (_resetLightProxyCountersCs == null || _computeWeightsCs == null ||
                _computeProxyCountsCs == null || _computeProxyBaselineOffsetsCs == null ||
                _createProxyJobsCs == null || _executeProxyJobsCs == null)
            {
                Debug.LogWarning("[NativeRtxptLightingUpdateBeginPass] Required proxy-build shaders missing — pass skipped.");
                return;
            }

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.LightingUpdateBegin", out var pd);

            // EnvMapBaker
            pd.BaseLayerCs          = _baseLayerCs;
            pd.BaseLayerDs          = _baseLayerDs;
            pd.ImportanceBakerCs    = _importanceBakerCs;
            pd.ImportanceBakerDs    = _importanceBakerDs;
            pd.EnvBakerCbPtr        = _envBakerCb.GetNativeBufferPtr();
            pd.ImportanceBakerCbPtr = _importanceBakerCb.GetNativeBufferPtr();
            var skyTex = _ctx.Setting?.environmentMap;
            pd.SkyTexturePtr    = skyTex != null ? skyTex.GetNativeTexturePtr() : Texture2D.blackTexture.GetNativeTexturePtr();
            pd.EnvCubeMip0Ptr   = _envCubeMip0Rt.GetNativeTexturePtr();
            pd.EnvCubeMip1Ptr   = _envCubeMip1Rt.GetNativeTexturePtr();
            pd.ImportanceMapPtr = _importanceMapRt.GetNativeTexturePtr();
            pd.RadianceMapPtr   = _radianceMapRt.GetNativeTexturePtr();
            pd.ImportanceMapRt  = _importanceMapRt;
            pd.RadianceMapRt    = _radianceMapRt;
            pd.DummyCubePtr     = _dummyCubeRt.GetNativeTexturePtr();
            pd.DummyTex2DPtr    = Texture2D.blackTexture.GetNativeTexturePtr();

            // EnvLightsBaker
            pd.BackupPastCs         = _backupPastCs;
            pd.BackupPastDs         = _backupPastDs;
            pd.SubdivideBaseCs      = _subdivideBaseCs;
            pd.SubdivideBaseDs      = _subdivideBaseDs;
            pd.SubdivideBoostCs     = _subdivideBoostCs;
            pd.SubdivideBoostDs     = _subdivideBoostDs;
            pd.FillLookupMapCs      = _fillLookupMapCs;
            pd.FillLookupMapDs      = _fillLookupMapDs;
            pd.MapPastToCurrentCs   = _mapPastToCurrentCs;
            pd.MapPastToCurrentDs   = _mapPastToCurrentDs;
            pd.EnvLightLookupMapPtr = _ctx.EnvLightLookupMapPtr;

            // Proxy build
            pd.ResetProxyCountersCs          = _resetLightProxyCountersCs;
            pd.ResetProxyCountersDs          = _resetLightProxyCountersDs;
            pd.ResetPastToCurrentCs          = _resetPastToCurrentHistoryCs;
            pd.ResetPastToCurrentDs          = _resetPastToCurrentHistoryDs;
            pd.ComputeWeightsCs              = _computeWeightsCs;
            pd.ComputeWeightsDs              = _computeWeightsDs;
            pd.ComputeProxyCountsCs          = _computeProxyCountsCs;
            pd.ComputeProxyCountsDs          = _computeProxyCountsDs;
            pd.ComputeProxyBaselineOffsetsCs = _computeProxyBaselineOffsetsCs;
            pd.ComputeProxyBaselineOffsetsDs = _computeProxyBaselineOffsetsDs;
            pd.CreateProxyJobsCs             = _createProxyJobsCs;
            pd.CreateProxyJobsDs             = _createProxyJobsDs;
            pd.ExecuteProxyJobsCs            = _executeProxyJobsCs;
            pd.ExecuteProxyJobsDs            = _executeProxyJobsDs;
            pd.TotalLightCount               = EnvQtTotalNodeCount + (uint)_analyticLightCount + _emissiveTotalTriCount;
            pd.HistoricTotalLightCount       = EnvQtTotalNodeCount + (uint)_analyticLightCount + _emissiveTotalTriCount;

            // BakeEmissiveTriangles
            pd.BakeEmissiveTrianglesCs = _bakeEmissiveTrianglesCs;
            pd.BakeEmissiveTrianglesDs = _bakeEmissiveTrianglesDs;
            pd.EmissiveTaskCount       = _emissiveTaskCount;
            var gpuSceneRef = _ctx.GpuScene;

            gpuSceneRef.GetSceneBufferPtrs(
                out pd.SubInstanceDataPtr, out pd.SubInstanceDataCount, out pd.SubInstanceDataStride,
                out pd.InstanceDataPtr, out pd.InstanceDataCount, out pd.InstanceDataStride,
                out pd.GeometryDataPtr, out pd.GeometryDataCount, out pd.GeometryDataStride,
                out pd.PTMaterialDataPtr, out pd.PTMaterialDataCount, out pd.PTMaterialDataStride);


            // Flip ping-pong AFTER filling passData so UploadLightData used same side
            _ping = !_ping;

            pd.Ctx = _ctx;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        // ====================================================================
        // ExecutePass — GPU dispatch in original RTXPT order
        // ====================================================================

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var buf = data.Ctx.Buffers;

            cmd.BeginSample("Rtxpt.LightingUpdateBegin");

            // ----------------------------------------------------------------
            // 1–2. EnvMapBaker
            // ----------------------------------------------------------------
            if (data.BaseLayerCs != null && data.ImportanceBakerCs != null)
            {
                cmd.BeginSample("Rtxpt.EnvMapBaker");

                // 1. BaseLayerCS — write env cube mip0 + mip1
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
                    data.BaseLayerCs.Dispatch(cmd, ds, BaseLayerGroupsXY, BaseLayerGroupsXY, 6);
                }

                // 2. ImportanceBakerCS — build importance + radiance maps, then generate mips
                {
                    var ds = data.ImportanceBakerDs;
                    ds.SetConstantBuffer("g_BuilderConsts", data.ImportanceBakerCbPtr);
                    ds.SetTexture("t_EnvMapCube", data.EnvCubeMip0Ptr);
                    ds.SetRWTexture("u_ImportanceMap", data.ImportanceMapPtr);
                    ds.SetRWTexture("u_RadianceMap", data.RadianceMapPtr);
                    data.ImportanceBakerCs.Dispatch(cmd, ds, ImportanceBakerGroupsXY, ImportanceBakerGroupsXY, 1);
                }

                cmd.GenerateMips(data.ImportanceMapRt);
                cmd.GenerateMips(data.RadianceMapRt);

                cmd.EndSample("Rtxpt.EnvMapBaker");
            }

            // Buffer pointers used by the remaining passes
            var pCtrl     = buf.LightControlBuffer.GetNativeBufferPtr();
            var pLights   = buf.LightBuffer.GetNativeBufferPtr();
            var pLightsEx = buf.LightExBuffer.GetNativeBufferPtr();
            var pScratch  = buf.LightScratchBuffer.GetNativeBufferPtr();
            var pScrList  = buf.ScratchListBuffer.GetNativeBufferPtr();
            var pWeights  = buf.LightWeightsBuffer.GetNativeBufferPtr();
            var pHistCur  = buf.HistoryRemapCurrentToPast.GetNativeBufferPtr();
            var pHistPas  = buf.HistoryRemapPastToCurrent.GetNativeBufferPtr();
            var pProxyCnt = buf.LightProxyCounters.GetNativeBufferPtr();
            var pProxies  = buf.LightSamplingProxies.GetNativeBufferPtr();

            int cCtrl     = buf.LightControlBuffer.count;
            int cLights   = buf.LightBuffer.count;
            int cLightsEx = buf.LightExBuffer.count;
            int cScrList  = buf.ScratchListBuffer.count;
            int cWeights  = buf.LightWeightsBuffer.count;
            int cHistCur  = buf.HistoryRemapCurrentToPast.count;
            int cHistPas  = buf.HistoryRemapPastToCurrent.count;
            int cProxyCnt = buf.LightProxyCounters.count;
            int cProxies  = buf.LightSamplingProxies.count;

            uint total            = data.TotalLightCount;
            uint historic         = data.HistoricTotalLightCount;
            var  envImportancePtr = data.Ctx.EnvRadianceAndImportanceMapPtr;
            var  envLookupMapPtr  = data.EnvLightLookupMapPtr;

            // ----------------------------------------------------------------
            // 3. ResetLightProxyCounters
            // ----------------------------------------------------------------
            {
                var ds = data.ResetProxyCountersDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_perLightProxyCounters", pProxyCnt, cProxyCnt, DXGI_FORMAT_R32_UINT);
                uint gx = (total + 1 + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS;
                data.ResetProxyCountersCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            // ----------------------------------------------------------------
            // 4. ResetPastToCurrentHistory
            // ----------------------------------------------------------------
            {
                uint items = Math.Max(historic, total);
                var  ds    = data.ResetPastToCurrentDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_historyRemapPastToCurrent", pHistPas, cHistPas, DXGI_FORMAT_R32_UINT);
                uint gx = Math.Max(1u, (items + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS);
                data.ResetPastToCurrentCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            // ----------------------------------------------------------------
            // 5. EnvLightsBackupPast
            // ----------------------------------------------------------------
            {
                var ds = data.BackupPastDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWStructuredBuffer("u_lightsBuffer", pLights, cLights, StrideLights);
                ds.SetRWTypedBuffer("u_scratchList", pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                uint gx = (EnvQtTotalNodeCount + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS;
                data.BackupPastCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            // ----------------------------------------------------------------
            // 6. EnvLightsSubdivideBase
            // ----------------------------------------------------------------
            {
                var ds = data.SubdivideBaseDs;
                ds.SetTexture("t_envRadianceAndImportanceMap", envImportancePtr);
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_scratchList", pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                data.SubdivideBaseCs.Dispatch(cmd, ds, 1, 1, 1);
            }

            // ----------------------------------------------------------------
            // 7. EnvLightsSubdivideBoost
            // ----------------------------------------------------------------
            {
                var ds = data.SubdivideBoostDs;
                ds.SetTexture("t_envRadianceAndImportanceMap", envImportancePtr);
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWStructuredBuffer("u_lightsBuffer", pLights, cLights, StrideLights);
                ds.SetRWStructuredBuffer("u_lightsExBuffer", pLightsEx, cLightsEx, StrideLightsEx);
                ds.SetRWTypedBuffer("u_scratchList", pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_historyRemapCurrentToPast", pHistCur, cHistCur, DXGI_FORMAT_R32_UINT);
                ds.SetRWTexture("u_envLightLookupMap", envLookupMapPtr);
                data.SubdivideBoostCs.Dispatch(cmd, ds, EnvQtUnboostedCount, 1, 1);
            }

            // ----------------------------------------------------------------
            // 8. BakeEmissiveTriangles
            // ----------------------------------------------------------------
            if (data.EmissiveTaskCount > 0)
            {
                cmd.BeginSample("Rtxpt.BakeEmissiveTriangles");
                var ds = data.BakeEmissiveTrianglesDs;
                // UAV outputs
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWBuffer("u_scratchBuffer", pScratch);
                ds.SetRWStructuredBuffer("u_lightsBuffer", pLights, cLights, StrideLights);
                ds.SetRWTypedBuffer("u_historyRemapCurrentToPast", pHistCur, cHistCur, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_historyRemapPastToCurrent", pHistPas, cHistPas, DXGI_FORMAT_R32_UINT);
                ds.SetRWStructuredBuffer("u_lightsExBuffer", pLightsEx, cLightsEx, StrideLightsEx);

                // SRV scene inputs
                ds.SetStructuredBuffer("t_SubInstanceData", data.SubInstanceDataPtr, data.SubInstanceDataCount, data.SubInstanceDataStride);
                ds.SetStructuredBuffer("t_InstanceData", data.InstanceDataPtr, data.InstanceDataCount, data.InstanceDataStride);
                ds.SetStructuredBuffer("t_GeometryData", data.GeometryDataPtr, data.GeometryDataCount, data.GeometryDataStride);
                ds.SetStructuredBuffer("t_PTMaterialData", data.PTMaterialDataPtr, data.PTMaterialDataCount, data.PTMaterialDataStride);
                // Bindless vertex/index buffers
                data.Ctx.GpuScene?.BindToShader(ds);
                // Dispatch: ceil(taskCount / 8) groups × [256,1,1] threads = taskCount × 32 threads total
                uint gxBake = Math.Max(1u, ((uint)data.EmissiveTaskCount + 7u) / 8u);
                data.BakeEmissiveTrianglesCs.Dispatch(cmd, ds, gxBake, 1, 1);
                cmd.EndSample("Rtxpt.BakeEmissiveTriangles");
            }

            // ----------------------------------------------------------------
            // 9. EnvLightFillLookupMap
            // ----------------------------------------------------------------
            {
                var ds = data.FillLookupMapDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWStructuredBuffer("u_lightsBuffer", pLights, cLights, StrideLights);
                ds.SetRWTexture("u_envLightLookupMap", envLookupMapPtr);
                data.FillLookupMapCs.Dispatch(cmd, ds, EnvQtTotalNodeCount, 1, 1);
            }

            // ----------------------------------------------------------------
            // 10. EnvLightsMapPastToCurrent
            // ----------------------------------------------------------------
            {
                var ds = data.MapPastToCurrentDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_scratchList", pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_historyRemapPastToCurrent", pHistPas, cHistPas, DXGI_FORMAT_R32_UINT);
                ds.SetRWTexture("u_envLightLookupMap", envLookupMapPtr);
                uint gx = (EnvQtTotalNodeCount + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS;
                data.MapPastToCurrentCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            // ----------------------------------------------------------------
            // 11. ProcessFeedbackHistoryPreFilter — TODO: optional, not yet implemented
            // 12. ProcessFeedbackHistoryP0        — TODO: not yet implemented
            // ----------------------------------------------------------------

            // ----------------------------------------------------------------
            // 13. ComputeWeights
            // ----------------------------------------------------------------
            {
                var ds = data.ComputeWeightsDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWStructuredBuffer("u_lightsBuffer", pLights, cLights, StrideLights);
                ds.SetRWStructuredBuffer("u_lightsExBuffer", pLightsEx, cLightsEx, StrideLightsEx);
                ds.SetRWTypedBuffer("u_lightWeights", pWeights, cWeights, DXGI_FORMAT_R32_FLOAT);
                ds.SetRWTypedBuffer("u_historyRemapCurrentToPast", pHistCur, cHistCur, DXGI_FORMAT_R32_UINT);
                uint gx = Math.Max(1u, (total + LLB_WEIGHTS_ITEMS_PER_GROUP - 1) / LLB_WEIGHTS_ITEMS_PER_GROUP);
                data.ComputeWeightsCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            // ----------------------------------------------------------------
            // 14. ComputeProxyCounts
            // ----------------------------------------------------------------
            {
                var ds = data.ComputeProxyCountsDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_scratchList", pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_lightWeights", pWeights, cWeights, DXGI_FORMAT_R32_FLOAT);
                ds.SetRWTypedBuffer("u_perLightProxyCounters", pProxyCnt, cProxyCnt, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_lightSamplingProxies", pProxies, cProxies, DXGI_FORMAT_R32_UINT);
                uint gx = Math.Max(1u, (total + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS);
                data.ComputeProxyCountsCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            // ----------------------------------------------------------------
            // 15. ComputeProxyBaselineOffsets  (single thread-group prefix-sum)
            // ----------------------------------------------------------------
            {
                var ds = data.ComputeProxyBaselineOffsetsDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_lightSamplingProxies", pProxies, cProxies, DXGI_FORMAT_R32_UINT);
                data.ComputeProxyBaselineOffsetsCs.Dispatch(cmd, ds, 1, 1, 1);
            }

            // ----------------------------------------------------------------
            // 16. CreateProxyJobs
            // ----------------------------------------------------------------
            {
                var ds = data.CreateProxyJobsDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWBuffer("u_scratchBuffer", pScratch);
                ds.SetRWTypedBuffer("u_scratchList", pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_perLightProxyCounters", pProxyCnt, cProxyCnt, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_lightSamplingProxies", pProxies, cProxies, DXGI_FORMAT_R32_UINT);
                uint gx = Math.Max(1u, (total + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS);
                data.CreateProxyJobsCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            // ----------------------------------------------------------------
            // 17. ExecuteProxyJobs
            // ----------------------------------------------------------------
            {
                var ds = data.ExecuteProxyJobsDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWBuffer("u_scratchBuffer", pScratch);
                ds.SetRWTypedBuffer("u_lightSamplingProxies", pProxies, cProxies, DXGI_FORMAT_R32_UINT);
                uint gx = Math.Max(1u, (LLB_MAX_PROXY_PROC_TASKS + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS);
                data.ExecuteProxyJobsCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            cmd.EndSample("Rtxpt.LightingUpdateBegin");
        }

        // ====================================================================
        // CPU helpers — EnvMapBaker constants
        // ====================================================================

        private void FillEnvBakerConstants(NativeRtxptSetting setting)
        {
            Array.Clear(s_envBakerBytes, 0, s_envBakerBytes.Length);
            int lightCount = 0;

            foreach (var light in Object.FindObjectsByType<Light>(FindObjectsSortMode.None))
            {
                if (!light.enabled || !light.gameObject.activeInHierarchy) continue;
                if (light.type != LightType.Directional) continue;
                if (lightCount >= 16) break;

                Color linear    = light.color.linear;
                float intensity = light.intensity;
                int   offset    = lightCount * 32;

                WriteF32(s_envBakerBytes, offset + 0, linear.r);
                WriteF32(s_envBakerBytes, offset + 4, linear.g);
                WriteF32(s_envBakerBytes, offset + 8, linear.b);
                WriteF32(s_envBakerBytes, offset + 12, intensity);

                Vector3 fwd = light.transform.forward;
                WriteF32(s_envBakerBytes, offset + 16, fwd.x);
                WriteF32(s_envBakerBytes, offset + 20, fwd.y);
                WriteF32(s_envBakerBytes, offset + 24, fwd.z);
                WriteF32(s_envBakerBytes, offset + 28, 0.1f); // angular size

                lightCount++;
            }

            float envIntensity = setting?.environmentMapIntensity ?? 1.0f;
            Color tint         = setting?.environmentMapTint ?? Color.white;
            bool  hasSky       = setting?.environmentMap != null;
            int   o            = 672;

            WriteF32(s_envBakerBytes, o + 0, tint.linear.r * envIntensity);
            WriteF32(s_envBakerBytes, o + 4, tint.linear.g * envIntensity);
            WriteF32(s_envBakerBytes, o + 8, tint.linear.b * envIntensity);
            WriteU32(s_envBakerBytes, o + 12, (uint)lightCount);
            WriteU32(s_envBakerBytes, o + 16, (uint)CubeDim);
            WriteU32(s_envBakerBytes, o + 20, (uint)CubeDimLowRes);
            WriteU32(s_envBakerBytes, o + 24, 0u); // ProcSkyEnabled = 0
            WriteU32(s_envBakerBytes, o + 28, hasSky ? 1u : 0u); // BackgroundSourceType
        }

        private static void FillImportanceBakerConstants()
        {
            Array.Clear(s_importanceBytes, 0, s_importanceBytes.Length);
            WriteU32(s_importanceBytes, 0, (uint)CubeDim);
            WriteU32(s_importanceBytes, 4, 1u);
            WriteU32(s_importanceBytes, 8, 0u);
            WriteU32(s_importanceBytes, 12, 0u);
            WriteU32(s_importanceBytes, 16, (uint)ImportanceMapDim);
            WriteU32(s_importanceBytes, 20, (uint)ImportanceMapDim);
            WriteU32(s_importanceBytes, 24, (uint)(ImportanceMapDim * ImportanceSamplesX));
            WriteU32(s_importanceBytes, 28, (uint)(ImportanceMapDim * ImportanceSamplesY));
            WriteU32(s_importanceBytes, 32, (uint)ImportanceSamplesX);
            WriteU32(s_importanceBytes, 36, (uint)ImportanceSamplesY);
            WriteF32(s_importanceBytes, 40, 1.0f / ImportanceSamples);
            WriteU32(s_importanceBytes, 44, 10u); // log2(1024)
        }

        // ====================================================================
        // CPU helpers — LightsBaker
        // ====================================================================

        private int CollectAndPackLights()
        {
            int count = 0;
            foreach (var light in Object.FindObjectsByType<Light>(FindObjectsSortMode.None))
            {
                if (light == null || !light.enabled) continue;
                if (count >= NativeRtxptBufferResources.MaxLights)
                {
                    Debug.LogWarning("[NativeRtxptLightingUpdateBeginPass] MaxLights exceeded; some lights ignored.");
                    break;
                }

                switch (light.type)
                {
                    case LightType.Point:
                        PackPointLight(light, ref s_lightsStaging[count], ref s_lightsExStaging[count]);
                        count++;
                        break;
                    case LightType.Spot:
                        PackSpotLight(light, ref s_lightsStaging[count], ref s_lightsExStaging[count]);
                        count++;
                        break;
                }
            }

            _dbgFrameCounter++;
            return count;
        }

        private void UploadLightData()
        {
            var buf = _ctx.Buffers;
            if (buf == null) return;

            // _ping was not yet flipped when UploadLightData runs (flip happens at end of RecordRenderGraph)
            uint currentOffset  = _ping ? 0u : WeightsCountHalf;
            uint historicOffset = _ping ? WeightsCountHalf : 0u;

            float envIntensity = _ctx.Setting?.environmentMapIntensity ?? 1.0f;
            Color envTint      = (_ctx.Setting?.environmentMapTint ?? Color.white).linear;

            ref var ctrl = ref s_controlStaging[0];
            ctrl                         = default;
            ctrl.TotalLightCount         = (uint)_analyticLightCount + EnvQtTotalNodeCount;
            ctrl.AnalyticLightCount      = (uint)_analyticLightCount;
            ctrl.EnvmapQuadNodeCount     = EnvQtTotalNodeCount;
            ctrl.ImportanceSamplingType  = 1;
            ctrl.HistoricTotalLightCount = (uint)_analyticLightCount + EnvQtTotalNodeCount;

            unsafe
            {
                ctrl._paddingBK[28] = currentOffset;
                ctrl._paddingBK[29] = historicOffset;
                float distantVsLocal = 1.0f;
                ctrl._paddingBK[0] = *(uint*)&distantVsLocal;
                ctrl._paddingBK[1] = 11u; // EnvMapImportanceMapMIPCount
                ctrl._paddingBK[2] = 1024u; // EnvMapImportanceMapResolution

                float one = 1f, zero = 0f;
                ctrl._paddingBK[88]  = *(uint*)&one;
                ctrl._paddingBK[89]  = *(uint*)&zero;
                ctrl._paddingBK[90]  = *(uint*)&zero;
                ctrl._paddingBK[91]  = *(uint*)&zero;
                ctrl._paddingBK[92]  = *(uint*)&zero;
                ctrl._paddingBK[93]  = *(uint*)&one;
                ctrl._paddingBK[94]  = *(uint*)&zero;
                ctrl._paddingBK[95]  = *(uint*)&zero;
                ctrl._paddingBK[96]  = *(uint*)&zero;
                ctrl._paddingBK[97]  = *(uint*)&zero;
                ctrl._paddingBK[98]  = *(uint*)&one;
                ctrl._paddingBK[99]  = *(uint*)&zero;
                ctrl._paddingBK[100] = *(uint*)&one;
                ctrl._paddingBK[101] = *(uint*)&zero;
                ctrl._paddingBK[102] = *(uint*)&zero;
                ctrl._paddingBK[103] = *(uint*)&zero;
                ctrl._paddingBK[104] = *(uint*)&zero;
                ctrl._paddingBK[105] = *(uint*)&one;
                ctrl._paddingBK[106] = *(uint*)&zero;
                ctrl._paddingBK[107] = *(uint*)&zero;
                ctrl._paddingBK[108] = *(uint*)&zero;
                ctrl._paddingBK[109] = *(uint*)&zero;
                ctrl._paddingBK[110] = *(uint*)&one;
                ctrl._paddingBK[111] = *(uint*)&zero;

                float cr = envTint.r * envIntensity, cg = envTint.g * envIntensity, cb = envTint.b * envIntensity;
                ctrl._paddingBK[112] = *(uint*)&cr;
                ctrl._paddingBK[113] = *(uint*)&cg;
                ctrl._paddingBK[114] = *(uint*)&cb;
                ctrl._paddingBK[115] = *(uint*)&one;
            }

            buf.LightControlBuffer.SetData(s_controlStaging);

            if (_analyticLightCount > 0)
            {
                buf.LightBuffer.SetData(s_lightsStaging, 0, (int)EnvQtTotalNodeCount, _analyticLightCount);
                buf.LightExBuffer.SetData(s_lightsExStaging, 0, (int)EnvQtTotalNodeCount, _analyticLightCount);
            }
        }

        // ====================================================================
        // Light packing helpers (mirrors LightsBaker.cpp)
        // ====================================================================

        private static void PackPointLight(Light light, ref RtxptPolymorphicLightInfo info, ref RtxptPolymorphicLightInfoEx infoEx)
        {
            info   = default;
            infoEx = default;
            var pos = light.transform.position;
            info.CenterX = pos.x;
            info.CenterY = pos.y;
            info.CenterZ = pos.z;
            const float r        = 0.01f;
            Vector3     radiance = new Vector3(light.color.linear.r, light.color.linear.g, light.color.linear.b) * light.intensity / (Mathf.PI * r * r);
            PackLightColor(radiance, ref info, (uint)RtxptLightType.Sphere);
            info.Scalars = Fp32ToFp16(r);
        }

        private static void PackSpotLight(Light light, ref RtxptPolymorphicLightInfo info, ref RtxptPolymorphicLightInfoEx infoEx)
        {
            info   = default;
            infoEx = default;
            var pos = light.transform.position;
            info.CenterX = pos.x;
            info.CenterY = pos.y;
            info.CenterZ = pos.z;
            const float r        = 0.01f;
            float       outerRad = Mathf.Deg2Rad * (light.spotAngle * 0.5f);
            float       innerRad = outerRad * 0.8f;
            float       softness = Mathf.Clamp01(1f - innerRad / outerRad);
            Vector3     radiance = new Vector3(light.color.linear.r, light.color.linear.g, light.color.linear.b) * light.intensity / (Mathf.PI * r * r);
            PackLightColor(radiance, ref info, (uint)RtxptLightType.Sphere);
            info.ColorTypeAndFlags         |= kShapingEnableBit;
            info.Scalars                   =  Fp32ToFp16(r);
            infoEx.PrimaryAxis             =  NDirToOctUnorm32(light.transform.forward);
            infoEx.CosConeAngleAndSoftness =  Fp32ToFp16(Mathf.Cos(outerRad)) | (Fp32ToFp16(softness) << 16);
        }

        private static void PackLightColor(Vector3 color, ref RtxptPolymorphicLightInfo info, uint typeCode)
        {
            info.ColorTypeAndFlags = typeCode << (int)kTypeShift;
            float maxR = Mathf.Max(color.x, Mathf.Max(color.y, color.z));
            if (maxR <= 0f) return;
            float logN     = Mathf.Clamp01((Mathf.Log(maxR, 2f) - kMinLog2Radiance) / (kMaxLog2Radiance - kMinLog2Radiance));
            uint  packed   = (uint)Mathf.Min(Mathf.Ceil(logN * 65534f) + 1f, 0xFFFF);
            float unpacked = Mathf.Pow(2f, ((packed - 1f) / 65534f) * (kMaxLog2Radiance - kMinLog2Radiance) + kMinLog2Radiance);
            uint  r8       = (uint)Mathf.RoundToInt(Mathf.Clamp01(color.x / unpacked) * 255f) & 0xFFu;
            uint  g8       = (uint)Mathf.RoundToInt(Mathf.Clamp01(color.y / unpacked) * 255f) & 0xFFu;
            uint  b8       = (uint)Mathf.RoundToInt(Mathf.Clamp01(color.z / unpacked) * 255f) & 0xFFu;
            info.ColorTypeAndFlags |= r8 | (g8 << 8) | (b8 << 16);
            info.LogRadiance       =  packed;
        }

        private static uint NDirToOctUnorm32(Vector3 n)
        {
            float absSum = Mathf.Abs(n.x) + Mathf.Abs(n.y) + Mathf.Abs(n.z);
            float px     = n.x / absSum, py = n.y / absSum;
            if (n.z < 0f)
            {
                float ox = (1f - Mathf.Abs(py)) * (px >= 0f ? 1f : -1f);
                float oy = (1f - Mathf.Abs(px)) * (py >= 0f ? 1f : -1f);
                px = ox;
                py = oy;
            }

            px = Mathf.Clamp01(px * 0.5f + 0.5f);
            py = Mathf.Clamp01(py * 0.5f + 0.5f);
            return ((uint)Mathf.RoundToInt(px * 0xFFFEu)) | (((uint)Mathf.RoundToInt(py * 0xFFFEu)) << 16);
        }

        private static uint Fp32ToFp16(float v)
        {
            uint  u      = (uint)BitConverter.ToInt32(BitConverter.GetBytes(v), 0);
            float scaled = v * 1.9259299444e-34f;
            uint  s      = (uint)BitConverter.ToInt32(BitConverter.GetBytes(scaled), 0);
            uint  sign   = u & 0x80000000u;
            uint  body   = s & 0x0FFFFFFFu;
            return ((sign >> 16) | (body >> 13)) & 0xFFFFu;
        }

        // ====================================================================
        // Byte writers
        // ====================================================================

        private static unsafe void WriteF32(byte[] buf, int offset, float v)
        {
            fixed (byte* p = &buf[offset]) *(float*)p = v;
        }

        private static unsafe void WriteU32(byte[] buf, int offset, uint v)
        {
            fixed (byte* p = &buf[offset]) *(uint*)p = v;
        }

        // ====================================================================
        // Render texture helpers
        // ====================================================================

        private void EnsureRenderTextures()
        {
            _envCubeMip0Rt   = EnsureCubeRT(ref _envCubeMip0Rt, CubeDim, RenderTextureFormat.ARGBHalf, true);
            _envCubeMip1Rt   = EnsureCubeRT(ref _envCubeMip1Rt, CubeDim / 2, RenderTextureFormat.ARGBHalf, true);
            _importanceMapRt = Ensure2DRT(ref _importanceMapRt, ImportanceMapDim, RenderTextureFormat.RFloat, true, useMipMap: true);
            _radianceMapRt   = Ensure2DRT(ref _radianceMapRt, ImportanceMapDim, RenderTextureFormat.ARGBHalf, true, useMipMap: true);
            _dummyCubeRt     = EnsureCubeRT(ref _dummyCubeRt, 4, RenderTextureFormat.ARGB32, false);
            EnsureLookupMapTexture();
        }

        private void EnsureLookupMapTexture()
        {
            if (_envLightLookupMapRt != null && _envLightLookupMapRt.IsCreated()) return;
            _envLightLookupMapRt?.Release();
            var desc = new RenderTextureDescriptor(EnvLookupMapDim, EnvLookupMapDim, GraphicsFormat.R32_UInt, 0)
            {
                enableRandomWrite = true,
                useMipMap         = false,
                dimension         = UnityEngine.Rendering.TextureDimension.Tex2D,
            };
            _envLightLookupMapRt = new RenderTexture(desc) { autoGenerateMips = false, hideFlags = HideFlags.HideAndDontSave };
            _envLightLookupMapRt.Create();
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

        private static RenderTexture EnsureCubeRT(ref RenderTexture rt, int size, RenderTextureFormat fmt, bool rw)
        {
            if (rt != null && rt.IsCreated()) return rt;
            rt?.Release();
            rt = new RenderTexture(size, size, 0, fmt)
            {
                dimension         = TextureDimension.Cube, useMipMap = false, autoGenerateMips = false,
                enableRandomWrite = rw, hideFlags                    = HideFlags.HideAndDontSave,
            };
            rt.Create();
            return rt;
        }

        private static RenderTexture Ensure2DRT(ref RenderTexture rt, int size, RenderTextureFormat fmt, bool rw, bool useMipMap = false)
        {
            if (rt != null && rt.IsCreated() && rt.useMipMap == useMipMap) return rt;
            rt?.Release();
            rt = new RenderTexture(size, size, 0, fmt)
            {
                dimension         = TextureDimension.Tex2D, useMipMap = useMipMap, autoGenerateMips = false,
                enableRandomWrite = rw, hideFlags                     = HideFlags.HideAndDontSave,
            };
            rt.Create();
            return rt;
        }

        private static void DestroyRT(ref RenderTexture rt)
        {
            if (rt == null) return;
            rt.Release();
            Object.DestroyImmediate(rt);
            rt = null;
        }
    }
}