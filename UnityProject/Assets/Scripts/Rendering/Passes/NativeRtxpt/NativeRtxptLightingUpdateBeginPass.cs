using System;
using System.Runtime.InteropServices;
using NativeRender;
using Unity.Mathematics;
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
    /// ScriptableRenderPass with the correct dispatch order:
    ///
    ///   CPU (Setup)
    ///     ControlDataSetup          — pack LightingControlData → LightControlBuffer
    ///     EnvmapAndAnalyticLightBuffers — pack point/spot lights → LightBuffer / LightExBuffer
    ///
    ///   GPU (ExecutePass)
    ///     1.  ResetLightProxyCounters
    ///     2.  ResetPastToCurrentHistory
    ///     3.  EnvLightsBackupPast
    ///     4.  EnvLightsSubdivideBase
    ///     5.  EnvLightsSubdivideBoost
    ///     6.  BakeEmissiveTriangles
    ///     7.  EnvLightFillLookupMap
    ///     8.  EnvLightsMapPastToCurrent
    ///     9.  ProcessFeedbackHistoryPreFilter
    ///     10. ProcessFeedbackHistoryP0
    ///     11. ComputeWeights
    ///     12. ComputeProxyCounts
    ///     13. ComputeProxyBaselineOffsets
    ///     14. CreateProxyJobs
    ///     15. ExecuteProxyJobs
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

        // Env quad-tree — derived from LightingConfig.h via LightingConfig.cs
        private const uint EnvQtBaseResolution   = LightingConfig.RTXPT_NEEAT_ENVMAP_QT_BASE_RESOLUTION;
        private const uint EnvQtSubdivisions     = LightingConfig.RTXPT_NEEAT_ENVMAP_QT_SUBDIVISIONS;
        private const uint EnvQtAdditionalNodes  = LightingConfig.RTXPT_NEEAT_ENVMAP_QT_ADDITIONAL_NODES;
        private const uint EnvQtUnboostedCount   = LightingConfig.RTXPT_NEEAT_ENVMAP_QT_UNBOOSTED_NODE_COUNT;
        private const uint EnvQtBoostSubdivision = LightingConfig.RTXPT_NEEAT_ENVMAP_QT_BOOST_SUBDIVISION;
        private const uint EnvQtBoostNodesMult   = LightingConfig.RTXPT_NEEAT_ENVMAP_QT_BOOST_NODES_MULT;
        private const uint EnvQtTotalNodeCount   = LightingConfig.RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT;

        // NEEATBaker.hlsli dispatch constants
        private const int  LLB_NUM_COMPUTE_THREADS_2D      = 8; // 2D tile dispatch thread count
        private const int  LLB_PREPROCESS_BLOCK_SIZE_INNER = 14; // outer=16, inner=outer-2
        private const uint LLB_NUM_COMPUTE_THREADS         = 128; // 1D dispatch thread count
        private const uint LLB_LOCAL_BLOCK_SIZE            = 32;
        private const uint LLB_WEIGHTS_ITEMS_PER_GROUP     = LLB_LOCAL_BLOCK_SIZE * LLB_NUM_COMPUTE_THREADS;
        private const uint LLB_MAX_PROXIES_PER_TASK        = 32;

        private static readonly uint LLB_MAX_PROXY_PROC_TASKS =
            (uint)NativeRtxptBufferResources.MaxLights +
            ((uint)NativeRtxptBufferResources.ProxySamplingCount + LLB_MAX_PROXIES_PER_TASK - 1) / LLB_MAX_PROXIES_PER_TASK;

        private const uint WeightsCountHalf = LightingConfig.RTXPT_LIGHTING_WEIGHTS_COUNT_HALF;

        // DXGI format constants
        private const uint DXGI_FORMAT_R32_FLOAT = 41u;
        private const uint DXGI_FORMAT_R32_UINT  = 42u;

        // Struct strides
        private static readonly int StrideCtrl     = Marshal.SizeOf<RtxptLightingControlData>();
        private static readonly int StrideLights   = Marshal.SizeOf<RtxptPolymorphicLightInfo>();
        private static readonly int StrideLightsEx = Marshal.SizeOf<RtxptPolymorphicLightInfoEx>();

        // EnvLightLookupMap dimension
        private const int EnvLookupMapDim = 1024;

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
        // GPU pipelines — NEE-AT feedback pre-processing (UpdateBegin)
        // ====================================================================

        private readonly NativeComputePipeline      _processFeedbackHistoryPreFilterCs;
        private readonly NativeComputeDescriptorSet _processFeedbackHistoryPreFilterDs;
        private readonly NativeComputePipeline      _processFeedbackHistoryP0Cs;
        private readonly NativeComputeDescriptorSet _processFeedbackHistoryP0Ds;

        // ====================================================================
        // Owned render textures
        // ====================================================================

        private RenderTexture _envLightLookupMapRt; // 1024×1024 2D R32_UINT UAV

        // ====================================================================
        // CPU staging
        // ====================================================================

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
        private uint                   _historicTotalLightCount;
        private bool                   _feedbackBufferFilled;
        private bool                   _previousFrameTemporalFeedbackAvailable;
        private Vector2                _localJitterSequence;
        private uint2                  _localSamplingTileJitter;
        private uint2                  _prevLocalSamplingTileJitter;
        private int2                   _prevRenderResolution;

        /// <summary>lightsBuffer index where emissive triangles start (= EnvQtTotalNodeCount + analyticLightCount).</summary>
        public uint EmissiveLightOffset => EnvQtTotalNodeCount + (uint)_analyticLightCount;

        /// <summary>Total emissive triangle-light count produced last frame.</summary>
        public uint EmissiveTriangleCount => _emissiveTotalTriCount;

        // ====================================================================
        // Constructor
        // ====================================================================

        public NativeRtxptLightingUpdateBeginPass(
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
            NativeComputeShader bakeEmissiveTrianglesCs,
            // NEE-AT feedback (begin half)
            NativeComputeShader processFeedbackHistoryPreFilterCs,
            NativeComputeShader processFeedbackHistoryP0Cs)
        {
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
            _resetLightProxyCountersCs         = new NativeComputePipeline(resetLightProxyCountersCs);
            _resetLightProxyCountersDs         = new NativeComputeDescriptorSet(_resetLightProxyCountersCs);
            _resetPastToCurrentHistoryCs       = new NativeComputePipeline(resetPastToCurrentHistoryCs);
            _resetPastToCurrentHistoryDs       = new NativeComputeDescriptorSet(_resetPastToCurrentHistoryCs);
            _computeWeightsCs                  = new NativeComputePipeline(computeWeightsCs);
            _computeWeightsDs                  = new NativeComputeDescriptorSet(_computeWeightsCs);
            _computeProxyCountsCs              = new NativeComputePipeline(computeProxyCountsCs);
            _computeProxyCountsDs              = new NativeComputeDescriptorSet(_computeProxyCountsCs);
            _computeProxyBaselineOffsetsCs     = new NativeComputePipeline(computeProxyBaselineOffsetsCs);
            _computeProxyBaselineOffsetsDs     = new NativeComputeDescriptorSet(_computeProxyBaselineOffsetsCs);
            _createProxyJobsCs                 = new NativeComputePipeline(createProxyJobsCs);
            _createProxyJobsDs                 = new NativeComputeDescriptorSet(_createProxyJobsCs);
            _executeProxyJobsCs                = new NativeComputePipeline(executeProxyJobsCs);
            _executeProxyJobsDs                = new NativeComputeDescriptorSet(_executeProxyJobsCs);
            _bakeEmissiveTrianglesCs           = new NativeComputePipeline(bakeEmissiveTrianglesCs);
            _bakeEmissiveTrianglesDs           = new NativeComputeDescriptorSet(_bakeEmissiveTrianglesCs);
            _processFeedbackHistoryPreFilterCs = new NativeComputePipeline(processFeedbackHistoryPreFilterCs);
            _processFeedbackHistoryPreFilterDs = new NativeComputeDescriptorSet(_processFeedbackHistoryPreFilterCs);
            _processFeedbackHistoryP0Cs        = new NativeComputePipeline(processFeedbackHistoryP0Cs);
            _processFeedbackHistoryP0Ds        = new NativeComputeDescriptorSet(_processFeedbackHistoryP0Cs);

            EnsureLookupMapTexture();
        }

        // ====================================================================
        // Dispose
        // ====================================================================

        public void Dispose()
        {
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

            // NEE-AT feedback (begin)
            _processFeedbackHistoryPreFilterDs?.Dispose();
            _processFeedbackHistoryPreFilterCs?.Dispose();
            _processFeedbackHistoryP0Ds?.Dispose();
            _processFeedbackHistoryP0Cs?.Dispose();

            // Render textures
            DestroyRT(ref _envLightLookupMapRt);
        }

        // ====================================================================
        // Setup — main thread, called once per frame before RecordRenderGraph
        // ====================================================================

        public void Setup(NativeRtxptPassContext ctx)
        {
            _dbgFrameCounter++;
            _ctx = ctx;
            EnsureLookupMapTexture();

            // Expose env-light lookup map pointer
            ctx.EnvLightLookupMapPtr = _envLightLookupMapRt != null && _envLightLookupMapRt.IsCreated()
                ? _envLightLookupMapRt.GetNativeTexturePtr()
                : IntPtr.Zero;

            // --- LightsBaker CPU staging work ---
            _analyticLightCount = CollectAndPackLights();

            // --- Emissive triangles: MUST run before BuildControlData so _emissiveTaskCount
            //     is known when we write BakerConstants.TriangleLightTaskCount.
            var gpuScene = ctx.GpuScene;

            uint emissiveLightOffset = EnvQtTotalNodeCount + (uint)_analyticLightCount;
            gpuScene.PrepareEmissiveTriangleTasks(emissiveLightOffset, ctx.Buffers.LightScratchBuffer);
            ctx.Buffers.RefreshLightScratchBufferPtr();
            _emissiveTaskCount     = gpuScene.LastEmissiveTaskCount;
            _emissiveTotalTriCount = gpuScene.LastEmissiveTriangleCount;
            BuildControlData();
        }

        // ====================================================================
        // RecordRenderGraph
        // ====================================================================

        private class PassData
        {
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
            internal GraphicsBuffer             LightControlBuffer;
            internal GraphicsBuffer             LightBuffer;
            internal GraphicsBuffer             LightExBuffer;
            internal RtxptLightingControlData[] ControlData;
            internal RtxptPolymorphicLightInfo[] LightData;
            internal RtxptPolymorphicLightInfoEx[] LightExData;
            internal int                        AnalyticLightCount;

            // --- Feedback pre-processing (begin) ---
            internal NativeComputePipeline      ProcessFeedbackHistoryPreFilterCs;
            internal NativeComputeDescriptorSet ProcessFeedbackHistoryPreFilterDs;
            internal NativeComputePipeline      ProcessFeedbackHistoryP0Cs;
            internal NativeComputeDescriptorSet ProcessFeedbackHistoryP0Ds;
            internal bool                       LastFrameFeedbackAvailable;

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
            pd.HistoricTotalLightCount       = _historicTotalLightCount;
            pd.LightControlBuffer            = _ctx.Buffers.LightControlBuffer;
            pd.LightBuffer                   = _ctx.Buffers.LightBuffer;
            pd.LightExBuffer                 = _ctx.Buffers.LightExBuffer;
            pd.ControlData                   = s_controlStaging;
            pd.LightData                     = s_lightsStaging;
            pd.LightExData                   = s_lightsExStaging;
            pd.AnalyticLightCount            = _analyticLightCount;

            // Feedback pre-processing (begin)
            pd.ProcessFeedbackHistoryPreFilterCs = _processFeedbackHistoryPreFilterCs;
            pd.ProcessFeedbackHistoryPreFilterDs = _processFeedbackHistoryPreFilterDs;
            pd.ProcessFeedbackHistoryP0Cs        = _processFeedbackHistoryP0Cs;
            pd.ProcessFeedbackHistoryP0Ds        = _processFeedbackHistoryP0Ds;
            pd.LastFrameFeedbackAvailable        = s_controlStaging[0].LastFrameTemporalFeedbackAvailable != 0;

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


            _historicTotalLightCount = pd.TotalLightCount;

            // Flip ping-pong AFTER filling passData so BuildControlData used same side
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

            cmd.BeginSample(RenderPassMarkers.RtxptLightingUpdateBegin);

            // CPU uploads.
            context.cmd.BeginSample(RenderPassMarkers.RtxptControlDataSetup);
            data.LightControlBuffer.SetData(data.ControlData, 0, 0, 1);
            context.cmd.EndSample(RenderPassMarkers.RtxptControlDataSetup);

            if (data.AnalyticLightCount > 0)
            {
                context.cmd.BeginSample(RenderPassMarkers.RtxptEnvmapAndAnalyticLightBuffers);
                data.LightBuffer.SetData(data.LightData, 0, (int)EnvQtTotalNodeCount, data.AnalyticLightCount);
                data.LightExBuffer.SetData(data.LightExData, 0, (int)EnvQtTotalNodeCount, data.AnalyticLightCount);
                context.cmd.EndSample(RenderPassMarkers.RtxptEnvmapAndAnalyticLightBuffers);
            }

            // Refresh cached pointers after CPU uploads before binding.
            buf.RefreshLightUploadBufferPtrs();

            var pCtrl     = buf.LightControlBufferPtr;
            var pLights   = buf.LightBufferPtr;
            var pLightsEx = buf.LightExBufferPtr;
            var pScratch  = buf.LightScratchBufferPtr;
            var pScrList  = buf.ScratchListBufferPtr;
            var pWeights  = buf.LightWeightsBufferPtr;
            var pHistCur  = buf.HistoryRemapCurrToPastPtr;
            var pHistPas  = buf.HistoryRemapPastToCurrPtr;
            var pProxyCnt = buf.LightProxyCountersPtr;
            var pProxies  = buf.LightSamplingProxiesPtr;

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
            // 1. ResetLightProxyCounters
            // ----------------------------------------------------------------
            {
                var ds = data.ResetProxyCountersDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_perLightProxyCounters", pProxyCnt, cProxyCnt, DXGI_FORMAT_R32_UINT);
                uint gx = (total + 1 + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS;
                cmd.BeginSample(RenderPassMarkers.RtxptResetLightProxyCounters);
                data.ResetProxyCountersCs.Dispatch(cmd, ds, gx, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptResetLightProxyCounters);
            }

            // ----------------------------------------------------------------
            // 2. ResetPastToCurrentHistory
            // ----------------------------------------------------------------
            {
                uint items = Math.Max(historic, total);
                var  ds    = data.ResetPastToCurrentDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_historyRemapPastToCurrent", pHistPas, cHistPas, DXGI_FORMAT_R32_UINT);
                uint gx = Math.Max(1u, (items + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS);
                cmd.BeginSample(RenderPassMarkers.RtxptResetPastToCurrentHistory);
                data.ResetPastToCurrentCs.Dispatch(cmd, ds, gx, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptResetPastToCurrentHistory);
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
                cmd.BeginSample(RenderPassMarkers.RtxptEnvLightsBackupPast);
                data.BackupPastCs.Dispatch(cmd, ds, gx, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptEnvLightsBackupPast);
            }

            // ----------------------------------------------------------------
            // 6. EnvLightsSubdivideBase
            // ----------------------------------------------------------------
            {
                var ds = data.SubdivideBaseDs;
                ds.SetTexture("t_envRadianceAndImportanceMap", envImportancePtr);
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_scratchList", pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                cmd.BeginSample(RenderPassMarkers.RtxptEnvLightsSubdivideBase);
                data.SubdivideBaseCs.Dispatch(cmd, ds, 1, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptEnvLightsSubdivideBase);
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
                cmd.BeginSample(RenderPassMarkers.RtxptEnvLightsSubdivideBoost);
                data.SubdivideBoostCs.Dispatch(cmd, ds, EnvQtUnboostedCount, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptEnvLightsSubdivideBoost);
            }

            // ----------------------------------------------------------------
            // 8. BakeEmissiveTriangles
            // ----------------------------------------------------------------
            if (data.EmissiveTaskCount > 0)
            {
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
                cmd.BeginSample(RenderPassMarkers.RtxptBakeEmissiveTriangles);
                data.BakeEmissiveTrianglesCs.Dispatch(cmd, ds, gxBake, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptBakeEmissiveTriangles);
            }

            // ----------------------------------------------------------------
            // 9. EnvLightFillLookupMap
            // ----------------------------------------------------------------
            {
                var ds = data.FillLookupMapDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWStructuredBuffer("u_lightsBuffer", pLights, cLights, StrideLights);
                ds.SetRWTexture("u_envLightLookupMap", envLookupMapPtr);
                cmd.BeginSample(RenderPassMarkers.RtxptEnvLightFillLookupMap);
                data.FillLookupMapCs.Dispatch(cmd, ds, EnvQtTotalNodeCount, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptEnvLightFillLookupMap);
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
                uint gx = (uint)(EnvQtTotalNodeCount + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS;
                cmd.BeginSample(RenderPassMarkers.RtxptEnvLightsMapPastToCurrent);
                data.MapPastToCurrentCs.Dispatch(cmd, ds, gx, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptEnvLightsMapPastToCurrent);
            }

            // ----------------------------------------------------------------
            // 11. ProcessFeedbackHistoryPreFilter / P0
            //     Original RTXPT only runs these when the previous frame's
            //     temporal feedback buffer is known to contain valid data.
            // ----------------------------------------------------------------
            if (data.LastFrameFeedbackAvailable)
            {
                {
                    var ds  = data.ProcessFeedbackHistoryPreFilterDs;
                    var ctx = data.Ctx;
                    ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                    ds.SetRWTexture("u_feedbackTotalWeight", ctx.FeedbackTotalWeightPtr);
                    ds.SetRWTexture("u_feedbackCandidates", ctx.FeedbackCandidatesPtr);
                    uint gx = (uint)(ctx.RenderResolution.x + LLB_PREPROCESS_BLOCK_SIZE_INNER - 1) / LLB_PREPROCESS_BLOCK_SIZE_INNER;
                    uint gy = (uint)(ctx.RenderResolution.y + LLB_PREPROCESS_BLOCK_SIZE_INNER - 1) / LLB_PREPROCESS_BLOCK_SIZE_INNER;
                    cmd.BeginSample(RenderPassMarkers.RtxptProcessFeedbackHistoryPreFilter);
                    data.ProcessFeedbackHistoryPreFilterCs.Dispatch(cmd, ds, gx, gy, 1);
                    cmd.EndSample(RenderPassMarkers.RtxptProcessFeedbackHistoryPreFilter);
                }

                {
                    var ds  = data.ProcessFeedbackHistoryP0Ds;
                    var ctx = data.Ctx;
                    ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                    ds.SetRWTypedBuffer("u_lightWeights", pWeights, cWeights, DXGI_FORMAT_R32_FLOAT);
                    ds.SetRWTexture("u_feedbackTotalWeight", ctx.FeedbackTotalWeightPtr);
                    ds.SetRWTexture("u_feedbackCandidates", ctx.FeedbackCandidatesPtr);
                    ds.SetRWTexture("u_feedbackTotalWeightBlended", ctx.FeedbackTotalWeightBlendedPtr);
                    ds.SetRWTexture("u_feedbackCandidatesBlended", ctx.FeedbackCandidatesBlendedPtr);
                    ds.SetRWTexture("u_ShaderDebugVizTextureBuffer", ctx.ShaderDebugVizPtr);
                    ds.SetRWTypedBuffer("u_historyRemapPastToCurrent", pHistPas, cHistPas, DXGI_FORMAT_R32_UINT);
                    ds.SetRWTypedBuffer("u_perLightProxyCounters", pProxyCnt, cProxyCnt, DXGI_FORMAT_R32_UINT);

                    uint gx = (uint)(ctx.RenderResolution.x + LLB_NUM_COMPUTE_THREADS_2D - 1) / LLB_NUM_COMPUTE_THREADS_2D;
                    uint gy = (uint)(ctx.RenderResolution.y + LLB_NUM_COMPUTE_THREADS_2D - 1) / LLB_NUM_COMPUTE_THREADS_2D;
                    cmd.BeginSample(RenderPassMarkers.RtxptProcessFeedbackHistoryP0);
                    data.ProcessFeedbackHistoryP0Cs.Dispatch(cmd, ds, gx, gy, 1);
                    cmd.EndSample(RenderPassMarkers.RtxptProcessFeedbackHistoryP0);
                }
            }

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
                cmd.BeginSample(RenderPassMarkers.RtxptComputeWeights);
                data.ComputeWeightsCs.Dispatch(cmd, ds, gx, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptComputeWeights);
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
                cmd.BeginSample(RenderPassMarkers.RtxptComputeProxyCounts);
                data.ComputeProxyCountsCs.Dispatch(cmd, ds, gx, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptComputeProxyCounts);
            }

            // ----------------------------------------------------------------
            // 15. ComputeProxyBaselineOffsets  (single thread-group prefix-sum)
            // ----------------------------------------------------------------
            {
                var ds = data.ComputeProxyBaselineOffsetsDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_lightSamplingProxies", pProxies, cProxies, DXGI_FORMAT_R32_UINT);
                cmd.BeginSample(RenderPassMarkers.RtxptComputeProxyBaselineOffsets);
                data.ComputeProxyBaselineOffsetsCs.Dispatch(cmd, ds, 1, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptComputeProxyBaselineOffsets);
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
                cmd.BeginSample(RenderPassMarkers.RtxptCreateProxyJobs);
                data.CreateProxyJobsCs.Dispatch(cmd, ds, gx, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptCreateProxyJobs);
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
                cmd.BeginSample(RenderPassMarkers.RtxptExecuteProxyJobs);
                data.ExecuteProxyJobsCs.Dispatch(cmd, ds, gx, 1, 1);
                cmd.EndSample(RenderPassMarkers.RtxptExecuteProxyJobs);
            }

            cmd.EndSample(RenderPassMarkers.RtxptLightingUpdateBegin);
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

        private void BuildControlData()
        {
            var buf = _ctx.Buffers;
            if (buf == null) return;

            // _ping was not yet flipped when BuildControlData runs (flip happens at end of RecordRenderGraph)
            uint currentOffset  = _ping ? 0u : WeightsCountHalf;
            uint historicOffset = _ping ? WeightsCountHalf : 0u;

            float envIntensity = _ctx.Setting?.environmentMapIntensity ?? 1.0f;
            Color envTint      = (_ctx.Setting?.environmentMapTint ?? Color.white).linear;

            ref var ctrl = ref s_controlStaging[0];
            ctrl                         = default;
            var setting = _ctx.Setting;
            // Original (LightsBaker.cpp:1022): TemporalFeedbackRequired depends only on
            // ImportanceSamplingType == NEEAT, not on the global UseNEE toggle.
            bool neeAtSelected      = setting != null && setting.neeType == NativeRtxptNeeType.NEEAT;
            bool neeAtEnabled       = setting != null && setting.useNEE && neeAtSelected;
            if (!neeAtSelected)
                _feedbackBufferFilled = false;

            bool lastFrameFeedback  = neeAtEnabled && _feedbackBufferFilled;
            bool lastFrameLocal     = _previousFrameTemporalFeedbackAvailable && lastFrameFeedback;
            uint totalLightCount    = EnvQtTotalNodeCount + (uint)_analyticLightCount + _emissiveTotalTriCount;
            int2 renderRes          = _ctx.RenderResolution;
            uint feedbackW          = (uint)Math.Max(1, renderRes.x);
            uint feedbackH          = (uint)Math.Max(1, renderRes.y);
            uint blendedW           = (uint)Math.Max(1, (renderRes.x + LightingConfig.RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE - 1) / LightingConfig.RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE);
            uint blendedH           = (uint)Math.Max(1, (renderRes.y + LightingConfig.RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE - 1) / LightingConfig.RTXPT_NEEAT_EARLY_FEEDBACK_TILE_SIZE);
            uint localW             = (uint)Math.Max(1, (renderRes.x + LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE - 1) / LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE);
            uint localH             = (uint)Math.Max(1, (renderRes.y + LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE - 1) / LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE);
            uint p0ThreadCount      = ((feedbackW + LLB_NUM_COMPUTE_THREADS_2D - 1) / LLB_NUM_COMPUTE_THREADS_2D)
                                    * ((feedbackH + LLB_NUM_COMPUTE_THREADS_2D - 1) / LLB_NUM_COMPUTE_THREADS_2D)
                                    * LLB_NUM_COMPUTE_THREADS_2D * LLB_NUM_COMPUTE_THREADS_2D;
 
            ctrl.TotalLightCount                    = totalLightCount;
            ctrl.AnalyticLightCount                 = (uint)_analyticLightCount;
            ctrl.EnvmapQuadNodeCount                = EnvQtTotalNodeCount;
            ctrl.TriangleLightCount                 = _emissiveTotalTriCount;
            ctrl.HistoricTotalLightCount            = _historicTotalLightCount;
            ctrl.LastFrameTemporalFeedbackAvailable = lastFrameFeedback ? 1u : 0u;
            ctrl.LastFrameLocalSamplesAvailable     = lastFrameLocal ? 1u : 0u;
            ctrl.ImportanceSamplingType             = setting != null && setting.useNEE ? (uint)setting.neeType : 0u;
            // LightsBaker.cpp:1022,1070 — depends only on ImportanceSamplingType == NEEAT.
            ctrl.TemporalFeedbackRequired           = neeAtSelected ? 1u : 0u;
            ctrl.TotalMaxFeedbackCount              = lastFrameFeedback ? p0ThreadCount : 0u;
            ctrl.GlobalFeedbackUseWeight            = lastFrameFeedback ? (setting?.neeatGlobalTemporalFeedbackWeight ?? 0.75f) : 0.0f;
            ctrl.LocalToGlobalSampleRatio           = lastFrameFeedback ? (setting?.neeatLocalToGlobalSampleRatio ?? 0.65f) : 0.0f;
            ctrl.TileBufferHeight                   = localH;
            ctrl.ScreenSpaceVsWorldSpaceThreshold   = setting?.neeatScreenSpaceVsWorldSpaceThreshold ?? 0.3f;
            ctrl.LocalSamplingResolutionX           = localW;
            ctrl.LocalSamplingResolutionY           = localH;
            UpdateLocalSamplingJitter();
            ctrl.LocalSamplingTileJitterX           = _localSamplingTileJitter.x;
            ctrl.LocalSamplingTileJitterY           = _localSamplingTileJitter.y;
            ctrl.LocalSamplingTileJitterPrevX       = _prevLocalSamplingTileJitter.x;
            ctrl.LocalSamplingTileJitterPrevY       = _prevLocalSamplingTileJitter.y;
 
            ref var bk = ref ctrl.BakerConstants;
            bk.CurrentWeightsBufferOffset       = currentOffset;
            bk.HistoricWeightsBufferOffset      = historicOffset;
            bk.DistantVsLocalRelativeImportance = (setting?.neeatDistantVsLocalImportance ?? 1.0f) * 0.0002f;
            bk.EnvMapImportanceMapMIPCount      = 11u;
            bk.EnvMapImportanceMapResolution    = 1024u;
            bk.TriangleLightTaskCount           = (uint)_emissiveTaskCount;
            bk.FeedbackResolutionX              = feedbackW;
            bk.FeedbackResolutionY              = feedbackH;
            bk.BlendedFeedbackResolutionX       = blendedW;
            bk.BlendedFeedbackResolutionY       = blendedH;

            // Prev-over-current viewport size (LightsBaker.cpp:1019). First frame: 1.0.
            int2 prevRes = _prevRenderResolution.x > 0 ? _prevRenderResolution : renderRes;
            bk.PrevOverCurrentViewportSizeX     = (float)prevRes.x / math.max(renderRes.x, 1);
            bk.PrevOverCurrentViewportSizeY     = (float)prevRes.y / math.max(renderRes.y, 1);
            _prevRenderResolution               = renderRes;

            bk.UpdateCounter                    = (uint)_dbgFrameCounter;
            bk.DepthDisocclusionThreshold       = setting?.neeatDepthDisocclusionThreshold ?? 1.5f;
            bk.EnableMotionReprojection         = (setting?.neeatEnableMotionReprojection ?? true) ? 1u : 0u;
            bk.ReservoirHistoryDropoff          = setting?.neeatReservoirHistoryDropoff ?? 0.005f;

            // Scene-relative info used by NEE-AT tile/light importance scoring.
            bk.SceneCameraPosX                  = _ctx.FrameState != null ? _ctx.FrameState.camPos.x : 0f;
            bk.SceneCameraPosY                  = _ctx.FrameState != null ? _ctx.FrameState.camPos.y : 0f;
            bk.SceneCameraPosZ                  = _ctx.FrameState != null ? _ctx.FrameState.camPos.z : 0f;
            bk.SceneAverageContentsDistance     = setting?.neeatSceneAverageContentsDistance ?? 10.0f;

            bk.ImportanceBoostIntensityDelta    = setting?.neeatImportanceBoostIntensityDelta ?? 0f;
            bk.ImportanceBoostFrustumMul        = setting?.neeatImportanceBoostFrustumMul ?? 0f;
            bk.ImportanceBoostFrustumFadeDistance = setting?.neeatImportanceBoostFrustumFadeDistance ?? 0f;

            // Frustum planes + corners (Gribb-Hartmann, LightsBaker.cpp:884 UpdateFrustumConsts).
            if (_ctx.FrameState != null)
                FillFrustumPlanesAndCorners(ref bk, _ctx.FrameState.worldToClip);

            bk.EnvMapParams = new RtxptLightsBakerEnvMapParams
            {
                TransformRow0    = new Vector4(1, 0, 0, 0),
                TransformRow1    = new Vector4(0, 1, 0, 0),
                TransformRow2    = new Vector4(0, 0, 1, 0),
                InvTransformRow0 = new Vector4(1, 0, 0, 0),
                InvTransformRow1 = new Vector4(0, 1, 0, 0),
                InvTransformRow2 = new Vector4(0, 0, 1, 0),
                ColorMultiplierR = envTint.r * envIntensity,
                ColorMultiplierG = envTint.g * envIntensity,
                ColorMultiplierB = envTint.b * envIntensity,
                Enabled          = 1.0f,
            };

            _previousFrameTemporalFeedbackAvailable = ctrl.TemporalFeedbackRequired != 0;
            if (ctrl.TemporalFeedbackRequired != 0)
                _feedbackBufferFilled = true;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Gribb-Hartmann frustum extraction from a column-vector worldToClip.
        // Mirrors LightsBaker::UpdateFrustumConsts (LightsBaker.cpp:884).
        // Donut uses row-vector convention (M[col][row]); Unity uses column-vector
        // (M[row][col]) — so row/col are swapped vs the original, but the result
        // (plane equations in world space) is identical.
        // Plane convention here matches donut's: dot(plane.xyz, p) - plane.w = signed distance.
        // ─────────────────────────────────────────────────────────────────────
        private static unsafe void FillFrustumPlanesAndCorners(ref RtxptLightsBakerConstants bk, Matrix4x4 wtc)
        {
            const float DISTANT_LIGHT_DISTANCE = 100000.0f;

            Vector4 r0 = wtc.GetRow(0);
            Vector4 r1 = wtc.GetRow(1);
            Vector4 r2 = wtc.GetRow(2);
            Vector4 r3 = wtc.GetRow(3);

            Vector4 left   = r3 + r0;
            Vector4 right  = r3 - r0;
            Vector4 top    = r3 - r1;
            Vector4 bottom = r3 + r1;
            Vector4 near   = r3 - r2;

            // Donut stores w as the negated homogeneous offset.
            left.w   = -left.w;
            right.w  = -right.w;
            top.w    = -top.w;
            bottom.w = -bottom.w;
            near.w   = -near.w;

            Normalize(ref left);
            Normalize(ref right);
            Normalize(ref top);
            Normalize(ref bottom);
            Normalize(ref near);

            // Far = inverted near pushed away by DISTANT_LIGHT_DISTANCE.
            Vector4 far = new Vector4(-near.x, -near.y, -near.z, -near.w - DISTANT_LIGHT_DISTANCE);

            // Order: Left Right Top Bottom Near Far.
            Vector4* planes = stackalloc Vector4[6] { left, right, top, bottom, near, far };

            fixed (float* dst = bk.FrustumPlanes)
            {
                for (int i = 0; i < 6; i++)
                {
                    dst[i * 4 + 0] = planes[i].x;
                    dst[i * 4 + 1] = planes[i].y;
                    dst[i * 4 + 2] = planes[i].z;
                    dst[i * 4 + 3] = planes[i].w;
                }
            }

            // Corners: intersect three planes per corner (8 corners for the box L/R × T/B × N/F).
            // Matches the original ordering: bit0=R/L (encoded oddly via XOR with bit1), bit1=B/T, bit2=F/N.
            fixed (float* dstC = bk.FrustumCorners)
            {
                for (int i = 0; i < 8; i++)
                {
                    bool bone = (i & 1) != 0;
                    bool btwo = (i & 2) != 0;
                    Vector4 a = (bone == btwo) ? right : left;
                    Vector4 b = (i & 2) != 0 ? bottom : top;
                    Vector4 c = (i & 4) != 0 ? far : near;
                    Vector3 corner = IntersectThreePlanes(a, b, c);
                    dstC[i * 4 + 0] = corner.x;
                    dstC[i * 4 + 1] = corner.y;
                    dstC[i * 4 + 2] = corner.z;
                    dstC[i * 4 + 3] = 0f;
                }
            }

            bk.DebugDrawFrustum = 0u;
        }

        private static void Normalize(ref Vector4 plane)
        {
            float lenSq = plane.x * plane.x + plane.y * plane.y + plane.z * plane.z;
            float scale = lenSq > 0f ? 1.0f / Mathf.Sqrt(lenSq) : 0f;
            plane.x *= scale;
            plane.y *= scale;
            plane.z *= scale;
            plane.w *= scale;
        }

        // Solve { dot(n_i, p) = d_i } for p, given three planes whose normals form an invertible basis.
        private static Vector3 IntersectThreePlanes(Vector4 a, Vector4 b, Vector4 c)
        {
            Matrix4x4 m = Matrix4x4.identity;
            m.SetRow(0, new Vector4(a.x, a.y, a.z, 0f));
            m.SetRow(1, new Vector4(b.x, b.y, b.z, 0f));
            m.SetRow(2, new Vector4(c.x, c.y, c.z, 0f));
            m.SetRow(3, new Vector4(0f,  0f,  0f, 1f));
            Matrix4x4 inv = m.inverse;
            Vector3 d = new Vector3(a.w, b.w, c.w);
            return inv.MultiplyPoint3x4(d);
        }

        private void UpdateLocalSamplingJitter()
        {
            _prevLocalSamplingTileJitter = _localSamplingTileJitter;

            const float g  = 1.324717957244746f;
            const float a1 = 1.0f / g;
            const float a2 = 1.0f / (g * g);

            if (((uint)_dbgFrameCounter % 1024u) == 0u)
                _localJitterSequence = Vector2.zero;

            _localJitterSequence.x = Mathf.Repeat(_localJitterSequence.x + a1, 1.0f);
            _localJitterSequence.y = Mathf.Repeat(_localJitterSequence.y + a2, 1.0f);

            uint maxJitter = (uint)(LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE - 1);
            _localSamplingTileJitter = new uint2(
                (uint)Mathf.Clamp((int)(_localJitterSequence.x * LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE), 0, (int)maxJitter),
                (uint)Mathf.Clamp((int)(_localJitterSequence.y * LightingConfig.RTXPT_LIGHTING_SAMPLING_BUFFER_TILE_SIZE), 0, (int)maxJitter));
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
        // Render texture helpers
        // ====================================================================

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

        private static void DestroyRT(ref RenderTexture rt)
        {
            if (rt == null) return;
            rt.Release();
            Object.DestroyImmediate(rt);
            rt = null;
        }
    }
}
