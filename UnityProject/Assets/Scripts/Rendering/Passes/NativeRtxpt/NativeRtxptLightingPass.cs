using System;
using System.Runtime.InteropServices;
using NativeRender;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using Object = UnityEngine.Object;

namespace PathTracing
{
    /// <summary>
    /// RTXPT lighting pass (Phase 1).
    ///
    /// CPU side: collects Unity analytic lights and uploads LightBuffer / LightExBuffer /
    /// LightControlBuffer via GraphicsBuffer.SetData (mirrors CollectAnalyticLightsCPU).
    ///
    /// GPU side: dispatches the full LightsBaker proxy-build pipeline using
    /// NativeComputePipeline, matching LightsBaker::UpdateFrame in C++:
    ///   1. ResetLightProxyCounters
    ///   2. ResetPastToCurrentHistory
    ///   3. ComputeWeights
    ///   4. ComputeProxyCounts
    ///   5. ComputeProxyBaselineOffsets
    ///   6. CreateProxyJobs
    ///   7. ExecuteProxyJobs
    ///
    /// ImportanceSamplingType = 1 (Power-based) — driven by GPU.
    /// </summary>
    /// <summary>
    /// DEPRECATED — absorbed into NativeRtxptLightingUpdateBeginPass.
    /// This class is retained only to avoid breaking serialized asset references.
    /// </summary>
    [Obsolete("Use NativeRtxptLightingUpdateBeginPass instead.")]
    public class NativeRtxptLightingPass : ScriptableRenderPass, IDisposable
    {
        // ---- constants mirrors from PolymorphicLight.h ----------------------
        private const uint  kTypeShift            = 24;
        private const uint  kShapingEnableBit     = 1u << 28;
        private const uint  kShapingUseMinFalloff = 1u << 30;
        private const float kMinLog2Radiance      = -8f;
        private const float kMaxLog2Radiance      = 40f;

        // Env quad-tree node count (mirrors RTXPT_NEEAT_ENVMAP_QT_TOTAL_NODE_COUNT in LightingConfig.h)
        private const uint EnvQtTotalNodeCount = 5368;

        // LLB constants (mirrors NEEATBaker.hlsli)
        private const uint LLB_NUM_COMPUTE_THREADS = 128;

        private const uint LLB_LOCAL_BLOCK_SIZE = 32;

        // ComputeWeights items-per-group = LLB_LOCAL_BLOCK_SIZE * LLB_NUM_COMPUTE_THREADS
        private const uint LLB_WEIGHTS_ITEMS_PER_GROUP = LLB_LOCAL_BLOCK_SIZE * LLB_NUM_COMPUTE_THREADS;

        // WeightsBufferOffset ping-pong half: mirrors RTXPT_LIGHTING_WEIGHTS_COUNT_HALF = MaxLights+1
        private const uint WeightsCountHalf = NativeRtxptBufferResources.MaxLights + 1;

        // For ExecuteProxyJobs we dispatch over LLB_MAX_PROXY_PROC_TASKS which must cover
        // the actual ProxyBuildTaskCount written by CreateProxyJobs on the GPU.
        // ProxyBuildTaskCount = TotalLightCount + ceil(SamplingProxyCount / LLB_MAX_PROXIES_PER_TASK).
        // Use ProxySamplingCount (the buffer size) as the upper bound for proxy tasks.
        private const uint LLB_MAX_PROXIES_PER_TASK = 32;

        private static readonly uint LLB_MAX_PROXY_PROC_TASKS =
            (uint)NativeRtxptBufferResources.MaxLights +
            ((uint)NativeRtxptBufferResources.ProxySamplingCount + LLB_MAX_PROXIES_PER_TASK - 1) / LLB_MAX_PROXIES_PER_TASK;

        // ---- GPU pipelines --------------------------------------------------
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

        // ---- CPU staging arrays --------------------------------------------
        private static readonly RtxptLightingControlData[]    s_controlStaging  = new RtxptLightingControlData[1];
        private static          RtxptPolymorphicLightInfo[]   s_lightsStaging   = new RtxptPolymorphicLightInfo[NativeRtxptBufferResources.MaxLights];
        private static          RtxptPolymorphicLightInfoEx[] s_lightsExStaging = new RtxptPolymorphicLightInfoEx[NativeRtxptBufferResources.MaxLights];

        // ---- state set by Setup each frame ---------------------------------
        private NativeRtxptPassContext _ctx;
        private int                    _analyticLightCount;
        private bool                   _ping = true; // ping-pong for weights buffer

        // ---- debug throttle ------------------------------------------------
        private int _dbgFrameCounter;

        // ---- GPU readback (static: ExecutePass is a static callback) ------
        private static bool                    s_weightsReadbackPending;
        private static uint                    s_weightsReadbackOffset; // which offset was current when we dispatched
        private static bool                    s_ctrlReadbackPending;
        private static AsyncGPUReadbackRequest s_ctrlReadback;
        private static bool                    s_proxyCntReadbackPending;
        private static uint                    s_proxyCntReadbackLightCount;
        private static bool                    s_proxiesReadbackPending;

        // ====================================================================
        // Constructor / Dispose
        // ====================================================================

        public NativeRtxptLightingPass(
            NativeComputeShader resetLightProxyCounters,
            NativeComputeShader resetPastToCurrentHistory,
            NativeComputeShader computeWeights,
            NativeComputeShader computeProxyCounts,
            NativeComputeShader computeProxyBaselineOffsets,
            NativeComputeShader createProxyJobs,
            NativeComputeShader executeProxyJobs)
        {
            if (resetLightProxyCounters != null)
            {
                _resetLightProxyCountersCs = new NativeComputePipeline(resetLightProxyCounters);
                _resetLightProxyCountersDs = new NativeComputeDescriptorSet(_resetLightProxyCountersCs);
            }

            if (resetPastToCurrentHistory != null)
            {
                _resetPastToCurrentHistoryCs = new NativeComputePipeline(resetPastToCurrentHistory);
                _resetPastToCurrentHistoryDs = new NativeComputeDescriptorSet(_resetPastToCurrentHistoryCs);
            }

            if (computeWeights != null)
            {
                _computeWeightsCs = new NativeComputePipeline(computeWeights);
                _computeWeightsDs = new NativeComputeDescriptorSet(_computeWeightsCs);
            }

            if (computeProxyCounts != null)
            {
                _computeProxyCountsCs = new NativeComputePipeline(computeProxyCounts);
                _computeProxyCountsDs = new NativeComputeDescriptorSet(_computeProxyCountsCs);
            }

            if (computeProxyBaselineOffsets != null)
            {
                _computeProxyBaselineOffsetsCs = new NativeComputePipeline(computeProxyBaselineOffsets);
                _computeProxyBaselineOffsetsDs = new NativeComputeDescriptorSet(_computeProxyBaselineOffsetsCs);
            }

            if (createProxyJobs != null)
            {
                _createProxyJobsCs = new NativeComputePipeline(createProxyJobs);
                _createProxyJobsDs = new NativeComputeDescriptorSet(_createProxyJobsCs);
            }

            if (executeProxyJobs != null)
            {
                _executeProxyJobsCs = new NativeComputePipeline(executeProxyJobs);
                _executeProxyJobsDs = new NativeComputeDescriptorSet(_executeProxyJobsCs);
            }
        }

        public void Dispose()
        {
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
        }

        // ====================================================================
        // Setup – called on the main thread before RecordRenderGraph
        // ====================================================================

        public void Setup(NativeRtxptPassContext ctx)
        {
            // Check results from the previous frame's readback
            CheckReadbackResults();

            _ctx                = ctx;
            _analyticLightCount = CollectAndPackLights();
            UploadLightData();
            // readback is requested at end of ExecutePass (after GPU dispatches)
        }

        // ====================================================================
        // RecordRenderGraph – GPU proxy-build pipeline
        // ====================================================================

        private class PassData
        {
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
            internal NativeRtxptPassContext     Ctx;
            internal uint                       TotalLightCount;
            internal uint                       HistoricTotalLightCount;
            internal uint                       CurrentWeightsOffset;
            internal uint                       HistoricWeightsOffset;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            // // Log shader assignment status once
            // if ((_dbgFrameCounter % 60) == 2)
            //     Debug.Log($"[Lighting] Shaders: ResetProxyCounters={_resetLightProxyCountersCs != null}" +
            //               $"  ResetHistory={_resetPastToCurrentHistoryCs != null}" +
            //               $"  ComputeWeights={_computeWeightsCs != null}" +
            //               $"  ProxyCounts={_computeProxyCountsCs != null}" +
            //               $"  BaselineOffsets={_computeProxyBaselineOffsetsCs != null}" +
            //               $"  CreateJobs={_createProxyJobsCs != null}" +
            //               $"  ExecuteJobs={_executeProxyJobsCs != null}");

            // Skip if any required shader is missing
            if (_resetLightProxyCountersCs == null || _computeWeightsCs == null ||
                _computeProxyCountsCs == null || _computeProxyBaselineOffsetsCs == null ||
                _createProxyJobsCs == null || _executeProxyJobsCs == null)
            {
                Debug.LogWarning("[NativeRtxptLightingPass] One or more lighting compute shaders are missing — GPU pass skipped.");
                return;
            }

            // if ((_dbgFrameCounter % 60) == 2)
            //     Debug.Log($"[Lighting] RecordRenderGraph: TotalLightCount={_analyticLightCount}  ping={!_ping}  currentOffset={(_ping ? 0u : WeightsCountHalf)}  AllShaders=OK");

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.LightsBaker", out var passData);
            passData.ResetProxyCountersCs          = _resetLightProxyCountersCs;
            passData.ResetProxyCountersDs          = _resetLightProxyCountersDs;
            passData.ResetPastToCurrentCs          = _resetPastToCurrentHistoryCs;
            passData.ResetPastToCurrentDs          = _resetPastToCurrentHistoryDs;
            passData.ComputeWeightsCs              = _computeWeightsCs;
            passData.ComputeWeightsDs              = _computeWeightsDs;
            passData.ComputeProxyCountsCs          = _computeProxyCountsCs;
            passData.ComputeProxyCountsDs          = _computeProxyCountsDs;
            passData.ComputeProxyBaselineOffsetsCs = _computeProxyBaselineOffsetsCs;
            passData.ComputeProxyBaselineOffsetsDs = _computeProxyBaselineOffsetsDs;
            passData.CreateProxyJobsCs             = _createProxyJobsCs;
            passData.CreateProxyJobsDs             = _createProxyJobsDs;
            passData.ExecuteProxyJobsCs            = _executeProxyJobsCs;
            passData.ExecuteProxyJobsDs            = _executeProxyJobsDs;
            passData.Ctx                           = _ctx;
            passData.TotalLightCount               = (uint)_analyticLightCount + EnvQtTotalNodeCount;
            passData.HistoricTotalLightCount       = (uint)_analyticLightCount + EnvQtTotalNodeCount; // simple: no history tracking
            passData.CurrentWeightsOffset          = _ping ? 0u : WeightsCountHalf;
            passData.HistoricWeightsOffset         = _ping ? WeightsCountHalf : 0u;
            _ping                                  = !_ping;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        // ====================================================================
        // Execute – GPU dispatch sequence
        // ====================================================================

        // DXGI format constants used for typed buffer binding
        private const uint DXGI_FORMAT_R32_FLOAT = 41u;
        private const uint DXGI_FORMAT_R32_UINT  = 42u;

        // Buffer element strides derived from struct sizes at runtime
        private static readonly int StrideCtrl     = Marshal.SizeOf<RtxptLightingControlData>();
        private static readonly int StrideLights   = Marshal.SizeOf<RtxptPolymorphicLightInfo>();
        private static readonly int StrideLightsEx = Marshal.SizeOf<RtxptPolymorphicLightInfoEx>();

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var  cmd      = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var  buf      = data.Ctx.Buffers;
            uint total    = data.TotalLightCount;
            uint historic = data.HistoricTotalLightCount;

            // Retrieve native buffer pointers
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

            // Buffer element counts for typed/structured bindings
            int cCtrl     = buf.LightControlBuffer.count;
            int cLights   = buf.LightBuffer.count;
            int cLightsEx = buf.LightExBuffer.count;
            int cScrList  = buf.ScratchListBuffer.count;
            int cWeights  = buf.LightWeightsBuffer.count;
            int cHistCur  = buf.HistoryRemapCurrentToPast.count;
            int cHistPas  = buf.HistoryRemapPastToCurrent.count;
            int cProxyCnt = buf.LightProxyCounters.count;
            int cProxies  = buf.LightSamplingProxies.count;

            cmd.BeginSample("Rtxpt.LightsBaker");
            // UnityEngine.Debug.Log($"[Lighting] ExecutePass: total={data.TotalLightCount}  cCtrl={data.Ctx.Buffers.LightControlBuffer.count}  cLights={data.Ctx.Buffers.LightBuffer.count}  cProxies={data.Ctx.Buffers.LightSamplingProxies.count}");

            // ---- 1. ResetLightProxyCounters --------------------------------
            // dispatch: div_ceil(TotalLightCount+1, 128)  (+1 for the "invalid" slot)
            {
                var ds = data.ResetProxyCountersDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_perLightProxyCounters", pProxyCnt, cProxyCnt, DXGI_FORMAT_R32_UINT);
                uint gx = (total + 1 + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS;
                data.ResetProxyCountersCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            // ---- 2. ResetPastToCurrentHistory ------------------------------
            if (data.ResetPastToCurrentCs != null)
            {
                uint items = Math.Max(historic, total);
                var  ds    = data.ResetPastToCurrentDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_historyRemapPastToCurrent", pHistPas, cHistPas, DXGI_FORMAT_R32_UINT);
                uint gx         = (items + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS;
                if (gx == 0) gx = 1;
                data.ResetPastToCurrentCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            // ---- 3. ComputeWeights -----------------------------------------
            {
                var ds = data.ComputeWeightsDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWStructuredBuffer("u_lightsBuffer", pLights, cLights, StrideLights);
                ds.SetRWStructuredBuffer("u_lightsExBuffer", pLightsEx, cLightsEx, StrideLightsEx);
                ds.SetRWTypedBuffer("u_lightWeights", pWeights, cWeights, DXGI_FORMAT_R32_FLOAT);
                ds.SetRWTypedBuffer("u_historyRemapCurrentToPast", pHistCur, cHistCur, DXGI_FORMAT_R32_UINT);
                uint gx         = (total + LLB_WEIGHTS_ITEMS_PER_GROUP - 1) / LLB_WEIGHTS_ITEMS_PER_GROUP;
                if (gx == 0) gx = 1;
                data.ComputeWeightsCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            // ---- 4. ComputeProxyCounts -------------------------------------
            {
                var ds = data.ComputeProxyCountsDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_scratchList", pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_lightWeights", pWeights, cWeights, DXGI_FORMAT_R32_FLOAT);
                ds.SetRWTypedBuffer("u_perLightProxyCounters", pProxyCnt, cProxyCnt, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_lightSamplingProxies", pProxies, cProxies, DXGI_FORMAT_R32_UINT);
                uint gx         = (total + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS;
                if (gx == 0) gx = 1;
                data.ComputeProxyCountsCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            // ---- 5. ComputeProxyBaselineOffsets ----------------------------
            // Original dispatches with (1,1,1) — single thread-group for prefix-sum
            {
                var ds = data.ComputeProxyBaselineOffsetsDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWTypedBuffer("u_lightSamplingProxies", pProxies, cProxies, DXGI_FORMAT_R32_UINT);
                data.ComputeProxyBaselineOffsetsCs.Dispatch(cmd, ds, 1, 1, 1);
            }

            // ---- 6. CreateProxyJobs ----------------------------------------
            {
                var ds = data.CreateProxyJobsDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWBuffer("u_scratchBuffer", pScratch); // RWByteAddressBuffer — no stride
                ds.SetRWTypedBuffer("u_scratchList", pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_perLightProxyCounters", pProxyCnt, cProxyCnt, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer("u_lightSamplingProxies", pProxies, cProxies, DXGI_FORMAT_R32_UINT);
                uint gx         = (total + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS;
                if (gx == 0) gx = 1;
                data.CreateProxyJobsCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            // ---- 7. ExecuteProxyJobs ---------------------------------------
            // ProxyBuildTaskCount is written on GPU; dispatch over max tasks, shader self-limits.
            {
                var ds = data.ExecuteProxyJobsDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl, cCtrl, StrideCtrl);
                ds.SetRWBuffer("u_scratchBuffer", pScratch); // RWByteAddressBuffer — no stride
                ds.SetRWTypedBuffer("u_lightSamplingProxies", pProxies, cProxies, DXGI_FORMAT_R32_UINT);
                uint gx         = (LLB_MAX_PROXY_PROC_TASKS + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS;
                if (gx == 0) gx = 1;

                // Debug.Log($"[Lighting] Dispatching ExecuteProxyJobs: ProxyBuildTaskCount=unknown (GPU-written)  dispatch gx={gx} over max {LLB_MAX_PROXY_PROC_TASKS} tasks");
                data.ExecuteProxyJobsCs.Dispatch(cmd, ds, gx, 1, 1);
            }

            cmd.EndSample("Rtxpt.LightsBaker");

            // // Insert readback INTO the command buffer so it executes on the GPU timeline,
            // // after all dispatches above. Reads LightWeightsBuffer[currentOffset..+4] to
            // // verify ComputeWeights actually wrote non-zero weights.
            // if (!s_weightsReadbackPending)
            // {
            //     s_weightsReadbackOffset = data.CurrentWeightsOffset;
            //     cmd.RequestAsyncReadback(buf.LightWeightsBuffer, OnWeightsReadback);
            //     s_weightsReadbackPending = true;
            // }
            //
            // // Also readback control buffer to verify SamplingProxyCount / ProxyBuildTaskCount.
            // if (!s_ctrlReadbackPending)
            // {
            //     cmd.RequestAsyncReadback(buf.LightControlBuffer, OnCtrlReadback);
            //     s_ctrlReadbackPending = true;
            // }
            //
            // // Readback per-light proxy counters to verify proxy distribution.
            // if (!s_proxyCntReadbackPending)
            // {
            //     s_proxyCntReadbackLightCount = total;
            //     cmd.RequestAsyncReadback(buf.LightProxyCounters, OnProxyCountersReadback);
            //     s_proxyCntReadbackPending = true;
            // }
            //
            // // Readback first few proxy list entries to verify light indices stored.
            // if (!s_proxiesReadbackPending)
            // {
            //     cmd.RequestAsyncReadback(buf.LightSamplingProxies, Mathf.Min(16, buf.LightSamplingProxies.count) * sizeof(uint), 0, OnProxiesReadback);
            //     s_proxiesReadbackPending = true;
            // }
        }

        private static void OnCtrlReadback(AsyncGPUReadbackRequest req)
        {
            s_ctrlReadbackPending = false;
            if (req.hasError)
            {
                Debug.LogError("[Lighting][Readback] LightControlBuffer error.");
                return;
            }

            var arr = req.GetData<RtxptLightingControlData>();
            if (arr.Length == 0) return;
            var   c           = arr[0];
            float weightsSumF = BitConverter.Int32BitsToSingle((int)c.WeightsSumUINT);
            Debug.Log($"[Lighting][Readback] Ctrl: TotalLightCount={c.TotalLightCount}" +
                      $"  SamplingProxyCount={c.SamplingProxyCount}" +
                      $"  ProxyBuildTaskCount={c.ProxyBuildTaskCount}" +
                      $"  WeightsSum={weightsSumF:F4} (raw={c.WeightsSumUINT})");
            if (c.SamplingProxyCount == 0)
                Debug.LogWarning("[Lighting][Readback] SamplingProxyCount=0 → NEE skipped!");
        }

        private static void OnProxyCountersReadback(AsyncGPUReadbackRequest req)
        {
            s_proxyCntReadbackPending = false;
            if (req.hasError)
            {
                Debug.LogError("[Lighting][Readback] LightProxyCounters error.");
                return;
            }

            var  data       = req.GetData<uint>();
            uint lightCount = s_proxyCntReadbackLightCount;
            var  sb         = new System.Text.StringBuilder("[Lighting][Readback] ProxyCounters:");
            for (uint i = 0; i < lightCount && i < (uint)data.Length; i++)
                sb.Append($"  [{i}]={data[(int)i]}");
            Debug.Log(sb.ToString());
        }

        private static void OnProxiesReadback(AsyncGPUReadbackRequest req)
        {
            s_proxiesReadbackPending = false;
            if (req.hasError)
            {
                Debug.LogError("[Lighting][Readback] LightSamplingProxies error.");
                return;
            }

            var data = req.GetData<uint>();
            var sb   = new System.Text.StringBuilder("[Lighting][Readback] ProxyList[0.." + (data.Length - 1) + "]:");
            for (int i = 0; i < data.Length; i++)
                sb.Append($"  [{i}]=0x{data[i]:X8}");
            Debug.Log(sb.ToString());
        }

        private static void OnWeightsReadback(AsyncGPUReadbackRequest req)
        {
            s_weightsReadbackPending = false;
            if (req.hasError)
            {
                Debug.LogError("[Lighting][Readback] LightWeightsBuffer readback error.");
                return;
            }

            var   weights = req.GetData<float>();
            uint  off     = s_weightsReadbackOffset;
            float w0      = off < (uint)weights.Length ? weights[(int)off] : -1f;
            float w1      = (off + 1) < (uint)weights.Length ? weights[(int)(off + 1)] : -1f;
            Debug.Log($"[Lighting][Readback] LightWeights[{off}]={w0:F4}  [{off + 1}]={w1:F4}  (len={weights.Length})");
            if (w0 == 0f && w1 == 0f)
                Debug.LogWarning("[Lighting][Readback] All weights are 0 at currentOffset → ComputeWeights produced nothing.");
        }

        // ====================================================================
        // Private helpers
        // ====================================================================

        /// <summary>
        /// Checks whether a previously requested GPU readback has completed and logs key fields.
        /// Called at the start of Setup() so we're back on the main thread.
        /// </summary>
        private void CheckReadbackResults()
        {
            // Readback is now handled via OnWeightsReadback callback (GPU-timeline sequenced).
        }

        /// <summary>
        /// Collects all enabled Point / Spot lights in the scene and packs them
        /// into the staging arrays.  Returns the number of lights packed.
        /// </summary>
        private int CollectAndPackLights()
        {
            int count  = 0;
            var lights = Object.FindObjectsByType<Light>(FindObjectsSortMode.None);
            foreach (var light in lights)
            {
                if (light == null || !light.enabled) continue;
                if (count >= NativeRtxptBufferResources.MaxLights)
                {
                    Debug.LogWarning("[NativeRtxptLightingPass] MaxLights exceeded; some lights ignored.");
                    break;
                }

                switch (light.type)
                {
                    case LightType.Point:
                        PackPointLight(light, ref s_lightsStaging[count], ref s_lightsExStaging[count]);
                        // if ((_dbgFrameCounter % 60) == 0)
                        //     Debug.Log($"[Lighting] Point '{light.name}' @ {light.transform.position}  intensity={light.intensity}  color={light.color}  → ColorTypeAndFlags=0x{s_lightsStaging[count].ColorTypeAndFlags:X8}  LogRadiance={s_lightsStaging[count].LogRadiance}");
                        count++;
                        break;
                    case LightType.Spot:
                        PackSpotLight(light, ref s_lightsStaging[count], ref s_lightsExStaging[count]);
                        // if ((_dbgFrameCounter % 60) == 0)
                        //     Debug.Log($"[Lighting] Spot  '{light.name}' @ {light.transform.position}  spotAngle={light.spotAngle}  intensity={light.intensity}  → ColorTypeAndFlags=0x{s_lightsStaging[count].ColorTypeAndFlags:X8}");
                        count++;
                        break;
                }
            }

            // if ((_dbgFrameCounter % 60) == 0)
            //     Debug.Log($"[Lighting] CollectAndPackLights: found {lights.Length} Light components, packed {count} analytic lights.");

            _dbgFrameCounter++;
            return count;
        }

        /// <summary>
        /// Uploads packed data to the GPU buffers via <c>GraphicsBuffer.SetData</c>.
        /// WeightsBufferOffsets are set here so the GPU shaders see correct ping-pong values.
        /// </summary>
        private void UploadLightData()
        {
            var buf = _ctx.Buffers;
            if (buf == null) return;

            // Ping-pong: _ping is toggled in RecordRenderGraph after passData is filled,
            // so here we use the same _ping value to fill the control buffer before GPU dispatch.
            uint currentOffset  = _ping ? 0u : WeightsCountHalf;
            uint historicOffset = _ping ? WeightsCountHalf : 0u;

            // Build control record
            ref var ctrl = ref s_controlStaging[0];
            ctrl                         = default;
            ctrl.TotalLightCount         = (uint)_analyticLightCount + EnvQtTotalNodeCount;
            ctrl.AnalyticLightCount      = (uint)_analyticLightCount;
            ctrl.EnvmapQuadNodeCount     = EnvQtTotalNodeCount;
            ctrl.ImportanceSamplingType  = 1; // Power-based importance sampling
            ctrl.HistoricTotalLightCount = (uint)_analyticLightCount + EnvQtTotalNodeCount;
            // WeightsSumUINT starts at 0; GPU fills it via InterlockedAdd in ComputeWeights.
            ctrl.WeightsSumUINT = 0;
            // ProxyBuildTaskCount is filled by GPU in CreateProxyJobs.
            ctrl.ProxyBuildTaskCount = 0;
            // SamplingProxyCount is filled by GPU. Pre-zero it.
            ctrl.SamplingProxyCount = 0;
            // ---- BakerConstants embedded in _paddingBK[] ------------------
            // LightsBakerConstants layout (LightingTypes.hlsli), all offsets from BakerConstants start:
            //  [0]  DistantVsLocalRelativeImportance  float
            //  [1]  EnvMapImportanceMapMIPCount        uint   (11 for 1024×1024)
            //  [2]  EnvMapImportanceMapResolution      uint   (1024)
            //  [28] CurrentWeightsBufferOffset         uint
            //  [29] HistoricWeightsBufferOffset        uint
            //  [88..99]  EnvMapParams.Transform        float3x4 row_major (local→world)
            //  [100..111] EnvMapParams.InvTransform    float3x4 row_major (world→local)
            //  [112..114] EnvMapParams.ColorMultiplier float3
            //  [115] EnvMapParams.Enabled              float
            float envIntensity = _ctx.Setting?.environmentMapIntensity ?? 1.0f;
            Color envTint      = (_ctx.Setting?.environmentMapTint ?? Color.white).linear;
            unsafe
            {
                ctrl._paddingBK[28] = currentOffset;  // BakerConstants.CurrentWeightsBufferOffset
                ctrl._paddingBK[29] = historicOffset; // BakerConstants.HistoricWeightsBufferOffset

                // DistantVsLocalRelativeImportance = 1.0 (equal weight for distant vs local)
                float distantVsLocal = 1.0f;
                ctrl._paddingBK[0] = *(uint*)&distantVsLocal;

                // EnvMapImportanceMapMIPCount: log2(1024)+1 = 11
                ctrl._paddingBK[1] = 11u;
                // EnvMapImportanceMapResolution: 1024
                ctrl._paddingBK[2] = 1024u;

                // EnvMapParams.Transform (local→world) and InvTransform (world→local)
                // Unity env map has no special rotation → both are identity float3x4
                // row_major float3x4: 3 rows × 4 columns
                float one = 1f, zero = 0f;
                // Transform rows
                ctrl._paddingBK[88]  = *(uint*)&one;  ctrl._paddingBK[89]  = *(uint*)&zero; ctrl._paddingBK[90]  = *(uint*)&zero; ctrl._paddingBK[91]  = *(uint*)&zero;
                ctrl._paddingBK[92]  = *(uint*)&zero; ctrl._paddingBK[93]  = *(uint*)&one;  ctrl._paddingBK[94]  = *(uint*)&zero; ctrl._paddingBK[95]  = *(uint*)&zero;
                ctrl._paddingBK[96]  = *(uint*)&zero; ctrl._paddingBK[97]  = *(uint*)&zero; ctrl._paddingBK[98]  = *(uint*)&one;  ctrl._paddingBK[99]  = *(uint*)&zero;
                // InvTransform rows (same as identity)
                ctrl._paddingBK[100] = *(uint*)&one;  ctrl._paddingBK[101] = *(uint*)&zero; ctrl._paddingBK[102] = *(uint*)&zero; ctrl._paddingBK[103] = *(uint*)&zero;
                ctrl._paddingBK[104] = *(uint*)&zero; ctrl._paddingBK[105] = *(uint*)&one;  ctrl._paddingBK[106] = *(uint*)&zero; ctrl._paddingBK[107] = *(uint*)&zero;
                ctrl._paddingBK[108] = *(uint*)&zero; ctrl._paddingBK[109] = *(uint*)&zero; ctrl._paddingBK[110] = *(uint*)&one;  ctrl._paddingBK[111] = *(uint*)&zero;

                // EnvMapParams.ColorMultiplier = tint * intensity
                float cr = envTint.r * envIntensity;
                float cg = envTint.g * envIntensity;
                float cb = envTint.b * envIntensity;
                ctrl._paddingBK[112] = *(uint*)&cr;
                ctrl._paddingBK[113] = *(uint*)&cg;
                ctrl._paddingBK[114] = *(uint*)&cb;

                // EnvMapParams.Enabled = 1.0
                ctrl._paddingBK[115] = *(uint*)&one;
            }

            // if ((_dbgFrameCounter % 60) == 1)  // just after counter incremented
            //     Debug.Log($"[Lighting] UploadLightData: TotalLightCount={ctrl.TotalLightCount}  ImportanceSamplingType={ctrl.ImportanceSamplingType}  currentOffset={currentOffset}  historicOffset={historicOffset}  StrideCtrl={StrideCtrl}  StrideLights={StrideLights}  StrideLightsEx={StrideLightsEx}");

            buf.LightControlBuffer.SetData(s_controlStaging);

            if (_analyticLightCount > 0)
            {
                // Analytic lights start after env quad nodes (indices 0..EnvQtTotalNodeCount-1 are reserved for GPU-filled env quad lights)
                buf.LightBuffer.SetData(s_lightsStaging, 0, (int)EnvQtTotalNodeCount, _analyticLightCount);
                buf.LightExBuffer.SetData(s_lightsExStaging, 0, (int)EnvQtTotalNodeCount, _analyticLightCount);
            }
        }

        // ====================================================================
        // Light packing  (mirrors ConvertLight() in LightsBaker.cpp)
        // ====================================================================

        private static void PackPointLight(Light light, ref RtxptPolymorphicLightInfo info,
            ref RtxptPolymorphicLightInfoEx infoEx)
        {
            info   = default;
            infoEx = default;

            var pos = light.transform.position;
            info.CenterX = pos.x;
            info.CenterY = pos.y;
            info.CenterZ = pos.z;

            // POLYLIGHT_POINT_ENABLE = 0 in PolymorphicLightPTConfig.h: point lights are
            // "handled by sphere".  Pack as kSphere with a small radius, matching the C++
            // LightsBaker path for point.radius > 0:
            //   radiance = flux / (PI * r^2)   (projected-area normalisation)
            //   Scalars  = fp16(radius)
            const float kPointRadius  = 0.01f; // 1 cm sphere approximating a point source
            float       projectedArea = Mathf.PI * kPointRadius * kPointRadius;

            Color   linear   = light.color.linear;
            Vector3 flux     = new Vector3(linear.r, linear.g, linear.b) * light.intensity;
            Vector3 radiance = flux / projectedArea;

            PackLightColor(radiance, ref info, (uint)RtxptLightType.Sphere);
            info.Scalars = Fp32ToFp16(kPointRadius);
        }

        private static void PackSpotLight(Light light, ref RtxptPolymorphicLightInfo info,
            ref RtxptPolymorphicLightInfoEx infoEx)
        {
            info   = default;
            infoEx = default;

            var pos = light.transform.position;
            info.CenterX = pos.x;
            info.CenterY = pos.y;
            info.CenterZ = pos.z;

            // Pack as kSphere + shaping (matches C++ LightsBaker spot with radius > 0)
            const float kSpotRadius   = 0.01f;
            float       projectedArea = Mathf.PI * kSpotRadius * kSpotRadius;

            // Unity spotAngle is the full cone angle; half-angle for outer
            float outerRad = Mathf.Deg2Rad * (light.spotAngle * 0.5f);
            float innerRad = outerRad * 0.8f;
            float softness = Mathf.Clamp01(1f - innerRad / outerRad);

            Color   linear   = light.color.linear;
            Vector3 flux     = new Vector3(linear.r, linear.g, linear.b) * light.intensity;
            Vector3 radiance = flux / projectedArea;

            PackLightColor(radiance, ref info, (uint)RtxptLightType.Sphere);
            info.ColorTypeAndFlags |= kShapingEnableBit;
            info.Scalars           =  Fp32ToFp16(kSpotRadius);

            // Extended shaping data
            var forward = light.transform.forward;
            infoEx.PrimaryAxis = NDirToOctUnorm32(forward);
            infoEx.CosConeAngleAndSoftness = Fp32ToFp16(Mathf.Cos(outerRad)) |
                                             (Fp32ToFp16(softness) << 16);
        }

        // ====================================================================
        // Encoding helpers (mirrors LightsBaker.cpp / PolymorphicLight.h)
        // ====================================================================

        /// <summary>
        /// Encodes <paramref name="color"/> into <c>ColorTypeAndFlags</c> + <c>LogRadiance</c>.
        /// Mirrors <c>packLightColor()</c> in <c>LightsBaker.cpp</c>.
        /// </summary>
        private static void PackLightColor(Vector3 color, ref RtxptPolymorphicLightInfo info, uint typeCode)
        {
            info.ColorTypeAndFlags = typeCode << (int)kTypeShift;

            float maxRadiance = Mathf.Max(color.x, Mathf.Max(color.y, color.z));
            if (maxRadiance <= 0f) return;

            float logN = Mathf.Clamp01(
                (Mathf.Log(maxRadiance, 2f) - kMinLog2Radiance) / (kMaxLog2Radiance - kMinLog2Radiance));
            uint packedRadiance = (uint)Mathf.Min(Mathf.Ceil(logN * 65534f) + 1f, 0xFFFF);

            // Unpack to find the quantised radiance used for colour normalisation
            float unpackedRadiance = Mathf.Pow(2f,
                ((packedRadiance - 1f) / 65534f) * (kMaxLog2Radiance - kMinLog2Radiance) + kMinLog2Radiance);

            float r = Mathf.Clamp01(color.x / unpackedRadiance);
            float g = Mathf.Clamp01(color.y / unpackedRadiance);
            float b = Mathf.Clamp01(color.z / unpackedRadiance);

            uint r8 = (uint)Mathf.RoundToInt(r * 255f) & 0xFFu;
            uint g8 = (uint)Mathf.RoundToInt(g * 255f) & 0xFFu;
            uint b8 = (uint)Mathf.RoundToInt(b * 255f) & 0xFFu;

            info.ColorTypeAndFlags |= r8 | (g8 << 8) | (b8 << 16);
            info.LogRadiance       =  packedRadiance;
        }

        /// <summary>
        /// Encodes a unit direction vector into an oct-mapped 32-bit value.
        /// Mirrors <c>NDirToOctUnorm32()</c> in <c>LightsBaker.cpp</c>.
        /// </summary>
        private static uint NDirToOctUnorm32(Vector3 n)
        {
            // Project onto L1 sphere
            float absSum = Mathf.Abs(n.x) + Mathf.Abs(n.y) + Mathf.Abs(n.z);
            float px     = n.x / absSum;
            float py     = n.y / absSum;

            if (n.z < 0f)
            {
                float ox = (1f - Mathf.Abs(py)) * (px >= 0f ? 1f : -1f);
                float oy = (1f - Mathf.Abs(px)) * (py >= 0f ? 1f : -1f);
                px = ox;
                py = oy;
            }

            // Map from [-1,1] to [0,1], then multiply by 0.5 + 0.5 again (matches C++ saturate(p*0.5+0.5))
            px = Mathf.Clamp01(px * 0.5f + 0.5f);
            py = Mathf.Clamp01(py * 0.5f + 0.5f);

            uint ux = (uint)Mathf.RoundToInt(px * 0xFFFEu);
            uint uy = (uint)Mathf.RoundToInt(py * 0xFFFEu);
            return ux | (uy << 16);
        }

        /// <summary>
        /// Converts a float32 value to a float16 bit pattern (returned as uint).
        /// Mirrors <c>fp32ToFp16()</c> in <c>LightsBaker.cpp</c>.
        /// </summary>
        private static uint Fp32ToFp16(float v)
        {
            // Use Unity's built-in Mathf.FloatToHalf equivalent via bit manipulation
            // Simple approach: clamp to fp16 range then convert
            uint u = (uint)BitConverter.ToInt32(BitConverter.GetBytes(v), 0);
            // Apply the 2^-112 multiplier trick to flush subnormals
            float scaled = v * 1.9259299444e-34f; // 2^-112
            uint  s      = (uint)BitConverter.ToInt32(BitConverter.GetBytes(scaled), 0);
            uint  sign   = u & 0x80000000u;
            uint  body   = s & 0x0FFFFFFFu;
            return ((sign >> 16) | (body >> 13)) & 0xFFFFu;
        }

        private static unsafe float UintBitsToFloat(uint v)
        {
            return *(float*)&v;
        }
    }
}