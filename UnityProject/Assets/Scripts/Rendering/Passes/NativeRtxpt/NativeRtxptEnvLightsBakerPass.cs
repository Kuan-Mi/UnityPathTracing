using System;
using System.Runtime.InteropServices;
using NativeRender;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Phase 1b: Env-light quad-tree baker.
    ///
    /// Runs after EnvMapBakerPass (which produces the importance map) and before
    /// NativeRtxptLightingPass (which uploads CPU analytic light data and runs proxy builds).
    ///
    /// Dispatch sequence mirrors LightsBaker::UpdateFrame steps 1.4–1.10:
    ///   1. EnvLightsBackupPast        – backup last-frame quad-tree into scratchList[0..TotalNodes)
    ///   2. EnvLightsSubdivideBase     – (1,1,1) groups; builds base quad-tree in scratchList
    ///   3. EnvLightsSubdivideBoost    – (UnboostedNodeCount,1,1) groups; boosts quad-tree; writes lightsBuffer
    ///   4. EnvLightsFillLookupMap     – (TotalNodeCount,1,1) groups; projects nodes → envLightLookupMap
    ///   5. EnvLightsMapPastToCurrent  – maps history indices in historyRemapPastToCurrent
    ///
    /// The resulting <c>EnvLightLookupMap</c> (1024×1024 R32_UINT) is exposed on
    /// <see cref="NativeRtxptPassContext.EnvLightLookupMapPtr"/> for the path tracer (t18 t_EnvLookupMap).
    /// </summary>
    public class NativeRtxptEnvLightsBakerPass : ScriptableRenderPass, IDisposable
    {
        // ---- Quad-tree constants (mirror LightingConfig.h) -----------------
        private const uint EnvQtBaseResolution    = 4;
        private const uint EnvQtSubdivisions      = 24;
        private const uint EnvQtAdditionalNodes   = 3 * EnvQtSubdivisions;                         // 72
        private const uint EnvQtUnboostedCount    = EnvQtBaseResolution * EnvQtBaseResolution       // 16
                                                  + EnvQtAdditionalNodes;                           // = 88
        private const uint EnvQtBoostSubdivision  = 20;
        private const uint EnvQtBoostNodesMult    = EnvQtBoostSubdivision * 3 + 1;                  // 61
        private const uint EnvQtTotalNodeCount    = EnvQtUnboostedCount * EnvQtBoostNodesMult;      // 5368

        private const uint LLB_NUM_COMPUTE_THREADS = 128;
        private const int  EnvLookupMapDim          = 1024; // must equal EnvMapImportanceSamplingBaker dim

        // ---- Buffer format ------------------------------------------------
        private const uint DXGI_FORMAT_R32_UINT = 42u;

        // ---- Struct strides -----------------------------------------------
        private static readonly int StrideCtrl     = Marshal.SizeOf<RtxptLightingControlData>();
        private static readonly int StrideLights   = Marshal.SizeOf<RtxptPolymorphicLightInfo>();
        private static readonly int StrideLightsEx = Marshal.SizeOf<RtxptPolymorphicLightInfoEx>();

        // ---- GPU pipelines ------------------------------------------------
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

        // ---- Owned RenderTexture ------------------------------------------
        private RenderTexture _envLightLookupMapRt; // 1024×1024 R32_UINT UAV

        // ---- Per-frame state ----------------------------------------------
        private NativeRtxptPassContext _ctx;
        private int                    _dbgFrameCounter;

        // Set to false to silence all debug logs and disable GPU readback requests.
        public static bool                     s_enableDebugLog = false;

        // ---- Async readback state (static: ExecutePass is a static callback) ----
        private static bool                    s_ctrlReadbackPending;
        private static bool                    s_scratchListReadbackPending;
        private static bool                    s_lightsReadbackPending;
        private static bool                    s_lookupMapReadbackPending;
        private static RenderTexture           s_lookupMapForReadback;

        // ===================================================================
        // Constructor / Dispose
        // ===================================================================

        public NativeRtxptEnvLightsBakerPass(
            NativeComputeShader backupPastCs,
            NativeComputeShader subdivideBaseCs,
            NativeComputeShader subdivideBoostCs,
            NativeComputeShader fillLookupMapCs,
            NativeComputeShader mapPastToCurrentCs)
        {
            if (backupPastCs != null)
            {
                _backupPastCs = new NativeComputePipeline(backupPastCs);
                _backupPastDs = new NativeComputeDescriptorSet(_backupPastCs);
            }
            if (subdivideBaseCs != null)
            {
                _subdivideBaseCs = new NativeComputePipeline(subdivideBaseCs);
                _subdivideBaseDs = new NativeComputeDescriptorSet(_subdivideBaseCs);
            }
            if (subdivideBoostCs != null)
            {
                _subdivideBoostCs = new NativeComputePipeline(subdivideBoostCs);
                _subdivideBoostDs = new NativeComputeDescriptorSet(_subdivideBoostCs);
            }
            if (fillLookupMapCs != null)
            {
                _fillLookupMapCs = new NativeComputePipeline(fillLookupMapCs);
                _fillLookupMapDs = new NativeComputeDescriptorSet(_fillLookupMapCs);
            }
            if (mapPastToCurrentCs != null)
            {
                _mapPastToCurrentCs = new NativeComputePipeline(mapPastToCurrentCs);
                _mapPastToCurrentDs = new NativeComputeDescriptorSet(_mapPastToCurrentCs);
            }

            EnsureLookupMapTexture();
        }

        public void Dispose()
        {
            _backupPastDs?.Dispose();       _backupPastCs?.Dispose();
            _subdivideBaseDs?.Dispose();    _subdivideBaseCs?.Dispose();
            _subdivideBoostDs?.Dispose();   _subdivideBoostCs?.Dispose();
            _fillLookupMapDs?.Dispose();    _fillLookupMapCs?.Dispose();
            _mapPastToCurrentDs?.Dispose(); _mapPastToCurrentCs?.Dispose();

            if (_envLightLookupMapRt != null)
            {
                _envLightLookupMapRt.Release();
                UnityEngine.Object.DestroyImmediate(_envLightLookupMapRt);
                _envLightLookupMapRt = null;
            }
        }

        // ===================================================================
        // Setup
        // ===================================================================

        public void Setup(NativeRtxptPassContext ctx)
        {
            _dbgFrameCounter++;
            _ctx = ctx;
            EnsureLookupMapTexture();

            // Log shader state on first frame
            if (s_enableDebugLog && _dbgFrameCounter == 1)
            {
                Debug.Log($"[EnvLightsBaker] Init: " +
                          $"BackupPast={_backupPastCs != null}  " +
                          $"SubdivideBase={_subdivideBaseCs != null}  " +
                          $"SubdivideBoost={_subdivideBoostCs != null}  " +
                          $"FillLookupMap={_fillLookupMapCs != null}  " +
                          $"MapPastToCurrent={_mapPastToCurrentCs != null}  " +
                          $"LookupMapCreated={_envLightLookupMapRt?.IsCreated()}  " +
                          $"TotalNodeCount={EnvQtTotalNodeCount}  UnboostedCount={EnvQtUnboostedCount}");
            }

            // Expose the lookup map pointer so the path tracer can bind t_EnvLookupMap (t18).
            ctx.EnvLightLookupMapPtr = _envLightLookupMapRt != null && _envLightLookupMapRt.IsCreated()
                ? _envLightLookupMapRt.GetNativeTexturePtr()
                : IntPtr.Zero;

            // Save RT ref for readback callbacks (static field)
            s_lookupMapForReadback = _envLightLookupMapRt;

            // Periodic pointer/buffer status log
            if (s_enableDebugLog && (_dbgFrameCounter % 60) == 2)
            {
                Debug.Log($"[EnvLightsBaker] frame={_dbgFrameCounter}  " +
                          $"EnvImportanceMapPtr={ctx.EnvImportanceMapPtr}  " +
                          $"EnvRadianceAndImportanceMapPtr={ctx.EnvRadianceAndImportanceMapPtr}  " +
                          $"EnvLightLookupMapPtr={ctx.EnvLightLookupMapPtr}  " +
                          $"ScratchListCount={ctx.Buffers?.ScratchListBuffer?.count}  " +
                          $"LightBufferCount={ctx.Buffers?.LightBuffer?.count}");
            }
        }

        // ===================================================================
        // RecordRenderGraph
        // ===================================================================

        private class PassData
        {
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
            internal NativeRtxptPassContext     Ctx;
            internal IntPtr                     EnvRadianceAndImportanceMapPtr;
            internal IntPtr                     EnvLightLookupMapPtr;
            internal int                        DbgFrame;
            internal bool                       EnableDebugLog;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (_subdivideBaseCs == null || _subdivideBoostCs == null || _fillLookupMapCs == null)
            {
                if (_dbgFrameCounter <= 3 || (_dbgFrameCounter % 60) == 1)
                    Debug.LogWarning($"[EnvLightsBaker] Shaders missing — pass skipped." +
                                     $" SubdivideBase={_subdivideBaseCs != null}" +
                                     $" SubdivideBoost={_subdivideBoostCs != null}" +
                                     $" FillLookupMap={_fillLookupMapCs != null}");
                return;
            }

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.EnvLightsBaker", out var pd);
            pd.BackupPastCs          = _backupPastCs;
            pd.BackupPastDs          = _backupPastDs;
            pd.SubdivideBaseCs       = _subdivideBaseCs;
            pd.SubdivideBaseDs       = _subdivideBaseDs;
            pd.SubdivideBoostCs      = _subdivideBoostCs;
            pd.SubdivideBoostDs      = _subdivideBoostDs;
            pd.FillLookupMapCs       = _fillLookupMapCs;
            pd.FillLookupMapDs       = _fillLookupMapDs;
            pd.MapPastToCurrentCs    = _mapPastToCurrentCs;
            pd.MapPastToCurrentDs    = _mapPastToCurrentDs;
            pd.Ctx                              = _ctx;
            pd.EnvRadianceAndImportanceMapPtr   = _ctx.EnvRadianceAndImportanceMapPtr;
            pd.EnvLightLookupMapPtr             = _ctx.EnvLightLookupMapPtr;
            pd.DbgFrame                         = _dbgFrameCounter;
            pd.EnableDebugLog                   = s_enableDebugLog;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        // ===================================================================
        // Execute
        // ===================================================================

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd      = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            var buf      = data.Ctx.Buffers;
            int dbgFrame = data.DbgFrame;

            var pCtrl    = buf.LightControlBuffer.GetNativeBufferPtr();
            var pLights  = buf.LightBuffer.GetNativeBufferPtr();
            var pLightsEx= buf.LightExBuffer.GetNativeBufferPtr();
            var pScrList = buf.ScratchListBuffer.GetNativeBufferPtr();
            var pHistCur = buf.HistoryRemapCurrentToPast.GetNativeBufferPtr();
            var pHistPas = buf.HistoryRemapPastToCurrent.GetNativeBufferPtr();

            int cCtrl    = buf.LightControlBuffer.count;
            int cLights  = buf.LightBuffer.count;
            int cLightsEx= buf.LightExBuffer.count;
            int cScrList = buf.ScratchListBuffer.count;
            int cHistCur = buf.HistoryRemapCurrentToPast.count;
            int cHistPas = buf.HistoryRemapPastToCurrent.count;

            var envImportancePtr = data.EnvRadianceAndImportanceMapPtr;
            var envLookupMapPtr  = data.EnvLightLookupMapPtr;

            // Log native pointers on early frames and periodically
            if (data.EnableDebugLog && (dbgFrame <= 3 || (dbgFrame % 60) == 2))
            {
                Debug.Log($"[EnvLightsBaker] ExecutePass frame={dbgFrame}" +
                          $"  importancePtr={envImportancePtr}  lookupMapPtr={envLookupMapPtr}" +
                          $"  pCtrl={pCtrl}  pScrList={pScrList}  pLights={pLights}" +
                          $"  cCtrl={cCtrl}  cScrList={cScrList}  cLights={cLights}");
            }

            cmd.BeginSample("Rtxpt.EnvLightsBaker");

            // ---- 1. EnvLightsBackupPast -----------------------------------
            if (data.BackupPastCs != null)
            {
                uint gx = (EnvQtTotalNodeCount + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS;
                if (data.EnableDebugLog && dbgFrame <= 3) Debug.Log($"[EnvLightsBaker] Dispatching BackupPast gx={gx}");
                var ds = data.BackupPastDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl,    cCtrl,    StrideCtrl);
                ds.SetRWStructuredBuffer("u_lightsBuffer",  pLights,  cLights,  StrideLights);
                ds.SetRWTypedBuffer(     "u_scratchList",   pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                data.BackupPastCs.Dispatch(cmd, data.BackupPastDs, gx, 1, 1);
            }

            // ---- 2. EnvLightsSubdivideBase --------------------------------
            {
                if (data.EnableDebugLog && dbgFrame <= 3) Debug.Log("[EnvLightsBaker] Dispatching SubdivideBase (1,1,1)");
                var ds = data.SubdivideBaseDs;
                ds.SetTexture(           "t_envRadianceAndImportanceMap", envImportancePtr);
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl,    cCtrl,    StrideCtrl);
                ds.SetRWTypedBuffer(     "u_scratchList",   pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                data.SubdivideBaseCs.Dispatch(cmd, data.SubdivideBaseDs, 1, 1, 1);
            }

            // ---- 3. EnvLightsSubdivideBoost --------------------------------
            {
                if (data.EnableDebugLog && dbgFrame <= 3) Debug.Log($"[EnvLightsBaker] Dispatching SubdivideBoost ({EnvQtUnboostedCount},1,1)");
                var ds = data.SubdivideBoostDs;
                ds.SetTexture(           "t_envRadianceAndImportanceMap", envImportancePtr);
                ds.SetRWStructuredBuffer("u_controlBuffer",             pCtrl,    cCtrl,    StrideCtrl);
                ds.SetRWStructuredBuffer("u_lightsBuffer",              pLights,  cLights,  StrideLights);
                ds.SetRWStructuredBuffer("u_lightsExBuffer",            pLightsEx,cLightsEx,StrideLightsEx);
                ds.SetRWTypedBuffer(     "u_scratchList",               pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer(     "u_historyRemapCurrentToPast", pHistCur, cHistCur, DXGI_FORMAT_R32_UINT);
                ds.SetRWTexture(         "u_envLightLookupMap",         envLookupMapPtr);
                data.SubdivideBoostCs.Dispatch(cmd, data.SubdivideBoostDs, EnvQtUnboostedCount, 1, 1);
            }

            // ---- 4. EnvLightsFillLookupMap --------------------------------
            {
                if (data.EnableDebugLog && dbgFrame <= 3) Debug.Log($"[EnvLightsBaker] Dispatching FillLookupMap ({EnvQtTotalNodeCount},1,1)");
                var ds = data.FillLookupMapDs;
                ds.SetRWStructuredBuffer("u_controlBuffer", pCtrl,  cCtrl,  StrideCtrl);
                ds.SetRWStructuredBuffer("u_lightsBuffer",  pLights,cLights,StrideLights);
                ds.SetRWTexture(         "u_envLightLookupMap", envLookupMapPtr);
                data.FillLookupMapCs.Dispatch(cmd, data.FillLookupMapDs, EnvQtTotalNodeCount, 1, 1);
            }

            // ---- 5. EnvLightsMapPastToCurrent ----------------------------
            if (data.MapPastToCurrentCs != null)
            {
                uint gx = (EnvQtTotalNodeCount + LLB_NUM_COMPUTE_THREADS - 1) / LLB_NUM_COMPUTE_THREADS;
                if (data.EnableDebugLog && dbgFrame <= 3) Debug.Log($"[EnvLightsBaker] Dispatching MapPastToCurrent gx={gx}");
                var ds = data.MapPastToCurrentDs;
                ds.SetRWStructuredBuffer("u_controlBuffer",            pCtrl,    cCtrl,    StrideCtrl);
                ds.SetRWTypedBuffer(     "u_scratchList",              pScrList, cScrList, DXGI_FORMAT_R32_UINT);
                ds.SetRWTypedBuffer(     "u_historyRemapPastToCurrent",pHistPas, cHistPas, DXGI_FORMAT_R32_UINT);
                ds.SetRWTexture(         "u_envLightLookupMap",        envLookupMapPtr);
                data.MapPastToCurrentCs.Dispatch(cmd, data.MapPastToCurrentDs, gx, 1, 1);
            }

            cmd.EndSample("Rtxpt.EnvLightsBaker");

            // ---- GPU readback requests (scheduled after all dispatches) ----
            if (!data.EnableDebugLog) return;

            // Readback LightControlBuffer to verify EnvmapQuadNodeCount
            if (!s_ctrlReadbackPending)
            {
                s_ctrlReadbackPending = true;
                cmd.RequestAsyncReadback(buf.LightControlBuffer, OnCtrlReadback);
            }

            // Readback full ScratchListBuffer — statistics will show how many entries are non-zero
            if (!s_scratchListReadbackPending)
            {
                s_scratchListReadbackPending = true;
                cmd.RequestAsyncReadback(buf.ScratchListBuffer, OnScratchListReadback);
            }

            // Readback full LightBuffer — statistics will show how many entries are non-zero
            if (!s_lightsReadbackPending)
            {
                s_lightsReadbackPending = true;
                cmd.RequestAsyncReadback(buf.LightBuffer, OnLightsReadback);
            }

            // Readback full EnvLightLookupMap — statistics will show how many pixels are non-zero
            if (!s_lookupMapReadbackPending && s_lookupMapForReadback != null
                && s_lookupMapForReadback.IsCreated())
            {
                s_lookupMapReadbackPending = true;
                cmd.RequestAsyncReadback(s_lookupMapForReadback, OnLookupMapReadback);
            }
        }

        // ===================================================================
        // Readback callbacks
        // ===================================================================

        private static void OnCtrlReadback(AsyncGPUReadbackRequest req)
        {
            s_ctrlReadbackPending = false;
            if (req.hasError) { Debug.LogError("[EnvLightsBaker][Readback] LightControlBuffer error."); return; }

            var arr = req.GetData<RtxptLightingControlData>();
            if (arr.Length == 0) return;
            var c = arr[0];
            Debug.Log($"[EnvLightsBaker][Readback] ControlBuffer:" +
                      $"  TotalLightCount={c.TotalLightCount}" +
                      $"  AnalyticLightCount={c.AnalyticLightCount}" +
                      $"  EnvmapQuadNodeCount={c.EnvmapQuadNodeCount}" +
                      $"  ImportanceSamplingType={c.ImportanceSamplingType}");
            if (c.EnvmapQuadNodeCount == 0)
                Debug.LogWarning("[EnvLightsBaker][Readback] EnvmapQuadNodeCount=0 — SubdivideBoost may not have run or ControlBuffer not initialised!");
        }

        private static void OnScratchListReadback(AsyncGPUReadbackRequest req)
        {
            s_scratchListReadbackPending = false;
            if (req.hasError) { Debug.LogError("[EnvLightsBaker][Readback] ScratchListBuffer error."); return; }

            var data = req.GetData<uint>();
            uint nonZero = 0;
            uint firstNonZeroIdx = 0;
            uint firstNonZeroVal = 0;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] != 0)
                {
                    if (nonZero == 0) { firstNonZeroIdx = (uint)i; firstNonZeroVal = data[i]; }
                    nonZero++;
                }
            }
            Debug.Log($"[EnvLightsBaker][Readback] ScratchList: total={data.Length}  nonZero={nonZero}" +
                      (nonZero > 0 ? $"  firstNonZero=[{firstNonZeroIdx}]=0x{firstNonZeroVal:X8}" : ""));
            if (nonZero == 0)
                Debug.LogWarning("[EnvLightsBaker][Readback] ScratchList all zeros — SubdivideBase may not have written quad-tree seeds!");
        }

        private static void OnLightsReadback(AsyncGPUReadbackRequest req)
        {
            s_lightsReadbackPending = false;
            if (req.hasError) { Debug.LogError("[EnvLightsBaker][Readback] LightBuffer error."); return; }

            var  data           = req.GetData<RtxptPolymorphicLightInfo>();
            uint nonZero        = 0;
            uint nonZeroLogRad  = 0;
            uint firstIdx       = 0;
            uint firstCTF       = 0;
            uint firstLRad      = 0;
            uint firstNZLRIdx   = 0;
            uint firstNZLRVal   = 0;
            for (int i = 0; i < data.Length; i++)
            {
                var li = data[i];
                if (li.ColorTypeAndFlags != 0 || li.LogRadiance != 0u)
                {
                    if (nonZero == 0) { firstIdx = (uint)i; firstCTF = li.ColorTypeAndFlags; firstLRad = li.LogRadiance; }
                    nonZero++;
                }
                if (li.LogRadiance != 0u)
                {
                    if (nonZeroLogRad == 0) { firstNZLRIdx = (uint)i; firstNZLRVal = li.LogRadiance; }
                    nonZeroLogRad++;
                }
            }
            Debug.Log($"[EnvLightsBaker][Readback] LightBuffer: total={data.Length}  nonZeroEntries={nonZero}  nonZeroLogRad={nonZeroLogRad}" +
                      (nonZero > 0 ? $"  firstNonZero=[{firstIdx}] CTF=0x{firstCTF:X8} LogRad=0x{firstLRad:X8}" : "") +
                      (nonZeroLogRad > 0 ? $"  firstNonZeroLR=[{firstNZLRIdx}]=0x{firstNZLRVal:X8}" : "  ALL LogRad=0!"));
            if (nonZero == 0)
                Debug.LogWarning("[EnvLightsBaker][Readback] LightBuffer all zeros — SubdivideBoost may not have written env quad nodes!");
            if (nonZero > 0 && nonZeroLogRad == 0)
                Debug.LogWarning("[EnvLightsBaker][Readback] All env quad entries have LogRad=0 — radiance is zero (check RGBA16F texture RGB channels)!");
        }

        private static void OnLookupMapReadback(AsyncGPUReadbackRequest req)
        {
            s_lookupMapReadbackPending = false;
            if (req.hasError) { Debug.LogError("[EnvLightsBaker][Readback] EnvLightLookupMap error."); return; }

            var  data        = req.GetData<uint>();
            uint nonZero     = 0;
            uint minVal      = uint.MaxValue;
            uint maxVal      = 0;
            uint firstNZIdx  = 0;
            uint firstNZVal  = 0;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] != 0)
                {
                    if (nonZero == 0) { firstNZIdx = (uint)i; firstNZVal = data[i]; }
                    nonZero++;
                    if (data[i] < minVal) minVal = data[i];
                    if (data[i] > maxVal) maxVal = data[i];
                }
            }
            Debug.Log($"[EnvLightsBaker][Readback] LookupMap: totalPixels={data.Length}  nonZero={nonZero}" +
                      (nonZero > 0 ? $"  min={minVal}  max={maxVal}  firstNonZero=[{firstNZIdx}]={firstNZVal}" : ""));
            if (nonZero == 0)
                Debug.LogWarning("[EnvLightsBaker][Readback] LookupMap all zeros — FillLookupMap may not have written light indices!");
        }

        // ===================================================================
        // Helpers
        // ===================================================================

        private void EnsureLookupMapTexture()
        {
            if (_envLightLookupMapRt != null && _envLightLookupMapRt.IsCreated()) return;
            _envLightLookupMapRt?.Release();

            // R32_UINT: Unity has no RenderTextureFormat for unsigned 32-bit int;
            // use GraphicsFormat.R32_UInt via RenderTextureDescriptor.
            var desc = new RenderTextureDescriptor(EnvLookupMapDim, EnvLookupMapDim, GraphicsFormat.R32_UInt, 0)
            {
                enableRandomWrite = true,
                useMipMap         = false,
                dimension         = UnityEngine.Rendering.TextureDimension.Tex2D,
            };
            _envLightLookupMapRt = new RenderTexture(desc)
            {
                autoGenerateMips = false,
                hideFlags        = HideFlags.HideAndDontSave,
            };
            _envLightLookupMapRt.Create();
        }
    }
}
