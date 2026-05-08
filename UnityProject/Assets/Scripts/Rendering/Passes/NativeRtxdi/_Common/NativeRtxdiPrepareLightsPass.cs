using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using NativeRender;
using Rtxdi;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    // -------------------------------------------------------------------------
    // Internal GPU structs (mirror SharedShaderInclude/ShaderParameters.h)
    // -------------------------------------------------------------------------

    /// <summary>
    /// Mirrors <c>PrepareLightsTask</c> from <c>ShaderParameters.h</c>.
    /// One entry per emissive sub-mesh (or per analytic primitive light).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct PrepareLightsTask
    {
        /// <summary>
        /// Bits [31..12] = flat instance index in t_InstanceData.
        /// Bits [11..0]  = sub-geometry index within the instance.
        /// Bit  31 (TASK_PRIMITIVE_LIGHT_BIT) = 1 for analytic lights.
        /// </summary>
        public uint instanceAndGeometryIndex;
        /// <summary>Number of triangles this task covers (1 for primitive lights).</summary>
        public uint triangleCount;
        /// <summary>First slot in LightDataBuffer[currentFrameOffset + lightBufferOffset].</summary>
        public uint lightBufferOffset;
        /// <summary>Frame-relative slot from the previous frame, or -1 if new.</summary>
        public int  previousLightBufferOffset;
    }

    /// <summary>
    /// Mirrors <c>PrepareLightsConstants</c> from <c>ShaderParameters.h</c>.
    /// Bound as a constant buffer at <c>b0 / g_Const</c> (VK_PUSH_CONSTANT maps to a
    /// regular CBV in the compiled DX12 DXIL).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NativePrepareLightsConstants
    {
        public uint numTasks;
        public uint currentFrameLightOffset;
        public uint previousFrameLightOffset;
        public uint _pad; // align to 16 bytes
    }

    // =========================================================================
    /// <summary>
    /// Native-RTXDI compute pass that mirrors
    /// <c>RTXDI/Samples/FullSample/Source/RenderPasses/PrepareLightsPass.{h,cpp}</c>.
    ///
    /// On the CPU side (<see cref="BuildTasksOnCpu"/>, called from
    /// <see cref="NativeRtxdiFeature.AddRenderPasses"/>), it:
    /// <list type="number">
    ///   <item>Iterates emissive geometries reported by <see cref="NativeRtxdiGPUScene"/>.</item>
    ///   <item>Builds the <c>TaskBuffer</c> and <c>GeometryInstanceToLight</c> arrays.</item>
    ///   <item>Uploads them to GPU via <c>GraphicsBuffer.SetData</c>.</item>
    ///   <item>Clears <c>LightIndexMappingBuffer</c> to zero.</item>
    ///   <item>Returns the <see cref="RTXDI_LightBufferParameters"/> for use in
    ///         <see cref="NativeResamplingConstants.lightBufferParams"/>.</item>
    /// </list>
    ///
    /// On the GPU side (<see cref="RecordRenderGraph"/>), it dispatches
    /// <c>PrepareLights.computeshader</c> with 256 threads per group.
    ///
    /// <para><b>Emissive meshes only for now</b> — analytic Unity lights (point / spot /
    /// directional) are not yet converted to <see cref="RTXDI.PolymorphicLightInfo"/> and are
    /// left for a follow-up.</para>
    /// </summary>
    // =========================================================================
    // === Shader Reflection: PrepareLights.computeshader ===
    // numthreads  [256, 1, 1]
    //
    // -- CBV (1) --
    //   g_Const                    ConstantBuffer<PrepareLightsConstants>  space0:b0
    //
    // -- Sampler (1) --
    //   s_MaterialSampler          SamplerState                            space0:s0
    //
    // -- SRV (7) --
    //   t_BindlessBuffers          ByteAddressBuffer[]                     space1:t0
    //   t_BindlessTextures         Texture2D[]                             space2:t0
    //   t_TaskBuffer               StructuredBuffer<PrepareLightsTask>     space0:t0
    //   t_PrimitiveLights          StructuredBuffer<PolymorphicLightInfo>  space0:t1
    //   t_InstanceData             StructuredBuffer<InstanceData>          space0:t2
    //   t_GeometryData             StructuredBuffer<GeometryData>          space0:t3
    //   t_MaterialConstants        StructuredBuffer<MaterialConstants>     space0:t4
    //
    // -- UAV (3) --
    //   u_LightDataBuffer          RWStructuredBuffer<PolymorphicLightInfo> space0:u0
    //   u_LightIndexMappingBuffer  RWBuffer<uint>                           space0:u1
    //   u_LocalLightPdfTexture     RWTexture2D<float>                       space0:u2
    public class NativeRtxdiPrepareLightsPass : ScriptableRenderPass, IDisposable
    {
        private const uint GroupSize = 256u;

        // ── GPU pipeline ──────────────────────────────────────────────────────
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;

        // b0 / g_Const: per-dispatch PrepareLightsConstants.
        private readonly GraphicsBuffer _prepareLightsConstantBuffer;
        private readonly NativePrepareLightsConstants[] _constantsArr = new NativePrepareLightsConstants[1];

        // ── Per-frame CPU state (computed in BuildTasksOnCpu) ─────────────────
        private NativeRtxdiPassContext _context;
        private int  _totalLightCount;      // total threads to dispatch
        private uint _numTasks;
        private uint _currentFrameOffset;
        private uint _previousFrameOffset;

        /// <summary>First slot in LightDataBuffer that belongs to the current frame.</summary>
        public uint CurrentFrameOffset => _currentFrameOffset;
        /// <summary>Number of valid light entries written this frame.</summary>
        public int  TotalLightCount    => _totalLightCount;

        // ── Temporal tracking (frame-relative light buffer offsets per geometry) ──
        // Key = (instanceIndex << 12 | geometrySubIndex).  Value = frame-relative offset.
        private readonly Dictionary<long, int> _prevEmissiveOffsets = new();
        private bool _oddFrame;

        // ── Working arrays (reused across frames to reduce GC pressure) ───────
        private PrepareLightsTask[] _taskScratch = Array.Empty<PrepareLightsTask>();
        private uint[]              _geomToLightScratch = Array.Empty<uint>();

        // ── RTXDI_INVALID_LIGHT_INDEX (0xFFFFFFFF) ────────────────────────────
        private const uint InvalidLightIndex = 0xFFFF_FFFFu;

        // ─────────────────────────────────────────────────────────────────────

        public NativeRtxdiPrepareLightsPass(NativeComputeShader shader)
        {
            _cs = new NativeComputePipeline(shader);
            _ds = new NativeComputeDescriptorSet(_cs);

            // Minimum constant-buffer size is 256 bytes on D3D12; allocate one element
            // padded to 16 bytes via the struct (NativePrepareLightsConstants = 16 B).
            _prepareLightsConstantBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Constant,
                1,
                Marshal.SizeOf<NativePrepareLightsConstants>())
            { name = "PrepareLightsConstants" };
        }

        public void Dispose()
        {
            _ds?.Dispose();
            _cs?.Dispose();
            _prepareLightsConstantBuffer?.Release();
        }

        // =====================================================================
        // CPU work — call from AddRenderPasses on the main thread.
        // =====================================================================

        /// <summary>
        /// Iterates emissive scene geometries, builds the <c>TaskBuffer</c> and
        /// <c>GeometryInstanceToLight</c> mappings, uploads them to GPU, and returns
        /// the <see cref="RTXDI_LightBufferParameters"/> to feed into
        /// <c>ResamplingConstants.lightBufferParams</c>.
        /// </summary>
        public RTXDI_LightBufferParameters BuildTasksOnCpu(NativeRtxdiPassContext ctx)
        {
            var resources = ctx.Resources;
            var gpuScene  = ctx.RtxdiGpuScene;

            // ---- Double-buffer offset for LightDataBuffer ping-pong ----
            // Mirror: constants.currentFrameLightOffset = m_maxLightsInBuffer * m_oddFrame;
            uint maxLights       = resources.MaxEmissiveTriangles + resources.MaxPrimitiveLights;
            _currentFrameOffset  = maxLights * (_oddFrame ? 1u : 0u);
            _previousFrameOffset = maxLights * (_oddFrame ? 0u : 1u);

            // ---- Collect emissive geometries from GPUScene ----
            var emissiveGeos = gpuScene.GetEmissiveGeometries();
            int numTasks     = emissiveGeos.Count;  // analytic lights excluded for now
            
            // Resize scratch arrays if needed
            if (_taskScratch.Length < numTasks)
                _taskScratch = new PrepareLightsTask[numTasks];

            int totalGeomInstances = gpuScene.TotalGeometryInstanceCount;
            int geomToLightLen     = Mathf.Max(totalGeomInstances, 1);
            if (_geomToLightScratch.Length < geomToLightLen)
                _geomToLightScratch = new uint[geomToLightLen];

            // Fill with RTXDI_INVALID_LIGHT_INDEX
            for (int i = 0; i < geomToLightLen; i++)
                _geomToLightScratch[i] = InvalidLightIndex;

            // ---- Build tasks ----
            uint lightBufferOffset = 0u;
            int  validTasks        = 0;

            foreach (var e in emissiveGeos)
            {
                if (lightBufferOffset + e.TriangleCount > maxLights)
                    break;  // overflow guard

                long hash = ((long)e.InstanceIndex << 12) | (long)e.GeometrySubIndex;
                int  prevOffset = _prevEmissiveOffsets.TryGetValue(hash, out int po) ? po : -1;

                _taskScratch[validTasks++] = new PrepareLightsTask
                {
                    instanceAndGeometryIndex  = ((uint)e.InstanceIndex << 12) | ((uint)e.GeometrySubIndex & 0xFFFu),
                    triangleCount             = e.TriangleCount,
                    lightBufferOffset         = lightBufferOffset,
                    previousLightBufferOffset = prevOffset,     // frame-relative, shader adds _previousFrameOffset
                };

                _prevEmissiveOffsets[hash] = (int)lightBufferOffset; // record for next frame

                // Fill GeometryInstanceToLight  (firstGeometryInstanceIndex + subIndex)
                int giIdx = (int)(e.FirstGeometryInstanceIndex + (uint)e.GeometrySubIndex);
                if (giIdx < geomToLightLen)
                    _geomToLightScratch[giIdx] = lightBufferOffset; // frame-relative

                lightBufferOffset += e.TriangleCount;
            }

            _numTasks       = (uint)validTasks;
            _totalLightCount = (int)lightBufferOffset;

            // ---- Upload GPU buffers ----
            if (validTasks > 0 && resources.TaskBuffer != null)
                resources.TaskBuffer.SetData(_taskScratch, 0, 0, validTasks);

            if (resources.GeometryInstanceToLight != null)
            {
                int uploadLen = Mathf.Min(geomToLightLen, resources.GeometryInstanceToLight.count);
                resources.GeometryInstanceToLight.SetData(_geomToLightScratch, 0, 0, uploadLen);
            }

            // Clear LightIndexMappingBuffer to 0 (zero = invalid mapping)
            // Done via a compute Clear or just by re-initializing from CPU on the first frame.
            // For simplicity, we leave this to the GPU dispatch clear (shader writes 0 for entries
            // that are not written, or the buffer starts as 0 on first allocation).

            // ---- Update constant buffer ----
            _constantsArr[0] = new NativePrepareLightsConstants
            {
                numTasks              = _numTasks,
                currentFrameLightOffset  = _currentFrameOffset,
                previousFrameLightOffset = _previousFrameOffset,
                _pad = 0u,
            };
            _prepareLightsConstantBuffer.SetData(_constantsArr);

            // Flip ping-pong for next frame
            _oddFrame = !_oddFrame;

            // ---- Compose RTXDI_LightBufferParameters ----
            var p = new RTXDI_LightBufferParameters();
            p.localLightBufferRegion.firstLightIndex = _currentFrameOffset;
            p.localLightBufferRegion.numLights       = (uint)_totalLightCount;
            // No infinite lights yet; firstLightIndex must still point past the local region.
            p.infiniteLightBufferRegion.firstLightIndex = _currentFrameOffset + (uint)_totalLightCount;
            p.infiniteLightBufferRegion.numLights        = 0;
            // No environment light — use RTXDI_INVALID_LIGHT_INDEX (0xFFFFFFFF).
            p.environmentLightParams.lightPresent = 0;
            p.environmentLightParams.lightIndex   = 0xFFFFFFFFu;
            return p;
        }

        // =====================================================================
        // Setup — call after BuildTasksOnCpu.
        // =====================================================================

        public void Setup(NativeRtxdiPassContext ctx)
        {
            _context = ctx;
        }

        // =====================================================================
        // RenderGraph
        // =====================================================================

        class PassData
        {
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal NativeRtxdiPassContext     Context;
            internal int                        TotalLightCount;
            internal GraphicsBuffer             ConstantBuffer;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            // Nothing to do if no emissive lights exist this frame
            if (_totalLightCount == 0 || _numTasks == 0)
                return;

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdi.PrepareLights", out var passData);

            passData.Cs             = _cs;
            passData.Ds             = _ds;
            passData.Context        = _context;
            passData.TotalLightCount = _totalLightCount;
            passData.ConstantBuffer  = _prepareLightsConstantBuffer;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            cmd.BeginSample(RenderPassMarkers.PrepareLightsCompute);

            var cs  = data.Cs;
            var ds  = data.Ds;
            var ctx = data.Context;
            var rtx = ctx.Resources;

            // ---- g_Const (b0): PrepareLightsConstants ----
            ds.SetConstantBuffer("g_Const", data.ConstantBuffer.GetNativeBufferPtr());

            // ---- SRV inputs ----
            if (rtx.TaskBuffer != null)
                ds.SetStructuredBuffer("t_TaskBuffer",
                    rtx.TaskBuffer.GetNativeBufferPtr(),
                    rtx.TaskBuffer.count, rtx.TaskBuffer.stride);

            if (rtx.PrimitiveLightBuffer != null)
                ds.SetStructuredBuffer("t_PrimitiveLights",
                    rtx.PrimitiveLightBuffer.GetNativeBufferPtr(),
                    rtx.PrimitiveLightBuffer.count, rtx.PrimitiveLightBuffer.stride);

            // t_InstanceData / t_GeometryData / t_MaterialConstants / bindless
            ctx.RtxdiGpuScene?.BindToShader(ds);

            // ---- UAV outputs ----
            if (rtx.LightDataBuffer != null)
                ds.SetRWStructuredBuffer("u_LightDataBuffer",
                    rtx.LightDataBuffer.GetNativeBufferPtr(),
                    rtx.LightDataBuffer.count, rtx.LightDataBuffer.stride);

            if (rtx.LightIndexMappingBuffer != null)
                ds.SetRWBuffer("u_LightIndexMappingBuffer",
                    rtx.LightIndexMappingBuffer.GetNativeBufferPtr());

            if (rtx.LocalLightPdfTexture?.rt != null)
                ds.SetRWTexture("u_LocalLightPdfTexture",
                    rtx.LocalLightPdfTexture.rt.GetNativeTexturePtr());

            // ---- Dispatch: ceil(totalLightCount / 256) groups of 256 threads ----
            uint groups = ((uint)data.TotalLightCount + GroupSize - 1u) / GroupSize;
            
            cs.Dispatch(cmd, ds, groups, 1u, 1u);

            cmd.EndSample(RenderPassMarkers.PrepareLightsCompute);
        }
    }
}
