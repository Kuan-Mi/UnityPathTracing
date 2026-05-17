using System;
using System.Runtime.InteropServices;
using mini;
using Rtxdi;
using RTXDI;
using Rtxdi.DI;
using Rtxdi.GI;
using Rtxdi.LightSampling;
using Rtxdi.PT;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// FullSample-aligned RTXDI resource bundle used by <see cref="NativeRtxdiFeature"/>.
    ///
    /// Mirrors <c>RTXDI/Samples/FullSample/Source/RtxdiResources.{h,cpp}</c>:
    ///   - <c>TaskBuffer</c>             : per-frame CPU-built tasks consumed by PrepareLights.computeshader.
    ///   - <c>PrimitiveLightBuffer</c>   : per-frame analytic primitive lights from the scene graph.
    ///   - <c>LightDataBuffer</c>        : double-sized RW buffer of <see cref="PolymorphicLightInfo"/>
    ///                                     written by PrepareLights and consumed by DI/GI sampling.
    ///   - <c>GeometryInstanceToLight</c>: instance/geometry → first-light offset mapping.
    ///   - <c>LightIndexMappingBuffer</c>: maps current-frame light indices to previous-frame indices
    ///                                     (also written by PrepareLights, ping-pong).
    ///   - <c>LocalLightPdfTexture</c>   : RWTexture2D mip chain holding per-light power for RIS.
    ///   - <c>EnvironmentPdfTexture</c>  : RWTexture2D mip chain for environment importance sampling.
    ///   - <c>RisBuffer</c> / <c>RisLightDataBuffer</c> / <c>NeighborOffsetsBuffer</c>
    ///   - DI / GI reservoir buffers + secondary GBuffer.
    ///
    /// Decoupled from <see cref="LightScene"/>; the NEW PrepareLights pass (port of
    /// <c>RTXDI/Samples/FullSample/Source/RenderPasses/PrepareLightsPass.{h,cpp}</c>) will own
    /// the CPU-side TaskBuffer / PrimitiveLightBuffer assembly and the GPU compute dispatch.
    /// </summary>
    public class NativeRtxdiResources : IDisposable
    {
        // ---- counts (stable for the lifetime of this object; resize triggers re-allocation upstream) ----
        public uint MaxEmissiveMeshes    { get; }
        public uint MaxEmissiveTriangles { get; }
        public uint MaxPrimitiveLights   { get; }
        public uint MaxGeometryInstances { get; }

        // ---- light data ----
        public GraphicsBuffer TaskBuffer                  { get; private set; }
        public GraphicsBuffer PrimitiveLightBuffer        { get; private set; }
        public GraphicsBuffer LightDataBuffer             { get; private set; }
        public GraphicsBuffer GeometryInstanceToLight     { get; private set; }
        public GraphicsBuffer LightIndexMappingBuffer     { get; private set; }

        public RTHandle       LocalLightPdfTexture        { get; private set; }
        public uint2          LocalLightPdfTextureSize    { get; private set; }
        public RTHandle       EnvironmentPdfTexture       { get; private set; }
        public uint2          EnvironmentPdfTextureSize   { get; private set; }

        // ---- RIS / sampling ----
        public ComputeBuffer  RisBuffer                   { get; private set; }
        public ComputeBuffer  RisLightDataBuffer          { get; private set; }
        public ComputeBuffer  NeighborOffsetsBuffer       { get; private set; }
        // u_RayCountBuffer (u12): per-frame ray counter used by shade passes for debug/stats.
        public GraphicsBuffer  RayCountBuffer             { get; private set; }

        // ---- reservoirs / secondary GBuffer ----
        public GraphicsBuffer LightReservoirBuffer        { get; private set; }
        public GraphicsBuffer GIReservoirBuffer           { get; private set; }
        public GraphicsBuffer PTReservoirBuffer           { get; private set; }
        public GraphicsBuffer SecondaryGBuffer            { get; private set; }

        private const int c_NumReSTIRDIReservoirBuffers = 3;
        private const int c_NumReSTIRGIReservoirBuffers = 2;
        private const int c_NumReSTIRPTReservoirBuffers = 2;

        private bool m_neighborOffsetsInitialized = false;

        public NativeRtxdiResources(
            ReSTIRDIContext              context,
            RISBufferSegmentAllocator    risBufferSegmentAllocator,
            uint                         maxEmissiveMeshes,
            uint                         maxEmissiveTriangles,
            uint                         maxPrimitiveLights,
            uint                         maxGeometryInstances,
            uint                         environmentMapWidth,
            uint                         environmentMapHeight)
        {
            MaxEmissiveMeshes    = maxEmissiveMeshes;
            MaxEmissiveTriangles = maxEmissiveTriangles;
            MaxPrimitiveLights   = maxPrimitiveLights;
            MaxGeometryInstances = maxGeometryInstances;

            int prepareLightsTaskStride = sizeof(uint) * 4; // PrepareLightsTask: 4 x uint
            int polymorphicLightStride  = Marshal.SizeOf<PolymorphicLightInfo>();

            // ---- TaskBuffer ----
            uint taskCount = math.max(maxEmissiveMeshes + maxPrimitiveLights, 1u);
            TaskBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                (int)taskCount,
                prepareLightsTaskStride) { name = "TaskBuffer" };

            // ---- PrimitiveLightBuffer ----
            uint primitiveCount = math.max(maxPrimitiveLights, 1u);
            PrimitiveLightBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                (int)primitiveCount,
                polymorphicLightStride) { name = "PrimitiveLightBuffer" };

            // ---- LightDataBuffer (double-buffered: current + previous frame side-by-side) ----
            uint totalLights = math.max(maxEmissiveTriangles + maxPrimitiveLights, 1u);
            LightDataBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                (int)(totalLights * 2u),
                polymorphicLightStride) { name = "LightDataBuffer" };

            // ---- GeometryInstanceToLight ----
            uint geomCount = math.max(maxGeometryInstances, 1u);
            GeometryInstanceToLight = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                (int)geomCount,
                sizeof(uint)) { name = "GeometryInstanceToLightBuffer" };

            // ---- LightIndexMappingBuffer (current↔prev mapping, *2 for ping-pong) ----
            LightIndexMappingBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured | GraphicsBuffer.Target.Raw,
                (int)(totalLights * 2u),
                sizeof(uint)) { name = "LightIndexMappingBuffer" };

            // ---- LocalLightPdfTexture ----
            uint maxLocalLights = maxEmissiveTriangles + maxPrimitiveLights;
            RtxdiUtils.ComputePdfTextureSize(maxLocalLights, out uint pdfW, out uint pdfH, out uint pdfMips);
            LocalLightPdfTextureSize = new uint2(pdfW, pdfH);
            UnityEngine.Debug.Log($"[NativeRtxdiResources] LocalLightPdfTexture: maxLocalLights={maxLocalLights} (maxEmissiveTriangles={maxEmissiveTriangles} + maxPrimitiveLights={maxPrimitiveLights}), pdfSize={pdfW}x{pdfH}, mips={pdfMips}");
            LocalLightPdfTexture = RTHandles.Alloc(
                width:             (int)pdfW,
                height:            (int)pdfH,
                colorFormat:       GraphicsFormat.R32_SFloat,
                enableRandomWrite: true,
                useMipMap:         true,
                autoGenerateMips:  false,
                name:              "LocalLightPdfTexture");

            // ---- EnvironmentPdfTexture ----
            uint envW = math.max(environmentMapWidth, 1u);
            uint envH = math.max(environmentMapHeight, 1u);
            EnvironmentPdfTextureSize = new uint2(envW, envH);
            EnvironmentPdfTexture = RTHandles.Alloc(
                width:           (int)envW,
                height:          (int)envH,
                colorFormat:     GraphicsFormat.R32_SFloat,
                enableRandomWrite: true,
                useMipMap:       true,
                autoGenerateMips:false,
                name:            "EnvironmentPdfTexture");

            // ---- runtime / reservoir / RIS resources ----
            var staticParams     = context.GetStaticParameters();
            var reservoirParams  = context.GetReservoirBufferParameters();

            NeighborOffsetsBuffer = new ComputeBuffer(
                (int)staticParams.NeighborOffsetCount,
                sizeof(float) * 2,
                ComputeBufferType.Default) { name = "NeighborOffsets" };
            InitializeNeighborOffsets(staticParams.NeighborOffsetCount);

            int reservoirStride   = Marshal.SizeOf<RTXDI_PackedDIReservoir>();
            int totalDIReservoirs = (int)reservoirParams.reservoirArrayPitch * c_NumReSTIRDIReservoirBuffers;
            if (totalDIReservoirs > 0)
            {
                LightReservoirBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, totalDIReservoirs, reservoirStride) { name = "LightReservoirBuffer" };
            }

            int giReservoirStride = Marshal.SizeOf<RTXDI_PackedGIReservoir>();
            int totalGIReservoirs = (int)reservoirParams.reservoirArrayPitch * c_NumReSTIRGIReservoirBuffers;
            if (totalGIReservoirs > 0)
            {
                GIReservoirBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, totalGIReservoirs, giReservoirStride) { name = "GIReservoirBuffer" };
            }

            int ptReservoirStride = Marshal.SizeOf<RTXDI_PackedPTReservoir>();
            int totalPTReservoirs = (int)reservoirParams.reservoirArrayPitch * c_NumReSTIRPTReservoirBuffers;
            if (totalPTReservoirs > 0)
            {
                PTReservoirBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured, totalPTReservoirs, ptReservoirStride) { name = "PTReservoirBuffer" };
            }

            int secondaryGBufferStride = Marshal.SizeOf<SecondaryGBufferData>();
            int totalSecondaryGBuffers = (int)reservoirParams.reservoirArrayPitch;
            SecondaryGBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured, totalSecondaryGBuffers, secondaryGBufferStride) { name = "SecondaryGBuffer" };

            uint totalSizeInElements = math.max(risBufferSegmentAllocator.GetTotalSizeInElements(), 1u);
            RisBuffer = new ComputeBuffer(
                (int)totalSizeInElements, sizeof(float) * 2, ComputeBufferType.Default) { name = "RisBuffer" };

            int lightDataStride = sizeof(uint) * 8;
            RisLightDataBuffer = new ComputeBuffer(
                (int)totalSizeInElements * 2, lightDataStride, ComputeBufferType.Default) { name = "RisLightDataBuffer" };

            // u_RayCountBuffer (u12): single uint written atomically by shading passes.
            RayCountBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Raw, 1, sizeof(uint)) { name = "RayCountBuffer" };
        }

        private void InitializeNeighborOffsets(uint neighborOffsetCount)
        {
            if (m_neighborOffsetsInitialized) return;

            var offsets = new Vector2[neighborOffsetCount];
            const int R = 250;
            const float phi2 = 1.0f / 1.3247179572447f;
            uint num = 0;
            float u = 0.5f;
            float v = 0.5f;
            while (num < neighborOffsetCount)
            {
                u += phi2;
                v += phi2 * phi2;
                if (u >= 1.0f) u -= 1.0f;
                if (v >= 1.0f) v -= 1.0f;
                float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
                if (rSq > 0.25f) continue;
                offsets[num++] = new Vector2((u - 0.5f) * R / 128.0f, (v - 0.5f) * R / 128.0f);
            }
            NeighborOffsetsBuffer.SetData(offsets);
            m_neighborOffsetsInitialized = true;
        }

        public void Dispose()
        {
            TaskBuffer?.Dispose();                  TaskBuffer = null;
            PrimitiveLightBuffer?.Dispose();        PrimitiveLightBuffer = null;
            LightDataBuffer?.Dispose();             LightDataBuffer = null;
            GeometryInstanceToLight?.Dispose();     GeometryInstanceToLight = null;
            LightIndexMappingBuffer?.Dispose();     LightIndexMappingBuffer = null;

            if (LocalLightPdfTexture  != null) { RTHandles.Release(LocalLightPdfTexture);  LocalLightPdfTexture  = null; }
            if (EnvironmentPdfTexture != null) { RTHandles.Release(EnvironmentPdfTexture); EnvironmentPdfTexture = null; }

            RisBuffer?.Release();                   RisBuffer = null;
            RisLightDataBuffer?.Release();          RisLightDataBuffer = null;
            RayCountBuffer?.Dispose();              RayCountBuffer = null;
            NeighborOffsetsBuffer?.Release();       NeighborOffsetsBuffer = null;

            LightReservoirBuffer?.Dispose();        LightReservoirBuffer = null;
            GIReservoirBuffer?.Dispose();           GIReservoirBuffer = null;
            PTReservoirBuffer?.Dispose();           PTReservoirBuffer = null;
            SecondaryGBuffer?.Dispose();            SecondaryGBuffer = null;
        }
    }
}
