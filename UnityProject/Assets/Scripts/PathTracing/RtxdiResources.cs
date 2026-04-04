using System;
using System.Runtime.InteropServices;
using Rtxdi;
using RTXDI;
using Rtxdi.DI;
using Rtxdi.GI;
using Rtxdi.LightSampling;
using Unity.Mathematics;
using UnityEngine;

namespace mini
{
    public struct SecondaryGBufferData
    {
        float3 worldPos;
        uint normal;

        uint2 throughputAndFlags; // .x = throughput.rg as float16, .y = throughput.b as float16, flags << 16
        uint diffuseAlbedo; // R11G11B10_UFLOAT
        uint specularAndRoughness; // R8G8B8A8_Gamma_UFLOAT

        float3 emission;
        float pdf;
    };

    public class RtxdiResources : IDisposable
    {
        private const int c_NumReSTIRDIReservoirBuffers = 3;
        private const int c_NumReSTIRGIReservoirBuffers = 2;

        private bool m_neighborOffsetsInitialized = false;

        private uint m_maxEmissiveMeshes;

        // public uint m_maxEmissiveTriangles;
        private uint m_maxGeometryInstances;


        // public GraphicsBuffer TaskBuffer { get; private set; }
        public GraphicsBuffer LightDataBuffer { get; private set; }

        // public GraphicsBuffer GeometryInstanceToLightBuffer { get; private set; }
        public ComputeBuffer NeighborOffsetsBuffer { get; private set; }
        public ComputeBuffer RisBuffer { get; private set; }

        public ComputeBuffer RisLightDataBuffer { get; private set; }
        public GraphicsBuffer LightReservoirBuffer { get; private set; }
        public GraphicsBuffer GIReservoirBuffer { get; private set; }
        public GraphicsBuffer SecondaryGBuffer { get; private set; }

        public GPUScene Scene;


        public unsafe RtxdiResources(
            ReSTIRDIContext context,
            RISBufferSegmentAllocator risBufferSegmentAllocator,
            GPUScene scene)

        {
            LightDataBuffer = scene._lightInfoBuffer;
            this.Scene = scene;
            // m_maxEmissiveMeshes = maxEmissiveMeshes;
            // m_maxEmissiveTriangles =scene.emissiveTriangleCount;
            // m_maxGeometryInstances = maxGeometryInstances;


            // // 1. TaskBuffer
            // // initial state: ShaderResource, canHaveUAVs = true
            // if (maxEmissiveMeshes > 0)
            // {
            //     TaskBuffer = new GraphicsBuffer(
            //         GraphicsBuffer.Target.Structured,
            //         (int)maxEmissiveMeshes,
            //         Marshal.SizeOf<PrepareLightsTask>()
            //     );
            //     TaskBuffer.name = "TaskBuffer";
            // }

            // 2. LightDataBuffer
            // 保存了全部灯光（三角面灯）的信息，数量就是全部的发光三角形数量。
            // initial state: ShaderResource, canHaveUAVs = true
            // if (maxEmissiveTriangles > 0)
            // {
            //     LightDataBuffer = new GraphicsBuffer(
            //         GraphicsBuffer.Target.Structured,
            //         (int)maxEmissiveTriangles,
            //         Marshal.SizeOf<RAB_LightInfo>()
            //     );
            //     LightDataBuffer.name = "LightDataBuffer";
            // }

            // 3. GeometryInstanceToLightBuffer
            // 每个几何实例对应一个 uint 索引，指向它关联的第一个灯光（面灯）在 LightDataBuffer 中的位置。
            // initial state: ShaderResource
            // if (maxGeometryInstances > 0)
            // {
            //     GeometryInstanceToLightBuffer = new GraphicsBuffer(
            //         GraphicsBuffer.Target.Structured,
            //         (int)maxGeometryInstances,
            //         sizeof(uint)
            //     );
            //     GeometryInstanceToLightBuffer.name = "GeometryInstanceToLightBuffer";
            // }

            // 获取参数
            var staticParams = context.GetStaticParameters();
            var reservoirParams = context.GetReservoirBufferParameters();

            // 4. NeighborOffsetsBuffer
            Debug.Log($"NeighborOffsetCount: {staticParams.NeighborOffsetCount}");

            NeighborOffsetsBuffer = new ComputeBuffer(
                (int)staticParams.NeighborOffsetCount,
                sizeof(Vector2),
                ComputeBufferType.Default
            );
            NeighborOffsetsBuffer.name = "NeighborOffsets";

            InitializeNeighborOffsets(staticParams.NeighborOffsetCount);

            // 5. LightReservoirBuffer
            // byteSize = sizeof(Packed) * pitch * numBuffers
            int reservoirStride = Marshal.SizeOf<RTXDI_PackedDIReservoir>();
            int totalReservoirs = (int)reservoirParams.reservoirArrayPitch * c_NumReSTIRDIReservoirBuffers;

            Debug.Log($"Creating LightReservoirBuffer with totalReservoirs: {totalReservoirs}, reservoirStride: {reservoirStride}");
            if (totalReservoirs > 0)
            {
                LightReservoirBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured,
                    totalReservoirs,
                    reservoirStride
                );
                LightReservoirBuffer.name = "LightReservoirBuffer";
            }

            int giReservoirStride = Marshal.SizeOf<RTXDI_PackedGIReservoir>();
            int totalGIReservoirs = (int)reservoirParams.reservoirArrayPitch * c_NumReSTIRGIReservoirBuffers;

            Debug.Log($"Creating GIReservoirBuffer with totalGIReservoirs: {totalGIReservoirs}, giReservoirStride: {giReservoirStride}");
            if (totalGIReservoirs > 0)
            {
                GIReservoirBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Structured,
                    totalGIReservoirs,
                    giReservoirStride
                );
                GIReservoirBuffer.name = "GIReservoirBuffer";
            }

            int secondaryGBufferStride = Marshal.SizeOf<SecondaryGBufferData>();
            int totalSecondaryGBuffers = (int)reservoirParams.reservoirArrayPitch;
            SecondaryGBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                totalSecondaryGBuffers,
                secondaryGBufferStride
            );
            SecondaryGBuffer.name = "SecondaryGBuffer";


            var totalSizeInElements = risBufferSegmentAllocator.GetTotalSizeInElements();
            Debug.Log($"Creating RisBuffer with totalSizeInElements: {totalSizeInElements}");

            RisBuffer = new ComputeBuffer(
                (int)math.max(totalSizeInElements, 1),
                sizeof(Vector2),
                ComputeBufferType.Default);
            RisBuffer.name = "RisBuffer";

            int lightDataStride = sizeof(uint) * 8;

            RisLightDataBuffer = new ComputeBuffer(
                (int)math.max(totalSizeInElements, 1),
                lightDataStride,
                ComputeBufferType.Default
            );
            RisLightDataBuffer.name = "RisLightDataBuffer";
        }


        void InitializeNeighborOffsets(uint neighborOffsetCount)
        {
            if (m_neighborOffsetsInitialized)
                return;


            var offsets = new Vector2[neighborOffsetCount];
            Array.Fill(offsets, Vector2.zero);

            {
                int R = 250;
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
                    if (rSq > 0.25f)
                        continue;

                    offsets[num++] = new Vector2((u - 0.5f) * R / 128.0f, (v - 0.5f) * R / 128.0f);
                }
            }

            // byte[] offsets = new byte[neighborOffsetCount * 2]; 
            //
            // RtxdiUtils.FillNeighborOffsetBuffer(offsets, neighborOffsetCount);

            NeighborOffsetsBuffer.SetData(offsets);

            m_neighborOffsetsInitialized = true;
        }

        public void Dispose()
        {
            NeighborOffsetsBuffer?.Release();
            LightReservoirBuffer?.Release();
        }
    }
}