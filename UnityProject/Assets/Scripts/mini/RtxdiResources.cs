using System.Runtime.InteropServices;
using Rtxdi;
using RTXDI;
using Rtxdi.DI;
using Unity.Mathematics;
using UnityEngine;

namespace mini
{
    public class RtxdiResources
    {
        private const int c_NumReSTIRDIReservoirBuffers = 3;

        private bool m_neighborOffsetsInitialized = false;
        private uint m_maxEmissiveMeshes;
        private uint m_maxEmissiveTriangles;
        private uint m_maxGeometryInstances;


        // public GraphicsBuffer TaskBuffer { get; private set; }
        public GraphicsBuffer LightDataBuffer { get; private set; }
        // public GraphicsBuffer GeometryInstanceToLightBuffer { get; private set; }
        public ComputeBuffer NeighborOffsetsBuffer { get; private set; }
        public GraphicsBuffer LightReservoirBuffer { get; private set; }

  
        

        public unsafe  RtxdiResources(
            ReSTIRDIContext context,
            uint maxEmissiveMeshes,
            uint maxEmissiveTriangles,
            uint maxGeometryInstances,
            GraphicsBuffer LightDataBuffer)
        { 
            this.LightDataBuffer = LightDataBuffer;
            // m_maxEmissiveMeshes = maxEmissiveMeshes;
            // m_maxEmissiveTriangles = maxEmissiveTriangles;
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
        }
        
        
        void InitializeNeighborOffsets( uint neighborOffsetCount)
        {
            if (m_neighborOffsetsInitialized)
                return;

            byte[] offsets = new byte[neighborOffsetCount * 2]; 

            RtxdiUtils.FillNeighborOffsetBuffer(offsets, neighborOffsetCount);
            NeighborOffsetsBuffer.SetData(offsets);
            
            m_neighborOffsetsInitialized = true;
        }
    }
}