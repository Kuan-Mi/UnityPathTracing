using System.Runtime.InteropServices;
using Rtxdi;
using RTXDI;
using Rtxdi.DI;
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
        public GraphicsBuffer NeighborOffsetsBuffer { get; private set; }
        public GraphicsBuffer LightReservoirBuffer { get; private set; }


        public int LightDataBufferSize;
        public int NeighborOffsetsBufferSize;
        public int LightReservoirBufferSize;
        

        public RtxdiResources(
            ReSTIRDIContext context,
            uint maxEmissiveMeshes,
            uint maxEmissiveTriangles,
            uint maxGeometryInstances,
            GraphicsBuffer LightDataBuffer)
        {
            this.LightDataBuffer = LightDataBuffer;
            LightDataBufferSize = (int)(maxEmissiveTriangles * Marshal.SizeOf<RAB_LightInfo>());
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
            // C++: format = nvrhi::Format::RG8_SNORM (2 bytes per element)
            // Unity 处理 Typed Buffer (Buffer<float2>) 比较麻烦，通常使用 Raw 缓冲区或 Texture1D。
            // 这里使用 Target.Raw (ByteAddressBuffer)，在 Shader 中需要手动解包，
            // 或者如果 Shader 只是将其视为 uint/short 数组，可以使用 Structured。
            // 为通用起见，这里使用 Raw，因为 byteSize 可能不是 stride 的整数倍。
            // 大小 = NeighborOffsetCount * 2 bytes
            uint neighborBufferSize = staticParams.NeighborOffsetCount * 2;
            
            Debug.Log($"Creating NeighborOffsetsBuffer with neighborBufferSize: {neighborBufferSize} bytes for NeighborOffsetCount: {staticParams.NeighborOffsetCount}");
            
            // 向上取整到 4 字节对齐，因为 Raw buffer 寻址通常是 4 字节
            int alignedNeighborSize = (int)((neighborBufferSize + 3) & ~3);
            
            Debug.Log($"Aligned NeighborOffsetsBuffer size: {alignedNeighborSize} bytes");

            NeighborOffsetsBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Raw,
                alignedNeighborSize / 4, // Count for Raw is in ints (4 bytes)
                4
            );
            NeighborOffsetsBuffer.name = "NeighborOffsets";
            NeighborOffsetsBufferSize = alignedNeighborSize;
            
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
                LightReservoirBufferSize = totalReservoirs * reservoirStride;
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