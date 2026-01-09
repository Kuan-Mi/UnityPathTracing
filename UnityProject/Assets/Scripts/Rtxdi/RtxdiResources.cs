using System;
using System.Runtime.InteropServices;
using DefaultNamespace;
using UnityEngine;

public class RtxdiResources : IDisposable
{
    // RTXDI 常量，通常为 2 或 3，取决于 SDK 版本
    private const int c_NumReSTIRDIReservoirBuffers = 3; 

    private bool m_neighborOffsetsInitialized = false;
    private uint m_maxEmissiveMeshes;
    private uint m_maxEmissiveTriangles;
    private uint m_maxGeometryInstances;

    // Public Buffers
    public GraphicsBuffer TaskBuffer { get; private set; }
    public GraphicsBuffer LightDataBuffer { get; private set; }
    public GraphicsBuffer GeometryInstanceToLightBuffer { get; private set; }
    public GraphicsBuffer NeighborOffsetsBuffer { get; private set; }
    public GraphicsBuffer LightReservoirBuffer { get; private set; }

    public unsafe RtxdiResources(
        ReSTIRDIContext context,
        uint maxEmissiveMeshes,
        uint maxEmissiveTriangles,
        uint maxGeometryInstances)
    {
        m_maxEmissiveMeshes = maxEmissiveMeshes;
        m_maxEmissiveTriangles = maxEmissiveTriangles;
        m_maxGeometryInstances = maxGeometryInstances;

        // 1. TaskBuffer
        // initial state: ShaderResource, canHaveUAVs = true
        if (maxEmissiveMeshes > 0)
        {
            TaskBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                (int)maxEmissiveMeshes,
                Marshal.SizeOf<PrepareLightsTask>()
            );
            TaskBuffer.name = "TaskBuffer";
        }

        // 2. LightDataBuffer
        // initial state: ShaderResource, canHaveUAVs = true
        if (maxEmissiveTriangles > 0)
        {
            LightDataBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                (int)maxEmissiveTriangles,
                Marshal.SizeOf<RAB_LightInfo>()
            );
            LightDataBuffer.name = "LightDataBuffer";
        }

        // 3. GeometryInstanceToLightBuffer
        // initial state: ShaderResource
        if (maxGeometryInstances > 0)
        {
            GeometryInstanceToLightBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                (int)maxGeometryInstances,
                sizeof(uint)
            );
            GeometryInstanceToLightBuffer.name = "GeometryInstanceToLightBuffer";
        }

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
        uint neighborBufferSize = staticParams->NeighborOffsetCount * 2;
        // 向上取整到 4 字节对齐，因为 Raw buffer 寻址通常是 4 字节
        int alignedNeighborSize = (int)((neighborBufferSize + 3) & ~3);
        
        NeighborOffsetsBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Raw, 
            alignedNeighborSize / 4, // Count for Raw is in ints (4 bytes)
            4 
        );
        NeighborOffsetsBuffer.name = "NeighborOffsets";

        // 5. LightReservoirBuffer
        // byteSize = sizeof(Packed) * pitch * numBuffers
        int reservoirStride = Marshal.SizeOf<RTXDI_PackedDIReservoir>();
        int totalReservoirs = (int)reservoirParams.reservoirArrayPitch * c_NumReSTIRDIReservoirBuffers;
        
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

    public void InitializeNeighborOffsets(uint neighborOffsetCount)
    {
        if (m_neighborOffsetsInitialized)
            return;

        // C++: std::vector<uint8_t> offsets(neighborOffsetCount * 2)
        int sizeInBytes = (int)neighborOffsetCount * 2;
        byte[] offsets = new byte[sizeInBytes];

        // 调用 Native 填充数据
        unsafe
        {
            fixed (byte* ptr = offsets)
            {
                RtxdiNative.FillNeighborOffsetBuffer((IntPtr)ptr, neighborOffsetCount);
            }
        }

        // 上传到 GPU
        // 注意：因为我们用的是 Raw buffer (stride=4)，SetData 传入 byte[] 会自动处理
        NeighborOffsetsBuffer.SetData(offsets);

        m_neighborOffsetsInitialized = true;
    }

    public uint GetMaxEmissiveMeshes() => m_maxEmissiveMeshes;
    public uint GetMaxEmissiveTriangles() => m_maxEmissiveTriangles;
    public uint GetMaxGeometryInstances() => m_maxGeometryInstances;

    public void Dispose()
    {
        TaskBuffer?.Dispose();
        TaskBuffer = null;

        LightDataBuffer?.Dispose();
        LightDataBuffer = null;

        GeometryInstanceToLightBuffer?.Dispose();
        GeometryInstanceToLightBuffer = null;

        NeighborOffsetsBuffer?.Dispose();
        NeighborOffsetsBuffer = null;

        LightReservoirBuffer?.Dispose();
        LightReservoirBuffer = null;
    }
}