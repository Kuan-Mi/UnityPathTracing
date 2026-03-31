#ifndef RAB_BUFFER_HLSLI
#define RAB_BUFFER_HLSLI

// G-buffer resources
Texture2D<float> t_GBufferDepth;
Texture2D<uint> t_GBufferNormals;
Texture2D<uint> t_GBufferGeoNormals;
Texture2D<uint> t_GBufferDiffuseAlbedo;
Texture2D<uint> t_GBufferSpecularRough;

Texture2D<float> t_PrevGBufferDepth;
Texture2D<uint> t_PrevGBufferNormals;
Texture2D<uint> t_PrevGBufferGeoNormals;
Texture2D<uint> t_PrevGBufferDiffuseAlbedo;
Texture2D<uint> t_PrevGBufferSpecularRough;
Texture2D<float2> t_PrevRestirLuminance;
Texture2D<float4> t_MotionVectors;
Texture2D<float4> t_DenoiserNormalRoughness;

RWTexture2D<float3> gOut_DirectLighting;

// RTXDI resources
StructuredBuffer<PolymorphicLightInfo> t_LightDataBuffer;
Buffer<float2> t_NeighborOffsets;
// Buffer<uint> t_LightIndexMappingBuffer;
// Texture2D t_EnvironmentPdfTexture;
Texture2D t_LocalLightPdfTexture;
StructuredBuffer<uint> t_GeometryInstanceToLight;

// Screen-sized UAVs
RWStructuredBuffer<RTXDI_PackedDIReservoir> u_LightReservoirs;
// RWTexture2D<float4> u_DiffuseLighting;
// RWTexture2D<float4> u_SpecularLighting;
// RWTexture2D<int2> u_TemporalSamplePositions;
// RWTexture2DArray<float4> u_Gradients;
// RWTexture2D<float2> u_RestirLuminance;
// RWStructuredBuffer<RTXDI_PackedGIReservoir> u_GIReservoirs;

// RTXDI UAVs
RWBuffer<uint2> u_RisBuffer;
RWBuffer<uint4> u_RisLightDataBuffer;

// Other
// ConstantBuffer<ResamplingConstants> g_Const;

#ifdef USE_RAY_QUERY
ConstantBuffer<ResamplingConstants> g_Const;
#else 
RWStructuredBuffer<ResamplingConstants> ResampleConstants;
#define g_Const ResampleConstants[0]
#endif

#define RTXDI_RIS_BUFFER u_RisBuffer
#define RTXDI_LIGHT_RESERVOIR_BUFFER u_LightReservoirs
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER t_NeighborOffsets


// Translate the light index between the current and previous frame.
// Do nothing as our lights are static in this sample.

// 不实现，因为我们在这个示例中使用的是静态光源。

// 将当前帧的光源索引转换到上一帧（如果 currentToPrevious 为 true ），或将上一帧的光源索引转换到当前帧（如果 currentToPrevious 为 false ）。
// 返回新的索引，如果光源在另一帧中不存在，则返回负数。
int RAB_TranslateLightIndex(uint lightIndex, bool currentToPrevious)
{
    return int(lightIndex);
}

#endif // RAB_BUFFER_HLSLI
