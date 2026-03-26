#ifndef RAB_BUFFER_HLSLI
#define RAB_BUFFER_HLSLI

Texture2D<float> gOut_ViewZ;
Texture2D<float4> gOut_Normal_Roughness;
Texture2D<float4> gOut_BaseColor_Metalness;
Texture2D<uint> gOut_GeoNormal;


// RTXDI：上一帧 GBuffer
Texture2D<float> gIn_PrevViewZ;
Texture2D<float4> gIn_PrevNormalRoughness;
Texture2D<float4> gIn_PrevBaseColorMetalness;
Texture2D<uint> gIn_PrevGeoNormal;

RWTexture2D<float3> gOut_DirectLighting;

Texture2D<float4> gOut_Mv;



// RTXDI resources
StructuredBuffer<PolymorphicLightInfo> t_LightDataBuffer;
Buffer<float2> t_NeighborOffsets;
Texture2D t_LocalLightPdfTexture;
StructuredBuffer<uint> t_GeometryInstanceToLight;

// Screen-sized UAVs
RWStructuredBuffer<RTXDI_PackedDIReservoir> u_LightReservoirs;

// RTXDI UAVs
RWBuffer<uint2> u_RisBuffer;


// Other


#define RTXDI_RIS_BUFFER u_RisBuffer
#define RTXDI_LIGHT_RESERVOIR_BUFFER u_LightReservoirs
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER t_NeighborOffsets



RWStructuredBuffer<ResamplingConstants> ResampleConstants;
#define g_Const ResampleConstants[0]




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
