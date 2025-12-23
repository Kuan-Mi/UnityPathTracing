#include "ml.hlsli"
#include "NRDInclude/NRD.hlsli"
#include "RayPayload.hlsl"
#include "GlobalResource.hlsl"
#include "Shared.hlsl"

Texture2D<float3> gIn_ComposedDiff;
Texture2D<float4> gIn_ComposedSpec_ViewZ;

RWTexture2D<float3> gOut_Composed; 

[shader("raygeneration")]
void MainRayGenShader()
{
    uint2 pixelPos = DispatchRaysIndex().xy;

    float2 pixelUv = float2(pixelPos + 0.5) / gRectSize;
    float2 sampleUv = pixelUv + gJitter;

    if (pixelUv.x > 1.0 || pixelUv.y > 1.0)
    {
        return;
    }

    float3 diff = gIn_ComposedDiff[ pixelPos ];
    float3 spec = gIn_ComposedSpec_ViewZ[ pixelPos ].xyz;
    float3 Lsum = diff + spec;
    
    // Apply exposure
    Lsum = ApplyExposure( Lsum );
    
    // Output
    gOut_Composed[ pixelPos ] = Lsum;

}
