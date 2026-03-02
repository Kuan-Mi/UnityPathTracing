#include "Include/Shared.hlsl"

// TLAS for inline ray queries (space1 avoids clashing with default SRV space)
RaytracingAccelerationStructure gWorldTlas;

// ---- Output ----
RWTexture3D<float4> _FroxelVolume;   // rgb = in-scatter * step, a = extinction * step

// ---- Parameters (set per-frame from C#) ----
float  _FogDensity;       // sigma_e (extinction coefficient)
float  _ScatterAlbedo;    // scattering / extinction
float  _HGG;              // Henyey-Greenstein g  (-1 .. 1)
float  _FogFar;           // volumetric depth range (world units)
float4 _SunColor;         // main light color * intensity (linear)
uint   _SliceCount;       // number of depth slices (e.g. 64)
uint   _FroxelW;          // froxel X resolution
uint   _FroxelH;          // froxel Y resolution

// Exponential depth distribution: slice k -> view-space depth
float SliceToViewZ(float k)
{
    // gNearZ is negative, so -gNearZ is positive
    return gNearZ * pow(_FogFar / -gNearZ, k / (float)_SliceCount);
}

// Henyey-Greenstein phase function
float PhaseHG(float cosTheta, float g)
{
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

[numthreads(8, 8, 8)]
void VolumetricShadow(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= _FroxelW || id.y >= _FroxelH || id.z >= _SliceCount)
        return;

    // Reconstruct froxel sample world position.
    // gJitter.x is in [-0.5, 0.5] (TAA pixel jitter); repurposed here as a per-frame
    // temporal offset along the slice axis so adjacent frames sample at different depths
    // within the same slice — TAA then averages them out, removing Z-banding.
    float viewZ     = SliceToViewZ((float)id.z + 0.5 + gJitter.x*1000);
    float2 uv       = ((float2)id.xy + 0.5 + gJitter*1000) / float2(_FroxelW, _FroxelH);
    float3 viewPos  = Geometry::ReconstructViewPosition(uv, gCameraFrustum, viewZ, gOrthoMode);
    float3 worldPos = Geometry::AffineTransform(gViewToWorld, viewPos);

    // Step length in metres: use fixed slice boundaries (not the jittered sample point)
    // so that the integrated extinction is independent of the jitter value.
    float viewZNear = SliceToViewZ((float)id.z);
    float viewZFar  = SliceToViewZ((float)id.z + 1.0);
    float stepM     = -(viewZFar - viewZNear) * gUnitToMetersMultiplier;

    // Inline shadow ray toward the sun
    RayDesc ray;
    ray.Origin    = worldPos;
    ray.Direction = gSunDirection.xyz;
    ray.TMin      = 0.05;
    ray.TMax      = 2000.0;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
    q.TraceRayInline(gWorldTlas, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    bool visible = (q.CommittedStatus() == COMMITTED_NOTHING);

    float3 scatter = 0.0;
    if (visible)
    {
        // Phase function evaluated for the view ray direction
        float3 rayDir  = normalize(worldPos - gCameraGlobalPos.xyz);
        float cosTheta = dot(rayDir, gSunDirection.xyz);
        float phase    = PhaseHG(cosTheta, _HGG);
        scatter        = _SunColor.rgb * _FogDensity * _ScatterAlbedo * phase;
    }

    // rgb = accumulated in-scatter contribution for this step
    // a   = extinction integral for this step (Beer-Lambert: exp(-a) per slice)
    _FroxelVolume[id] = float4(scatter * stepM, _FogDensity * stepM);
}
