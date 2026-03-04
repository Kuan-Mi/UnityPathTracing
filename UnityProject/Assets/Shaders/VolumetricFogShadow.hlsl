#include "Include/Shared.hlsl"

// TLAS for inline ray queries (space1 avoids clashing with default SRV space)
RaytracingAccelerationStructure gWorldTlas;

// ---- Output ----
RWTexture3D<float4> _FroxelVolume;   // rgb = in-scatter * step, a = extinction * step

// ---- Temporal history (previous frame, read-only) ----
Texture3D<float4>   _FroxelVolumeHistory;
SamplerState        sampler_FroxelVolumeHistory;  // Unity auto-binds (bilinear clamp)

// ---- Parameters (set per-frame from C#) ----
float  _FogDensity;       // sigma_e (extinction coefficient)
float  _ScatterAlbedo;    // scattering / extinction
float  _HGG;              // Henyey-Greenstein g  (-1 .. 1)
float  _FogFar;           // volumetric depth range (world units)
float4 _SunColor;         // main light color * intensity (linear)
uint   _SliceCount;       // number of depth slices (e.g. 64)
uint   _FroxelW;          // froxel X resolution
uint   _FroxelH;          // froxel Y resolution
float  _TemporalBlend;    // blend weight for current frame (0 = all history, 1 = all current)

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

StructuredBuffer<uint4> gIn_ScramblingRanking;
StructuredBuffer<uint4> gIn_Sobol;

float2 GetBlueNoise(uint2 pixelPos, uint seed = 0)
{
    // 缓存效率低 多0.2ms
    // return Rng::Hash::GetFloat2();
    // https://eheitzresearch.wordpress.com/772-2/
    // https://belcour.github.io/blog/research/publication/2019/06/17/sampling-bluenoise.html

    // Sample index
    uint sampleIndex = (gFrameIndex + seed) & (BLUE_NOISE_TEMPORAL_DIM - 1);

    // sampleIndex = 3;
    // pixelPos /= 8;

    uint2 uv = pixelPos & (BLUE_NOISE_SPATIAL_DIM - 1);
    uint index = uv.x + uv.y * BLUE_NOISE_SPATIAL_DIM;
    uint3 A = gIn_ScramblingRanking[index].xyz;

    // return float2(A.x/256.0 , A.y / 256.0);
    uint rankedSampleIndex = sampleIndex ^ A.z;
    // return float2(rankedSampleIndex / float(BLUE_NOISE_TEMPORAL_DIM), 0);

    uint4 B = gIn_Sobol[rankedSampleIndex & 255];
    float4 blue = (float4(B ^ A.xyxy) + 0.5) * (1.0 / 256.0);

    // ( Optional ) Randomize in [ 0; 1 / 256 ] area to get rid of possible banding
    uint d = Sequence::Bayer4x4ui(pixelPos, gFrameIndex);
    float2 dither = (float2(d & 3, d >> 2) + 0.5) * (1.0 / 4.0);
    blue += (dither.xyxy - 0.5) * (1.0 / 256.0);

    return saturate(blue.xy);
}

// ---- Shared per-froxel evaluation logic (used by both compute and raytrace entries) ----
void EvaluateFroxel(uint3 id, bool visible)
{
    float2 temporalJitter = GetBlueNoise(id.xy, 12345); // TODO: add UI to disable blue noise and use pure RNG instead

    float viewZ     = SliceToViewZ((float)id.z + temporalJitter.x);
    float2 uv       = ((float2)id.xy + temporalJitter) / float2(_FroxelW, _FroxelH);
    float3 viewPos  = Geometry::ReconstructViewPosition(uv, gCameraFrustum, viewZ, gOrthoMode);
    float3 worldPos = Geometry::AffineTransform(gViewToWorld, viewPos);

    // Step length in metres: use fixed slice boundaries (not the jittered sample point)
    // so that the integrated extinction is independent of the jitter value.
    float viewZNear = SliceToViewZ((float)id.z);
    float viewZFar  = SliceToViewZ((float)id.z + 1.0);
    float stepM     = -(viewZFar - viewZNear) * gUnitToMetersMultiplier;

    float3 scatter = 0.001;
    if (visible)
    {
        // Phase function evaluated for the view ray direction
        float3 rayDir  = normalize(worldPos - gCameraGlobalPos.xyz);
        float cosTheta = dot(rayDir, gSunDirection.xyz);
        float phase    = PhaseHG(cosTheta, _HGG);
        scatter        = _SunColor.rgb * _FogDensity * _ScatterAlbedo * phase;
    }

    float4 current = float4(scatter * stepM, _FogDensity * stepM);

    // ---- Temporal accumulation ----
    // Reproject this voxel's world position into the previous frame's froxel volume.
    float2 prevScreenUV = Geometry::GetScreenUv(gWorldToClipPrev, worldPos);
    float  prevViewZ    = Geometry::AffineTransform(gWorldToViewPrev, worldPos).z;
    // Invert SliceToViewZ: sliceUV = log(viewZ/gNearZ) / log(_FogFar / -gNearZ)  (mirrors VolumetricIntegrate.compute)
    float  prevSliceUV  = saturate(log(prevViewZ / gNearZ) / log(_FogFar / -gNearZ));
    float3 prevUVW      = float3(prevScreenUV, prevSliceUV);
    bool   prevValid    = all(prevUVW >= 0.0) && all(prevUVW <= 1.0);

    float4 history = _FroxelVolumeHistory.SampleLevel(sampler_FroxelVolumeHistory, prevUVW, 0);
    float  blend   = prevValid ? _TemporalBlend : 1.0;
    // rgb = accumulated in-scatter contribution for this step
    // a   = extinction integral for this step (Beer-Lambert: exp(-a) per slice)
    _FroxelVolume[id] = lerp(history, current, blend);
}

// ---- Ray Tracing entry points (VolumetricFogShadow.raytrace) ----
#ifdef VOLUMETRIC_FOG_RAYTRACE

struct VolShadowPayload
{
    bool visible;
};

// Miss shader: ray reached TMax without hitting any geometry → sun is visible
[shader("miss")]
void VolumetricShadowMiss(inout VolShadowPayload payload)
{
    payload.visible = true;
}

// Ray generation shader: one invocation per froxel voxel (dispatched as W x H x SliceCount)
[shader("raygeneration")]
void VolumetricShadowRayGen()
{
    uint3 id = DispatchRaysIndex().xyz;
    if (id.x >= _FroxelW || id.y >= _FroxelH || id.z >= _SliceCount)
        return;

    Rng::Hash::Initialize(id.xy, gFrameIndex);

    // Reconstruct world-space voxel centre for ray origin
    float2 temporalJitter = GetBlueNoise(id.xy, 12345);
    float viewZ    = SliceToViewZ((float)id.z + temporalJitter.x);
    float2 uv      = ((float2)id.xy + temporalJitter) / float2(_FroxelW, _FroxelH);
    float3 viewPos = Geometry::ReconstructViewPosition(uv, gCameraFrustum, viewZ, gOrthoMode);
    float3 worldPos = Geometry::AffineTransform(gViewToWorld, viewPos);

    // Shadow ray toward the sun using the full DXR pipeline
    RayDesc ray;
    ray.Origin    = worldPos;
    ray.Direction = gSunDirection.xyz;
    ray.TMin      = 0.05;
    ray.TMax      = 2000.0;

    VolShadowPayload payload;
    payload.visible = false;

    // RAY_FLAG_SKIP_CLOSEST_HIT_SHADER: no closest-hit execution needed, only miss matters
    // RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH: terminate on first opaque hit for performance
    TraceRay(gWorldTlas,
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             0xFF, 0, 1, 0, ray, payload);

    EvaluateFroxel(id, payload.visible);
}

#endif // VOLUMETRIC_FOG_RAYTRACE
