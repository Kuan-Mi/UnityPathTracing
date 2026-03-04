#include "Include/Shared.hlsl"
#include "Include/RayTracingShared.hlsl"

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
uint   _EmissiveRayCount;       // number of random probe rays for emissive in-scatter (0 = off)
float  _EmissiveIntensityScale; // manual scale for emissive contribution to fog

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

float3 UniformSphereDir(float2 u)
{
    float cosTheta = 2.0 * u.x - 1.0;
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi      = 2.0 * 3.14159265 * u.y;
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

float3 CastEmissiveRay(float3 origin, float3 direction,out float hitDist)
{
    hitDist = INF;
    GeometryProps geometryProps0;
    MaterialProps materialProps0;
    CastRay(origin, direction, 0.0, 1000.0, GetConeAngleFromRoughness(0.0, 0.0), (gOnScreen == SHOW_INSTANCE_INDEX || gOnScreen == SHOW_NORMAL) ? GEOMETRY_ALL : FLAG_NON_TRANSPARENT, geometryProps0, materialProps0);

    if (geometryProps0.IsMiss())
        return float3(0, 0, 0);
    hitDist = geometryProps0.hitT;
    return materialProps0.Lemi;
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
 
    float2 mipAndCone = float2(0, 0); // not needed for shadow ray, but must be provided for visibility query
    
    float hitT = CastVisibilityRay_AnyHit( worldPos, gSunDirection.xyz, 0.0, INF, mipAndCone, gWorldTlas,FLAG_NON_TRANSPARENT,0);

    
    // ---- Emissive in-scatter: N random sphere rays sampling nearby emissive surfaces ----
    // Uses the VolFogShadow closest-hit pass in material shaders to read emission.
    // Monte Carlo estimator: scatter += (4π / N) * Σ [ Lemi_i * phase(viewRay, emissiveDir_i) ]
    float3 emissiveScatter = 0.0;
    if (_EmissiveRayCount > 0)
    {
        float3 viewRayDir = normalize(worldPos - gCameraGlobalPos.xyz);
    
        [loop]
        for (uint i = 0; i < _EmissiveRayCount; i++)
        {
            float2 rndE       = Rng::Hash::GetFloat2();
            float3 emissiveDir = UniformSphereDir(rndE);
            
            float hitDist = 0;
            float3 Lemi        = CastEmissiveRay(worldPos, emissiveDir, hitDist);
            
            // 如果击中了发光体 (Lemi > 0) 且距离有效
            if (any(Lemi > 0.0) && hitDist < 1000.0)
            {
                // 核心修复：计算透射率 transmittance
                // Beer-Lambert: exp(-Density * Distance)
                float transmittance = exp(-_FogDensity * hitDist);
            
                float cosTheta = dot(viewRayDir, emissiveDir);
                float phase    = PhaseHG(cosTheta, _HGG);
            
                // 累加时乘上透射率
                emissiveScatter += Lemi * transmittance * phase;
            }
        }
    
        // Monte Carlo weight: integrate over the sphere (PDF = 1/4π → multiply by 4π / N)
        emissiveScatter *= (4.0 * 3.14159265) / (float)_EmissiveRayCount;
        emissiveScatter *= _FogDensity * _ScatterAlbedo * _EmissiveIntensityScale;
    }

    // Step length in metres: use fixed slice boundaries (not the jittered sample point)
    // so that the integrated extinction is independent of the jitter value.
    float viewZNear = SliceToViewZ((float)id.z);
    float viewZFar  = SliceToViewZ((float)id.z + 1.0);
    float stepM     = -(viewZFar - viewZNear) * gUnitToMetersMultiplier;

    float3 scatter = 0;
    if (hitT == INF)
    {
        // Phase function evaluated for the view ray direction
        float3 rayDir  = normalize(worldPos - gCameraGlobalPos.xyz);
        float cosTheta = dot(rayDir, gSunDirection.xyz);
        float phase    = PhaseHG(cosTheta, _HGG);
        scatter        = _SunColor.rgb * _FogDensity * _ScatterAlbedo * phase;
    }

    scatter += emissiveScatter;
    
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
