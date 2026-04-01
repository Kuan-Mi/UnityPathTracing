// ──────────────────────────────────────────────────────────────────────────────
// GBufferRaster.hlsl
//
// Rasterization-based G-Buffer fill pass.
// Writes the identical output layout as the ray-tracing GBuffer.hlsl so that
// the downstream NRD / path-tracing passes are unaffected.
//
// Uses Multiple Render Targets (MRT); output layout (matches GBuffer.hlsl RT):
//   SV_Target0  ViewDepth      R32_Float    – linear view-space Z
//   SV_Target1  DiffuseAlbedo  R32_UInt     – Pack_R11G11B10_UFLOAT(albedo)
//   SV_Target2  SpecularRough  R32_UInt     – Pack_R8G8B8A8_Gamma_UFLOAT(Rf0, roughness)
//   SV_Target3  Normals        R32_UInt     – ndirToOctUnorm32(material normal)
//   SV_Target4  GeoNormals     R32_UInt     – ndirToOctUnorm32(geometry normal)
//   SV_Target5  Emissive       R16G16B16A16 – emissive.rgb, viewZAndTaaMask
//   SV_Target6  MotionVectors  R16G16B16A16 – motion.xy + delta-viewZ, viewZAndTaaMask
//
// This file is #included from the "GBufferRaster" HLSLPROGRAM in Lit.shader
// after the UnityPerMaterial cbuffer and texture declarations.
// ──────────────────────────────────────────────────────────────────────────────

#include "Assets/Shaders/Include/Shared.hlsl"
#include "Assets/Shaders/donut/utils.hlsli"
#include "Assets/Shaders/donut/packing.hlsli"

// ── Fragment output struct (MRT) ──────────────────────────────────────────────
struct GBR_FragOutput
{
    float viewDepth : SV_Target0;
    uint diffuseAlbedo : SV_Target1;
    uint specularRough : SV_Target2;
    uint normals : SV_Target3;
    uint geoNormals : SV_Target4;
    float4 emissive : SV_Target5;
    float4 motionVectors : SV_Target6;
};

// ── Vertex input / output ─────────────────────────────────────────────────────
struct GBR_Attributes
{
    float4 positionOS : POSITION;
    float3 normalOS : NORMAL0;
    float4 tangentOS : TANGENT;
    float2 uv : TEXCOORD0;
};

struct GBR_Varyings
{
    float4 positionCS : SV_POSITION;
    float3 positionWS : TEXCOORD0;
    float3 prevPositionWS : TEXCOORD1;
    float3 normalWS : TEXCOORD2;
    float4 tangentWS : TEXCOORD3; // xyz = world tangent, w = bitangent sign
    float2 uv : TEXCOORD4;
};

// ── Vertex shader ─────────────────────────────────────────────────────────────
GBR_Varyings GBufferRasterVert(GBR_Attributes IN)
{
    GBR_Varyings OUT;

    float3 posOS = IN.positionOS.xyz;

    OUT.positionWS = mul(GetObjectToWorldMatrix(), float4(posOS, 1.0)).xyz;
    OUT.positionCS = TransformWorldToHClip(OUT.positionWS);

    // Previous-frame world position for motion vectors.
    // GetPrevObjectToWorldMatrix() returns unity_MatrixPreviousM, which Unity
    // populates automatically when DepthTextureMode.MotionVectors is enabled.
    // For skinned meshes a dedicated previous-frame vertex stream would give
    // exact motion vectors; for rigid bodies this is exact.
    OUT.prevPositionWS = mul(GetPrevObjectToWorldMatrix(), float4(posOS, 1.0)).xyz;

    OUT.normalWS = TransformObjectToWorldNormal(IN.normalOS);
    OUT.tangentWS = float4(TransformObjectToWorldDir(IN.tangentOS.xyz), IN.tangentOS.w);
    OUT.uv = TRANSFORM_TEX(IN.uv, _BaseMap);

    // Flip clip-space Y to produce a vertically-inverted image.
    OUT.positionCS.y = -OUT.positionCS.y;

    // Apply TAA jitter – matches GBuffer.hlsl: sampleUv = pixelUv + gJitter.
    // gJitter is in UV space; after the Y-flip above, clip-space Y and UV Y
    // share the same +Y-downward convention, so we scale uniformly by 2
    // (UV [0,1] → NDC [-1,1]) and multiply by w to stay in clip space.
    
    OUT.positionCS.xy += (float2(-gJitter.x,gJitter.y)) *2 * OUT.positionCS.w;

    return OUT;
}

// ── Fragment shader ───────────────────────────────────────────────────────────
GBR_FragOutput GBufferRasterFrag(GBR_Varyings IN)
{
    float2 uv = IN.uv;

    // ── Albedo ────────────────────────────────────────────────────────────────
    float4 baseMapSample = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, uv);
    float3 albedoRaw = _BaseColor.rgb * baseMapSample.rgb;

    // ── Alpha test ────────────────────────────────────────────────────────────
    #if _ALPHATEST_ON
    clip(baseMapSample.a * _BaseColor.a - _Cutoff);
    #endif

    // ── Roughness / Metallic ──────────────────────────────────────────────────
    float roughness, metallic;
    #if _METALLICSPECGLOSSMAP
    float4 metallicSample = SAMPLE_TEXTURE2D(_MetallicGlossMap, sampler_MetallicGlossMap, uv);
    // Bistro channel layout: roughness in G (inverted smoothness), metallic in B.
    // Standard URP layout (commented): smoothness in A, metallic in R.
    float smooth = (1.0 - metallicSample.g) * _Smoothness;
    roughness = 1.0 - smooth;
    metallic = metallicSample.b;
    #else
    roughness = 1.0 - _Smoothness;
    metallic = _Metallic;
    #endif

    // ── Geometry normal (vertex normal, world space) ──────────────────────────
    float3 geoNormalWS = normalize(IN.normalWS);

    // ── Material normal (with optional normal map) ────────────────────────────
    float3 matNormalWS;
    #if _NORMALMAP
    float3 tangentWS = normalize(IN.tangentWS.xyz);
    float3 bitangentWS = normalize(cross(geoNormalWS, tangentWS)) * IN.tangentWS.w;
    float3x3 tbn = float3x3(tangentWS, bitangentWS, geoNormalWS);
    float4 nSample = SAMPLE_TEXTURE2D(_BumpMap, sampler_BumpMap, uv);
    float3 tangentN = UnpackNormalScale(nSample, _BumpScale);
    matNormalWS = normalize(mul(tangentN, tbn));
    #else
    matNormalWS = geoNormalWS;
    #endif

    // ── Emissive ──────────────────────────────────────────────────────────────
    float3 emissive = float3(0.0, 0.0, 0.0);
    #if _EMISSION
    emissive = _EmissionColor.rgb
        * SAMPLE_TEXTURE2D(_EmissionMap, sampler_EmissionMap, uv).rgb;
    #endif

    // ── BRDF conversion ───────────────────────────────────────────────────────
    // Matches GBuffer.hlsl: BRDF::ConvertBaseColorMetalnessToAlbedoRf0
    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0(albedoRaw, metallic, albedo, Rf0);

    // ── View-space depth ──────────────────────────────────────────────────────
    float3 X = IN.positionWS;
    float3 Xprev = IN.prevPositionWS;
    float viewZ = Geometry::AffineTransform(gWorldToView, X).z;


    // ── Motion vectors ────────────────────────────────────────────────────────
    // GetMotion() replicates the RT version exactly:
    //   motion.xy = (prevScreenUV - curScreenUV) * gRectSize   [pixel units]
    //   motion.z  = prevViewZ - curViewZ                        [view-space delta]
    float3 motion = GetMotion(X, Xprev);

    // ── MRT output ────────────────────────────────────────────────────────────
    GBR_FragOutput o;
    o.viewDepth = viewZ;
    o.diffuseAlbedo = Pack_R11G11B10_UFLOAT(albedo);
    o.specularRough = Pack_R8G8B8A8_Gamma_UFLOAT(float4(Rf0, roughness));
    o.normals = ndirToOctUnorm32(matNormalWS);
    o.geoNormals = ndirToOctUnorm32(geoNormalWS);
    o.emissive = float4(emissive, 1);
    o.motionVectors = float4(motion, 1);
    // o.motionVectors = 0;
    return o;
}
