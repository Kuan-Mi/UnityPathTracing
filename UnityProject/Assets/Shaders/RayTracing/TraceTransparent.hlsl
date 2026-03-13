#include "Assets/Shaders/Include/ml.hlsli"
#include "Assets/Shaders/NRD/NRD.hlsli"
#include "Assets/Shaders/Include/Shared.hlsl"
#include "Assets/Shaders/Include/RayTracingShared.hlsl"

Texture2D<float3> gIn_ComposedDiff;
Texture2D<float4> gIn_ComposedSpec_ViewZ;

RWTexture2D<float4> gOut_Normal_Roughness;
RWTexture2D<float3> gOut_Composed;
RWTexture2D<float4> gInOut_Mv;

struct TraceTransparentDesc
{
    // Geometry properties
    GeometryProps geometryProps;

    // Pixel position
    uint2 pixelPos;

    // Is reflection or refraction in first segment?
    bool isReflection;
};

float3 TraceTransparent(TraceTransparentDesc desc)
{
    float eta = BRDF::IOR::Air / BRDF::IOR::Glass;

    GeometryProps geometryProps = desc.geometryProps;
    float pathThroughput = 1.0;
    bool isReflection = desc.isReflection;
    float bayer = Sequence::Bayer4x4(desc.pixelPos, gFrameIndex);

    MaterialProps materialProps;

    [loop]
    for (uint bounce = 1; bounce <= PT_DELTA_BOUNCES_NUM; bounce++)
    {
        // Reflection or refraction?
        float NoV = abs(dot(geometryProps.N, geometryProps.V));
        float F = BRDF::FresnelTerm_Dielectric(eta, NoV);

        // 第一次时，固定为反射或折射，之后根据概率随机
        if (bounce == 1)
            pathThroughput *= isReflection ? F : 1.0 - F;
        // else if (bounce == 2)
        // {
        //     isReflection = !isReflection; // TODO: verify corner cases
        //     pathThroughput *= isReflection ? F : 1.0 - F;
        // }
        else
        {
            float rnd = frac(bayer + Sequence::Halton(bounce, 3)); // "Halton( bounce, 2 )" works worse than others

            [flatten]
            if( gDenoiserType == DENOISER_REFERENCE || gRR )
                rnd = Rng::Hash::GetFloat( );
            else
                F = clamp( F, PT_GLASS_MIN_F, 1.0 - PT_GLASS_MIN_F ); // TODO: needed?

            isReflection = rnd < F; // TODO: if "F" is clamped, "pathThroughput" should be adjusted too
        }
        
        uint flags = bounce == PT_DELTA_BOUNCES_NUM ? FLAG_NON_TRANSPARENT : GEOMETRY_ALL;


        float3 Xoffset, ray;
        eta = GetDeltaEventRay(geometryProps, isReflection, eta, Xoffset, ray);

        CastRay(Xoffset, ray, 0.0, INF, GetConeAngleFromRoughness(geometryProps.mip, 0.0), flags, geometryProps, materialProps);


        bool isAir = eta < 1.0;
        float extinction = isAir ? 0.0 : 1.0; // TODO: tint color?
        if (!geometryProps.IsMiss()) // TODO: fix for non-convex geometry
            pathThroughput *= exp(-extinction * geometryProps.hitT * gUnitToMetersMultiplier);

        // Is opaque hit found?
        if (!geometryProps.Has(FLAG_TRANSPARENT)) // TODO: stop if pathThroughput is low
            break;
    }


    float4 Lcached = float4(materialProps.Lemi, 0.0);
    if (!geometryProps.IsMiss())
    {
        // L1 cache - reproject previous frame, carefully treating specular
        float3 prevLdiff, prevLspec;
        float reprojectionWeight = ReprojectIrradiance(false, !isReflection, gIn_ComposedDiff, gIn_ComposedSpec_ViewZ, geometryProps, desc.pixelPos, prevLdiff, prevLspec);
        // reprojectionWeight = 0;
        Lcached = float4(prevLdiff + prevLspec, reprojectionWeight);
        // Lcached = float4(reprojectionWeight,reprojectionWeight,reprojectionWeight, 1);

        // L2 cache - SHARC
        HashGridParameters hashGridParams;
        hashGridParams.cameraPosition = gCameraGlobalPos.xyz;
        hashGridParams.sceneScale = SHARC_SCENE_SCALE;
        hashGridParams.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
        hashGridParams.levelBias = SHARC_GRID_LEVEL_BIAS;

        float3 Xglobal = GetGlobalPos( geometryProps.X );
        uint level = HashGridGetLevel( Xglobal, hashGridParams );
        float voxelSize = HashGridGetVoxelSize( level, hashGridParams );

        float2 rndScaled = ImportanceSampling::Cosine::GetRay( Rng::Hash::GetFloat2( ) ).xy;
        rndScaled *= voxelSize;
        rndScaled *= USE_SHARC_DITHERING * float( USE_SHARC_DEBUG == 0 );

        float3x3 mBasis = Geometry::GetBasis( geometryProps.N );
        Xglobal += mBasis[ 0 ] * rndScaled.x + mBasis[ 1 ] * rndScaled.y;

        SharcHitData sharcHitData;
        sharcHitData.positionWorld = Xglobal;
        sharcHitData.materialDemodulation = GetMaterialDemodulation( geometryProps, materialProps );
        sharcHitData.normalWorld = geometryProps.N;
        sharcHitData.emissive = materialProps.Lemi;

        HashMapData hashMapData;
        hashMapData.capacity = SHARC_CAPACITY;
        hashMapData.hashEntriesBuffer = gInOut_SharcHashEntriesBuffer;

        SharcParameters sharcParams;
        sharcParams.gridParameters = hashGridParams;
        sharcParams.hashMapData = hashMapData;
        sharcParams.radianceScale = SHARC_RADIANCE_SCALE;
        sharcParams.enableAntiFireflyFilter = SHARC_ANTI_FIREFLY;
        sharcParams.accumulationBuffer = gInOut_SharcAccumulated;
        sharcParams.resolvedBuffer = gInOut_SharcResolved;

        bool isSharcAllowed = Rng::Hash::GetFloat( ) > Lcached.w; // is needed?
        isSharcAllowed &= gSHARC && NRD_MODE < OCCLUSION; // trivial

        // isSharcAllowed = false;
        float3 sharcRadiance;
        if (isSharcAllowed && SharcGetCachedRadiance( sharcParams, sharcHitData, sharcRadiance, false))
        {
            Lcached = float4( sharcRadiance, 1.0 );
            // Lcached = float4( 0,1,0, 1.0 );
            
        }

        // Cache miss - compute lighting, if not found in caches
        if (Rng::Hash::GetFloat() > Lcached.w)
        {
            float3 L = GetLighting(geometryProps, materialProps, LIGHTING | SHADOW) + materialProps.Lemi;
            Lcached.xyz = max(Lcached.xyz, L);
            
            // Lcached.xyz = float3(1,0,1);
        }
    }

    // Output
    return Lcached.xyz * pathThroughput;
}

//========================================================================================
// MAIN
//========================================================================================

[shader("raygeneration")]
void MainRayGenShader()
{
    uint2 pixelPos = DispatchRaysIndex().xy;

    float2 pixelUv = float2(pixelPos + 0.5) * gInvRectSize;
    float2 sampleUv = pixelUv + gJitter;

    // Do not generate NANs for unused threads
    if (pixelUv.x > 1.0 || pixelUv.y > 1.0)
        return;

    Rng::Hash::Initialize(pixelPos, gFrameIndex);

    float3 diff = gIn_ComposedDiff[pixelPos];
    float3 spec = gIn_ComposedSpec_ViewZ[pixelPos].xyz;
    float3 Lsum = diff + spec * float(gOnScreen == SHOW_FINAL);

    // Primary ray for transparent geometry only
    float3 cameraRayOrigin = (float3)0;
    float3 cameraRayDirection = (float3)0;
    GetCameraRay(cameraRayOrigin, cameraRayDirection, sampleUv);

    float viewZAndTaaMask = gInOut_Mv[pixelPos].w;
    float viewZ = Math::Sign(gNearZ) * abs(viewZAndTaaMask) / FP16_VIEWZ_SCALE; // viewZ before PSR
    float3 Xv = Geometry::ReconstructViewPosition(sampleUv, gCameraFrustum, viewZ, gOrthoMode);
    float tOpaque = gOrthoMode == 0 ? length(Xv) : abs(Xv.z);

    // 减去 ray origin（近平面）到相机原点的距离
    float3 XvNear = Geometry::ReconstructViewPosition(sampleUv, gCameraFrustum, abs(gNearZ), gOrthoMode);
    float tNear = gOrthoMode == 0 ? length(XvNear) : abs(gNearZ);

    float tmin0 = max(0.0, tOpaque - tNear); // ← 修正后的 tmax

    GeometryProps geometryPropsT;
    MaterialProps materialPropsT;

    CastRay(cameraRayOrigin, cameraRayDirection, 0.0, tmin0, GetConeAngleFromRoughness(0.0, 0.0), FLAG_TRANSPARENT  , geometryPropsT, materialPropsT);

    // Trace delta events
    if (!geometryPropsT.IsMiss() && geometryPropsT.hitT < tmin0 && gOnScreen == SHOW_FINAL)
    {
        // Append "glass" mask to "hair" mask
        viewZAndTaaMask = -abs(viewZAndTaaMask);

        float3 mvT = GetMotion(geometryPropsT.X, geometryPropsT.Xprev);
        // gInOut_Mv[pixelPos] = float4(mvT, viewZAndTaaMask);


        // Patch guides for RR
        [branch]
        if( gRR )
            gOut_Normal_Roughness[ pixelPos ] = NRD_FrontEnd_PackNormalAndRoughness( geometryPropsT.N, 0.0, 0 );

        TraceTransparentDesc desc;
        desc.geometryProps = geometryPropsT;
        desc.pixelPos = pixelPos;

        desc.isReflection = true;
        float3 reflection = TraceTransparent(desc);
        Lsum = reflection;

        desc.isReflection = false;
        float3 refraction = TraceTransparent(desc);
        Lsum += refraction;
    }

    // Apply exposure
    Lsum = ApplyExposure(Lsum);

    // Output
    gOut_Composed[pixelPos] = Lsum;
}
