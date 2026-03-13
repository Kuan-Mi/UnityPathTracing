float3 evalSingleScatteringTransmission(
    GeometryProps geo,
    MaterialProps mat,
    float3 vectorToLight,
    float3 Csun,
    const RTXCR_SubsurfaceMaterialData subsurfaceMaterialData,
    const RTXCR_SubsurfaceInteraction subsurfaceInteraction)
{
    float3 radiance = 0.0;

    RTXCR_SubsurfaceMaterialCoefficients sssMaterialCoeffcients = RTXCR_ComputeSubsurfaceMaterialCoefficients(subsurfaceMaterialData);

    for (int bsdfSampleIndex = 0; bsdfSampleIndex < gSssTransmissionBsdfSampleCount; ++bsdfSampleIndex)
    {
        // Step 4.1: generate cosine-weighted refraction ray into the volume
        const float3 refractedRayDirection = RTXCR_CalculateRefractionRay(subsurfaceInteraction, Rng::Hash::GetFloat2());
        const float3 hitPos = subsurfaceInteraction.centerPosition;


        float thickness = 0.0f;
        float3 backPosition;
        float2 mipConeRefr = GetConeAngleFromRoughness(geo.mip, 0.0);

        {
            // Trace the refraction ray to find the backface exit
            // 追踪折射射线找到背面出口
            GeometryProps refrProps;
            MaterialProps refrMatProps;
            float3 transmissionRayOrigin = geo.GetXoffset(-geo.N, PT_BOUNCE_RAY_OFFSET);
            CastRay(transmissionRayOrigin, refractedRayDirection, 0.0, INF, mipConeRefr, FLAG_NON_TRANSPARENT, refrProps, refrMatProps);

            // if (refrProps.IsMiss()) {
            //     return float3(1.0, 1.0, 0.0); // 【测试2】纯黄色！如果物体变黄了，说明射线穿透了体积但没有碰到任何东西！(99%是背面剔除或者模型不是闭合的)
            // } else {
            //     return float3(0.0, 1.0, 0.0); // 纯绿色。说明射线成功找到了出口！
            // }

            if (refrProps.IsMiss() || refrProps.instanceIndex != geo.instanceIndex)
                continue;

            thickness = refrProps.hitT;
            backPosition = transmissionRayOrigin + thickness * refractedRayDirection;
            float3 backN = -refrMatProps.N;

            // Offset exit point along backface normal for shadow ray
            float3 backPositionOffset = refrProps.GetXoffset(backN, PT_BOUNCE_RAY_OFFSET);

            // Cast shadow ray from the backface exit toward the sun
            float shadowHitT = CastVisibilityRay_AnyHit(backPositionOffset, vectorToLight, 0.0, INF, GetConeAngleFromAngularRadius(refrProps.mip, gTanSunAngularRadius), gWorldTlas, FLAG_NON_TRANSPARENT, 0);

            // if (shadowHitT == INF) {
            //     return float3(0.0, 1.0, 1.0); // 【测试3】纯青色！说明光线没被遮挡，能照到背面！
            // } else {
            //     return float3(1.0, 0.0, 1.0); // 纯洋红色！说明物体的背面被其他东西（或者自身错误偏移）遮挡了，一直在阴影里。
            // }

            if (shadowHitT == INF)
            {
                float3 transmissionBsdf = RTXCR_EvaluateBoundaryTerm(mat.N, vectorToLight, refractedRayDirection, backN, thickness, sssMaterialCoeffcients);
                radiance += Csun * transmissionBsdf * RTXCR_PI;
            }
        }
        // Step 4.2: single scattering — uniform stepping along the refraction ray
        float stepSize = thickness / (gSsTransmissionPerBsdfScatteringSampleCount + 1);
        float accumulatedT = 0.0;
        float3 scatteringThroughput = 0.0;

        for (int sampleIndex = 0; sampleIndex < gSsTransmissionPerBsdfScatteringSampleCount; ++sampleIndex)
        {
            const float currentT = accumulatedT + stepSize;
            accumulatedT = currentT;

            if (currentT >= thickness)
                break;

            float3 samplePosition = hitPos + currentT * refractedRayDirection;

            // Sample a scattering direction with HG phase function
            // 采样一个散射方向，使用Henyey-Greenstein相函数
            float3 scatteringDirection = RTXCR_SampleDirectionHenyeyGreenstein(Rng::Hash::GetFloat2(), subsurfaceMaterialData.g, refractedRayDirection);

            // Trace scattering ray to find exit boundary
            // 追踪散射射线找到出口边界
            GeometryProps scatteringGeoProps;
            MaterialProps scatteringMatProps;
            CastRay(samplePosition, scatteringDirection, 0.0, INF, mipConeRefr, FLAG_NON_TRANSPARENT, scatteringGeoProps, scatteringMatProps);


            // 找到散射出口了
            if (!scatteringGeoProps.IsMiss() && scatteringGeoProps.instanceIndex == geo.instanceIndex)
            {
                // float3 scatteringBoundaryPosition = samplePosition + scatteringProps.hitT * scatteringDirection;
                float3 scatteringSampleGeometryNormal = -scatteringGeoProps.N;

                // Offset and cast shadow ray from scattering exit
                float3 scatteringBoundaryPosition = scatteringGeoProps.GetXoffset(scatteringSampleGeometryNormal, PT_SHADOW_RAY_OFFSET);
                float scatteringShadowHitT = CastVisibilityRay_AnyHit(
                    scatteringBoundaryPosition, vectorToLight, 0.0, INF,
                    GetConeAngleFromAngularRadius(scatteringGeoProps.mip, gTanSunAngularRadius),
                    gWorldTlas, FLAG_NON_TRANSPARENT, 0);


                // if (scatteringShadowHitT == INF) {
                //     return float3(0.0, 1.0, 1.0); // 【测试3】纯青色！说明光线没被遮挡，能照到背面！
                // } else {
                //     return float3(1.0, 0.0, 1.0); // 纯洋红色！说明物体的背面被其他东西（或者自身错误偏移）遮挡了，一直在阴影里。
                // }


                if (scatteringShadowHitT == INF)
                {
                    float totalScatteringDistance = currentT + scatteringGeoProps.hitT;
                    float3 ssTransmissionBsdf = RTXCR_EvaluateSingleScattering(vectorToLight, scatteringSampleGeometryNormal, totalScatteringDistance, sssMaterialCoeffcients);

                    scatteringThroughput += Csun * ssTransmissionBsdf * stepSize; // Li * BSDF / PDF
                }
            }
        }

        radiance += scatteringThroughput;
    }

    radiance /= max(gSssTransmissionBsdfSampleCount, 1);
    return radiance;
}

float3 EvaluateDirectionalLights(GeometryProps geo, MaterialProps mat, bool isSSS)
{
    // -----------------------------------------------------------------------
    // Extract materials
    // -----------------------------------------------------------------------
    float3 albedo, Rf0;
    BRDF::ConvertBaseColorMetalnessToAlbedoRf0(mat.baseColor.xyz, mat.metalness, albedo, Rf0);

    float3 Xshadow = geo.X;

    float3 Csun = GetSunIntensity(gSunDirection.xyz);
    float3 Csky = GetSkyIntensity(-geo.V);

    float NoL_geom = saturate(dot(geo.N, gSunDirection.xyz));
    float minThresh = isSSS ? gSssMinThreshold : 0.03;
    float shadow = Math::SmoothStep(minThresh, 0.1, NoL_geom);

    // Don't early-out for SSS: transmission contributes even when back-lit
    if (shadow == 0.0 && !isSSS)
        return 0.0;

    // Pseudo sky importance sampling
    float3 Cimp = lerp(Csky, Csun, Math::SmoothStep(0.0, 0.2, mat.roughness));
    Cimp *= Math::SmoothStep(-0.01, 0.05, gSunDirection.z);

    // -----------------------------------------------------------------------
    // Common BRDF
    // -----------------------------------------------------------------------
    float3 L = gSunDirection.xyz;
    float3 V = geo.V;
    float3 H = normalize(L + V);

    float NoL = saturate(dot(mat.N, L));
    float NoH = saturate(dot(mat.N, H));
    float VoH = saturate(dot(V, H));
    float NoV = abs(dot(mat.N, V));

    float D = BRDF::DistributionTerm(mat.roughness, NoH);
    float G = BRDF::GeometryTermMod(mat.roughness, NoL, NoV, VoH, NoH);
    float3 F = BRDF::FresnelTerm(Rf0, VoH);
    float Kdiff = BRDF::DiffuseTerm(mat.roughness, NoL, NoV, VoH);

    float3 Cspec = saturate(F * D * G * NoL);
    float3 Cdiff = Kdiff * Csun * albedo * NoL;

    float3 lighting = Cspec * Cimp;

    // -----------------------------------------------------------------------
    // SSS: Burley sample -> exit point -> override Cdiff + shadow origin
    // -----------------------------------------------------------------------
    GeometryProps shadowGeo = geo;
    float3 transmissionRadiance = 0.0;

#if( RTXCR_INTEGRATION == 1 )
    if (isSSS)
    {
        // 定义次表面材质
        RTXCR_SubsurfaceMaterialData subsurfaceMaterialData = (RTXCR_SubsurfaceMaterialData)0;
        subsurfaceMaterialData.transmissionColor = albedo;
        // 使用Lemi作为散射颜色只是为了测试
        subsurfaceMaterialData.scatteringColor = mat.scatteringColor;
        subsurfaceMaterialData.scale = gSssScale / gUnitToMetersMultiplier;
        subsurfaceMaterialData.g = gSssAnisotropy;

        // ---------------------------------------------------------------
        // 首先是 Evaluate Diffusion Profile
        // ---------------------------------------------------------------

        // 定义次表面交互
        float3x3 basis = Geometry::GetBasis(geo.N);
        RTXCR_SubsurfaceInteraction subsurfaceInteraction = RTXCR_CreateSubsurfaceInteraction(geo.X, basis[2], basis[0], basis[1]);

        // 生成次表面样本并评估BSSDF，（采样点和权重）
        RTXCR_SubsurfaceSample subsurfaceSample = (RTXCR_SubsurfaceSample)0;
        RTXCR_EvalBurleyDiffusionProfile(subsurfaceMaterialData,
                                         subsurfaceInteraction,
                                         gSssMaxSampleRadius / gUnitToMetersMultiplier,
                                         true,
                                         Rng::Hash::GetFloat2(),
                                         subsurfaceSample);

        // 从采样点向太阳方向投射锥形光线，找到第一个交点（如果有的话）
        float2 mipConeSSS = GetConeAngleFromRoughness(geo.mip, 0.0);
        GeometryProps sssProps;
        MaterialProps sssMaterialProps;
        CastRay(subsurfaceSample.samplePosition, -subsurfaceInteraction.normal,
                0.0, INF, mipConeSSS, FLAG_NON_TRANSPARENT, sssProps, sssMaterialProps);

        // 如果交点有效且是皮肤材质，使用这个交点来计算SSS贡献并更新阴影原点
        if (!sssProps.IsMiss() && sssProps.instanceIndex == geo.instanceIndex)
        {
            Xshadow = sssProps.X;
            shadowGeo = sssProps;
            float NoL_sss = saturate(dot(sssMaterialProps.N, L));
            Cdiff = RTXCR_EvalBssrdf(subsurfaceSample, Csun, NoL_sss);
        }

        transmissionRadiance = evalSingleScatteringTransmission(geo, mat, gSunDirection.xyz, Csun, subsurfaceMaterialData, subsurfaceInteraction);
    }
#endif

    lighting += Cdiff * (1.0 - F);
    lighting *= shadow;

    // -----------------------------------------------------------------------
    // Shadow ray: jitter within sun angular radius for soft shadows
    // (applies to surface reflection + diffusion profile, NOT transmission)
    // -----------------------------------------------------------------------
    if (Color::Luminance(lighting) != 0)
    {
        float2 rnd = Rng::Hash::GetFloat2();
        rnd = ImportanceSampling::Cosine::GetRay(rnd).xy;
        rnd *= gTanSunAngularRadius;

        float3 sunDirection = normalize(gSunBasisX.xyz * rnd.x + gSunBasisY.xyz * rnd.y + gSunDirection.xyz);
        float2 mipAndCone = GetConeAngleFromAngularRadius(shadowGeo.mip, gTanSunAngularRadius);

        Xshadow = shadowGeo.GetXoffset(sunDirection, PT_SHADOW_RAY_OFFSET);
        float hitT = CastVisibilityRay_AnyHit(Xshadow, sunDirection, 0.0, INF, mipAndCone, gWorldTlas, FLAG_NON_TRANSPARENT, 0);
        lighting *= float(hitT == INF);
    }

#if( RTXCR_INTEGRATION == 1 )
    lighting += transmissionRadiance;
#endif

    return lighting;
}
