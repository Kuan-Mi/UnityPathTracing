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
        RTXCR_SubsurfaceMaterialData sssMat = (RTXCR_SubsurfaceMaterialData)0;
        sssMat.transmissionColor = albedo;
        sssMat.scatteringColor = gSssScatteringColor;
        sssMat.scale = gSssScale / gUnitToMetersMultiplier;
        sssMat.g = 0.0;

        float3 Xoff = geo.GetXoffset(geo.N, PT_SHADOW_RAY_OFFSET);
        float3x3 basis = Geometry::GetBasis(geo.N);
        RTXCR_SubsurfaceInteraction sssInteraction =
            RTXCR_CreateSubsurfaceInteraction(Xoff, basis[2], basis[0], basis[1]);

        RTXCR_SubsurfaceSample sssSample = (RTXCR_SubsurfaceSample)0;
        RTXCR_EvalBurleyDiffusionProfile(sssMat, sssInteraction,
                                         gSssMaxSampleRadius / gUnitToMetersMultiplier, true, Rng::Hash::GetFloat2(), sssSample);

        float2 mipConeSSS = GetConeAngleFromRoughness(geo.mip, 0.0);
        GeometryProps sssProps;
        MaterialProps sssMaterialProps;
        CastRay(sssSample.samplePosition, -sssInteraction.normal,
                0.0, INF, mipConeSSS, FLAG_NON_TRANSPARENT, sssProps, sssMaterialProps);

        if (!sssProps.IsMiss() && sssProps.Has(FLAG_SKIN))
        {
            Xshadow = sssProps.X;
            shadowGeo = sssProps;
            float NoL_sss = saturate(dot(sssMaterialProps.N, L));
            Cdiff = RTXCR_EvalBssrdf(sssSample, Csun, NoL_sss);
        }

        // ---------------------------------------------------------------
        // SSS Transmission: boundary term + single scattering (Step 4)
        // Following the RTXCR SDK integration guide.
        // ---------------------------------------------------------------
        #define SSS_TRANSMISSION_BSDF_SAMPLE_COUNT       1
        #define SSS_TRANSMISSION_SCATTERING_SAMPLE_COUNT 10

        RTXCR_SubsurfaceMaterialCoefficients sssCoeffs =
            RTXCR_ComputeSubsurfaceMaterialCoefficients(sssMat);

        for (int bsdfSampleIdx = 0; bsdfSampleIdx < SSS_TRANSMISSION_BSDF_SAMPLE_COUNT; ++bsdfSampleIdx)
        {
            // Step 4.1: generate cosine-weighted refraction ray into the volume
            float3 refrDir = RTXCR_CalculateRefractionRay(sssInteraction,
                                                           Rng::Hash::GetFloat2());

            // Trace the refraction ray to find the backface exit
            float2 mipConeRefr = GetConeAngleFromRoughness(geo.mip, 0.0);
            GeometryProps refrProps;
            MaterialProps refrMatProps;
            float3 refrOrigin = geo.GetXoffset(-geo.N, PT_SHADOW_RAY_OFFSET);
            CastRay(refrOrigin, refrDir,
                    0.0, INF, mipConeRefr, FLAG_NON_TRANSPARENT, refrProps, refrMatProps);

            // if (refrProps.IsMiss()) {
            //     return float3(1.0, 1.0, 0.0); // 【测试2】纯黄色！如果物体变黄了，说明射线穿透了体积但没有碰到任何东西！(99%是背面剔除或者模型不是闭合的)
            // } else {
            //     return float3(0.0, 1.0, 0.0); // 纯绿色。说明射线成功找到了出口！
            // }
            
            
            if (refrProps.IsMiss())
                continue;

            float thickness = refrProps.hitT;
            float3 backPosition = refrOrigin + thickness * refrDir;
            float3 backN = -refrMatProps.N;

            // Offset exit point along backface normal for shadow ray
            float3 backPositionOffset = refrProps.GetXoffset(backN, PT_SHADOW_RAY_OFFSET);

            // Cast shadow ray from the backface exit toward the sun
            float shadowHitT = CastVisibilityRay_AnyHit(
                backPositionOffset, L, 0.0, INF,
                GetConeAngleFromAngularRadius(refrProps.mip, gTanSunAngularRadius),
                gWorldTlas, FLAG_NON_TRANSPARENT, 0);

            // if (shadowHitT == INF) {
            //     return float3(0.0, 1.0, 1.0); // 【测试3】纯青色！说明光线没被遮挡，能照到背面！
            // } else {
            //     return float3(1.0, 0.0, 1.0); // 纯洋红色！说明物体的背面被其他东西（或者自身错误偏移）遮挡了，一直在阴影里。
            // }
            
            if (shadowHitT == INF)
            {
                // Boundary term: Li * BSDF * PI
                // (PI comes from cosine-lobe PDF cancellation: cosTheta / (cosTheta/pi) = pi)
                float3 boundaryBsdf = RTXCR_EvaluateBoundaryTerm(
                    sssInteraction.normal, L, refrDir, backN, thickness, sssCoeffs);
                transmissionRadiance += Csun * boundaryBsdf * RTXCR_PI;
            }

            // Step 4.2: single scattering — uniform stepping along the refraction ray
            float stepSize = thickness / (SSS_TRANSMISSION_SCATTERING_SAMPLE_COUNT + 1);
            float accumulatedT = 0.0;
            float3 scatteringThroughput = 0.0;

            for (int scatterIdx = 0; scatterIdx < SSS_TRANSMISSION_SCATTERING_SAMPLE_COUNT; ++scatterIdx)
            {
                float currentT = accumulatedT + stepSize;
                accumulatedT = currentT;

                if (currentT >= thickness)
                    break;

                float3 samplePos = refrOrigin + currentT * refrDir;

                // Sample a scattering direction with HG phase function
                float3 scatterDir = RTXCR_SampleDirectionHenyeyGreenstein(
                    Rng::Hash::GetFloat2(), sssMat.g, refrDir);

                // Trace scattering ray to find exit boundary
                GeometryProps ssRayProps;
                MaterialProps ssRayMatProps;
                CastRay(samplePos, scatterDir,
                        0.0, INF, mipConeRefr, FLAG_NON_TRANSPARENT, ssRayProps, ssRayMatProps);
                
                
                // 找到散射出口了
                if (!ssRayProps.IsMiss())
                {
                    float3 scatterExitPos = samplePos + ssRayProps.hitT * scatterDir;
                    float3 scatterExitN =  - ssRayMatProps.N;

                    // Offset and cast shadow ray from scattering exit
                    float3 scatterExitOffset = ssRayProps.GetXoffset(scatterExitN, PT_SHADOW_RAY_OFFSET);
                    float ssShadowHitT = CastVisibilityRay_AnyHit(
                        scatterExitOffset, L, 0.0, INF,
                        GetConeAngleFromAngularRadius(ssRayProps.mip, gTanSunAngularRadius),
                        gWorldTlas, FLAG_NON_TRANSPARENT, 0);

                    
                    // if (ssShadowHitT == INF) {
                    //     return float3(0.0, 1.0, 1.0); // 【测试3】纯青色！说明光线没被遮挡，能照到背面！
                    // } else {
                    //     return float3(1.0, 0.0, 1.0); // 纯洋红色！说明物体的背面被其他东西（或者自身错误偏移）遮挡了，一直在阴影里。
                    // }
            
                    
                    if (ssShadowHitT == INF)
                    {
                        float totalScatteringDist = currentT + ssRayProps.hitT;
                        float3 ssContrib = RTXCR_EvaluateSingleScattering(
                            L, scatterExitN, totalScatteringDist, sssCoeffs);
                        // Li * BSDF / PDF, stepSize is the numerical integration weight
                        scatteringThroughput += Csun * ssContrib * stepSize;
                    }
                }
            }

            transmissionRadiance += scatteringThroughput;
        }

        transmissionRadiance /= max(SSS_TRANSMISSION_BSDF_SAMPLE_COUNT, 1);
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

    // Add transmission after shadow — it has its own per-exit shadow rays
#if( RTXCR_INTEGRATION == 1 )
    lighting += transmissionRadiance;
#endif

    return lighting;
}
