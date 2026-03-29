#ifndef RAB_VISIBILITY_TEST_HLSLI
#define RAB_VISIBILITY_TEST_HLSLI

RayDesc setupVisibilityRay(RAB_Surface surface, RAB_LightSample lightSample, float offset = 0.001)
{
    float3 L = lightSample.position - surface.worldPos;

    RayDesc ray;
    ray.TMin = offset;
    ray.TMax = length(L) - offset;
    ray.Direction = normalize(L);
    ray.Origin = surface.worldPos;

    return ray;
}
bool GetFinalVisibility(RAB_Surface surface,float3 samplePosition)
{
    float3 L = samplePosition - surface.worldPos;
    float offset = 0.001;
    float TMax = length(L) - offset;
    
    bool visibility = CastVisibilityRay_AnyHit( surface.worldPos, normalize(L), 0.001, TMax, float2(0,0), gWorldTlas,FLAG_NON_TRANSPARENT,0);

    return  visibility;
}
// Tests the visibility between a surface and a light sample.
// Returns true if there is nothing between them.
bool RAB_GetConservativeVisibility(RAB_Surface surface, RAB_LightSample lightSample)
{
    
    float3 L = lightSample.position - surface.worldPos;
    float offset = 0.001;
    float TMax = length(L) - offset;
    
    bool visibility = CastVisibilityRay_AnyHit( surface.worldPos, normalize(L), 0.001, TMax, float2(0,0), gWorldTlas,FLAG_NON_TRANSPARENT,0);

    return  visibility;
    
    // RayDesc ray = setupVisibilityRay(surface, lightSample);
    //
    // RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rayQuery;
    //
    // rayQuery.TraceRayInline(SceneBVH, RAY_FLAG_NONE, INSTANCE_MASK_OPAQUE, ray);
    //
    // rayQuery.Proceed();
    //
    // bool visible = (rayQuery.CommittedStatus() == COMMITTED_NOTHING);
    //
    // return visible;
}

// Tests the visibility between a surface and a light sample on the previous frame.
// Since the scene is static in this sample app, it's equivalent to RAB_GetConservativeVisibility.
bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface,
    RAB_LightSample lightSample)
{
    return RAB_GetConservativeVisibility(currentSurface, lightSample);
}

//追踪一条可见光线，返回表面与光样本之间的近似保守可见性。保守意味着如果无法确定，则假定光线可见。
//例如，保守可见性可以跳过经过 alpha 测试或半透明的表面，从而显著提升性能。
//但是，如果此保守可见性与最终可见性之间存在显著差异，则会导致噪声增加。

// 此函数用于光线追踪偏差校正的空间重采样函数中。为了保持结果无偏，还应根据以相同方式计算的可见性来保留或丢弃初始样本。
bool RAB_GetConservativeVisibility(RAB_Surface surface, float3 samplePosition)
{
    
    float3 L = samplePosition - surface.worldPos;
    float offset = 0.001;
    float TMax = length(L) - offset;
    
    bool visibility = CastVisibilityRay_AnyHit( surface.worldPos, normalize(L), 0.001, TMax, float2(0,0), gWorldTlas,FLAG_NON_TRANSPARENT,0);

    return  visibility;
}

//与 RAB_GetConservativeVisibility 相同的可见性光线追踪，但适用于源自上一帧的表面和光样本。

//当前一帧的 TLAS 和 BLAS 数据可用时，实现应使用该前一帧数据和 previousSurface 参数。
//当前一帧的加速度结构不可用时，实现应使用 currentSurface 参数，但这会导致结果暂时出现偏差，并且在某些情况下会增加噪声。
//具体来说，融合时空重采样算法在处理动画对象时会产生非常明显的噪声结果。
bool RAB_GetTemporalConservativeVisibility(RAB_Surface currentSurface, RAB_Surface previousSurface, float3 samplePosition)
{
    return  RAB_GetConservativeVisibility(currentSurface, samplePosition);
}

#endif // RAB_VISIBILITY_TEST_HLSLI
