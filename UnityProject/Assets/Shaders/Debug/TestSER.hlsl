// ---- 资源绑定 ----
RaytracingAccelerationStructure SceneBVH : register(t0);
RWTexture2D<float4> OutputTexture : register(u0);

// 用于模拟复杂材质的常量
#define HEAVY_WORK_ITERATIONS 5000 
#define USE_SER // 注释掉这一行来关闭重排序进行对比测试

cbuffer SceneConstants : register(b0)
{
    float4x4 viewProjInv;
    float3 cameraPos;
    float _scenePad;
};


struct RayPayload {
    float3 color;
    uint matType; // 0 为轻，1 为重
};

// 模拟复杂计算的函数
float3 ComplexMaterialShading(float3 color, uint iterations) {
    float3 temp = color;
    for(uint i = 0; i < iterations; i++) {
        // 使用一些无法被编译器优化掉的数学运算
        temp = sin(temp) * cos(temp + 0.1f) + sqrt(temp * temp + 0.01f);
        if(i % 2 == 0) temp = frac(temp * 1.1f);
    }
    return temp;
}

[shader("raygeneration")]
void RayGenShader() {
    uint2 idx = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    // --- 简单的射线生成 ---

    float2 uv = (float2(idx) + 0.5f) / float2(dim);
    float2 ndc = uv * 2.0f - 1.0f;

    float4 target = mul(viewProjInv, float4(ndc, 1, 1));
    target /= target.w;

    RayDesc ray;
    ray.Origin = cameraPos;
    ray.Direction = normalize(target.xyz - cameraPos);
    ray.TMin = 0.001f;
    ray.TMax = 1000.0f;

    RayPayload payload = { float3(0,0,0), 0 };

    // 1. 追踪射线，获取 HitObject
    dx::HitObject hit = dx::HitObject::TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    // 2. 预判工作量 (这是关键！)
    // 我们根据 PrimitiveIndex 奇偶性来人为制造分歧
    // 在实际应用中，这里可以是 hit.LoadLocalRootTableConstant() 获取材质类型
    // uint isHeavy = (hit.GetPrimitiveIndex() % 2 == 0) ? 1 : 0;
    uint isHeavy = (idx.x % 2 == 0) ? 1 : 0;

    
#ifdef USE_SER
    // 3. 执行重排序
    // 强制 GPU 把所有的 heavy 线程排在一起，light 线程排在一起
    dx::MaybeReorderThread(hit, isHeavy, 1); 
#endif

    // 4. 执行着色
    if (hit.IsHit()) {
        // 如果是 Heavy 材质，执行高负载计算
        if (isHeavy == 1) {
            payload.color = ComplexMaterialShading(float3(1, 0.5, 0.2), HEAVY_WORK_ITERATIONS);
        } else {
            payload.color = float3(0.2, 0.5, 1.0); // Light 材质
        }
    } else {
        payload.color = float3(0.05, 0.05, 0.05); // Miss
    }

    OutputTexture[idx] = float4(payload.color, 1.0f);
}

[shader("miss")]
void MissShader(inout RayPayload payload) {}

[shader("closesthit")]
void ClosestHitShader(inout RayPayload payload, BuiltInTriangleIntersectionAttributes attr) {}