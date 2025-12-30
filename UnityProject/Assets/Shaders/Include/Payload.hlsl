
struct MainRayPayload
{
    float3 X; // 命中点的世界空间坐标
    float3 Xprev;
    float4 T; // 切线向量（xyz）和副切线符号（w）
    float3 N; // 法线向量（世界空间）
    float hitT; // 光线命中的距离（t值），INF表示未命中
    float curvature; // 曲率估算值（用于材质、去噪等）
    float2 mipAndCone;
    uint instanceIndex; // 命中的实例索引（用于查找InstanceData）
    
    uint textureOffsetAndFlags;

    float3 matN;

    float3 Lemi;
    float3 baseColor;
    float roughness;
    float metalness;

    float3 GetXoffset(float3 offsetDir, float amount)
    {
        float viewZ = Geometry::AffineTransform(gWorldToView, X).z;
        amount *= gUnproject * abs(viewZ);
        return X + offsetDir * max(amount, 0.00001);
    }

    bool IsMiss()
    {
        return hitT == INF;
    }
    
    
    void SetFlag(uint flag)
    {
        textureOffsetAndFlags |= (flag << FLAG_FIRST_BIT);
    }

};
