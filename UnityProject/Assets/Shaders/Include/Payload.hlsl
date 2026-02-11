struct MainRayPayload
{
    // float3 X; // 命中点的世界空间坐标
    float3 Xprev;
    float4 T; // 切线向量（xyz）和副切线符号（w）
    float2 N; // 法线向量（世界空间）
    float hitT; // 光线命中的距离（t值），INF表示未命中
    float curvature; // 曲率估算值（用于材质、去噪等）
    
    float2 mipAndCone;
    uint instanceIndexAndFlags;
    uint Lemi;
    
    float2 matN;
    uint baseColor;
    uint roughnessAndMetalness;
    // float metalness;

    bool IsMiss()
    {
        return hitT == INF;
    }

    void SetFlag(uint flag)
    {
        instanceIndexAndFlags |= (flag << FLAG_FIRST_BIT);
    }

    bool Has(uint flag)
    {
        return (instanceIndexAndFlags & (flag << FLAG_FIRST_BIT)) != 0;
    }

    #define INSTANCE_INDEX_MASK 0x00FFFFFF

    void SetInstanceIndex(uint index)
    {
        instanceIndexAndFlags = (instanceIndexAndFlags & ~INSTANCE_INDEX_MASK) | (index & INSTANCE_INDEX_MASK);
    }

    uint GetInstanceIndex()
    {
        return (instanceIndexAndFlags & INSTANCE_INDEX_MASK);
    }
};
