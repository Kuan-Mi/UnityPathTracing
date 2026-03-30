#ifndef RAB_MATERIAL_HLSLI
#define RAB_MATERIAL_HLSLI

static const float kMinRoughness = 0.05f;

// 存储表面的材质信息
struct RAB_Material
{
    float3 diffuseAlbedo;
    float3 specularF0;
    float roughness;
};

// 返回一个空的材质实例
RAB_Material RAB_EmptyMaterial()
{
    RAB_Material material = (RAB_Material)0;

    return material;
}

// 获取表面的漫反射反照率（albedo）。
float3 GetDiffuseAlbedo(RAB_Material material)
{
    return material.diffuseAlbedo;
}

// 获取表面的镜面反射F0值。
float3 GetSpecularF0(RAB_Material material)
{
    return material.specularF0;
}

// 获取表面的粗糙度值。
float GetRoughness(RAB_Material material)
{
    return material.roughness;
}

RAB_Material GetGBufferMaterial(
    int2 pixelPosition,
    Texture2D<uint> diffuseAlbedoTexture, 
    Texture2D<uint> specularRoughTexture)
{
    RAB_Material material = RAB_EmptyMaterial();

    if (any(pixelPosition >= gRectSize))
        return material;

    material.diffuseAlbedo = Unpack_R11G11B10_UFLOAT(diffuseAlbedoTexture[pixelPosition]).rgb;
    float4 specularRough = Unpack_R8G8B8A8_Gamma_UFLOAT(specularRoughTexture[pixelPosition]);
    material.roughness = specularRough.a;
    material.specularF0 = specularRough.rgb;

    return material;
}


RAB_Material RAB_GetGBufferMaterial(
    int2 pixelPosition,  bool previousFrame)
{
    if(previousFrame)
    {
        return GetGBufferMaterial(
            pixelPosition,
            t_PrevGBufferDiffuseAlbedo,
            t_PrevGBufferSpecularRough);
    }
    else
    {
        return GetGBufferMaterial(
            pixelPosition,
            t_GBufferDiffuseAlbedo,
            t_GBufferSpecularRough);
    }
    //
    // RAB_Material material = RAB_EmptyMaterial();
    //
    // if (any(pixelPosition >= gRectSize))
    //     return material;
    //
    // float4 packedBaseColorMetalness = baseColorMetalness[pixelPosition];
    // float3 BaseColor = Color::FromSrgb(packedBaseColorMetalness.xyz);
    // float Metalness = packedBaseColorMetalness.w;
    //
    // float3 albedo, Rf0;
    // BRDF::ConvertBaseColorMetalnessToAlbedoRf0(BaseColor, Metalness, albedo, Rf0);
    //
    //
    // material.diffuseAlbedo = albedo;
    //
    // material.roughness = roughness;
    // material.specularF0 = Rf0;
    //
    // return material;
}

bool RAB_AreMaterialsSimilar(RAB_Material a, RAB_Material b)
{
    const float roughnessThreshold = 0.5;
    const float reflectivityThreshold = 0.25;
    const float albedoThreshold = 0.25;

    if (!RTXDI_CompareRelativeDifference(a.roughness, b.roughness, roughnessThreshold))
        return false;

    if (abs(calcLuminance(a.specularF0) - calcLuminance(b.specularF0)) > reflectivityThreshold)
        return false;
    
    if (abs(calcLuminance(a.diffuseAlbedo) - calcLuminance(b.diffuseAlbedo)) > albedoThreshold)
        return false;

    return true;
}


#endif // RAB_MATERIAL_HLSLI
