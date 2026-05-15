namespace PathTracing
{
    // 0 showValidation     Blend Alpha
    // 1 showShadow         解码后输出阴影
    // 2 showMv             VM
    // 3 ShowNormal         解码后输出法线 转到NRD坐标系
    // 4 showOut            Blend Alpha
    // 5 showAlpha          灰度输出
    // 6 ShowRoughness      解码后输出粗糙度
    // 7 ShowRadiance       解码后RGB输出
    public enum ShowPass : int
    {
        Validation,
        Shadow,
        Mv,
        Normal,
        Out,
        Alpha,
        Roughness,
        Radiance,
        NoiseShadow,
        Dlss,
        ViewZ,
        Gradient,
        // Rtxdi GBuffer passes (R32_UINT uint-load)
        RtxdiViewDepth,      // 12
        RtxdiDiffuseAlbedo,  // 13  R11G11B10 → float3 albedo
        RtxdiSpecularF0,     // 14  R8G8B8A8_Gamma → float3 F0
        RtxdiRoughness,      // 15  R8G8B8A8_Gamma → float roughness
        RtxdiNormal,         // 16  oct32 → float3 normal
        RtxdiGeoNormal,      // 17  oct32 → float3 geo-normal
        PdfTextureMip,       // 18  R32_Float mip slice → log-scale heat map
        ShowGradientArray,   // 19  Texture2DArray slice (gradient array debug)
    }
}