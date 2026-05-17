using UnityEngine;

namespace PathTracing
{
    public static class ShaderIDs
    {
        
        public static readonly int paramsID = Shader.PropertyToID("GlobalConstants");
        
        public static readonly int g_HashEntriesID = Shader.PropertyToID("gInOut_SharcHashEntriesBuffer");
        public static readonly int g_AccumulationBufferID = Shader.PropertyToID("gInOut_SharcAccumulated");
        public static readonly int g_ResolvedBufferID = Shader.PropertyToID("gInOut_SharcResolved");
        
        
        
        
        public static int g_ScramblingRankingID = Shader.PropertyToID("gIn_ScramblingRanking");
        public static int g_SobolID = Shader.PropertyToID("gIn_Sobol");

        // 测试用
        public static int g_OutputID = Shader.PropertyToID("g_Output");

        //  传入NRD的无噪声资源
        public static int g_MvID = Shader.PropertyToID("gOut_Mv");
        public static int g_ViewZID = Shader.PropertyToID("gOut_ViewZ");
        public static int g_Normal_RoughnessID = Shader.PropertyToID("gOut_Normal_Roughness");
        public static int g_BaseColor_MetalnessID = Shader.PropertyToID("gOut_BaseColor_Metalness");

        // 不传入NRD的资源
        public static int g_DirectLightingID = Shader.PropertyToID("gOut_DirectLighting");
        public static int g_DirectEmissionID = Shader.PropertyToID("gOut_DirectEmission");
        public static int g_PsrThroughputID = Shader.PropertyToID("gOut_PsrThroughput");

        // 传入NRD的有噪声资源
        public static int g_ShadowDataID = Shader.PropertyToID("gOut_ShadowData");
        public static int g_DiffID = Shader.PropertyToID("gOut_Diff");
        public static int g_SpecID = Shader.PropertyToID("gOut_Spec");
        
        public static int gIn_PrevComposedDiffID = Shader.PropertyToID("gIn_PrevComposedDiff");
        public static int gIn_PrevComposedSpec_PrevViewZID = Shader.PropertyToID("gIn_PrevComposedSpec_PrevViewZ");


        public static int g_AccelStructID = Shader.PropertyToID("gWorldTlas");


        // TraceOpaque
        public static int gIn_ViewZID = Shader.PropertyToID("gIn_ViewZ");
        public static int gIn_Normal_RoughnessID = Shader.PropertyToID("gIn_Normal_Roughness");
        public static int gIn_BaseColor_MetalnessID = Shader.PropertyToID("gIn_BaseColor_Metalness");
        public static int gIn_DirectLightingID = Shader.PropertyToID("gIn_DirectLighting");
        public static int gIn_DirectEmissionID = Shader.PropertyToID("gIn_DirectEmission");
        public static int gIn_ShadowID = Shader.PropertyToID("gIn_Shadow");
        public static int gIn_DiffID = Shader.PropertyToID("gIn_Diff");
        public static int gIn_SpecID = Shader.PropertyToID("gIn_Spec");
        public static int gIn_PsrThroughputID = Shader.PropertyToID("gIn_PsrThroughput");
        
        public static int gOut_ComposedDiffID = Shader.PropertyToID("gOut_ComposedDiff");
        public static int gOut_ComposedSpec_ViewZID = Shader.PropertyToID("gOut_ComposedSpec_ViewZ");

        public static int gWorldTlasID = Shader.PropertyToID("gWorldTlas");
        public static int gIn_InstanceDataID = Shader.PropertyToID("gIn_InstanceData");
        public static int gIn_PrimitiveDataID = Shader.PropertyToID("gIn_PrimitiveData");


        // TraceTransparency
        public static int gIn_ComposedDiffID = Shader.PropertyToID("gIn_ComposedDiff");
        public static int gIn_ComposedSpec_ViewZID = Shader.PropertyToID("gIn_ComposedSpec_ViewZ");
        public static int gOut_ComposedID = Shader.PropertyToID("gOut_Composed");

        // Spot lights
        public static int gIn_SpotLightsID  = Shader.PropertyToID("gIn_SpotLights");
        // public static int gOut_SpotDirectID = Shader.PropertyToID("gOut_SpotDirect");
        // public static int gIn_SpotDirectID  = Shader.PropertyToID("gIn_SpotDirect");

        // Area lights
        public static int gIn_AreaLightsID  = Shader.PropertyToID("gIn_AreaLights");

        // Point lights
        public static int gIn_PointLightsID = Shader.PropertyToID("gIn_PointLights");

        // TAA
        public static int gIn_MvID = Shader.PropertyToID("gIn_Mv");
        public static int gIn_ComposedID = Shader.PropertyToID("gIn_Composed");
        public static int gIn_HistoryID = Shader.PropertyToID("gIn_History");
        public static int gOut_ResultID = Shader.PropertyToID("gOut_Result");
        public static int gOut_DebugID = Shader.PropertyToID("gOut_Debug");

        // RTXDI：上一帧 GBuffer
        public static int gIn_PrevViewZID = Shader.PropertyToID("gIn_PrevViewZ");
        public static int gIn_PrevNormalRoughnessID = Shader.PropertyToID("gIn_PrevNormalRoughness");
        public static int gIn_PrevBaseColorMetalnessID = Shader.PropertyToID("gIn_PrevBaseColorMetalness");
        
        
        public static int t_LightDataBufferID = Shader.PropertyToID("t_LightDataBuffer");
        public static int t_NeighborOffsetsID = Shader.PropertyToID("t_NeighborOffsets");
        
        public static int u_LightReservoirsID = Shader.PropertyToID("u_LightReservoirs");
        
        
        public static readonly int GInOutMv = Shader.PropertyToID("gInOut_Mv");

        // GBuffer 输入纹理 (t_ = texture read)
        public static int t_GBufferDepthID = Shader.PropertyToID("t_GBufferDepth");
        public static int t_GBufferDiffuseAlbedoID = Shader.PropertyToID("t_GBufferDiffuseAlbedo");
        public static int t_GBufferSpecularRoughID = Shader.PropertyToID("t_GBufferSpecularRough");
        public static int t_GBufferNormalsID = Shader.PropertyToID("t_GBufferNormals");
        public static int t_GBufferGeoNormalsID = Shader.PropertyToID("t_GBufferGeoNormals");
        public static int t_LocalLightPdfTextureID = Shader.PropertyToID("t_LocalLightPdfTexture");
        public static int t_PrevGBufferDepthID = Shader.PropertyToID("t_PrevGBufferDepth");
        public static int t_PrevGBufferDiffuseAlbedoID = Shader.PropertyToID("t_PrevGBufferDiffuseAlbedo");
        public static int t_PrevGBufferSpecularRoughID = Shader.PropertyToID("t_PrevGBufferSpecularRough");
        public static int t_PrevGBufferNormalsID = Shader.PropertyToID("t_PrevGBufferNormals");
        public static int t_PrevGBufferGeoNormalsID = Shader.PropertyToID("t_PrevGBufferGeoNormals");
        public static int t_MotionVectorsID = Shader.PropertyToID("t_MotionVectors");

        // GBuffer 输出纹理 (u_ = UAV write)
        public static int u_ViewDepthID = Shader.PropertyToID("u_ViewDepth");
        public static int u_DiffuseAlbedoID = Shader.PropertyToID("u_DiffuseAlbedo");
        public static int u_SpecularRoughID = Shader.PropertyToID("u_SpecularRough");
        public static int u_NormalsID = Shader.PropertyToID("u_Normals");
        public static int u_GeoNormalsID = Shader.PropertyToID("u_GeoNormals");
        public static int u_EmissiveID = Shader.PropertyToID("u_Emissive");
        public static int u_MotionVectorsID = Shader.PropertyToID("u_MotionVectors");
        public static int u_LocalLightPdfTextureID = Shader.PropertyToID("u_LocalLightPdfTexture");
        public static int gOut_GeoNormalID = Shader.PropertyToID("gOut_GeoNormal");
        public static int gIn_PrevGeoNormalID = Shader.PropertyToID("gIn_PrevGeoNormal");

        // GI 辐照度与自发光
        public static int gIn_EmissiveLightingID = Shader.PropertyToID("gIn_EmissiveLighting");

        // DLSS RR 输入/输出纹理
        public static int gIn_ViewDepthID = Shader.PropertyToID("gIn_ViewDepth");
        public static int gIn_DiffuseAlbedoID = Shader.PropertyToID("gIn_DiffuseAlbedo");
        public static int gIn_SpecularRoughID = Shader.PropertyToID("gIn_SpecularRough");
        public static int gIn_NormalsID = Shader.PropertyToID("gIn_Normals");
        public static int gOut_DiffAlbedoID = Shader.PropertyToID("gOut_DiffAlbedo");
        public static int gOut_SpecAlbedoID = Shader.PropertyToID("gOut_SpecAlbedo");
        public static int gOut_SpecHitDistanceID = Shader.PropertyToID("gOut_SpecHitDistance");

        // RTXDI 缓冲区
        public static int t_GeometryInstanceToLightID = Shader.PropertyToID("t_GeometryInstanceToLight");
        public static int u_RisBufferID = Shader.PropertyToID("u_RisBuffer");
        public static int u_RisLightDataBufferID = Shader.PropertyToID("u_RisLightDataBuffer");
        public static int ResampleConstantsID = Shader.PropertyToID("ResampleConstants");
        public static int u_GIReservoirsID = Shader.PropertyToID("u_GIReservoirs");
        public static int u_SecondaryGBufferID = Shader.PropertyToID("u_SecondaryGBuffer");

        // 常量缓冲区
        public static int g_ConstID = Shader.PropertyToID("g_Const");

        // Mip 生成参数
        public static int _SourceMipID = Shader.PropertyToID("_SourceMip");
        public static int _TargetMipID = Shader.PropertyToID("_TargetMip");
        public static int _SrcMipLevelID = Shader.PropertyToID("_SrcMipLevel");
        public static int _TargetSizeID = Shader.PropertyToID("_TargetSize");

        // 参考 Path Tracing 参数
        public static int _ReferenceBounceNumID = Shader.PropertyToID("_ReferenceBounceNum");
        public static int g_ConvergenceStepID = Shader.PropertyToID("g_ConvergenceStep");
        public static int g_splitID = Shader.PropertyToID("g_split");

        // 自动曝光 (AE) 参数
        public static int _AE_ExposureBufferID = Shader.PropertyToID("_AE_ExposureBuffer");
        public static int _AE_HistogramBufferID = Shader.PropertyToID("_AE_HistogramBuffer");
        public static int _AE_ComposedTextureID = Shader.PropertyToID("_AE_ComposedTexture");
        public static int _AE_TexWidthID = Shader.PropertyToID("_AE_TexWidth");
        public static int _AE_TexHeightID = Shader.PropertyToID("_AE_TexHeight");
        public static int _AE_EVMinID = Shader.PropertyToID("_AE_EVMin");
        public static int _AE_EVMaxID = Shader.PropertyToID("_AE_EVMax");
        public static int _AE_LowPercentID = Shader.PropertyToID("_AE_LowPercent");
        public static int _AE_HighPercentID = Shader.PropertyToID("_AE_HighPercent");
        public static int _AE_SpeedUpID = Shader.PropertyToID("_AE_SpeedUp");
        public static int _AE_SpeedDownID = Shader.PropertyToID("_AE_SpeedDown");
        public static int _AE_DeltaTimeID = Shader.PropertyToID("_AE_DeltaTime");
        public static int _AE_ExposureCompensationID = Shader.PropertyToID("_AE_ExposureCompensation");
        public static int _AE_MinExposureID = Shader.PropertyToID("_AE_MinExposure");
        public static int _AE_MaxExposureID = Shader.PropertyToID("_AE_MaxExposure");

        // 累积 Pass 参数
        public static int gIn_noiseID = Shader.PropertyToID("gIn_noise");
        public static int gIn_AccumulatedID = Shader.PropertyToID("gIn_Accumulated");
    }
}