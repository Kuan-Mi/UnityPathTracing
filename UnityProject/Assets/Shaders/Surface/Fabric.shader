Shader "Custom/Fabric"
{
    // -----------------------------------------------------------------------
    // Properties 与 AVP_Fabric 完全一致，材质可直接从 AVP_Fabric 切换过来
    // -----------------------------------------------------------------------
    Properties
    {
        Vector1_4784DC7("Tiling", Float) = 1
        Color_512C93D4("Albedo Color", Color) = (1, 1, 1, 0)
        [NoScaleOffset]Texture2D_2D0D32F9("Albedo", 2D) = "white" {}
        AlbedoNoisePower("AlbedoNoisePower", Range(0, 1)) = 1
        [NoScaleOffset]Texture2D_1C6F9DC8("Mask", 2D) = "white" {}
        Vector1_CC012422("Specular", Range(0, 1)) = 0
        Vector1_430999C3("Smoothness", Range(0, 1)) = 0
        Vector1_E6267AA5("OcclusionPower", Range(0, 1)) = 0
        [Normal][NoScaleOffset]Texture2D_C4AB2143("Normals", 2D) = "bump" {}
        Vector1_4548C0F8("NormalPower", Range(-4, 4)) = 1
        Vector1_18BBE2E1("TilingDetail", Float) = 1
        Color_A43A3DFB("RimColor", Color) = (0, 0, 0, 0)
        Vector1_BD133B8D("TilingRimNoise", Float) = 1
        [NoScaleOffset]Texture2D_26D18CAA("RimNoise", 2D) = "white" {}
        Vector1_CB152CD2("RimPower", Range(0, 10)) = 2
        [NoScaleOffset]AlbedoDetail("AlbedoDetail", 2D) = "white" {}
        [Normal][NoScaleOffset]Texture2D_D3552121("NormalDetail", 2D) = "bump" {}
        Vector1_5514B0AB("NormalDetailPower", Range(-4, 4)) = 1
        [NoScaleOffset]Texture2D_C388EAF4("OcclusionDetail", 2D) = "white" {}
        Vector1_54B720F6("OcclusionDetailPower", Range(0, 1)) = 0
        Thickness("Thickness", Range(0, 1)) = 0
        [HideInInspector]_QueueOffset("_QueueOffset", Float) = 0
        [HideInInspector]_QueueControl("_QueueControl", Float) = -1
        [HideInInspector][NoScaleOffset]unity_Lightmaps("unity_Lightmaps", 2DArray) = "" {}
        [HideInInspector][NoScaleOffset]unity_LightmapsInd("unity_LightmapsInd", 2DArray) = "" {}
        [HideInInspector][NoScaleOffset]unity_ShadowMasks("unity_ShadowMasks", 2DArray) = "" {}
    }

    // -----------------------------------------------------------------------
    // SubShader 1: URP 光栅化通道, 直接复用 AVP_Fabric 的所有 Pass
    // -----------------------------------------------------------------------
    SubShader
    {
        Tags
        {
            "RenderPipeline" = "UniversalPipeline"
            "RenderType"     = "Opaque"
            "UniversalMaterialType" = "Lit"
            "Queue"          = "Geometry"
        }

        UsePass "AVP_Fabric/Universal Forward"
        UsePass "AVP_Fabric/GBuffer"
        UsePass "AVP_Fabric/ShadowCaster"
        UsePass "AVP_Fabric/DepthOnly"
        UsePass "AVP_Fabric/DepthNormals"
        UsePass "AVP_Fabric/Meta"
    }

    // -----------------------------------------------------------------------
    // SubShader 2: 光线追踪通道
    // -----------------------------------------------------------------------
    SubShader
    {
        Pass
        {
            Name "Test2"
            Tags { "LightMode" = "RayTracing" }

            HLSLPROGRAM
            #include "UnityRaytracingMeshUtils.cginc"
            #include "Assets/Shaders/Include/ml.hlsli"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            #pragma shader_feature_local_raytracing _SSS
            #pragma shader_feature_local_raytracing _SKINNEDMESH
            
            // ------------------------------------------------------------
            // 材质 CBuffer —— 名称与 AVP_Fabric ShaderGraph 生成代码完全匹配
            // ------------------------------------------------------------
            CBUFFER_START(UnityPerMaterial)
                float  Vector1_4784DC7;         // 主 Tiling
                float4 Color_512C93D4;           // Albedo Color
                float4 Texture2D_2D0D32F9_TexelSize;
                float  AlbedoNoisePower;
                float4 Texture2D_1C6F9DC8_TexelSize;
                float  Vector1_CC012422;         // Specular intensity
                float  Vector1_430999C3;         // Smoothness intensity
                float  Vector1_E6267AA5;         // OcclusionPower
                float4 Texture2D_C4AB2143_TexelSize;
                float  Vector1_4548C0F8;         // NormalPower
                float  Vector1_18BBE2E1;         // TilingDetail
                float4 Color_A43A3DFB;           // RimColor
                float  Vector1_BD133B8D;         // TilingRimNoise
                float4 Texture2D_26D18CAA_TexelSize;
                float  Vector1_CB152CD2;         // RimPower
                float4 AlbedoDetail_TexelSize;
                float4 Texture2D_D3552121_TexelSize;
                float  Vector1_5514B0AB;         // NormalDetailPower
                float4 Texture2D_C388EAF4_TexelSize;
                float  Vector1_54B720F6;         // OcclusionDetailPower
                float  Thickness;
            CBUFFER_END

            // 贴图声明 --------------------------------------------------------
            TEXTURE2D(Texture2D_2D0D32F9);      // Albedo
            SAMPLER(sampler_Texture2D_2D0D32F9);
            TEXTURE2D(Texture2D_1C6F9DC8);      // Mask (R=Spec, G=Occ, A=Smooth)
            SAMPLER(sampler_Texture2D_1C6F9DC8);
            TEXTURE2D(Texture2D_C4AB2143);      // Normal (主法线)
            SAMPLER(sampler_Texture2D_C4AB2143);
            TEXTURE2D(Texture2D_26D18CAA);      // RimNoise
            SAMPLER(sampler_Texture2D_26D18CAA);
            TEXTURE2D(AlbedoDetail);            // Albedo Detail
            SAMPLER(sampler_AlbedoDetail);
            TEXTURE2D(Texture2D_D3552121);      // Normal Detail
            SAMPLER(sampler_Texture2D_D3552121);
            TEXTURE2D(Texture2D_C388EAF4);      // Occlusion Detail
            SAMPLER(sampler_Texture2D_C388EAF4);

            #include "Assets/Shaders/Include/Shared.hlsl"
            #include "Assets/Shaders/Include/Payload.hlsl"
            #include "Assets/Shaders/Include/SurfaceRayTracingCommon.hlsl"

            #pragma multi_compile_local RAY_TRACING_PROCEDURAL_GEOMETRY

            #pragma raytracing test
            // #pragma enable_d3d11_debug_symbols
            #pragma use_dxc
            #pragma enable_ray_tracing_shader_debug_symbols
            #pragma require Native16Bit
            #pragma require int64

            // NormalStrength helper (ShaderGraph-compatible)
            float3 FabricNormalStrength(float3 n, float strength)
            {
                return float3(n.rg * strength, lerp(1.0, n.b, saturate(strength)));
            }

            // // ============================================================
            // // AnyHit: Fabric 为 Opaque（无 AlphaClip），直接 pass
            // // ============================================================
            // [shader("anyhit")]
            // void AnyHitMain(inout MainRayPayload payload, AttributeData attribs)
            // {
            //     // Fabric 是不透明材质，无需任何处理
            // }

            // ============================================================
            // ClosestHit
            // ============================================================
            [shader("closesthit")]
            void ClosestHitMain(inout MainRayPayload payload : SV_RayPayload,
                                AttributeData attribs : SV_IntersectionAttributes)
            {
                // ----------------------------------------------------------
                // 1. 获取并插值顶点属性
                // ----------------------------------------------------------
                uint3 triIdx = UnityRayTracingFetchTriangleIndices(PrimitiveIndex());
                Vertex v0 = FetchVertex(triIdx.x);
                Vertex v1 = FetchVertex(triIdx.y);
                Vertex v2 = FetchVertex(triIdx.z);
                
                float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                                     attribs.barycentrics.x,
                                     attribs.barycentrics.y);
                
                Vertex v = InterpolateVertices(v0, v1, v2, bary);
                
                // ----------------------------------------------------------
                // 2. 几何法线与方向
                // ----------------------------------------------------------
                bool isFrontFace = (HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE);
                float3 normalOS = isFrontFace ? v.normal : -v.normal;
                float3 normalWS = normalize(mul(normalOS, (float3x3)WorldToObject()));
                
                float3 rayDir = WorldRayDirection();
                payload.hitT  = RayTCurrent();
                
                // Curvature
                payload.curvature = ComputeVertexCurvature(v0.normal, v1.normal, v2.normal);
                
                // Mip level (UV tiled by main tiling factor)
                float mip = ComputeRayMipLevel(
                    v0.uv, v1.uv, v2.uv,
                    v0.position, v1.position, v2.position,
                    normalWS, rayDir,
                    payload.hitT, payload.mipAndCone.y, Vector1_4784DC7);
                payload.mipAndCone.x += mip;
                
                // ----------------------------------------------------------
                // 5. 纹理坐标
                // ----------------------------------------------------------
                float2 mainUV    = v.uv * Vector1_4784DC7;
                float2 detailUV  = v.uv * Vector1_18BBE2E1;
                float2 rimUV     = v.uv * Vector1_BD133B8D;
                
                // ----------------------------------------------------------
                // 6. 法线计算（镜像 SurfaceDescriptionFunction 的逻辑）
                //    NormalDetail(detail tiling) blended with MainNormal(main tiling)
                // ----------------------------------------------------------
                float3 tangentWS = normalize(mul(v.tangent.xyz, (float3x3)WorldToObject()));
                
                // 主法线贴图（使用主 Tiling UV，NormalPower = Vector1_4548C0F8）
                float4 mainNormalSample = Texture2D_C4AB2143.SampleLevel(sampler_Texture2D_C4AB2143, mainUV, mip);
                float3 mainNormalTS = UnpackNormal(mainNormalSample);
                float3 mainNormalScaled = FabricNormalStrength(mainNormalTS, Vector1_4548C0F8);
                
                // 细节法线贴图（使用 Detail Tiling UV，NormalDetailPower = Vector1_5514B0AB）
                float4 detailNormalSample = Texture2D_D3552121.SampleLevel(
                    sampler_Texture2D_D3552121, detailUV, mip);
                float3 detailNormalTS = UnpackNormal(detailNormalSample);
                float3 detailNormalScaled = FabricNormalStrength(detailNormalTS, Vector1_5514B0AB);
                
                // NormalBlend（与 ShaderGraph Unity_NormalBlend 等价）
                float3 blendedNormalTS = BlendNormals(detailNormalScaled, mainNormalScaled);
                
                // 切线空间 → 世界空间
                float3 bitangentWS = cross(normalWS, tangentWS) * v.tangent.w;
                half3x3 tbn = half3x3(tangentWS, bitangentWS, normalWS);
                float3 matWorldNormal = normalize(TransformTangentToWorld(blendedNormalTS, tbn));
                
                // ----------------------------------------------------------
                // 7. Albedo 计算（镜像 SurfaceDescriptionFunction）
                //    BaseColor = (abs(RimNoise)^AlbedoNoisePower * AlbedoDetail * MainAlbedo * AlbedoColor)
                //              + (RimPower * RimColor * RimNoise)  [Rim 贡献]
                // ----------------------------------------------------------
                // RimNoise 采样
                float4 rimNoiseSample = Texture2D_26D18CAA.SampleLevel(
                    sampler_Texture2D_26D18CAA, rimUV, mip);
                
                // Albedo 噪声调制
                float4 albedoNoise = pow(abs(rimNoiseSample), AlbedoNoisePower);
                
                // AlbedoDetail 贴图
                float4 albedoDetailSample = AlbedoDetail.SampleLevel(
                    sampler_AlbedoDetail, detailUV, mip);
                
                // 主 Albedo 贴图
                float4 mainAlbedoSample = Texture2D_2D0D32F9.SampleLevel(
                    sampler_Texture2D_2D0D32F9, mainUV, mip);
                
                // 结合：albedoNoise * albedoDetail * mainAlbedo * AlbedoColor
                float3 albedo = (albedoNoise * albedoDetailSample * mainAlbedoSample
                                 * Color_512C93D4).rgb;
                
                // Rim 光照（基于世界空间 NdotV）
                // 原 ShaderGraph 中使用 TangentSpace NdotV，等价于世界空间 NdotV
                float3 viewDir   = -normalize(rayDir);
                float  NdotV     = saturate(dot(matWorldNormal, viewDir));
                float  rimFactor = pow(abs(1.0 - NdotV), Vector1_CB152CD2);
                float3 rimColor  = rimFactor * Color_A43A3DFB.rgb * rimNoiseSample.rgb;
                albedo += rimColor;
                
                // ----------------------------------------------------------
                // 8. 材质参数（Mask: R=Specular, G=Occlusion, A=Smoothness）
                // ----------------------------------------------------------
                float4 maskSample = Texture2D_1C6F9DC8.SampleLevel(
                    sampler_Texture2D_1C6F9DC8, mainUV, mip);
                
                float specularIntensity = maskSample.r * Vector1_CC012422;
                float smoothness        = maskSample.a * Vector1_430999C3;
                float roughness         = 1.0 - smoothness;
                
                // 布料为电介质（Specular Workflow），metallic = 0
                // 用 specularIntensity 做轻微的 metallic 近似（提升菲涅耳 f0）
                // f0 = lerp(0.04, albedo, metallic) —— fabric 通常 specular 极低
                float metallic = 0.0;
                
                // ----------------------------------------------------------
                // 9. Emission（AVP_Fabric 无 Emission，置零）
                // ----------------------------------------------------------
                payload.Lemi = Packing::EncodeRgbe(float3(0.0, 0.0, 0.0));
                
                // ----------------------------------------------------------
                // 10. 运动向量：Xprev
                // ----------------------------------------------------------
                float3 worldPos    = mul(ObjectToWorld3x4(), float4(v.position, 1.0)).xyz;
#if _SKINNEDMESH
                float3 prevWorldPosition = mul(GetPrevObjectToWorldMatrix(), float4(v.lastPos, 1.0)).xyz;
#else
                float3 prevWorldPosition = mul(GetPrevObjectToWorldMatrix(), float4(v.position, 1.0)).xyz;
#endif
                payload.Xprev = prevWorldPosition;
                
                // ----------------------------------------------------------
                // 11. 填充 Payload
                // ----------------------------------------------------------
                payload.N    = Packing::EncodeUnitVector(normalWS);
                payload.matN = Packing::EncodeUnitVector(matWorldNormal);
                
                payload.roughnessAndMetalness = Packing::Rg16fToUint(float2(roughness, metallic));
                payload.baseColor             = Packing::RgbaToUint(float4(albedo, 1.0), 8, 8, 8, 8);
                
                uint instanceIndex = InstanceIndex();
                payload.SetInstanceIndex(instanceIndex);
                
                
                uint flag = FLAG_NON_TRANSPARENT;
                #if  _SURFACE_TYPE_TRANSPARENT
                flag = FLAG_TRANSPARENT;
                #endif
                
                #if _SKIN
                flag |= FLAG_SKIN;
                #endif
                
                
                payload.SetFlag(flag);
            }
            ENDHLSL
        }
    }

    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
