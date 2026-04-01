Shader "RayTracing/Lit"
{
    Properties
    {
        // Specular vs Metallic workflow
        _WorkflowMode("WorkflowMode", Float) = 1.0

        [MainTexture] _BaseMap("Albedo", 2D) = "white" {}
        [MainColor] _BaseColor("Color", Color) = (1,1,1,1)

        _Cutoff("Alpha Cutoff", Range(0.0, 1.0)) = 0.5

        _IOR("IOR", Range(0.0, 2.0)) = 1.5

        _Smoothness("Smoothness", Range(0.0, 1.0)) = 0.5
        _SmoothnessTextureChannel("Smoothness texture channel", Float) = 0

        _Metallic("Metallic", Range(0.0, 1.0)) = 0.0
        _MetallicGlossMap("Metallic", 2D) = "white" {}

        _SpecColor("Specular", Color) = (0.2, 0.2, 0.2)
        _SpecGlossMap("Specular", 2D) = "white" {}

        [ToggleOff] _SpecularHighlights("Specular Highlights", Float) = 1.0
        [ToggleOff] _EnvironmentReflections("Environment Reflections", Float) = 1.0

        _BumpScale("Scale", Float) = 1.0
        _BumpMap("Normal Map", 2D) = "bump" {}

        _Parallax("Scale", Range(0.005, 0.08)) = 0.005
        _ParallaxMap("Height Map", 2D) = "black" {}

        _OcclusionStrength("Strength", Range(0.0, 1.0)) = 1.0
        _OcclusionMap("Occlusion", 2D) = "white" {}

        [HDR] _EmissionColor("Color", Color) = (0,0,0)
        _EmissionMap("Emission", 2D) = "white" {}

        _DetailMask("Detail Mask", 2D) = "white" {}
        _DetailAlbedoMapScale("Scale", Range(0.0, 2.0)) = 1.0
        _DetailAlbedoMap("Detail Albedo x2", 2D) = "linearGrey" {}
        _DetailNormalMapScale("Scale", Range(0.0, 2.0)) = 1.0
        [Normal] _DetailNormalMap("Normal Map", 2D) = "bump" {}

        // SRP batching compatibility for Clear Coat (Not used in Lit)
        [HideInInspector] _ClearCoatMask("_ClearCoatMask", Float) = 0.0
        [HideInInspector] _ClearCoatSmoothness("_ClearCoatSmoothness", Float) = 0.0

        // Blending state
        _Surface("__surface", Float) = 0.0
        _Blend("__blend", Float) = 0.0
        _Cull("__cull", Float) = 2.0
        [ToggleUI] _AlphaClip("__clip", Float) = 0.0
        [HideInInspector] _SrcBlend("__src", Float) = 1.0
        [HideInInspector] _DstBlend("__dst", Float) = 0.0
        [HideInInspector] _SrcBlendAlpha("__srcA", Float) = 1.0
        [HideInInspector] _DstBlendAlpha("__dstA", Float) = 0.0
        [HideInInspector] _ZWrite("__zw", Float) = 1.0
        [HideInInspector] _BlendModePreserveSpecular("_BlendModePreserveSpecular", Float) = 1.0
        [HideInInspector] _AlphaToMask("__alphaToMask", Float) = 0.0
        [HideInInspector] _AddPrecomputedVelocity("_AddPrecomputedVelocity", Float) = 0.0
        [HideInInspector] _XRMotionVectorsPass("_XRMotionVectorsPass", Float) = 1.0

        [ToggleUI] _ReceiveShadows("Receive Shadows", Float) = 1.0
        [Toggle(_SSS)] _SSS("SSS", Float) = 0.0
        _SSSScatteringColor("SSS Scattering Color", Color) = (1, 0.5, 0.3, 1)
        _SSSScatteringScale("SSS Scattering Scale", Float) = 1.0
        [Toggle(_SKINNEDMESH)] _SKINNEDMESH("Skinned Mesh", Float) = 0.0
        // Editmode props
        _QueueOffset("Queue offset", Float) = 0.0

        // ObsoleteProperties
        [HideInInspector] _MainTex("BaseMap", 2D) = "white" {}
        [HideInInspector] _Color("Base Color", Color) = (1, 1, 1, 1)
        [HideInInspector] _GlossMapScale("Smoothness", Float) = 0.0
        [HideInInspector] _Glossiness("Smoothness", Float) = 0.0
        [HideInInspector] _GlossyReflections("EnvironmentReflections", Float) = 0.0

        [HideInInspector][NoScaleOffset]unity_Lightmaps("unity_Lightmaps", 2DArray) = "" {}
        [HideInInspector][NoScaleOffset]unity_LightmapsInd("unity_LightmapsInd", 2DArray) = "" {}
        [HideInInspector][NoScaleOffset]unity_ShadowMasks("unity_ShadowMasks", 2DArray) = "" {}
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "UniversalMaterialType" = "Lit"
            "IgnoreProjector" = "True"
        }
        LOD 300

        UsePass "Universal Render Pipeline/Lit/ForwardLit"
        UsePass "Universal Render Pipeline/Lit/ShadowCaster"
        UsePass "Universal Render Pipeline/Lit/GBuffer"
        UsePass "Universal Render Pipeline/Lit/DepthOnly"
        UsePass "Universal Render Pipeline/Lit/DepthNormals"
        UsePass "Universal Render Pipeline/Lit/Meta"
        UsePass "Universal Render Pipeline/Lit/Universal2D"
        UsePass "Universal Render Pipeline/Lit/MotionVectors"
        UsePass "Universal Render Pipeline/Lit/XRMotionVectors"

        // ──────────────────────────────────────────────────────────────────────
        // Rasterization G-Buffer pass
        // Writes the same seven G-Buffer textures as the ray-tracing GBuffer.hlsl
        // so the downstream NRD / path-tracing passes can be driven by either.
        // ──────────────────────────────────────────────────────────────────────
        Pass
        {
            Name "GBufferRaster"

            Tags
            {
                "LightMode" = "GBufferRaster"
            }

            Cull   Front
            ZWrite On
            ZTest  LEqual

            HLSLPROGRAM
            #pragma target 4.5

            #pragma vertex   GBufferRasterVert
            #pragma fragment GBufferRasterFrag

            #pragma shader_feature_local _NORMALMAP
            #pragma shader_feature_local _METALLICSPECGLOSSMAP
            #pragma shader_feature_local _EMISSION
            #pragma shader_feature_local _SSS

            #pragma multi_compile_instancing
            #pragma instancing_options renderinglayer

            #pragma use_dxc

            #include "Assets/Shaders/Include/ml.hlsli"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseMap_ST;
                float4 _BaseMap_TexelSize;
                float4 _DetailAlbedoMap_ST;
                float4 _BaseColor;
                float4 _SpecColor;
                float4 _EmissionColor;
                float _Cutoff;
                float _Smoothness;
                float _Metallic;
                float _BumpScale;
                float _Parallax;
                float _OcclusionStrength;
                float _ClearCoatMask;
                float _ClearCoatSmoothness;
                float _DetailAlbedoMapScale;
                float _DetailNormalMapScale;
                float _Surface;
            CBUFFER_END

            float4 _SSSScatteringColor;
            float  _SSSScatteringScale;

            TEXTURE2D(_BaseMap);          SAMPLER(sampler_BaseMap);
            TEXTURE2D(_BumpMap);          SAMPLER(sampler_BumpMap);
            TEXTURE2D(_EmissionMap);      SAMPLER(sampler_EmissionMap);
            TEXTURE2D(_MetallicGlossMap); SAMPLER(sampler_MetallicGlossMap);

            #include "Assets/Shaders/RayTracing/GBufferRaster.hlsl"

            ENDHLSL
        }
    }
    SubShader
    {
        Pass
        {
            Name "Test2"

            Tags
            {
                "LightMode"="RayTracing"
            }
            HLSLPROGRAM
            #include "UnityRaytracingMeshUtils.cginc"
            #include "Assets/Shaders/Include/ml.hlsli"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"


            CBUFFER_START(UnityPerMaterial)
                float4 _BaseMap_ST;
                float4 _BaseMap_TexelSize;
                float4 _DetailAlbedoMap_ST;
                float4 _BaseColor;
                float4 _SpecColor;
                float4 _EmissionColor;
                float _Cutoff;
                float _Smoothness;
                float _Metallic;
                float _BumpScale;
                float _Parallax;
                float _OcclusionStrength;
                float _ClearCoatMask;
                float _ClearCoatSmoothness;
                float _DetailAlbedoMapScale;
                float _DetailNormalMapScale;
                float _Surface;
            CBUFFER_END

            float4 _SSSScatteringColor;
            float _SSSScatteringScale;

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);
            TEXTURE2D(_BumpMap);
            SAMPLER(sampler_BumpMap);
            TEXTURE2D(_EmissionMap);
            SAMPLER(sampler_EmissionMap);


            TEXTURE2D(_ParallaxMap);
            SAMPLER(sampler_ParallaxMap);
            TEXTURE2D(_OcclusionMap);
            SAMPLER(sampler_OcclusionMap);
            TEXTURE2D(_DetailMask);
            SAMPLER(sampler_DetailMask);
            TEXTURE2D(_DetailAlbedoMap);
            SAMPLER(sampler_DetailAlbedoMap);
            TEXTURE2D(_DetailNormalMap);
            SAMPLER(sampler_DetailNormalMap);
            TEXTURE2D(_MetallicGlossMap);
            SAMPLER(sampler_MetallicGlossMap);
            TEXTURE2D(_SpecGlossMap);
            SAMPLER(sampler_SpecGlossMap);
            TEXTURE2D(_ClearCoatMap);
            SAMPLER(sampler_ClearCoatMap);


            #include "Assets/Shaders/Include/Shared.hlsl"
            #include "Assets/Shaders/Include/Payload.hlsl"

            #pragma shader_feature_local_raytracing _EMISSION
            #pragma shader_feature_local_raytracing _NORMALMAP
            #pragma shader_feature_local_raytracing _METALLICSPECGLOSSMAP
            #pragma shader_feature_local_raytracing _SURFACE_TYPE_TRANSPARENT
            #pragma shader_feature_local_raytracing _SSS
            #pragma shader_feature_local_raytracing _SKINNEDMESH

            #pragma multi_compile_local RAY_TRACING_PROCEDURAL_GEOMETRY

            #pragma raytracing test
            // #pragma enable_d3d11_debug_symbols
            #pragma use_dxc
            #pragma enable_ray_tracing_shader_debug_symbols
            #pragma require Native16Bit
            #pragma require int64

            #include "Assets/Shaders/Include/SurfaceRayTracingCommon.hlsl"

            [shader("anyhit")]
            void AnyHitMain(inout MainRayPayload payload, AttributeData attribs)
            {
                #if _SURFACE_TYPE_TRANSPARENT
                if (payload.Has(FLAG_IGNORE_WHEN_TRANSPARENT))
                {
                    IgnoreHit();
                }
                #else
                // 1. 获取顶点索引
                uint3 triangleIndices = UnityRayTracingFetchTriangleIndices(PrimitiveIndex());

                // 2. 获取三个顶点的 UV（为了性能，AnyHit 通常只取 UV，不计算法线等复杂属性）
                float2 uv0 = UnityRayTracingFetchVertexAttribute2(triangleIndices.x, kVertexAttributeTexCoord0);
                float2 uv1 = UnityRayTracingFetchVertexAttribute2(triangleIndices.y, kVertexAttributeTexCoord0);
                float2 uv2 = UnityRayTracingFetchVertexAttribute2(triangleIndices.z, kVertexAttributeTexCoord0);

                // 3. 计算插值 UV
                float3 barycentricCoords = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                              attribs.barycentrics.x, attribs.barycentrics.y);
                float2 uv = uv0 * barycentricCoords.x + uv1 * barycentricCoords.y + uv2 * barycentricCoords.z;

                // 4. 采样 Alpha 通道
                // 注意：在 AnyHit 中采样通常使用 SampleLevel 0 以保证性能，或者根据 RayT 计算一个近似 Mip
                float4 baseColor = _BaseMap.SampleLevel(sampler_BaseMap, _BaseMap_ST.xy * uv + _BaseMap_ST.zw, 0);
                float alpha = baseColor.a * _BaseColor.a;

                // 5. Alpha Test 判定
                // 如果透明度小于阈值，则调用 IgnoreHit()，光线将忽略此次相交
                if (alpha < _Cutoff)
                {
                    IgnoreHit();
                }
                #endif
            }

            [shader("closesthit")]
            void ClosestHitMain(inout MainRayPayload payload : SV_RayPayload, AttributeData attribs : SV_IntersectionAttributes)
            {
                // ----------------------------------------------------------
                // 1. 获取并插值顶点属性
                // ----------------------------------------------------------
                uint3 triangleIndices = UnityRayTracingFetchTriangleIndices(PrimitiveIndex());
                Vertex v0 = FetchVertex(triangleIndices.x);
                Vertex v1 = FetchVertex(triangleIndices.y);
                Vertex v2 = FetchVertex(triangleIndices.z);

                float3 bary = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);

                Vertex v = InterpolateVertices(v0, v1, v2, bary);

                // ----------------------------------------------------------
                // 2. 几何法线与方向
                // ----------------------------------------------------------
                bool isFrontFace = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;
                float3 normalOS = isFrontFace ? v.normal : -v.normal;
                float3 normalWS = normalize(mul(normalOS, (float3x3)WorldToObject()));

                float3 rayDir = WorldRayDirection();
                payload.hitT = RayTCurrent();

                // Curvature
                payload.curvature = ComputeVertexCurvature(v0.normal, v1.normal, v2.normal);

                // Mip level
                float mip = ComputeRayMipLevel(
                    v0.uv, v1.uv, v2.uv,
                    v0.position, v1.position, v2.position,
                    normalWS, rayDir,
                    payload.hitT, payload.mipAndCone.y, 1.0);
                payload.mipAndCone.x += mip;

                // ----------------------------------------------------------
                // 6. 法线计算
                // ----------------------------------------------------------

                #if _NORMALMAP
                float3 tangentWS = normalize(mul(v.tangent.xyz, (float3x3)WorldToObject()));

                float2 normalUV = v.uv;

                float4 n = _BumpMap.SampleLevel(sampler_BumpMap, _BaseMap_ST.xy * normalUV + _BaseMap_ST.zw, mip);
                float3 tangentNormal = UnpackNormalScale(n, _BumpScale);

                float3 bitangent = cross(normalWS.xyz, tangentWS.xyz);
                half3x3 tangentToWorld = half3x3(tangentWS.xyz, bitangent.xyz, normalWS.xyz);

                float3 matWorldNormal = TransformTangentToWorld(tangentNormal, tangentToWorld);

                #else
                float3 matWorldNormal = normalWS;
                #endif

                // ----------------------------------------------------------
                // 7. Albedo 计算
                // ----------------------------------------------------------
                float3 albedo = _BaseColor.xyz * _BaseMap.SampleLevel(sampler_BaseMap, _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw, mip).xyz;


                float roughness;
                float metallic;

                #if _METALLICSPECGLOSSMAP

                float4 vv = _MetallicGlossMap.SampleLevel(sampler_MetallicGlossMap, _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw, mip);

                // float smooth = vv.a * _Smoothness;
                // roughness = 1 - smooth;
                // metallic = vv.r;

                // for Bistro
                float smooth = (1 - vv.g) * _Smoothness;
                roughness = 1 - smooth;
                metallic = vv.b;

                #else

                roughness = 1 - _Smoothness;
                metallic = _Metallic;


                #endif

                #if _SSS

                float3 scattering = _SSSScatteringColor.xyz * _SSSScatteringScale;

                payload.Lemi = Packing::EncodeRgbe(scattering);
                #elif _EMISSION
                float3 emission = _EmissionColor.xyz * _EmissionMap.SampleLevel(sampler_EmissionMap, v.uv, mip).xyz;
                payload.Lemi = Packing::EncodeRgbe(emission);
                #else
                payload.Lemi = Packing::EncodeRgbe(float3(0, 0, 0));
                #endif

                #if !_SSS
                // todo packed error
                float emissionLevel = Color::Luminance(payload.Lemi);
                emissionLevel = saturate(emissionLevel * 50.0);

                // 临时禁用发光对金属度和粗糙度的影响，后续可以考虑更合理的方式（比如单独一个参数控制发光对金属度和粗糙度的影响程度）
                // metallic = lerp(metallic, 0.0, emissionLevel);
                // roughness = lerp(roughness, 1.0, emissionLevel);
                #endif


                float3 dielectricSpecular = float3(0.04, 0.04, 0.04);
                float3 _SpecularColor = lerp(dielectricSpecular, albedo, metallic);


                // Instance
                payload.SetInstanceIndex(InstanceID() + GeometryIndex());

                // 法线
                payload.N = Packing::EncodeUnitVector(normalWS);
                payload.matN = Packing::EncodeUnitVector(matWorldNormal);

                // float3 worldPosition = mul(ObjectToWorld3x4(), float4(v.position, 1.0)).xyz;
                #if _SKINNEDMESH
                float3 prevWorldPosition = mul(GetPrevObjectToWorldMatrix(), float4(v.lastPos, 1.0)).xyz;
                #else
                float3 prevWorldPosition = mul(GetPrevObjectToWorldMatrix(), float4(v.position, 1.0)).xyz;
                #endif

                // 位置
                payload.Xprev = prevWorldPosition;
                payload.roughnessAndMetalness = Packing::Rg16fToUint(float2(roughness, metallic));
                payload.baseColor = Packing::RgbaToUint(float4(albedo, 1.0), 8, 8, 8, 8);

                uint flag = FLAG_NON_TRANSPARENT;
                #if  _SURFACE_TYPE_TRANSPARENT
                flag = FLAG_TRANSPARENT;
                #endif

                #if _SSS
                flag |= FLAG_SKIN;
                #endif

                payload.SetFlag(flag);

                #if _EMISSION

                payload.primitiveIndex = PrimitiveIndex();

                #else

                payload.primitiveIndex = INF;
                #endif
                payload.barycentrics = attribs.barycentrics;
            }
            ENDHLSL
        }


        Pass
        {
            Name "RTXDI"

            Tags
            {
                "LightMode"="RayTracing"
            }
            HLSLPROGRAM
            #include "UnityRaytracingMeshUtils.cginc"
            #include "Assets/Shaders/Include/ml.hlsli"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"


            CBUFFER_START(UnityPerMaterial)
                float4 _BaseMap_ST;
                float4 _BaseMap_TexelSize;
                float4 _DetailAlbedoMap_ST;
                float4 _BaseColor;
                float4 _SpecColor;
                float4 _EmissionColor;
                float _Cutoff;
                float _Smoothness;
                float _Metallic;
                float _BumpScale;
                float _Parallax;
                float _OcclusionStrength;
                float _ClearCoatMask;
                float _ClearCoatSmoothness;
                float _DetailAlbedoMapScale;
                float _DetailNormalMapScale;
                float _Surface;
            CBUFFER_END

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            #include "Assets/Shaders/Include/Shared.hlsl"
            #include "Assets/Shaders/Include/Payload.hlsl"

            #pragma shader_feature_local_raytracing _EMISSION
            #pragma shader_feature_local_raytracing _SURFACE_TYPE_TRANSPARENT

            #pragma multi_compile_local RAY_TRACING_PROCEDURAL_GEOMETRY

            #pragma raytracing test
            // #pragma enable_d3d11_debug_symbols
            #pragma use_dxc
            // #pragma enable_ray_tracing_shader_debug_symbols
            #pragma require Native16Bit
            #pragma require int64

            #include "Assets/Shaders/Include/SurfaceRayTracingCommon.hlsl"

            [shader("anyhit")]
            void AnyHitMain(inout LightPayload payload, AttributeData attribs)
            {
                #if _SURFACE_TYPE_TRANSPARENT
                // if (payload.Has(FLAG_IGNORE_WHEN_TRANSPARENT))
                {
                    IgnoreHit();
                }
                #else
                // 1. 获取顶点索引
                uint3 triangleIndices = UnityRayTracingFetchTriangleIndices(PrimitiveIndex());

                // 2. 获取三个顶点的 UV（为了性能，AnyHit 通常只取 UV，不计算法线等复杂属性）
                float2 uv0 = UnityRayTracingFetchVertexAttribute2(triangleIndices.x, kVertexAttributeTexCoord0);
                float2 uv1 = UnityRayTracingFetchVertexAttribute2(triangleIndices.y, kVertexAttributeTexCoord0);
                float2 uv2 = UnityRayTracingFetchVertexAttribute2(triangleIndices.z, kVertexAttributeTexCoord0);

                // 3. 计算插值 UV
                float3 barycentricCoords = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                              attribs.barycentrics.x, attribs.barycentrics.y);
                float2 uv = uv0 * barycentricCoords.x + uv1 * barycentricCoords.y + uv2 * barycentricCoords.z;

                // 4. 采样 Alpha 通道
                // 注意：在 AnyHit 中采样通常使用 SampleLevel 0 以保证性能，或者根据 RayT 计算一个近似 Mip
                float4 baseColor = _BaseMap.SampleLevel(sampler_BaseMap, _BaseMap_ST.xy * uv + _BaseMap_ST.zw, 0);
                float alpha = baseColor.a * _BaseColor.a;

                // 5. Alpha Test 判定
                // 如果透明度小于阈值，则调用 IgnoreHit()，光线将忽略此次相交
                if (alpha < _Cutoff)
                {
                    IgnoreHit();
                }
                #endif
            }

            [shader("closesthit")]
            void ClosestHitMain(inout LightPayload payload : SV_RayPayload, AttributeData attribs : SV_IntersectionAttributes)
            {
                payload.instanceIndex = InstanceID() + GeometryIndex();
                payload.primitiveIndex = PrimitiveIndex();
                payload.barycentrics = attribs.barycentrics;
            }
            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
    CustomEditor "LitRayTracingShader"
}