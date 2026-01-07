Shader "Custom/LitWithRayTracing"
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
            #include "ml.hlsli"

            // #define half float


            // #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"

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


            #include "Include/Shared.hlsl"
            #include "Include/Payload.hlsl"

            #pragma shader_feature_local_raytracing _EMISSION
            #pragma shader_feature_local_raytracing _NORMALMAP
            #pragma shader_feature_local_raytracing _METALLICSPECGLOSSMAP
            #pragma shader_feature_local_raytracing _SURFACE_TYPE_TRANSPARENT

            #pragma multi_compile_local RAY_TRACING_PROCEDURAL_GEOMETRY

            #pragma raytracing test
            #pragma enable_d3d11_debug_symbols
            #pragma use_dxc
            #pragma enable_ray_tracing_shader_debug_symbols

            // #pragma require Int64BufferAtomics
            #pragma require Native16Bit
            // #pragma require inlineraytracing
            // #pragma target 6.4
            

            struct AttributeData
            {
                float2 barycentrics;
            };


            #if RAY_TRACING_PROCEDURAL_GEOMETRY

            [shader("intersection")]
            void IntersectionMain()
            {
                AttributeData attr;
                attr.barycentrics = float2(0.0, 0.0);
                ReportHit(0, 0.0, attr);
            }

            #endif


            struct PrimitiveData
            {
                float16_t2 uv0;
                float16_t2 uv1;
                float16_t2 uv2;
                float worldArea;

                float16_t2 n0;
                float16_t2 n1;
                float16_t2 n2;
                float uvArea;

                float16_t2 t0;
                float16_t2 t1;
                float16_t2 t2;
                float bitangentSign;
            };

            struct InstanceData
            {
                // For static: mObjectToWorld
                // For dynamic: mWorldToWorldPrev
                float4 mOverloadedMatrix0;
                float4 mOverloadedMatrix1;
                float4 mOverloadedMatrix2;

                float16_t4 baseColorAndMetalnessScale;
                float16_t4 emissionAndRoughnessScale;

                float16_t2 normalUvScale;
                uint32_t textureOffsetAndFlags;
                uint32_t primitiveOffset;
                float scale; // TODO: handling object scale embedded into the transformation matrix (assuming uniform scale), sign represents triangle winding

                uint32_t morphPrimitiveOffset;
                uint32_t unused1;
                uint32_t unused2;
                uint32_t unused3;
            };


            StructuredBuffer<InstanceData> gIn_InstanceData;
            StructuredBuffer<PrimitiveData> gIn_PrimitiveData;


            struct Vertex
            {
                float3 position;
                float3 normal;
                float4 tangent;
                float2 uv;
            };

            float LengthSquared(float3 v)
            {
                return dot(v, v);
            }


            #define MAX_MIP_LEVEL                       11.0

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
            void ClosestHitMain(inout MainRayPayload payload : SV_RayPayload,
                                AttributeData attribs : SV_IntersectionAttributes)
            {
                payload.hitT = RayTCurrent();

                uint instanceIndex = InstanceID();
                InstanceData instanceData = gIn_InstanceData[instanceIndex];

                uint primitiveIndex = instanceData.primitiveOffset + PrimitiveIndex();
                PrimitiveData primitiveData = gIn_PrimitiveData[primitiveIndex];

                float worldArea = primitiveData.worldArea * instanceData.scale * instanceData.scale;

                float3x3 mObjectToWorld = ObjectToWorld3x4();
                bool isFrontFace = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;

                float flip = isFrontFace ? -1.0 : 1.0;

                // Barycentrics
                float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                                             attribs.barycentrics.x, attribs.barycentrics.y);

                // Normal
                float3 n0 = Packing::DecodeUnitVector(primitiveData.n0, true);
                float3 n1 = Packing::DecodeUnitVector(primitiveData.n1, true);
                float3 n2 = Packing::DecodeUnitVector(primitiveData.n2, true);

                float3 N = barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2;


                N = Geometry::RotateVector(mObjectToWorld, N);

                N = normalize(mul(N, (float3x3)WorldToObject()));

                N = normalize(N * flip);
                payload.N = Packing::EncodeUnitVector(-N);

                // Curvature
                float dnSq0 = Math::LengthSquared(n0 - n1);
                float dnSq1 = Math::LengthSquared(n1 - n2);
                float dnSq2 = Math::LengthSquared(n2 - n0);
                float dnSq = max(dnSq0, max(dnSq1, dnSq2));
                payload.curvature = sqrt(dnSq / worldArea);

                float3 direction = WorldRayDirection();

                // Mip level
                float NoRay = abs(dot(direction, -N));
                float a = payload.hitT * payload.mipAndCone.y;
                a *= Math::PositiveRcp(NoRay);
                a *= sqrt(primitiveData.uvArea / worldArea);

                float mip = log2(a);
                mip += MAX_MIP_LEVEL;
                mip = max(mip, 0.0);
                payload.mipAndCone.x += mip;

                // Uv
                float2 uv = barycentrics.x * primitiveData.uv0 + barycentrics.y * primitiveData.uv1 + barycentrics.z * primitiveData.uv2;

                // Tangent
                float3 t0 = Packing::DecodeUnitVector(primitiveData.t0, true);
                float3 t1 = Packing::DecodeUnitVector(primitiveData.t1, true);
                float3 t2 = Packing::DecodeUnitVector(primitiveData.t2, true);

                float3 T = barycentrics.x * t0 + barycentrics.y * t1 + barycentrics.z * t2;
                T = Geometry::RotateVector(mObjectToWorld, T);
                T = normalize(T);
                payload.T = float4(T, primitiveData.bitangentSign);

                #if _NORMALMAP
                float3 tangentWS = T;

                // float2 normalUV = float2(v.uv.x, 1 - v.uv.y); // 修正UV翻转问题
                float2 normalUV = uv; // 修正UV翻转问题

                float4 n = _BumpMap.SampleLevel(sampler_BumpMap, _BaseMap_ST.xy * normalUV + _BaseMap_ST.zw, mip);

                float3 tangentNormal = UnpackNormalScale(n, _BumpScale);

                float3 bitangent = cross(N.xyz, tangentWS.xyz);
                half3x3 tangentToWorld = half3x3(tangentWS.xyz, bitangent.xyz, N.xyz);

                float3 matWorldNormal = TransformTangentToWorld(tangentNormal, tangentToWorld);

                #else
                float3 matWorldNormal = N;
                #endif

                payload.matN = Packing::EncodeUnitVector(-matWorldNormal);

                float3 albedo = _BaseColor.xyz * _BaseMap.SampleLevel(sampler_BaseMap, _BaseMap_ST.xy * uv + _BaseMap_ST.zw, mip).xyz;


                float roughness;
                float metallic;

                #if _METALLICSPECGLOSSMAP

                float4 vv = _MetallicGlossMap.SampleLevel(sampler_MetallicGlossMap, _BaseMap_ST.xy * uv + _BaseMap_ST.zw, mip);
                // metallic = vv.r;
                roughness = vv.g * (1 - _Smoothness);
                metallic = vv.b;

                #else

                roughness = 1 - _Smoothness;
                metallic = _Metallic;

                #endif

                #if _EMISSION
                float3 emission = _EmissionColor.xyz * _EmissionMap.SampleLevel(sampler_EmissionMap, uv, mip).xyz;
                payload.Lemi = Packing::EncodeRgbe(emission);
                #else
                payload.Lemi = Packing::EncodeRgbe(float3(0, 0, 0));

                #endif

                float emissionLevel = Color::Luminance(payload.Lemi);
                emissionLevel = saturate(emissionLevel * 50.0);

                metallic = lerp(metallic, 0.0, emissionLevel);
                roughness = lerp(roughness, 1.0, emissionLevel);


                // Instance
                // payload.instanceIndex = instanceIndex;
                payload.SetInstanceIndex(instanceIndex);


                // float4x4 prev = GetPrevObjectToWorldMatrix();
                // float4x4 prev = unity_MatrixPreviousM;


                // float3 worldPosition = mul(ObjectToWorld3x4(), float4(v.position, 1.0)).xyz;
                //
                // float3 prevWorldPosition = mul(GetPrevObjectToWorldMatrix(), float4(v.position, 1.0)).xyz;

                // 位置
                // payload.X = worldPosition;
                // payload.Xprev = prevWorldPosition;
                // payload.roughness = roughness; 

                payload.roughnessAndMetalness = Packing::Rg16fToUint(float2(roughness, metallic));
                payload.baseColor = Packing::RgbaToUint(float4(albedo, 1.0), 8, 8, 8, 8);

                uint flag = FLAG_NON_TRANSPARENT;
                #if  _SURFACE_TYPE_TRANSPARENT
                flag = FLAG_TRANSPARENT;
                #endif
                payload.SetFlag(flag);
            }
            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
    CustomEditor "UnityEditor.Rendering.Universal.ShaderGUI.LitShader"
}