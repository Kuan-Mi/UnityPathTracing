Shader "RayTracing/Skin"
{
    Properties
    {
        [MainTexture] _BaseMap("Albedo", 2D) = "white" {}
        [MainColor] _BaseColor("Color", Color) = (1,1,1,1)

        _ScatteringColor ("Scattering Color", Color) = (1, 0.5, 0.3, 1)
        _Smoothness("Smoothness", Range(0.0, 1.0)) = 0.5

        _Metallic("Metallic", Range(0.0, 1.0)) = 0.0
        _MetallicGlossMap("Metallic", 2D) = "white" {}

        _BumpScale("Scale", Float) = 1.0
        _BumpMap("Normal Map", 2D) = "bump" {}

        // Micro normal (skin pore detail)
        [Normal] _MicroNormalMap("Micro Normal Map", 2D) = "bump" {}
        _MicroNormalStrength("Micro Normal Strength", Range(0.0, 2.0)) = 1.0
        _MicroNormalTiling("Micro Normal Tiling", Float) = 4.0

        // Ray Tracing
        [Toggle] _SSS("SSS (Ray Tracing)", Float) = 1.0
        [Toggle] _SKINNEDMESH("Skinned Mesh (Ray Tracing)", Float) = 0.0
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
                "LightMode" = "RayTracing"
            }
            HLSLPROGRAM
            #include "UnityRaytracingMeshUtils.cginc"
            #include "Assets/Shaders/Include/ml.hlsli"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            float4 _BaseMap_ST;
            float4 _BaseMap_TexelSize;
            float _Smoothness;
            float _Metallic;
            float _BumpScale;
            // Micro normal
            float _MicroNormalStrength;
            float _MicroNormalTiling;

            float4 _BaseColor;
            float3 _ScatteringColor;

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);
            TEXTURE2D(_BumpMap);
            SAMPLER(sampler_BumpMap);
            TEXTURE2D(_MetallicGlossMap);
            SAMPLER(sampler_MetallicGlossMap);
            TEXTURE2D(_MicroNormalMap);
            SAMPLER(sampler_MicroNormalMap);

            #include "Assets/Shaders/Include/Shared.hlsl"
            #include "Assets/Shaders/Include/Payload.hlsl"
            #include "Assets/Shaders/Include/SurfaceRayTracingCommon.hlsl"

            #pragma shader_feature_local_raytracing _EMISSION
            #pragma shader_feature_local_raytracing _NORMALMAP
            #pragma shader_feature_local_raytracing _METALLICSPECGLOSSMAP
            #pragma shader_feature_local_raytracing _SSS
            #pragma shader_feature_local_raytracing _SKINNEDMESH

            #pragma multi_compile_local RAY_TRACING_PROCEDURAL_GEOMETRY

            #pragma raytracing test
            #pragma enable_d3d11_debug_symbols
            #pragma use_dxc
            #pragma enable_ray_tracing_shader_debug_symbols
            #pragma require Native16Bit
            #pragma require int64


            [shader("anyhit")]
            void AnyHitMain(inout MainRayPayload payload, AttributeData attribs)
            {
            }

            [shader("closesthit")]
            void ClosestHitMain(inout MainRayPayload payload : SV_RayPayload, AttributeData attribs : SV_IntersectionAttributes)
            {
                // ----------------------------------------------------------
                // 1. Fetch and interpolate vertex attributes
                // ----------------------------------------------------------
                uint3 triangleIndices = UnityRayTracingFetchTriangleIndices(PrimitiveIndex());
                Vertex v0 = FetchVertex(triangleIndices.x);
                Vertex v1 = FetchVertex(triangleIndices.y);
                Vertex v2 = FetchVertex(triangleIndices.z);

                payload.curvature = ComputeVertexCurvature(v0.normal, v1.normal, v2.normal);

                float3 barycentricCoords = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                          attribs.barycentrics.x, attribs.barycentrics.y);

                Vertex v = InterpolateVertices(v0, v1, v2, barycentricCoords);

                bool isFrontFace = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;
                float3 normalOS = isFrontFace ? v.normal : -v.normal;
                float3 normalWS = normalize(mul(normalOS, (float3x3)WorldToObject()));

                float3 direction = WorldRayDirection();
                payload.hitT = RayTCurrent();

                // Mip level
                float mip = ComputeRayMipLevel(
                    v0.uv, v1.uv, v2.uv,
                    v0.position, v1.position, v2.position,
                    normalWS, direction,
                    payload.hitT, payload.mipAndCone.y, 1.0);
                payload.mipAndCone.x += mip;

                // ----------------------------------------------------------
                // 3. Normal mapping: main normal blended with micro normal
                // ----------------------------------------------------------
                float3 tangentWS = normalize(mul(v.tangent.xyz, (float3x3)WorldToObject()));
                float3 bitangentWS = cross(normalWS, tangentWS) * v.tangent.w;
                half3x3 tbn = half3x3(tangentWS, bitangentWS, normalWS);

                float3 matWorldNormal = normalWS;

                float2 mainUV = _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw;
                float4 mainNormalSample = _BumpMap.SampleLevel(sampler_BumpMap, mainUV, mip);
                float3 mainNormalTS = UnpackNormalScale(mainNormalSample, _BumpScale);

                // Micro normal uses its own independent tiling
                float2 microUV = v.uv * _MicroNormalTiling;
                float4 microNormalSample = _MicroNormalMap.SampleLevel(sampler_MicroNormalMap, microUV, mip);
                float3 microNormalTS = UnpackNormalScale(microNormalSample, _MicroNormalStrength);

                // Blend: micro detail on top of main normal
                float3 blendedNormalTS = BlendNormals(mainNormalTS, microNormalTS);

                matWorldNormal = normalize(TransformTangentToWorld(blendedNormalTS, tbn));


                // ----------------------------------------------------------
                // 4. Albedo
                // ----------------------------------------------------------
                float2 baseUV = _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw;
                float3 albedo = _BaseColor.xyz * _BaseMap.SampleLevel(sampler_BaseMap, baseUV, mip).xyz;

                // ----------------------------------------------------------
                // 5. Roughness & Metallic
                // ----------------------------------------------------------
                float roughness;
                float metallic;

                float4 metallicSample = _MetallicGlossMap.SampleLevel(sampler_MetallicGlossMap, baseUV, mip);
                float smooth = metallicSample.a * _Smoothness;
                roughness = 1.0 - smooth;
                metallic = metallicSample.r;


                // Skin is biological tissue, clamp metallic to near zero
                metallic = min(metallic, 0.04);

                // ----------------------------------------------------------
                // 6. Emission
                // ----------------------------------------------------------
                payload.Lemi = Packing::EncodeRgbe(_ScatteringColor);

                // ----------------------------------------------------------
                // 7. Fill payload
                // ----------------------------------------------------------
                uint instanceIndex = InstanceIndex();
                payload.SetInstanceIndex(instanceIndex);

                float3 worldPosition = mul(ObjectToWorld3x4(), float4(v.position, 1.0)).xyz;
#if _SKINNEDMESH
                float3 prevWorldPosition = mul(GetPrevObjectToWorldMatrix(), float4(v.lastPos, 1.0)).xyz;
#else
                float3 prevWorldPosition = mul(GetPrevObjectToWorldMatrix(), float4(v.position, 1.0)).xyz;
#endif
                payload.Xprev = prevWorldPosition;

                payload.N = Packing::EncodeUnitVector(normalWS);
                payload.matN = Packing::EncodeUnitVector(matWorldNormal);

                payload.roughnessAndMetalness = Packing::Rg16fToUint(float2(roughness, metallic));
                payload.baseColor = Packing::RgbaToUint(float4(albedo, 1.0), 8, 8, 8, 8);

#if _SSS
                payload.SetFlag(FLAG_NON_TRANSPARENT | FLAG_SKIN);
#else
                payload.SetFlag(FLAG_NON_TRANSPARENT);
#endif
            }
            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
    CustomEditor "SkinRayTracingShader"
}