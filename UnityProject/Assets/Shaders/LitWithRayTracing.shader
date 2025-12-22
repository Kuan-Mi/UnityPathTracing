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
            Name "Test"

            Tags
            {
                "LightMode"="RayTracing"
            }
            HLSLPROGRAM
            #include "UnityRaytracingMeshUtils.cginc"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "_Utils.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"

            #pragma shader_feature_local_raytracing _EMISSION
            #pragma shader_feature_local_raytracing _NORMALMAP
            #pragma shader_feature_local_raytracing _METALLICSPECGLOSSMAP
            #pragma shader_feature_local_raytracing _SURFACE_TYPE_TRANSPARENT

            #pragma multi_compile_local RAY_TRACING_PROCEDURAL_GEOMETRY

            #pragma max_recursion_depth 2

            #pragma raytracing test

            float _IOR;

            struct AttributeData
            {
                float2 barycentrics;
            };


            #include "RayPayload.hlsl"
            #include "GlobalResource.hlsl"

            #if RAY_TRACING_PROCEDURAL_GEOMETRY

            [shader("intersection")]
            void IntersectionMain()
            {
                AttributeData attr;
                attr.barycentrics = float2(0.0, 0.0);
                ReportHit(0, 0.0, attr);
            }

            #endif

            struct Vertex
            {
                float3 position;
                float3 normal;
                float4 tangent;
                float2 uv;
            };


            Vertex FetchVertex(uint vertexIndex)
            {
                Vertex v;
                // 从 Unity 提供的 Vertex Attribute 结构中读取数据
                v.position = UnityRayTracingFetchVertexAttribute3(vertexIndex, kVertexAttributePosition);
                v.normal = UnityRayTracingFetchVertexAttribute3(vertexIndex, kVertexAttributeNormal);
                v.tangent = UnityRayTracingFetchVertexAttribute4(vertexIndex, kVertexAttributeTangent);
                v.uv = UnityRayTracingFetchVertexAttribute2(vertexIndex, kVertexAttributeTexCoord0);
                return v;
            }

            // 手动插值顶点属性
            Vertex InterpolateVertices(Vertex v0, Vertex v1, Vertex v2, float3 barycentrics)
            {
                Vertex v;
                #define INTERPOLATE_ATTRIBUTE(attr) v.attr = v0.attr * barycentrics.x + v1.attr * barycentrics.y + v2.attr * barycentrics.z
                INTERPOLATE_ATTRIBUTE(position);
                INTERPOLATE_ATTRIBUTE(normal);
                INTERPOLATE_ATTRIBUTE(tangent);
                INTERPOLATE_ATTRIBUTE(uv);
                return v;
            }


            [shader("closesthit")]
            void ClosestHitMain(inout RayPayload payload : SV_RayPayload,
                                AttributeData attribs : SV_IntersectionAttributes)
            {
                // if (payload.bounceIndexOpaque == g_BounceCountOpaque)
                // {
                //     payload.bounceIndexOpaque += 1;
                //     return;
                // }

                uint3 triangleIndices = UnityRayTracingFetchTriangleIndices(PrimitiveIndex());
                Vertex v0 = FetchVertex(triangleIndices.x);
                Vertex v1 = FetchVertex(triangleIndices.y);
                Vertex v2 = FetchVertex(triangleIndices.z);

                float3 barycentricCoords = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                                                  attribs.barycentrics.x, attribs.barycentrics.y);

                Vertex v = InterpolateVertices(v0, v1, v2, barycentricCoords);


                bool isFrontFace = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;

                float3 normalOS = isFrontFace ? v.normal : -v.normal;

                float3 normalWS = normalize(mul(normalOS, (float3x3)WorldToObject()));

                #if _NORMALMAP
                float3 tangentWS = normalize(mul(v.tangent.xyz, (float3x3)WorldToObject()));
                float4 n = _BumpMap.SampleLevel(sampler_BumpMap, _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw, 0);
                float3 tangentNormal = UnpackNormalScale(n, _BumpScale);

                float3 bitangent = cross(normalWS.xyz, tangentWS.xyz);
                half3x3 tangentToWorld = half3x3(tangentWS.xyz, bitangent.xyz, normalWS.xyz);

                float3 worldNormal = TransformTangentToWorld(tangentNormal, tangentToWorld);

                #else
                float3 worldNormal = normalWS;
                #endif

                float3 worldPosition = mul(ObjectToWorld(), float4(v.position, 1)).xyz;

                float3 albedo = _BaseColor.xyz * _BaseMap.SampleLevel(sampler_BaseMap,
                                                                      _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw,
                                                                      0).xyz;


                albedo = float3(0, 1, 0);

                float4 vv = _MetallicGlossMap.SampleLevel(sampler_MetallicGlossMap,
                                                                          _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw, 0);
                float metallic = vv.r * _Metallic;
                float smooth = vv.a * _Smoothness;

                float3 dielectricSpecular = float3(0.04, 0.04, 0.04);
                float3 _SpecularColor = lerp(dielectricSpecular, albedo, metallic);

                // float avgF0 = dot(_SpecularColor, float3(0.3333, 0.3333, 0.3333));
                // float _IOR = (1 + sqrt(avgF0)) / (1 - sqrt(avgF0));

                // _IOR = 0.5f;

                #if _SURFACE_TYPE_TRANSPARENT

                worldNormal = normalize(worldNormal + RandomUnitVector(payload.rngState) * (1 - smooth));

                float3 reflectionRayDir = reflect(WorldRayDirection(), worldNormal);

                // _IOR = 1.5; // 水的折射率
                float indexOfRefraction = isFrontFace ? 1.0 / _IOR : _IOR;

                float3 refractionRayDir = refract(WorldRayDirection(), worldNormal, indexOfRefraction); // 折射方向
                float fresnelFactorTransparent = FresnelReflectAmountTransparent(
                    isFrontFace ? 1 : _IOR, isFrontFace ? _IOR : 1, WorldRayDirection(), worldNormal);
                // 是否进行折射
                float doRefraction = (RandomFloat01(payload.rngState) > fresnelFactorTransparent) ? 1 : 0;
                float3 bounceRayDir = lerp(reflectionRayDir, refractionRayDir, doRefraction);

                float pushOff = doRefraction ? -K_RAY_ORIGIN_PUSH_OFF : K_RAY_ORIGIN_PUSH_OFF;
                float3 albedoTransparent = !isFrontFace
                               ? exp(-(1 - albedo) * RayTCurrent() * _BaseColor.a * 20)
                               : float3(1, 1, 1);


                #endif

                #if _EMISSION
                float3 emission = _EmissionColor.xyz * _EmissionMap.SampleLevel(sampler_EmissionMap, v.uv, 0).xyz;
                payload.emission = emission;
                #else
                payload.emission = float3(0, 0, 0);
                #endif


                #if _SURFACE_TYPE_TRANSPARENT

                payload.k = (doRefraction == 1) ? 1 - fresnelFactorTransparent : fresnelFactorTransparent;
                payload.albedo = albedoTransparent;
                payload.bounceRayOrigin = worldPosition + pushOff * worldNormal;
                payload.bounceRayDirection = bounceRayDir;
                payload.bounceIndexTransparent = payload.bounceIndexTransparent + 1;

                #else

                payload.bounceIndexOpaque = payload.bounceIndexOpaque + 1;
                float3 reflectionRayDir = reflect(WorldRayDirection(), worldNormal);
                float3 diffuseRayDir = normalize(worldNormal + RandomUnitVector(payload.rngState));
                float3 specularRayDir = normalize(lerp(diffuseRayDir, reflectionRayDir, smooth));

                float fresnelFactor = FresnelReflectAmountOpaque(1, _IOR, WorldRayDirection(), worldNormal);
                float specularChance = lerp(metallic, 1, fresnelFactor * smooth);
                float doSpecular = (RandomFloat01(payload.rngState) < specularChance) ? 1 : 0;
                float3 reflectedRayDir = lerp(diffuseRayDir, specularRayDir, doSpecular);

                payload.k = (doSpecular == 1) ? specularChance : 1 - specularChance;
                payload.albedo = lerp(albedo, _SpecularColor.xyz, doSpecular);
                payload.bounceRayOrigin = worldPosition + K_RAY_ORIGIN_PUSH_OFF * worldNormal;
                payload.bounceRayDirection = reflectedRayDir;

                #endif


                // payload.worldFaceNormal = worldNormal;
                // payload.albedo = worldNormal;
            }
            ENDHLSL
        }
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
            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"

            #pragma shader_feature_local_raytracing _EMISSION
            #pragma shader_feature_local_raytracing _NORMALMAP
            #pragma shader_feature_local_raytracing _METALLICSPECGLOSSMAP
            #pragma shader_feature_local_raytracing _SURFACE_TYPE_TRANSPARENT

            #pragma multi_compile_local RAY_TRACING_PROCEDURAL_GEOMETRY

            #pragma max_recursion_depth 2

            #pragma raytracing test

            struct AttributeData
            {
                float2 barycentrics;
            };


            #include "RayPayload.hlsl"
            #include "GlobalResource.hlsl"

            #if RAY_TRACING_PROCEDURAL_GEOMETRY

            [shader("intersection")]
            void IntersectionMain()
            {
                AttributeData attr;
                attr.barycentrics = float2(0.0, 0.0);
                ReportHit(0, 0.0, attr);
            }

            #endif

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

            Vertex FetchVertex(uint vertexIndex)
            {
                Vertex v;
                // 从 Unity 提供的 Vertex Attribute 结构中读取数据
                v.position = UnityRayTracingFetchVertexAttribute3(vertexIndex, kVertexAttributePosition);
                v.normal = UnityRayTracingFetchVertexAttribute3(vertexIndex, kVertexAttributeNormal);
                v.tangent = UnityRayTracingFetchVertexAttribute4(vertexIndex, kVertexAttributeTangent);
                v.uv = UnityRayTracingFetchVertexAttribute2(vertexIndex, kVertexAttributeTexCoord0);
                return v;
            }

            // 手动插值顶点属性
            Vertex InterpolateVertices(Vertex v0, Vertex v1, Vertex v2, float3 barycentrics)
            {
                Vertex v;
                #define INTERPOLATE_ATTRIBUTE(attr) v.attr = v0.attr * barycentrics.x + v1.attr * barycentrics.y + v2.attr * barycentrics.z
                INTERPOLATE_ATTRIBUTE(position);
                INTERPOLATE_ATTRIBUTE(normal);
                INTERPOLATE_ATTRIBUTE(tangent);
                INTERPOLATE_ATTRIBUTE(uv);
                return v;
            }

            #define MAX_MIP_LEVEL                       11.0

            [shader("closesthit")]
            void ClosestHitMain(inout MainRayPayload payload : SV_RayPayload,
                                AttributeData attribs : SV_IntersectionAttributes)
            {
                uint3 triangleIndices = UnityRayTracingFetchTriangleIndices(PrimitiveIndex());
                Vertex v0 = FetchVertex(triangleIndices.x);
                Vertex v1 = FetchVertex(triangleIndices.y);
                Vertex v2 = FetchVertex(triangleIndices.z);

                float3 n0 = v0.normal;
                float3 n1 = v1.normal;
                float3 n2 = v2.normal;

                // Curvature
                float dnSq0 = LengthSquared(n0 - n1);
                float dnSq1 = LengthSquared(n1 - n2);
                float dnSq2 = LengthSquared(n2 - n0);
                float dnSq = max(dnSq0, max(dnSq1, dnSq2));

                payload.curvature = sqrt(dnSq);

                float3 barycentricCoords = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y,
                                                  attribs.barycentrics.x, attribs.barycentrics.y);

                Vertex v = InterpolateVertices(v0, v1, v2, barycentricCoords);


                bool isFrontFace = HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;

                float3 normalOS = isFrontFace ? v.normal : -v.normal;

                float3 normalWS = normalize(mul(normalOS, (float3x3)WorldToObject()));


                float3 direction = WorldRayDirection();

                // 长度
                payload.hitT = RayTCurrent();

                // Mip level

                // 2. 计算 UV 空间面积 (uvArea)
                float2 uvE1 = v1.uv - v0.uv;
                float2 uvE2 = v2.uv - v0.uv;
                // 使用 2D 叉乘公式计算面积
                float uvArea = abs(uvE1.x * uvE2.y - uvE2.x * uvE1.y) * 0.5f;

                // 3. 计算世界空间面积 (worldArea)
                // 注意：v0.position 是模型空间，需要考虑物体的缩放
                float3 edge1 = v1.position - v0.position;
                float3 edge2 = v2.position - v0.position;
                float3 crossProduct = cross(edge1, edge2);

                // 将模型空间的面积向量转换到世界空间，从而自动处理非统一缩放
                // 使用 ObjectToWorld 的转置逆矩阵或直接变换向量（视缩放情况而定）
                float3 worldCrossProduct = mul((float3x3)ObjectToWorld(), crossProduct);
                float worldArea = length(crossProduct) * 0.5f;

                float NoRay = abs(dot(direction, normalWS));
                float a = payload.hitT * payload.mipAndCone.y;
                a *= Math::PositiveRcp(NoRay);
                a *= sqrt(uvArea / max(worldArea, 1e-10f));

                float mip = log2(a);
                mip += MAX_MIP_LEVEL;
                mip = max(mip, 0.0);

                // mip = payload.mipAndCone.y;

                // mip = 0;
                payload.mipAndCone.x += mip;

                #if _NORMALMAP
                float3 tangentWS = normalize(mul(v.tangent.xyz, (float3x3)WorldToObject()));

                float2 normalUV = float2(v.uv.x, 1 - v.uv.y); // 修正UV翻转问题

                float2 packedNormal = _BumpMap.SampleLevel(sampler_BumpMap, _BaseMap_ST.xy * normalUV + _BaseMap_ST.zw,
                         mip).xy;

                float4 T = float4(tangentWS, 1);

                float3 N = Geometry::TransformLocalNormal(packedNormal, T, normalWS);

                // float3 tangentNormal = UnpackNormalScale(n, _BumpScale);

                // float3 bitangent = cross(normalWS.xyz, tangentWS.xyz);
                // half3x3 tangentToWorld = half3x3(tangentWS.xyz, bitangent.xyz, normalWS.xyz);

                // float3 worldNormal = TransformTangentToWorld(tangentNormal, tangentToWorld);

                float3 worldNormal = N;
                #else
                float3 worldNormal = normalWS;
                #endif

                float3 albedo = _BaseColor.xyz * _BaseMap.SampleLevel(sampler_BaseMap,
           _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw, mip).xyz;

                // albedo = randomColor;
                // albedo = float3(ccR, ccG, ccB);

                float roughness;
                float metallic;

                #if _METALLICSPECGLOSSMAP

                float4 vv = _MetallicGlossMap.SampleLevel(sampler_MetallicGlossMap, _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw, mip);
                // metallic = vv.r;
                roughness = vv.g * (1 - _Smoothness);
                metallic = vv.b;

                #else

                roughness = 1 - _Smoothness;
                metallic = _Metallic;

                #endif


                float3 dielectricSpecular = float3(0.04, 0.04, 0.04);
                float3 _SpecularColor = lerp(dielectricSpecular, albedo, metallic);


                // Instance
                uint instanceIndex = InstanceIndex();
                payload.instanceIndex = instanceIndex;

                float3x3 mObjectToWorld = (float3x3)ObjectToWorld();


                float4x4 prev = GetPrevObjectToWorldMatrix();

                float3x3 mPrevObjectToWorld = (float3x3)prev;
                // 法线
                payload.N = normalWS;
                payload.matN = worldNormal;

                float3 worldPosition = mul(ObjectToWorld(), float4(v.position, 1)).xyz;

                float3 prevWorldPosition = mul(mPrevObjectToWorld, v.position);
                // 位置
                payload.X = worldPosition;
                payload.Xprev = prevWorldPosition;
                payload.roughness = roughness;
                payload.baseColor = albedo;
                payload.metalness = metallic;

                #if _EMISSION
                float3 emission = _EmissionColor.xyz * _EmissionMap.SampleLevel(sampler_EmissionMap, v.uv, mip).xyz;
                payload.Lemi = emission;
                #else
                payload.Lemi = float3(0, 0, 0);
                #endif
            }
            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
    CustomEditor "UnityEditor.Rendering.Universal.ShaderGUI.LitShader"
}