Shader "Custom/TerrainLitWithRay"
{
    Properties
    {
        [HideInInspector] [ToggleUI] _EnableHeightBlend("EnableHeightBlend", Float) = 0.0
        _HeightTransition("Height Transition", Range(0, 1.0)) = 0.0
        // Layer count is passed down to guide height-blend enable/disable, due
        // to the fact that heigh-based blend will be broken with multipass.
        [HideInInspector] [PerRendererData] _NumLayersCount ("Total Layer Count", Float) = 1.0

        // set by terrain engine
        [HideInInspector] _Control("Control (RGBA)", 2D) = "red" {}
        [HideInInspector] _Splat3("Layer 3 (A)", 2D) = "grey" {}
        [HideInInspector] _Splat2("Layer 2 (B)", 2D) = "grey" {}
        [HideInInspector] _Splat1("Layer 1 (G)", 2D) = "grey" {}
        [HideInInspector] _Splat0("Layer 0 (R)", 2D) = "grey" {}
        [HideInInspector] _Normal3("Normal 3 (A)", 2D) = "bump" {}
        [HideInInspector] _Normal2("Normal 2 (B)", 2D) = "bump" {}
        [HideInInspector] _Normal1("Normal 1 (G)", 2D) = "bump" {}
        [HideInInspector] _Normal0("Normal 0 (R)", 2D) = "bump" {}
        [HideInInspector] _Mask3("Mask 3 (A)", 2D) = "grey" {}
        [HideInInspector] _Mask2("Mask 2 (B)", 2D) = "grey" {}
        [HideInInspector] _Mask1("Mask 1 (G)", 2D) = "grey" {}
        [HideInInspector] _Mask0("Mask 0 (R)", 2D) = "grey" {}
        [HideInInspector][Gamma] _Metallic0("Metallic 0", Range(0.0, 1.0)) = 0.0
        [HideInInspector][Gamma] _Metallic1("Metallic 1", Range(0.0, 1.0)) = 0.0
        [HideInInspector][Gamma] _Metallic2("Metallic 2", Range(0.0, 1.0)) = 0.0
        [HideInInspector][Gamma] _Metallic3("Metallic 3", Range(0.0, 1.0)) = 0.0
        [HideInInspector] _Smoothness0("Smoothness 0", Range(0.0, 1.0)) = 0.5
        [HideInInspector] _Smoothness1("Smoothness 1", Range(0.0, 1.0)) = 0.5
        [HideInInspector] _Smoothness2("Smoothness 2", Range(0.0, 1.0)) = 0.5
        [HideInInspector] _Smoothness3("Smoothness 3", Range(0.0, 1.0)) = 0.5

        // used in fallback on old cards & base map
        [HideInInspector] _MainTex("BaseMap (RGB)", 2D) = "grey" {}
        [HideInInspector] _BaseColor("Main Color", Color) = (1,1,1,1)

        [HideInInspector] _TerrainHolesTexture("Holes Map (RGB)", 2D) = "white" {}

        [ToggleUI] _EnableInstancedPerPixelNormal("Enable Instanced per-pixel normal", Float) = 1.0
    }

    SubShader
    {

        Tags { "Queue" = "Geometry-100" "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "UniversalMaterialType" = "Lit" "IgnoreProjector" = "False" "TerrainCompatible" = "True"}


        UsePass "Universal Render Pipeline/Terrain/Lit/ForwardLit"
        UsePass "Universal Render Pipeline/Terrain/Lit/ShadowCaster"
        UsePass "Universal Render Pipeline/Terrain/Lit/GBuffer"
        UsePass "Universal Render Pipeline/Terrain/Lit/DepthOnly"
        UsePass "Universal Render Pipeline/Terrain/Lit/DepthNormals"
        UsePass "Universal Render Pipeline/Terrain/Lit/SceneSelectionPass"
        UsePass "Universal Render Pipeline/Terrain/Lit/Meta"
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
            #include "Packages/com.unity.render-pipelines.universal/Shaders/Terrain/TerrainLitInput.hlsl"
            #include "Include/Shared.hlsl"
            #include "Include/Payload.hlsl"

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

            [shader("anyhit")]
            void AnyHitMain(inout MainRayPayload payload, AttributeData attribs)
            {
 
            }

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

                // float2 normalUV = float2(v.uv.x, 1 - v.uv.y); // 修正UV翻转问题
                float2 normalUV = (v.uv); // 修正UV翻转问题

                // float2 packedNormal = _BumpMap.SampleLevel(sampler_BumpMap, _BaseMap_ST.xy * normalUV + _BaseMap_ST.zw,
                //                        mip).xy;

                float4 T = float4(tangentWS, 1);

                // float3 N = Geometry::TransformLocalNormal(packedNormal, T, normalWS);
                
                // N = normalWS;

                // float3 tangentNormal = UnpackNormalScale(n, _BumpScale);

                // float3 bitangent = cross(normalWS.xyz, tangentWS.xyz);
                // half3x3 tangentToWorld = half3x3(tangentWS.xyz, bitangent.xyz, normalWS.xyz);

                // float3 worldNormal = TransformTangentToWorld(tangentNormal, tangentToWorld);

                float3 worldNormal = normalWS;
                #else
                float3 worldNormal = normalWS;
                #endif

                // float3 albedo = _BaseColor.xyz * _BaseMap.SampleLevel(sampler_BaseMap, _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw, mip).xyz;

float3 albedo = float3(1, 1, 1);
                float roughness;
                float metallic;

                #if _METALLICSPECGLOSSMAP

                float4 vv = _MetallicGlossMap.SampleLevel(sampler_MetallicGlossMap, _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw, mip);
                metallic = vv.r;
                float smooth = vv.a * _Smoothness;
                roughness =   (1 - smooth); 
                #else

                roughness = 1  ;
                metallic = 0;

                #endif

                #if _EMISSION
                float3 emission = _EmissionColor.xyz * _EmissionMap.SampleLevel(sampler_EmissionMap, v.uv, mip).xyz;
                payload.Lemi =  Packing::EncodeRgbe( emission);
                // payload.Lemi = _EmissionMap.SampleLevel(sampler_EmissionMap, v.uv, mip).xyz;
                #else
                payload.Lemi = Packing::EncodeRgbe( float3(0, 0, 0));
                #endif

                float emissionLevel = Color::Luminance(payload.Lemi);
                emissionLevel = saturate(emissionLevel * 50.0);

                metallic = lerp(metallic, 0.0, emissionLevel);
                roughness = lerp(roughness, 1.0, emissionLevel);


                float3 dielectricSpecular = float3(0.04, 0.04, 0.04);
                float3 _SpecularColor = lerp(dielectricSpecular, albedo, metallic);


                // Instance
                uint instanceIndex = InstanceIndex();
                // payload.instanceIndex = instanceIndex;
                payload.SetInstanceIndex(instanceIndex);

                float3x3 mObjectToWorld = (float3x3)ObjectToWorld();


                float4x4 prev = GetPrevObjectToWorldMatrix();
                // float4x4 prev = unity_MatrixPreviousM;

                float3x3 mPrevObjectToWorld = (float3x3)prev;
                // 法线
                payload.N = normalWS;
                payload.matN = worldNormal;

                float3 worldPosition = mul(ObjectToWorld3x4(), float4(v.position, 1.0)).xyz;

                float3 prevWorldPosition = mul(GetPrevObjectToWorldMatrix(), float4(v.position, 1.0)).xyz;

                // 位置
                // payload.X = worldPosition;
                payload.Xprev = prevWorldPosition;
                payload.roughness = roughness;
                payload.baseColor = Packing::RgbaToUint(float4(albedo, 1.0), 8,8,8,8);
                payload.metalness = metallic;

                uint flag = FLAG_NON_TRANSPARENT;
                #if  _SURFACE_TYPE_TRANSPARENT
                flag = FLAG_TRANSPARENT;
                #endif
                payload.SetFlag(flag);
            }
            ENDHLSL
        }
    }

    CustomEditor "UnityEditor.Rendering.Universal.TerrainLitShaderGUI"

    Fallback "Hidden/Universal Render Pipeline/FallbackError"
}