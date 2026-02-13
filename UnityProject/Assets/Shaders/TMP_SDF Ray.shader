Shader "TextMeshPro/Distance Field Ray"
{

    Properties
    {
        _FaceTex ("Face Texture", 2D) = "white" {}
        _FaceUVSpeedX ("Face UV Speed X", Range(-5, 5)) = 0.0
        _FaceUVSpeedY ("Face UV Speed Y", Range(-5, 5)) = 0.0
        _FaceColor ("Face Color", Color) = (1,1,1,1)
        _FaceDilate ("Face Dilate", Range(-1,1)) = 0

        _OutlineColor ("Outline Color", Color) = (0,0,0,1)
        _OutlineTex ("Outline Texture", 2D) = "white" {}
        _OutlineUVSpeedX ("Outline UV Speed X", Range(-5, 5)) = 0.0
        _OutlineUVSpeedY ("Outline UV Speed Y", Range(-5, 5)) = 0.0
        _OutlineWidth ("Outline Thickness", Range(0, 1)) = 0
        _OutlineSoftness ("Outline Softness", Range(0,1)) = 0

        _Bevel ("Bevel", Range(0,1)) = 0.5
        _BevelOffset ("Bevel Offset", Range(-0.5,0.5)) = 0
        _BevelWidth ("Bevel Width", Range(-.5,0.5)) = 0
        _BevelClamp ("Bevel Clamp", Range(0,1)) = 0
        _BevelRoundness ("Bevel Roundness", Range(0,1)) = 0

        _LightAngle ("Light Angle", Range(0.0, 6.2831853)) = 3.1416
        _SpecularColor ("Specular", Color) = (1,1,1,1)
        _SpecularPower ("Specular", Range(0,4)) = 2.0
        _Reflectivity ("Reflectivity", Range(5.0,15.0)) = 10
        _Diffuse ("Diffuse", Range(0,1)) = 0.5
        _Ambient ("Ambient", Range(1,0)) = 0.5

        _BumpMap ("Normal map", 2D) = "bump" {}
        _BumpOutline ("Bump Outline", Range(0,1)) = 0
        _BumpFace ("Bump Face", Range(0,1)) = 0

        _ReflectFaceColor ("Reflection Color", Color) = (0,0,0,1)
        _ReflectOutlineColor("Reflection Color", Color) = (0,0,0,1)
        _Cube ("Reflection Cubemap", Cube) = "black" { /* TexGen CubeReflect */ }
        _EnvMatrixRotation ("Texture Rotation", vector) = (0, 0, 0, 0)


        _UnderlayColor ("Border Color", Color) = (0,0,0, 0.5)
        _UnderlayOffsetX ("Border OffsetX", Range(-1,1)) = 0
        _UnderlayOffsetY ("Border OffsetY", Range(-1,1)) = 0
        _UnderlayDilate ("Border Dilate", Range(-1,1)) = 0
        _UnderlaySoftness ("Border Softness", Range(0,1)) = 0

        _GlowColor ("Color", Color) = (0, 1, 0, 0.5)
        _GlowOffset ("Offset", Range(-1,1)) = 0
        _GlowInner ("Inner", Range(0,1)) = 0.05
        _GlowOuter ("Outer", Range(0,1)) = 0.05
        _GlowPower ("Falloff", Range(1, 0)) = 0.75

        _WeightNormal ("Weight Normal", float) = 0
        _WeightBold ("Weight Bold", float) = 0.5

        _ShaderFlags ("Flags", float) = 0
        _ScaleRatioA ("Scale RatioA", float) = 1
        _ScaleRatioB ("Scale RatioB", float) = 1
        _ScaleRatioC ("Scale RatioC", float) = 1

        _MainTex ("Font Atlas", 2D) = "white" {}
        _TextureWidth ("Texture Width", float) = 512
        _TextureHeight ("Texture Height", float) = 512
        _GradientScale ("Gradient Scale", float) = 5.0
        _ScaleX ("Scale X", float) = 1.0
        _ScaleY ("Scale Y", float) = 1.0
        _PerspectiveFilter ("Perspective Correction", Range(0, 1)) = 0.875
        _Sharpness ("Sharpness", Range(-1,1)) = 0

        _VertexOffsetX ("Vertex OffsetX", float) = 0
        _VertexOffsetY ("Vertex OffsetY", float) = 0

        _MaskCoord ("Mask Coordinates", vector) = (0, 0, 32767, 32767)
        _ClipRect ("Clip Rect", vector) = (-32767, -32767, 32767, 32767)
        _MaskSoftnessX ("Mask SoftnessX", float) = 0
        _MaskSoftnessY ("Mask SoftnessY", float) = 0

        _StencilComp ("Stencil Comparison", Float) = 8
        _Stencil ("Stencil ID", Float) = 0
        _StencilOp ("Stencil Operation", Float) = 0
        _StencilWriteMask ("Stencil Write Mask", Float) = 255
        _StencilReadMask ("Stencil Read Mask", Float) = 255

        _CullMode ("Cull Mode", Float) = 0
        _ColorMask ("Color Mask", Float) = 15
    }

    SubShader
    {

        Tags
        {
            "Queue"="Transparent"
            "IgnoreProjector"="True"
            "RenderType"="Transparent"
        }

        Stencil
        {
            Ref [_Stencil]
            Comp [_StencilComp]
            Pass [_StencilOp]
            ReadMask [_StencilReadMask]
            WriteMask [_StencilWriteMask]
        }

        Cull [_CullMode]
        ZWrite Off
        Lighting Off
        Fog
        {
            Mode Off
        }
        ZTest [unity_GUIZTestMode]
        Blend One OneMinusSrcAlpha
        ColorMask [_ColorMask]

        Pass
        {
            CGPROGRAM
            #pragma target 3.0
            #pragma vertex VertShader
            #pragma fragment PixShader
            #pragma shader_feature __ BEVEL_ON
            #pragma shader_feature __ UNDERLAY_ON UNDERLAY_INNER
            #pragma shader_feature __ GLOW_ON

            #pragma multi_compile __ UNITY_UI_CLIP_RECT
            #pragma multi_compile __ UNITY_UI_ALPHACLIP

            #include "UnityCG.cginc"
            #include "UnityUI.cginc"
            #include "Assets/TextMesh Pro/Shaders/TMPro_Properties.cginc"
            #include "Assets/TextMesh Pro/Shaders/TMPro.cginc"

            struct vertex_t
            {
                UNITY_VERTEX_INPUT_INSTANCE_ID
                float4 position : POSITION;
                float3 normal : NORMAL;
                fixed4 color : COLOR;
                float4 texcoord0 : TEXCOORD0;
                float2 texcoord1 : TEXCOORD1;
            };

            struct pixel_t
            {
                UNITY_VERTEX_INPUT_INSTANCE_ID
                UNITY_VERTEX_OUTPUT_STEREO
                float4 position : SV_POSITION;
                fixed4 color : COLOR;
                float2 atlas : TEXCOORD0; // Atlas
                float4 param : TEXCOORD1; // alphaClip, scale, bias, weight
                float4 mask : TEXCOORD2; // Position in object space(xy), pixel Size(zw)
                float3 viewDir : TEXCOORD3;

                #if (UNDERLAY_ON || UNDERLAY_INNER)
                float4 texcoord2 : TEXCOORD4; // u,v, scale, bias
                fixed4 underlayColor : COLOR1;
                #endif

                float4 textures : TEXCOORD5;
            };

            // Used by Unity internally to handle Texture Tiling and Offset.
            float4 _FaceTex_ST;
            float4 _OutlineTex_ST;
            float _UIMaskSoftnessX;
            float _UIMaskSoftnessY;
            int _UIVertexColorAlwaysGammaSpace;

            pixel_t VertShader(vertex_t input)
            {
                pixel_t output;

                UNITY_INITIALIZE_OUTPUT(pixel_t, output);
                UNITY_SETUP_INSTANCE_ID(input);
                UNITY_TRANSFER_INSTANCE_ID(input, output);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(output);

                float bold = step(input.texcoord0.w, 0);

                float4 vert = input.position;
                vert.x += _VertexOffsetX;
                vert.y += _VertexOffsetY;

                float4 vPosition = UnityObjectToClipPos(vert);

                float2 pixelSize = vPosition.w;
                pixelSize /= float2(_ScaleX, _ScaleY) * abs(mul((float2x2)UNITY_MATRIX_P, _ScreenParams.xy));
                float scale = rsqrt(dot(pixelSize, pixelSize));
                scale *= abs(input.texcoord0.w) * _GradientScale * (_Sharpness + 1);
                if (UNITY_MATRIX_P[3][3] == 0) scale = lerp(abs(scale) * (1 - _PerspectiveFilter), scale, abs(dot(UnityObjectToWorldNormal(input.normal.xyz), normalize(WorldSpaceViewDir(vert)))));

                float weight = lerp(_WeightNormal, _WeightBold, bold) / 4.0;
                weight = (weight + _FaceDilate) * _ScaleRatioA * 0.5;

                float bias = (.5 - weight) + (.5 / scale);

                float alphaClip = (1.0 - _OutlineWidth * _ScaleRatioA - _OutlineSoftness * _ScaleRatioA);

                #if GLOW_ON
                alphaClip = min(alphaClip, 1.0 - _GlowOffset * _ScaleRatioB - _GlowOuter * _ScaleRatioB);
                #endif

                alphaClip = alphaClip / 2.0 - (.5 / scale) - weight;

                #if (UNDERLAY_ON || UNDERLAY_INNER)
                float4 underlayColor = _UnderlayColor;
                underlayColor.rgb *= underlayColor.a;

                float bScale = scale;
                bScale /= 1 + ((_UnderlaySoftness * _ScaleRatioC) * bScale);
                float bBias = (0.5 - weight) * bScale - 0.5 - ((_UnderlayDilate * _ScaleRatioC) * 0.5 * bScale);

                float x = -(_UnderlayOffsetX * _ScaleRatioC) * _GradientScale / _TextureWidth;
                float y = -(_UnderlayOffsetY * _ScaleRatioC) * _GradientScale / _TextureHeight;
                float2 bOffset = float2(x, y);
                #endif

                // Generate UV for the Masking Texture
                float4 clampedRect = clamp(_ClipRect, -2e10, 2e10);
                float2 maskUV = (vert.xy - clampedRect.xy) / (clampedRect.zw - clampedRect.xy);

                // Support for texture tiling and offset
                float2 textureUV = input.texcoord1;
                float2 faceUV = TRANSFORM_TEX(textureUV, _FaceTex);
                float2 outlineUV = TRANSFORM_TEX(textureUV, _OutlineTex);


                if (_UIVertexColorAlwaysGammaSpace && !IsGammaSpace())
                {
                    input.color.rgb = UIGammaToLinear(input.color.rgb);
                }
                output.position = vPosition;
                output.color = input.color;
                output.atlas = input.texcoord0;
                output.param = float4(alphaClip, scale, bias, weight);
                const half2 maskSoftness = half2(max(_UIMaskSoftnessX, _MaskSoftnessX), max(_UIMaskSoftnessY, _MaskSoftnessY));
                output.mask = half4(vert.xy * 2 - clampedRect.xy - clampedRect.zw, 0.25 / (0.25 * maskSoftness + pixelSize.xy));
                output.viewDir = mul((float3x3)_EnvMatrix, _WorldSpaceCameraPos.xyz - mul(unity_ObjectToWorld, vert).xyz);
                #if (UNDERLAY_ON || UNDERLAY_INNER)
                output.texcoord2 = float4(input.texcoord0 + bOffset, bScale, bBias);
                output.underlayColor = underlayColor;
                #endif
                output.textures = float4(faceUV, outlineUV);

                return output;
            }


            fixed4 PixShader(pixel_t input) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(input);

                float c = tex2D(_MainTex, input.atlas).a;

                #ifndef UNDERLAY_ON
                clip(c - input.param.x);
                #endif

                float scale = input.param.y;
                float bias = input.param.z;
                float weight = input.param.w;
                float sd = (bias - c) * scale;

                float outline = (_OutlineWidth * _ScaleRatioA) * scale;
                float softness = (_OutlineSoftness * _ScaleRatioA) * scale;

                half4 faceColor = _FaceColor;
                half4 outlineColor = _OutlineColor;

                faceColor.rgb *= input.color.rgb;

                faceColor *= tex2D(_FaceTex, input.textures.xy + float2(_FaceUVSpeedX, _FaceUVSpeedY) * _Time.y);
                outlineColor *= tex2D(_OutlineTex, input.textures.zw + float2(_OutlineUVSpeedX, _OutlineUVSpeedY) * _Time.y);

                faceColor = GetColor(sd, faceColor, outlineColor, outline, softness);

                #if BEVEL_ON
                float3 dxy = float3(0.5 / _TextureWidth, 0.5 / _TextureHeight, 0);
                float3 n = GetSurfaceNormal(input.atlas, weight, dxy);

                float3 bump = UnpackNormal(tex2D(_BumpMap, input.textures.xy + float2(_FaceUVSpeedX, _FaceUVSpeedY) * _Time.y)).xyz;
                bump *= lerp(_BumpFace, _BumpOutline, saturate(sd + outline * 0.5));
                n = normalize(n - bump);

                float3 light = normalize(float3(sin(_LightAngle), cos(_LightAngle), -1.0));

                float3 col = GetSpecular(n, light);
                faceColor.rgb += col * faceColor.a;
                faceColor.rgb *= 1 - (dot(n, light) * _Diffuse);
                faceColor.rgb *= lerp(_Ambient, 1, n.z * n.z);

                fixed4 reflcol = texCUBE(_Cube, reflect(input.viewDir, -n));
                faceColor.rgb += reflcol.rgb * lerp(_ReflectFaceColor.rgb, _ReflectOutlineColor.rgb, saturate(sd + outline * 0.5)) * faceColor.a;
                #endif

                #if UNDERLAY_ON
                float d = tex2D(_MainTex, input.texcoord2.xy).a * input.texcoord2.z;
                faceColor += input.underlayColor * saturate(d - input.texcoord2.w) * (1 - faceColor.a);
                #endif

                #if UNDERLAY_INNER
                float d = tex2D(_MainTex, input.texcoord2.xy).a * input.texcoord2.z;
                faceColor += input.underlayColor * (1 - saturate(d - input.texcoord2.w)) * saturate(1 - sd) * (1 - faceColor.a);
                #endif

                #if GLOW_ON
                float4 glowColor = GetGlowColor(sd, scale);
                faceColor.rgb += glowColor.rgb * glowColor.a;
                #endif

                // Alternative implementation to UnityGet2DClipping with support for softness.
                #if UNITY_UI_CLIP_RECT
                half2 m = saturate((_ClipRect.zw - _ClipRect.xy - abs(input.mask.xy)) * input.mask.zw);
                faceColor *= m.x * m.y;
                #endif

                #if UNITY_UI_ALPHACLIP
                clip(faceColor.a - 0.001);
                #endif

                return faceColor * input.color.a;
            }
            ENDCG
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
            #include "ml.hlsli"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // UI Editable properties
            uniform sampler2D _FaceTex; // Alpha : Signed Distance
            uniform float _FaceUVSpeedX;
            uniform float _FaceUVSpeedY;
            uniform float4 _FaceColor; // RGBA : Color + Opacity
            uniform float _FaceDilate; // v[ 0, 1]
            uniform float _OutlineSoftness; // v[ 0, 1]

            uniform sampler2D _OutlineTex; // RGBA : Color + Opacity
            uniform float _OutlineUVSpeedX;
            uniform float _OutlineUVSpeedY;
            uniform float4 _OutlineColor; // RGBA : Color + Opacity
            uniform float _OutlineWidth; // v[ 0, 1]

            uniform float _Bevel; // v[ 0, 1]
            uniform float _BevelOffset; // v[-1, 1]
            uniform float _BevelWidth; // v[-1, 1]
            uniform float _BevelClamp; // v[ 0, 1]
            uniform float _BevelRoundness; // v[ 0, 1]

            uniform sampler2D _BumpMap; // Normal map
            uniform float _BumpOutline; // v[ 0, 1]
            uniform float _BumpFace; // v[ 0, 1]

            uniform samplerCUBE _Cube; // Cube / sphere map
            uniform float4 _ReflectFaceColor; // RGB intensity
            uniform float4 _ReflectOutlineColor;
            //uniform float		_EnvTiltX;					// v[-1, 1]
            //uniform float		_EnvTiltY;					// v[-1, 1]
            uniform float3 _EnvMatrixRotation;
            uniform float4x4 _EnvMatrix;

            uniform float4 _SpecularColor; // RGB intensity
            uniform float _LightAngle; // v[ 0,Tau]
            uniform float _SpecularPower; // v[ 0, 1]
            uniform float _Reflectivity; // v[ 5, 15]
            uniform float _Diffuse; // v[ 0, 1]
            uniform float _Ambient; // v[ 0, 1]

            uniform float4 _UnderlayColor; // RGBA : Color + Opacity
            uniform float _UnderlayOffsetX; // v[-1, 1]
            uniform float _UnderlayOffsetY; // v[-1, 1]
            uniform float _UnderlayDilate; // v[-1, 1]
            uniform float _UnderlaySoftness; // v[ 0, 1]

            uniform float4 _GlowColor; // RGBA : Color + Intesity
            uniform float _GlowOffset; // v[-1, 1]
            uniform float _GlowOuter; // v[ 0, 1]
            uniform float _GlowInner; // v[ 0, 1]
            uniform float _GlowPower; // v[ 1, 1/(1+4*4)]

            // API Editable properties
            uniform float _ShaderFlags;
            uniform float _WeightNormal;
            uniform float _WeightBold;

            uniform float _ScaleRatioA;
            uniform float _ScaleRatioB;
            uniform float _ScaleRatioC;

            uniform float _VertexOffsetX;
            uniform float _VertexOffsetY;

            //uniform float		_UseClipRect;
            uniform float _MaskID;
            uniform sampler2D _MaskTex;
            uniform float4 _MaskCoord;
            uniform float4 _ClipRect; // bottom left(x,y) : top right(z,w)
            uniform float _MaskSoftnessX;
            uniform float _MaskSoftnessY;

            // Font Atlas properties
            uniform sampler2D _MainTex;
            uniform float _TextureWidth;
            uniform float _TextureHeight;
            uniform float _GradientScale;
            uniform float _ScaleX;
            uniform float _ScaleY;
            uniform float _PerspectiveFilter;
            uniform float _Sharpness;


            float2 UnpackUV(float uv)
            {
                float2 output;
                output.x = floor(uv / 4096);
                output.y = uv - 4096 * output.x;

                return output * 0.001953125;
            }

            float4 GetColor(half d, float4 faceColor, float4 outlineColor, half outline, half softness)
            {
                half faceAlpha = 1 - saturate((d - outline * 0.5 + softness * 0.5) / (1.0 + softness));
                half outlineAlpha = saturate((d + outline * 0.5)) * sqrt(min(1.0, outline));

                faceColor.rgb *= faceColor.a;
                outlineColor.rgb *= outlineColor.a;

                faceColor = lerp(faceColor, outlineColor, outlineAlpha);

                faceColor *= faceAlpha;

                return faceColor;
            }

            float3 GetSurfaceNormal(float4 h, float bias)
            {
                bool raisedBevel = step(1, fmod(_ShaderFlags, 2));

                h += bias + _BevelOffset;

                float bevelWidth = max(.01, _OutlineWidth + _BevelWidth);

                // Track outline
                h -= .5;
                h /= bevelWidth;
                h = saturate(h + .5);

                if (raisedBevel) h = 1 - abs(h * 2.0 - 1.0);
                h = lerp(h, sin(h * 3.141592 / 2.0), _BevelRoundness);
                h = min(h, 1.0 - _BevelClamp);
                h *= _Bevel * bevelWidth * _GradientScale * -2.0;

                float3 va = normalize(float3(1.0, 0.0, h.y - h.x));
                float3 vb = normalize(float3(0.0, -1.0, h.w - h.z));

                return cross(va, vb);
            }

            float3 GetSurfaceNormal(float2 uv, float bias, float3 delta)
            {
                // Read "height field"
                float4 h = {
                    tex2D(_MainTex, uv - delta.xz).a,
                    tex2D(_MainTex, uv + delta.xz).a,
                    tex2D(_MainTex, uv - delta.zy).a,
                    tex2D(_MainTex, uv + delta.zy).a
                };

                return GetSurfaceNormal(h, bias);
            }

            float3 GetSpecular(float3 n, float3 l)
            {
                float spec = pow(max(0.0, dot(n, l)), _Reflectivity);
                return _SpecularColor.rgb * spec * _SpecularPower;
            }

            float4 GetGlowColor(float d, float scale)
            {
                float glow = d - (_GlowOffset * _ScaleRatioB) * 0.5 * scale;
                float t = lerp(_GlowInner, (_GlowOuter * _ScaleRatioB), step(0.0, glow)) * 0.5 * scale;
                glow = saturate(abs(glow / (1.0 + t)));
                glow = 1.0 - pow(glow, _GlowPower);
                glow *= sqrt(min(1.0, t)); // Fade off glow thinner than 1 screen pixel
                return float4(_GlowColor.rgb, saturate(_GlowColor.a * glow * 2));
            }

            float4 BlendARGB(float4 overlying, float4 underlying)
            {
                overlying.rgb *= overlying.a;
                underlying.rgb *= underlying.a;
                float3 blended = overlying.rgb + ((1 - overlying.a) * underlying.rgb);
                float alpha = underlying.a + (1 - underlying.a) * overlying.a;
                return float4(blended, alpha);
            }


            #include "Include/Shared.hlsl"
            #include "Include/Payload.hlsl"

            #pragma shader_feature_raytracing _USEPACK

            #pragma shader_feature_local_raytracing _EMISSION
            #pragma shader_feature_local_raytracing _NORMALMAP
            #pragma shader_feature_local_raytracing _METALLICSPECGLOSSMAP
            #pragma shader_feature_local_raytracing _SURFACE_TYPE_TRANSPARENT

            #pragma multi_compile_local RAY_TRACING_PROCEDURAL_GEOMETRY

            #pragma raytracing test
            #pragma enable_d3d11_debug_symbols
            #pragma use_dxc
            #pragma enable_ray_tracing_shader_debug_symbols
            #pragma require Native16Bit
            #pragma require int64

            struct AttributeData
            {
                float2 barycentrics;
            };

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
                // 1. 获取顶点索引
                uint3 triangleIndices = UnityRayTracingFetchTriangleIndices(PrimitiveIndex());

                // 2. 获取三个顶点的 UV（为了性能，AnyHit 通常只取 UV，不计算法线等复杂属性）
                float2 uv0 = UnityRayTracingFetchVertexAttribute2(triangleIndices.x, kVertexAttributeTexCoord0);
                float2 uv1 = UnityRayTracingFetchVertexAttribute2(triangleIndices.y, kVertexAttributeTexCoord0);
                float2 uv2 = UnityRayTracingFetchVertexAttribute2(triangleIndices.z, kVertexAttributeTexCoord0);

                // 3. 计算插值 UV
                float3 barycentricCoords = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
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
            }

            [shader("closesthit")]
            void ClosestHitMain(inout MainRayPayload payload : SV_RayPayload, AttributeData attribs : SV_IntersectionAttributes)
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

                float4 n = _BumpMap.SampleLevel(sampler_BumpMap, _BaseMap_ST.xy * normalUV + _BaseMap_ST.zw, mip);

                // float4 T = float4(tangentWS, 1);

                // float3 N = Geometry::TransformLocalNormal(packedNormal, T, normalWS);

                float3 tangentNormal = UnpackNormalScale(n, _BumpScale);

                float3 bitangent = cross(normalWS.xyz, tangentWS.xyz);
                half3x3 tangentToWorld = half3x3(tangentWS.xyz, bitangent.xyz, normalWS.xyz);

                float3 matWorldNormal = TransformTangentToWorld(tangentNormal, tangentToWorld);
                // worldNormal = tangentNormal; 
                // float3 worldNormal = N;
                #else
                float3 matWorldNormal = normalWS;
                #endif

                float3 albedo = _BaseColor.xyz * _BaseMap.SampleLevel(sampler_BaseMap, _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw, mip).xyz;


                float roughness;
                float metallic;

                #if _METALLICSPECGLOSSMAP

                float4 vv = _MetallicGlossMap.SampleLevel(sampler_MetallicGlossMap, _BaseMap_ST.xy * v.uv + _BaseMap_ST.zw, mip);

                roughness = (1 - vv.a) * (1 - _Smoothness);
                metallic = vv.r;

                // roughness = vv.g * (1 - _Smoothness);
                // metallic = vv.b;

                #else

                roughness = 1 - _Smoothness;
                metallic = _Metallic;

                #endif

                #if _EMISSION
                float3 emission = _EmissionColor.xyz * _EmissionMap.SampleLevel(sampler_EmissionMap, v.uv, mip).xyz;
                payload.Lemi = Packing::EncodeRgbe(emission);
                #else
                payload.Lemi = Packing::EncodeRgbe(float3(0, 0, 0));

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
                payload.N = Packing::EncodeUnitVector(normalWS);
                payload.matN = Packing::EncodeUnitVector(matWorldNormal);

                float3 worldPosition = mul(ObjectToWorld3x4(), float4(v.position, 1.0)).xyz;

                float3 prevWorldPosition = mul(GetPrevObjectToWorldMatrix(), float4(v.position, 1.0)).xyz;

                // 位置
                // payload.X = worldPosition;
                payload.Xprev = prevWorldPosition;
                // payload.roughness = roughness; 

                payload.roughnessAndMetalness = Packing::Rg16fToUint(float2(roughness, metallic));

                // albedo *= float3(0, 1.0, 0);

                payload.baseColor = Packing::RgbaToUint(float4(albedo, 1.0), 8, 8, 8, 8);
                // payload.metalness = metallic;
                uint flag = FLAG_NON_TRANSPARENT;
                #if  _SURFACE_TYPE_TRANSPARENT
                flag = FLAG_TRANSPARENT;
                #endif
                payload.SetFlag(flag);
            }
            ENDHLSL
        }
    }

    Fallback "TextMeshPro/Mobile/Distance Field"
    CustomEditor "TMPro.EditorUtilities.TMP_SDFShaderGUI"
}