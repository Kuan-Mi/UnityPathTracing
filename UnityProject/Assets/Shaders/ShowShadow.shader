Shader "KM/ShowShadow"
{
    SubShader
    {
        Tags
        {
            "RenderPipeline"="UniversalPipeline"
            "RenderType"="Opaque"
        }

        Pass
        {
            Name "ShowShadow"
            ZWrite Off
            ZTest Always
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5

            #include "NRDInclude/NRD.hlsli"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // Blitter 会自动绑定
            TEXTURE2D(_BlitTexture);
            SAMPLER(sampler_BlitTexture);

            struct Attributes
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings Vert(Attributes input)
            {
                Varyings o;
                o.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
                o.uv = GetFullScreenTriangleTexCoord(input.vertexID);
                return o;
            }

            float4 Frag(Varyings i) : SV_Target
            {
                // float OUT_SHADOW_TRANSLUCENCY = SAMPLE_TEXTURE2D(_BlitTexture, sampler_BlitTexture, i.uv).r;
                //
                //
                // float shadow = SIGMA_BackEnd_UnpackShadow(OUT_SHADOW_TRANSLUCENCY);
                //
                // shadow = OUT_SHADOW_TRANSLUCENCY;
                // float4 color = float4(shadow, shadow, shadow, 1);   
                
                return SAMPLE_TEXTURE2D(_BlitTexture, sampler_BlitTexture, i.uv);

 
            }
            ENDHLSL
        }
        Pass
        {
            Name "ShowMV"
            // 【重要】混合模式：保证只显示箭头，不黑屏
            Blend SrcAlpha OneMinusSrcAlpha
            // 【重要】总是显示在最上层
            ZTest Always
            ZWrite Off
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // 你的 Motion Vector 贴图
            TEXTURE2D(_BlitTexture);
            SAMPLER(sampler_BlitTexture);

            struct Attributes
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings Vert(Attributes input)
            {
                Varyings o;
                o.positionCS = GetFullScreenTriangleVertexPosition(input.vertexID);
                o.uv = GetFullScreenTriangleTexCoord(input.vertexID);
                return o;
            }

            // 画线段的距离场函数
            float DistanceToSegment(float2 p, float2 a, float2 b)
            {
                float2 pa = p - a;
                float2 ba = b - a;
                float baLenSq = dot(ba, ba);
                // 防除零保护
                if (baLenSq < 0.0001) return length(pa);
                float h = saturate(dot(pa, ba) / baLenSq);
                return length(pa - ba * h);
            }

            // 箭头绘制函数
            float DrawArrow(float2 p, float2 start, float2 dir, float len)
            {
                float2 end = start + dir * len;

                // 1. 箭身
                float dLine = DistanceToSegment(p, start, end);

                // 2. 箭头头部 (根据线长动态调整大小，限制最小最大值)
                float headSize = clamp(len * 0.35, 3.0, 10.0);
                float2 n = float2(-dir.y, dir.x); // 法线

                float2 h0 = end;
                float2 h1 = end - dir * headSize + n * headSize * 0.5;
                float2 h2 = end - dir * headSize - n * headSize * 0.5;

                float dHead = DistanceToSegment(p, h0, h1);
                dHead = min(dHead, DistanceToSegment(p, h0, h2));

                float d = min(dLine, dHead);

                // 抗锯齿边缘
                return smoothstep(1.5, 0.5, d);
            }

            float4 Frag(Varyings i) : SV_Target
            {
                float2 gScreenSize = _ScreenParams.xy;

                // --- 配置参数 ---
                float gGridSize = 40.0; // 网格大小（像素），建议设小一点比如40，80太稀疏
                float gArrowScale = 2.0; // 箭头长度缩放
                float gMinThreshold = 0.5; // 最小移动阈值（像素），小于此值不画
                // ----------------

                float2 pixelPos = i.uv * gScreenSize;

                // 1. 计算网格中心
                float2 cellId = floor(pixelPos / gGridSize);
                float2 cellCenter = cellId * gGridSize + gGridSize * 0.5;
                float2 centerUv = cellCenter / gScreenSize;

                // 2. 采样 Motion Vector
                // 根据你的代码：motion.xy 已经是像素单位 (Pixel Units)
                float3 motion = SAMPLE_TEXTURE2D(_BlitTexture, sampler_BlitTexture, centerUv).xyz;

                // 3. 提取速度向量
                // 你的代码是 prev - current (指向上一帧)
                // 如果想让箭头指向物体“前进”的方向，需要取反 (-motion.xy)
                // 如果想看“轨迹”来源，则保持原样。这里默认取反以符合直觉。
                float2 velocityPixels = -motion.xy;

                float speed = length(velocityPixels);

                // 4. 阈值判断 (像素单位)
                if (speed < gMinThreshold)
                    return float4(0, 0, 0, 0); // 透明

                // 5. 归一化方向
                float2 dir = velocityPixels / speed;

                // 6. 计算箭头显示长度
                // 限制最长不超过网格大小，防止过于杂乱
                float drawLen = min(speed * gArrowScale, gGridSize * 0.5);

                // 7. 绘制
                float alpha = DrawArrow(
                    pixelPos,
                    cellCenter,
                    dir,
                    // float2(0,1), // 测试用固定方向
                    drawLen
                );

                // 8. 颜色 (RG表示方向，B固定)
                float3 color = float3(abs(dir.x), abs(dir.y), 0.2);

                // 如果想要纯色高亮，可以用下面这行：
                // color = float3(1.0, 1.0, 0.0); // 黄色

                return float4(color, 0);
                return float4(color, alpha);
            }
            ENDHLSL
        }
    }
}