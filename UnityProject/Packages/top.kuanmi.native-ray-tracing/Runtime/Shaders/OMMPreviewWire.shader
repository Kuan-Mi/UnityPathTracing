// OMMPreviewWire.shader
// Wireframe overlay for the OMM UV-space preview.
// Drawn on top of OMMPreviewFill when the Wireframe option is enabled.

Shader "NativeRender/OMMPreviewWire"
{
    Properties {}

    SubShader
    {
        Tags
        {
            "RenderType"="Transparent"
            "Queue"="Transparent+100"
        }

        Pass
        {
            Name "OMMWireframe"
            ZWrite Off
            ZTest Always
            Cull Off

            HLSLPROGRAM
            #pragma vertex   vert_wire
            #pragma geometry geom_wire
            #pragma fragment frag_wire
            #pragma target   4.0

            #include "UnityCG.cginc"

            struct appdata_wire
            {
                float2 texcoord : TEXCOORD0;
            };

            struct v2f_wire
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 bary : TEXCOORD1;
            };

            v2f_wire vert_wire(appdata_wire v)
            {
                v2f_wire o;
                float2 ndc = v.texcoord * 2.0 - 1.0;
                ndc.y = -ndc.y;
                o.pos = float4(ndc, 0.5, 1.0);
                o.uv = v.texcoord;
                o.bary = float3(0, 0, 0);
                return o;
            }

            [maxvertexcount(3)]
            void geom_wire(triangle v2f_wire input[3],
                     inout TriangleStream<v2f_wire> stream)
            {
                input[0].bary = float3(1, 0, 0);
                input[1].bary = float3(0, 1, 0);
                input[2].bary = float3(0, 0, 1);
                stream.Append(input[0]);
                stream.Append(input[1]);
                stream.Append(input[2]);
                stream.RestartStrip();
            }

            float4 frag_wire(v2f_wire i) : SV_Target
            {
                float3 d = fwidth(i.bary);
                float3 a3 = step(d, i.bary); // 0 = near edge, 1 = interior
                if (min(a3.x, min(a3.y, a3.z)) > 0.0) discard;
                return float4(1, 1, 1, 1); // white wire, visible over fill colors
            }
            ENDHLSL
        }
    }
}