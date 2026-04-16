// OMMPreviewFill.shader
// Renders a mesh in UV space, coloring each pixel by its OMM micro-triangle state.
// Fill pass only. See OMMPreviewWire.shader for the wireframe overlay.

Shader "NativeRender/OMMPreviewFill"
{
    Properties
    {
        _AlphaTex ("Alpha Texture", 2D) = "white" {}
        _AlphaCutoff ("Alpha Cutoff", Range(0,1)) = 0.5
        _ColorizeModes ("Colorize Modes", Int) = 1
        _DrawContour ("Draw Contour", Int) = 1
        _HighlightOMM ("Highlight OMM", Int) = -1
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }

        Pass
        {
            Name "OMMFill"
            ZWrite Off
            ZTest Always
            Cull Off

            HLSLPROGRAM
            #pragma vertex   vert
            #pragma geometry geom
            #pragma fragment frag
            #pragma target   4.0

            #include "UnityCG.cginc"

            StructuredBuffer<int>    _OmmIndexBuf;
            StructuredBuffer<uint2>  _OmmDescBuf;
            StructuredBuffer<uint>   _OmmArrayBuf;
            Texture2D    _AlphaTex;
            SamplerState sampler_AlphaTex;
            float _AlphaCutoff;
            int   _ColorizeModes;
            int   _DrawContour;
            int   _HighlightOMM;
            int   _PrimitiveOffset;

            struct appdata
            {
                float2 texcoord : TEXCOORD0;
                uint   vertexId : SV_VertexID;
            };

            struct v2f
            {
                float4 pos    : SV_POSITION;
                float2 uv     : TEXCOORD0;
                uint   primId : TEXCOORD1;
                float3 bary   : TEXCOORD2;
            };

            v2f vert(appdata v)
            {
                v2f o;
                float2 ndc = v.texcoord * 2.0 - 1.0;
                ndc.y = -ndc.y;
                o.pos   = float4(ndc, 0.5, 1.0);
                o.uv    = v.texcoord;
                o.primId = 0;
                o.bary  = float3(0, 0, 0);
                return o;
            }

            [maxvertexcount(3)]
            void geom(triangle v2f input[3], uint primId : SV_PrimitiveID,
                      inout TriangleStream<v2f> stream)
            {
                input[0].bary  = float3(1, 0, 0); input[0].primId = primId;
                input[1].bary  = float3(0, 1, 0); input[1].primId = primId;
                input[2].bary  = float3(0, 0, 1); input[2].primId = primId;
                stream.Append(input[0]);
                stream.Append(input[1]);
                stream.Append(input[2]);
                stream.RestartStrip();
            }

            uint PrefixEor2(uint x)
            {
                x ^= (x >> 1) & 0x7fff7fff;
                x ^= (x >> 2) & 0x3fff3fff;
                x ^= (x >> 4) & 0x0fff0fff;
                x ^= (x >> 8) & 0x00ff00ff;
                return x;
            }

            uint InterleaveBits2(uint x, uint y)
            {
                x = (x & 0xffff) | (y << 16);
                x = ((x >> 8) & 0x0000ff00) | ((x << 8) & 0x00ff0000) | (x & 0xff0000ff);
                x = ((x >> 4) & 0x00f000f0) | ((x << 4) & 0x0f000f00) | (x & 0xf00ff00f);
                x = ((x >> 2) & 0x0c0c0c0c) | ((x << 2) & 0x30303030) | (x & 0xc3c3c3c3);
                x = ((x >> 1) & 0x22222222) | ((x << 1) & 0x44444444) | (x & 0x99999999);
                return x;
            }

            uint DBary2Index(uint u, uint v, uint w, uint level)
            {
                uint coordMask = (1u << level) - 1u;
                uint b0 = ~(u ^ w) & coordMask;
                uint t  = (u ^ v) & b0;
                uint c  = (((u & v & w) | (~u & ~v & ~w)) & coordMask) << 16;
                uint f  = PrefixEor2(t | c) ^ u;
                uint b1 = (f & ~b0) | t;
                return InterleaveBits2(b0, b1);
            }

            uint Bary2Index(float2 bc, uint level, out bool isUpright)
            {
                float steps = float(1u << level);
                uint iu = (uint)(steps * bc.x);
                uint iv = (uint)(steps * bc.y);
                uint iw = (uint)(steps * (1.0 - bc.x - bc.y));
                isUpright = ((iu & 1u) ^ (iv & 1u) ^ (iw & 1u)) != 0u;
                return DBary2Index(iu, iv, iw, level);
            }

            float3 MicroStateColor(int state)
            {
                if (_ColorizeModes)
                {
                    if (state == 0) return float3(0, 0, 1);
                    if (state == 1) return float3(0, 1, 0);
                    if (state == 2) return float3(1, 0, 1);
                    return float3(1, 1, 0);
                }
                else
                {
                    if (state == 0) return float3(0, 0, 0);
                    if (state == 1) return float3(1, 1, 1);
                    if (state == 2) return float3(1, 0, 0);
                    return float3(0, 0, 0);
                }
            }

            float4 frag(v2f i) : SV_Target
            {
                if (_DrawContour)
                {
                    float2 uvDX = ddx(i.uv);
                    float2 uvDY = ddy(i.uv);
                    float a00 = _AlphaTex.Sample(sampler_AlphaTex, i.uv).a;
                    float a10 = _AlphaTex.Sample(sampler_AlphaTex, i.uv + float2(uvDX.x, 0)).a;
                    float a01 = _AlphaTex.Sample(sampler_AlphaTex, i.uv + float2(0, uvDY.y)).a;
                    float a11 = _AlphaTex.Sample(sampler_AlphaTex, i.uv + uvDX + uvDY).a;
                    float4 alpha = float4(a00, a10, a01, a11);
                    if (any(alpha < _AlphaCutoff) && any(alpha >= _AlphaCutoff))
                        return float4(0.8, 0, 0, 1);
                }

                int ommIndex = _OmmIndexBuf[i.primId + _PrimitiveOffset];

                if (ommIndex < 0)
                {
                    int specialState = -(ommIndex + 1);
                    float3 c = MicroStateColor(specialState) * 0.5;
                    return float4(c, 1);
                }

                float highlight = (_HighlightOMM >= 0) ? 0.5 : 1.0;
                if (_HighlightOMM >= 0 && ommIndex == _HighlightOMM)
                    highlight = 1.0;

                uint2 desc       = _OmmDescBuf[ommIndex];
                uint byteOffset  = desc.x;
                uint subdivLevel = desc.y & 0xffffu;
                uint fmt         = (desc.y >> 16u) & 0xffffu;
                bool is2State    = (fmt == 1u);

                bool isUpright;
                uint microIndex = Bary2Index(i.bary.yz, subdivLevel, isUpright);

                uint statesPerDW = is2State ? 32u : 16u;
                uint dwordIndex  = (byteOffset / 4u) + (microIndex / statesPerDW);
                uint stateDW     = _OmmArrayBuf[dwordIndex];
                uint bitShift    = (is2State ? 1u : 2u) * (microIndex % statesPerDW);
                uint state       = (stateDW >> bitShift) & (is2State ? 0x1u : 0x3u);

                float3 clr = MicroStateColor((int)state) * 0.5;
                if (isUpright) clr *= 0.5;
                clr *= highlight;

                float alphaLerp = _AlphaTex.Sample(sampler_AlphaTex, i.uv).r;
                clr += 0.01 * alphaLerp;

                return float4(clr, 1);
            }
            ENDHLSL
        }
    }
}
