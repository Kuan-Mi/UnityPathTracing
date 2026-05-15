#pragma pack_matrix(row_major)

// Minimal test shader: fills a Texture2DArray (2 slices) with known colors.
//   Slice 0 → red   (1, 0, 0, 1)
//   Slice 1 → green (0, 1, 0, 1)
// Used to verify that NriTextureArrayResource wraps a D3D12 Texture2DArray
// correctly and that the UAV binding works end-to-end.

struct GradientArrayTestConstants
{
    uint width;
    uint height;
};

ConstantBuffer<GradientArrayTestConstants> g_Const : register(b0);
RWTexture2DArray<float4> u_Output : register(u0);

[numthreads(8, 8, 1)]
void main(uint2 id : SV_DispatchThreadID)
{
    if (id.x >= g_Const.width || id.y >= g_Const.height)
        return;

    // Write a distinct solid color per slice so we can verify both slices are written.
    u_Output[int3(id, 0)] = float4(1.0, 1.0, 0.0, 1.0); // Slice 0 = red
    u_Output[int3(id, 1)] = float4(0.0, 0.0, 1.0, 1.0); // Slice 1 = green
}
