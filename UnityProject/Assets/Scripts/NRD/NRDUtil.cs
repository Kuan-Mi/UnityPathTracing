using Nrd;
using UnityEngine.Experimental.Rendering;

namespace NRD
{
    public static class NRDUtil
    {
        public static DXGI_FORMAT GetDXGIFormat(GraphicsFormat format)
        {
            switch (format)
            {
                // --- 8-bit Formats ---
                case GraphicsFormat.R8_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R8_UNORM;
                case GraphicsFormat.R8_SNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R8_SNORM;
                case GraphicsFormat.R8_UInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R8_UINT;
                case GraphicsFormat.R8_SInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R8_SINT;
                // 注意: DXGI 没有 R8_SRGB，通常在视图层处理或使用 UNORM
                case GraphicsFormat.R8_SRGB:
                    return DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;

                // --- 16-bit Formats (Two 8-bit components) ---
                case GraphicsFormat.R8G8_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R8G8_UNORM;
                case GraphicsFormat.R8G8_SNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R8G8_SNORM;
                case GraphicsFormat.R8G8_UInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R8G8_UINT;
                case GraphicsFormat.R8G8_SInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R8G8_SINT;
                case GraphicsFormat.R8G8_SRGB:
                    return DXGI_FORMAT.DXGI_FORMAT_UNKNOWN; // DXGI 无 R8G8_SRGB

                // --- 24-bit Formats (Three 8-bit components) ---
                // DXGI 现代硬件通常不支持原生 24位 R8G8B8，通常是 32位或压缩格式
                case GraphicsFormat.R8G8B8_UNorm:
                case GraphicsFormat.R8G8B8_SNorm:
                case GraphicsFormat.R8G8B8_UInt:
                case GraphicsFormat.R8G8B8_SInt:
                case GraphicsFormat.R8G8B8_SRGB:
                case GraphicsFormat.B8G8R8_UNorm:
                case GraphicsFormat.B8G8R8_SNorm:
                case GraphicsFormat.B8G8R8_UInt:
                case GraphicsFormat.B8G8R8_SInt:
                case GraphicsFormat.B8G8R8_SRGB:
                    return DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;

                // --- 32-bit Formats (Four 8-bit components) ---
                case GraphicsFormat.R8G8B8A8_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R8G8B8A8_UNORM;
                case GraphicsFormat.R8G8B8A8_SRGB:
                    return DXGI_FORMAT.DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                case GraphicsFormat.R8G8B8A8_SNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R8G8B8A8_SNORM;
                case GraphicsFormat.R8G8B8A8_UInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R8G8B8A8_UINT;
                case GraphicsFormat.R8G8B8A8_SInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R8G8B8A8_SINT;

                case GraphicsFormat.B8G8R8A8_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM;
                case GraphicsFormat.B8G8R8A8_SRGB:
                    return DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
                // DXGI 没有 B8G8R8A8 的 SNorm/UInt/SInt 定义
                case GraphicsFormat.B8G8R8A8_SNorm:
                case GraphicsFormat.B8G8R8A8_UInt:
                case GraphicsFormat.B8G8R8A8_SInt:
                    return DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;

                // --- 16-bit Single Component ---
                case GraphicsFormat.R16_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R16_UNORM;
                case GraphicsFormat.R16_SNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R16_SNORM;
                case GraphicsFormat.R16_UInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R16_UINT;
                case GraphicsFormat.R16_SInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R16_SINT;
                case GraphicsFormat.R16_SFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_R16_FLOAT;

                // --- 32-bit Two Component (16-bit each) ---
                case GraphicsFormat.R16G16_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R16G16_UNORM;
                case GraphicsFormat.R16G16_SNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R16G16_SNORM;
                case GraphicsFormat.R16G16_UInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R16G16_UINT;
                case GraphicsFormat.R16G16_SInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R16G16_SINT;
                case GraphicsFormat.R16G16_SFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_R16G16_FLOAT;

                // --- 48-bit Three Component (16-bit each) ---
                // DXGI 不支持 R16G16B16
                case GraphicsFormat.R16G16B16_UNorm:
                case GraphicsFormat.R16G16B16_SNorm:
                case GraphicsFormat.R16G16B16_UInt:
                case GraphicsFormat.R16G16B16_SInt:
                case GraphicsFormat.R16G16B16_SFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;

                // --- 64-bit Four Component (16-bit each) ---
                case GraphicsFormat.R16G16B16A16_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_UNORM;
                case GraphicsFormat.R16G16B16A16_SNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_SNORM;
                case GraphicsFormat.R16G16B16A16_UInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_UINT;
                case GraphicsFormat.R16G16B16A16_SInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_SINT;
                case GraphicsFormat.R16G16B16A16_SFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_R16G16B16A16_FLOAT;

                // --- 32-bit Single Component ---
                case GraphicsFormat.R32_UInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R32_UINT;
                case GraphicsFormat.R32_SInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R32_SINT;
                case GraphicsFormat.R32_SFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_R32_FLOAT;

                // --- 64-bit Two Component (32-bit each) ---
                case GraphicsFormat.R32G32_UInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R32G32_UINT;
                case GraphicsFormat.R32G32_SInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R32G32_SINT;
                case GraphicsFormat.R32G32_SFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_R32G32_FLOAT;

                // --- 96-bit Three Component (32-bit each) ---
                case GraphicsFormat.R32G32B32_UInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R32G32B32_UINT;
                case GraphicsFormat.R32G32B32_SInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R32G32B32_SINT;
                case GraphicsFormat.R32G32B32_SFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_R32G32B32_FLOAT;

                // --- 128-bit Four Component (32-bit each) ---
                case GraphicsFormat.R32G32B32A32_UInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R32G32B32A32_UINT;
                case GraphicsFormat.R32G32B32A32_SInt:
                    return DXGI_FORMAT.DXGI_FORMAT_R32G32B32A32_SINT;
                case GraphicsFormat.R32G32B32A32_SFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_R32G32B32A32_FLOAT;

                // --- Packed Formats ---
                case GraphicsFormat.R4G4B4A4_UNormPack16:
                    // 注意: DXGI 只有 A4B4G4R4 或 B4G4R4A4，这取决于具体的位布局是否匹配
                    return DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;
                case GraphicsFormat.B4G4R4A4_UNormPack16:
                    return DXGI_FORMAT.DXGI_FORMAT_B4G4R4A4_UNORM;

                case GraphicsFormat.R5G6B5_UNormPack16:
                    return DXGI_FORMAT.DXGI_FORMAT_UNKNOWN; // DXGI通常是 B5G6R5
                case GraphicsFormat.B5G6R5_UNormPack16:
                    return DXGI_FORMAT.DXGI_FORMAT_B5G6R5_UNORM;

                case GraphicsFormat.R5G5B5A1_UNormPack16:
                    return DXGI_FORMAT.DXGI_FORMAT_UNKNOWN; // DXGI通常是 B5G5R5A1
                case GraphicsFormat.B5G5R5A1_UNormPack16:
                    return DXGI_FORMAT.DXGI_FORMAT_B5G5R5A1_UNORM;

                case GraphicsFormat.E5B9G9R9_UFloatPack32:
                    return DXGI_FORMAT.DXGI_FORMAT_R9G9B9E5_SHAREDEXP;

                case GraphicsFormat.B10G11R11_UFloatPack32:
                    return DXGI_FORMAT.DXGI_FORMAT_R11G11B10_FLOAT; // 注意 Unity 和 DXGI 的 RG 顺序可能在命名上相反，但内存布局通常一致

                case GraphicsFormat.A2B10G10R10_UNormPack32:
                    return DXGI_FORMAT.DXGI_FORMAT_R10G10B10A2_UNORM;
                case GraphicsFormat.A2B10G10R10_UIntPack32:
                    return DXGI_FORMAT.DXGI_FORMAT_R10G10B10A2_UINT;

                // --- Depth / Stencil ---
                case GraphicsFormat.D16_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_D16_UNORM;
                case GraphicsFormat.D24_UNorm_S8_UInt:
                    return DXGI_FORMAT.DXGI_FORMAT_D24_UNORM_S8_UINT;
                case GraphicsFormat.D32_SFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_D32_FLOAT;
                case GraphicsFormat.D32_SFloat_S8_UInt:
                    return DXGI_FORMAT.DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
                case GraphicsFormat.S8_UInt:
                    // 通常映射到 R8_UINT 或特定的 Depth 格式，但在 DXGI 中通常没有纯 S8 格式作为 View
                    return DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;

                // --- Compressed Formats (BC / DXT) ---
                case GraphicsFormat.RGBA_DXT1_SRGB: // 曾用名 RGB_DXT1_SRGB
                    return DXGI_FORMAT.DXGI_FORMAT_BC1_UNORM_SRGB;
                case GraphicsFormat.RGBA_DXT1_UNorm: // 曾用名 RGB_DXT1_UNorm
                    return DXGI_FORMAT.DXGI_FORMAT_BC1_UNORM;

                case GraphicsFormat.RGBA_DXT3_SRGB:
                    return DXGI_FORMAT.DXGI_FORMAT_BC2_UNORM_SRGB;
                case GraphicsFormat.RGBA_DXT3_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_BC2_UNORM;

                case GraphicsFormat.RGBA_DXT5_SRGB:
                    return DXGI_FORMAT.DXGI_FORMAT_BC3_UNORM_SRGB;
                case GraphicsFormat.RGBA_DXT5_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_BC3_UNORM;

                case GraphicsFormat.R_BC4_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_BC4_UNORM;
                case GraphicsFormat.R_BC4_SNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_BC4_SNORM;

                case GraphicsFormat.RG_BC5_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_BC5_UNORM;
                case GraphicsFormat.RG_BC5_SNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_BC5_SNORM;

                case GraphicsFormat.RGB_BC6H_UFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_BC6H_UF16;
                case GraphicsFormat.RGB_BC6H_SFloat:
                    return DXGI_FORMAT.DXGI_FORMAT_BC6H_SF16;

                case GraphicsFormat.RGBA_BC7_SRGB:
                    return DXGI_FORMAT.DXGI_FORMAT_BC7_UNORM_SRGB;
                case GraphicsFormat.RGBA_BC7_UNorm:
                    return DXGI_FORMAT.DXGI_FORMAT_BC7_UNORM;

                // --- Video / YUV ---
                case GraphicsFormat.YUV2:
                    return DXGI_FORMAT.DXGI_FORMAT_YUY2;

                // --- Mobile / Unsupported on Desktop DXGI ---
                // PVRTC, ETC, ASTC 等格式在标准 DXGI 中不存在
                default:
                    return DXGI_FORMAT.DXGI_FORMAT_UNKNOWN;
            }
        }
    }
}