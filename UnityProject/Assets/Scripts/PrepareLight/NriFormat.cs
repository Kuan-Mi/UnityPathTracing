namespace DefaultNamespace
{
    public enum NriFormat : byte
    {
        UNKNOWN, // -  -  -  -  -  -  -  -  -  -

        // Plain: 8 bits per channel
        R8_UNORM, // +  +  +  -  +  -  +  +  +  -
        R8_SNORM, // +  +  +  -  +  -  +  +  +  -
        R8_UINT, // +  +  +  -  -  -  +  +  +  - // SHADING_RATE compatible, see NRI_SHADING_RATE macro
        R8_SINT, // +  +  +  -  -  -  +  +  +  -

        RG8_UNORM, // +  +  +  -  +  -  +  +  +  - // "AccelerationStructure" compatible (requires "tiers.rayTracing >= 2")
        RG8_SNORM, // +  +  +  -  +  -  +  +  +  - // "AccelerationStructure" compatible (requires "tiers.rayTracing >= 2")
        RG8_UINT, // +  +  +  -  -  -  +  +  +  -
        RG8_SINT, // +  +  +  -  -  -  +  +  +  -

        BGRA8_UNORM, // +  +  +  -  +  -  +  +  +  -
        BGRA8_SRGB, // +  -  +  -  +  -  -  -  -  -

        RGBA8_UNORM, // +  +  +  -  +  -  +  +  +  - // "AccelerationStructure" compatible (requires "tiers.rayTracing >= 2")
        RGBA8_SRGB, // +  -  +  -  +  -  -  -  -  -
        RGBA8_SNORM, // +  +  +  -  +  -  +  +  +  - // "AccelerationStructure" compatible (requires "tiers.rayTracing >= 2")
        RGBA8_UINT, // +  +  +  -  -  -  +  +  +  -
        RGBA8_SINT, // +  +  +  -  -  -  +  +  +  -

        // Plain: 16 bits per channel
        R16_UNORM, // +  +  +  -  +  -  +  +  +  -
        R16_SNORM, // +  +  +  -  +  -  +  +  +  -
        R16_UINT, // +  +  +  -  -  -  +  +  +  -
        R16_SINT, // +  +  +  -  -  -  +  +  +  -
        R16_SFLOAT, // +  +  +  -  +  -  +  +  +  -

        RG16_UNORM, // +  +  +  -  +  -  +  +  +  - // "AccelerationStructure" compatible (requires "tiers.rayTracing >= 2")
        RG16_SNORM, // +  +  +  -  +  -  +  +  +  - // "AccelerationStructure" compatible
        RG16_UINT, // +  +  +  -  -  -  +  +  +  -
        RG16_SINT, // +  +  +  -  -  -  +  +  +  -
        RG16_SFLOAT, // +  +  +  -  +  -  +  +  +  - // "AccelerationStructure" compatible

        RGBA16_UNORM, // +  +  +  -  +  -  +  +  +  - // "AccelerationStructure" compatible (requires "tiers.rayTracing >= 2")
        RGBA16_SNORM, // +  +  +  -  +  -  +  +  +  - // "AccelerationStructure" compatible
        RGBA16_UINT, // +  +  +  -  -  -  +  +  +  -
        RGBA16_SINT, // +  +  +  -  -  -  +  +  +  -
        RGBA16_SFLOAT, // +  +  +  -  +  -  +  +  +  - // "AccelerationStructure" compatible

        // Plain: 32 bits per channel
        R32_UINT, // +  +  +  -  -  +  +  +  +  +
        R32_SINT, // +  +  +  -  -  +  +  +  +  +
        R32_SFLOAT, // +  +  +  -  +  +  +  +  +  +

        RG32_UINT, // +  +  +  -  -  -  +  +  +  -
        RG32_SINT, // +  +  +  -  -  -  +  +  +  -
        RG32_SFLOAT, // +  +  +  -  +  -  +  +  +  - // "AccelerationStructure" compatible

        RGB32_UINT, // +  -  -  -  -  -  +  -  +  -
        RGB32_SINT, // +  -  -  -  -  -  +  -  +  -
        RGB32_SFLOAT, // +  -  -  -  -  -  +  -  +  - // "AccelerationStructure" compatible

        RGBA32_UINT, // +  +  +  -  -  -  +  +  +  -
        RGBA32_SINT, // +  +  +  -  -  -  +  +  +  -
        RGBA32_SFLOAT, // +  +  +  -  +  -  +  +  +  -

        // Packed: 16 bits per pixel
        B5_G6_R5_UNORM, // +  -  +  -  +  -  -  -  -  -
        B5_G5_R5_A1_UNORM, // +  -  +  -  +  -  -  -  -  -
        B4_G4_R4_A4_UNORM, // +  -  +  -  +  -  -  -  -  -

        // Packed: 32 bits per pixel
        R10_G10_B10_A2_UNORM, // +  +  +  -  +  -  +  +  +  - // "AccelerationStructure" compatible (requires "tiers.rayTracing >= 2")
        R10_G10_B10_A2_UINT, // +  +  +  -  -  -  +  +  +  -
        R11_G11_B10_UFLOAT, // +  +  +  -  +  -  +  +  +  -
        R9_G9_B9_E5_UFLOAT, // +  -  -  -  -  -  -  -  -  -

        // Block-compressed
        BC1_RGBA_UNORM, // +  -  -  -  -  -  -  -  -  -
        BC1_RGBA_SRGB, // +  -  -  -  -  -  -  -  -  -
        BC2_RGBA_UNORM, // +  -  -  -  -  -  -  -  -  -
        BC2_RGBA_SRGB, // +  -  -  -  -  -  -  -  -  -
        BC3_RGBA_UNORM, // +  -  -  -  -  -  -  -  -  -
        BC3_RGBA_SRGB, // +  -  -  -  -  -  -  -  -  -
        BC4_R_UNORM, // +  -  -  -  -  -  -  -  -  -
        BC4_R_SNORM, // +  -  -  -  -  -  -  -  -  -
        BC5_RG_UNORM, // +  -  -  -  -  -  -  -  -  -
        BC5_RG_SNORM, // +  -  -  -  -  -  -  -  -  -
        BC6H_RGB_UFLOAT, // +  -  -  -  -  -  -  -  -  -
        BC6H_RGB_SFLOAT, // +  -  -  -  -  -  -  -  -  -
        BC7_RGBA_UNORM, // +  -  -  -  -  -  -  -  -  -
        BC7_RGBA_SRGB, // +  -  -  -  -  -  -  -  -  -

        // Depth-stencil
        D16_UNORM, // -  -  -  +  -  -  -  -  -  -
        D24_UNORM_S8_UINT, // -  -  -  +  -  -  -  -  -  -
        D32_SFLOAT, // -  -  -  +  -  -  -  -  -  -
        D32_SFLOAT_S8_UINT_X24, // -  -  -  +  -  -  -  -  -  -

        // Depth-stencil (SHADER_RESOURCE)
        R24_UNORM_X8, // .x - depth    // +  -  -  -  -  -  -  -  -  -
        X24_G8_UINT, // .y - stencil  // +  -  -  -  -  -  -  -  -  -
        R32_SFLOAT_X8_X24, // .x - depth    // +  -  -  -  -  -  -  -  -  -
        X32_G8_UINT_X24 // .y - stencil  // +  -  -  -  -  -  -  -  -  -
    }
}