using System;
using System.Runtime.InteropServices;
using Nrd;


namespace Nrd
{
    public enum ResourceType : uint
    {
        //=============================================================================================================================
        // NON-NOISY INPUTS
        //=============================================================================================================================

        // 3D world-space motion (RGBA16f+) or 2D screen-space motion (RG16f+), MVs must be non-jittered, MV = previous - current
        IN_MV,

        // Data must match encoding in "NRD_FrontEnd_PackNormalAndRoughness" and "NRD_FrontEnd_UnpackNormalAndRoughness" (RGBA8+)
        IN_NORMAL_ROUGHNESS,

        // Linear view depth for primary rays (R16f+)
        IN_VIEWZ,

        // (Optional) User-provided history confidence in range 0-1, i.e. antilag (R8+)
        // Used only if "CommonSettings::isHistoryConfidenceAvailable = true" and "NRD_SUPPORTS_HISTORY_CONFIDENCE = 1"
        IN_DIFF_CONFIDENCE,
        IN_SPEC_CONFIDENCE,

        // (Optional) User-provided disocclusion threshold selector in range 0-1 (R8+)
        // Disocclusion threshold is mixed between "disocclusionThreshold" and "disocclusionThresholdAlternate"
        // Used only if "CommonSettings::isDisocclusionThresholdMixAvailable = true" and "NRD_SUPPORTS_DISOCCLUSION_THRESHOLD_MIX = 1"
        IN_DISOCCLUSION_THRESHOLD_MIX,

        // (Optional) Base color (can be decoupled to diffuse and specular albedo based on metalness) and metalness (RGBA8+)
        // Used only if "CommonSettings::isBaseColorMetalnessAvailable = true" and "NRD_SUPPORTS_BASECOLOR_METALNESS = 1".
        // Currently used only by REBLUR (if Temporal Stabilization pass is available and "stabilizationStrength != 0")
        // to patch MV if specular (virtual) motion prevails on diffuse (surface) motion
        IN_BASECOLOR_METALNESS,

        //=============================================================================================================================
        // NOISY INPUTS
        //=============================================================================================================================

        // Radiance and hit distance (RGBA16f+)
        //      REBLUR: use "REBLUR_FrontEnd_PackRadianceAndNormHitDist" for encoding
        //      RELAX: use "RELAX_FrontEnd_PackRadianceAndHitDist" for encoding
        IN_DIFF_RADIANCE_HITDIST,
        IN_SPEC_RADIANCE_HITDIST,

        // Hit distance (R8+)
        //      REBLUR: use "REBLUR_FrontEnd_GetNormHitDist" for encoding
        IN_DIFF_HITDIST,
        IN_SPEC_HITDIST,

        // Sampling direction and normalized hit distance (RGBA8+)
        //      REBLUR: use "REBLUR_FrontEnd_PackDirectionalOcclusion" for encoding
        IN_DIFF_DIRECTION_HITDIST,

        // SH data (2x RGBA16f+)
        //      REBLUR: use "REBLUR_FrontEnd_PackSh" for encoding
        //      RELAX: use "RELAX_FrontEnd_PackSh" for encoding
        IN_DIFF_SH0,
        IN_DIFF_SH1,
        IN_SPEC_SH0,
        IN_SPEC_SH1,

        // Penumbra and optional translucency (R16f+ and RGBA8+ for translucency)
        //      SIGMA: use "SIGMA_FrontEnd_PackPenumbra" for penumbra properties encoding
        //      SIGMA: use "SIGMA_FrontEnd_PackTranslucency" for translucency encoding
        IN_PENUMBRA,
        IN_TRANSLUCENCY,

        // Some signal (R8+)
        IN_SIGNAL,

        //=============================================================================================================================
        // OUTPUTS
        //=============================================================================================================================

        // IMPORTANT: Most of denoisers do not write into output pixels outside of "CommonSettings::denoisingRange"!

        // Radiance and normalized hit distance (occlusion) or history length
        //      REBLUR: use "REBLUR_BackEnd_UnpackRadianceAndNormHitDist" for decoding (R11G11B10f+)
        //          .w = diffuse or specular occlusion (default) or history length in frames if "ReblurSettings::returnHistoryLengthInsteadOfOcclusion = true"
        //      RELAX: use "RELAX_BackEnd_UnpackRadiance" for decoding (R11G11B10f+)
        //          .w = diffuse history length in frames
        OUT_DIFF_RADIANCE_HITDIST,
        OUT_SPEC_RADIANCE_HITDIST,

        // SH data
        //      REBLUR: use "REBLUR_BackEnd_UnpackSh" for decoding (2x RGBA16f+)
        //          .normHitDist = diffuse or specular occlusion (default) or history length in frames if "ReblurSettings::returnHistoryLengthInsteadOfOcclusion = true"
        //      RELAX: use "RELAX_BackEnd_UnpackSh" for decoding (2x RGBA16f+)
        //          .normHitDist = diffuse history length in frames
        OUT_DIFF_SH0,
        OUT_DIFF_SH1,
        OUT_SPEC_SH0,
        OUT_SPEC_SH1,

        // Normalized hit distance (R8+)
        OUT_DIFF_HITDIST,
        OUT_SPEC_HITDIST,

        // Bent normal and normalized hit distance (RGBA8+)
        //      REBLUR: use "REBLUR_BackEnd_UnpackDirectionalOcclusion" for decoding
        OUT_DIFF_DIRECTION_HITDIST,

        // Shadow and optional transcluceny (R8+ or RGBA8+)
        //      SIGMA: use "SIGMA_BackEnd_UnpackShadow" for decoding
        OUT_SHADOW_TRANSLUCENCY, // IMPORTANT: used as history if "stabilizationStrength != 0"

        // Denoised signal (R8+)
        OUT_SIGNAL,

        // (Optional) Debug output (RGBA8+), .w = transparency
        // Used if "CommonSettings::enableValidation = true"
        OUT_VALIDATION,

        //=============================================================================================================================
        // POOLS
        //=============================================================================================================================

        // Can be reused after denoising
        TRANSIENT_POOL,

        // Dedicated to NRD, can't be reused
        PERMANENT_POOL,

        MAX_NUM,
    };
}

namespace Nri
{
    [Flags]
    public enum AccessBits : uint
    {
        NONE = 0, // Mapped to "COMMON" (aka "GENERAL" access), if AgilitySDK is not available, leading to potential discrepancies with VK

        // Buffer                                // Access  Compatible "StageBits" (including ALL)
        INDEX_BUFFER = (1 << 0), // R   INDEX_INPUT
        VERTEX_BUFFER = (1 << 1), // R   VERTEX_SHADER
        CONSTANT_BUFFER = (1 << 2), // R   GRAPHICS_SHADERS, COMPUTE_SHADER, RAY_TRACING_SHADERS
        ARGUMENT_BUFFER = (1 << 3), // R   INDIRECT
        SCRATCH_BUFFER = (1 << 4), // RW  ACCELERATION_STRUCTURE, MICROMAP

        // Attachment
        COLOR_ATTACHMENT = (1 << 5), // RW  COLOR_ATTACHMENT
        SHADING_RATE_ATTACHMENT = (1 << 6), // R   FRAGMENT_SHADER
        DEPTH_STENCIL_ATTACHMENT_READ = (1 << 7), // R   DEPTH_STENCIL_ATTACHMENT
        DEPTH_STENCIL_ATTACHMENT_WRITE = (1 << 8), //  W  DEPTH_STENCIL_ATTACHMENT

        // Acceleration structure
        ACCELERATION_STRUCTURE_READ = (1 << 9), // R   COMPUTE_SHADER, RAY_TRACING_SHADERS, ACCELERATION_STRUCTURE
        ACCELERATION_STRUCTURE_WRITE = (1 << 10), //  W  ACCELERATION_STRUCTURE

        // Micromap
        MICROMAP_READ = (1 << 11), // R   MICROMAP, ACCELERATION_STRUCTURE
        MICROMAP_WRITE = (1 << 12), //  W  MICROMAP

        // Shader resource
        SHADER_RESOURCE = (1 << 13), // R   GRAPHICS_SHADERS, COMPUTE_SHADER, RAY_TRACING_SHADERS
        SHADER_RESOURCE_STORAGE = (1 << 14), // RW  GRAPHICS_SHADERS, COMPUTE_SHADER, RAY_TRACING_SHADERS, CLEAR_STORAGE + shaders
        SHADER_BINDING_TABLE = (1 << 15), // R   RAY_TRACING_SHADERS

        // Copy
        COPY_SOURCE = (1 << 16), // R   COPY
        COPY_DESTINATION = (1 << 17), //  W  COPY

        // Resolve
        RESOLVE_SOURCE = (1 << 18), // R   RESOLVE
        RESOLVE_DESTINATION = (1 << 19), //  W  RESOLVE

        // Clear storage
        CLEAR_STORAGE = (1 << 20) //  W  CLEAR_STORAGE
    }

    public enum Layout : uint
    {
        // Compatible "AccessBits":
        // Special
        UNDEFINED, // https://microsoft.github.io/DirectX-Specs/d3d/D3D12EnhancedBarriers.html#d3d12_barrier_layout_undefined
        GENERAL, // ~ALL access, but potentially not optimal (required for "SharingMode::SIMULTANEOUS")
        PRESENT, // NONE (use "after.stages = StageBits::NONE")

        // Access specific
        COLOR_ATTACHMENT, // COLOR_ATTACHMENT
        SHADING_RATE_ATTACHMENT, // SHADING_RATE_ATTACHMENT
        DEPTH_STENCIL_ATTACHMENT, // DEPTH_STENCIL_ATTACHMENT_WRITE
        DEPTH_STENCIL_READONLY, // DEPTH_STENCIL_ATTACHMENT_READ, SHADER_RESOURCE
        SHADER_RESOURCE, // SHADER_RESOURCE
        SHADER_RESOURCE_STORAGE, // SHADER_RESOURCE_STORAGE
        COPY_SOURCE, // COPY_SOURCE
        COPY_DESTINATION, // COPY_DESTINATION
        RESOLVE_SOURCE, // RESOLVE_SOURCE
        RESOLVE_DESTINATION // RESOLVE_DESTINATION
    }
    
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct NriResourceState
    {
        public AccessBits accessBits;
        public Layout layout;
        public uint stageBits;
    }
    
    [Serializable]
    [StructLayout(LayoutKind.Sequential,Pack = 1)]
    public struct NrdResourceInput
    {
        public ResourceType type;
        public IntPtr texture;
        public NriResourceState state;
    }
}