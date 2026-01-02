#pragma once
#include <cstdint>
#include <d3d12.h>
#include <NRD.h>
#include <NRDSettings.h>
#include <NRIDescs.h>


    enum class MyResourceType : uint32_t
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
        
        TaaHistory,
        TaaHistoryPrev,
        PsrThroughput,
        
        // dlss
        RRGuide_DiffAlbedo,
        RRGuide_SpecAlbedo,
        RRGuide_SpecHitDistance,
        RRGuide_Normal_Roughness,
        DlssOutput,
        Composed,

        MAX_NUM,
    };



#pragma pack(push, 1)
namespace nri
{
    struct Texture;
}

struct FrameData
{
    nrd::CommonSettings commonSettings;
    nrd::SigmaSettings sigmaSettings;
    nrd::ReblurSettings reblurSettings;

    uint16_t width;
    uint16_t height;

    int instanceId;
};

struct NriResourceState
{
    nri::AccessBits accessBits;
    uint32_t layout;
    nri::StageBits stageBits;
};

struct NrdResourceInput
{
    nrd::ResourceType type;
    nri::Texture* texture;
    NriResourceState state;
};
#pragma pack(pop)
