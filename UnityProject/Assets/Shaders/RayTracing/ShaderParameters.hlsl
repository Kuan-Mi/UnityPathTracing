
#ifndef SHADER_PARAMETERS_H
#define SHADER_PARAMETERS_H


#include <Assets/Shaders/Rtxdi/DI/ReSTIRDIParameters.h>
#include <Assets/Shaders/Rtxdi/GI/ReSTIRGIParameters.h>
#include <Assets/Shaders/Rtxdi/ReGIR/ReGIRParameters.h>


#define BACKGROUND_DEPTH 65504.f


struct ResamplingConstants
{
    RTXDI_RuntimeParameters runtimeParams;

    RTXDI_LightBufferParameters lightBufferParams;
    RTXDI_RISBufferSegmentParameters localLightsRISBufferSegmentParams;
    RTXDI_RISBufferSegmentParameters environmentLightRISBufferSegmentParams;

    ReSTIRDI_Parameters restirDI;
    ReGIR_Parameters regir;

    uint frameIndex;
    uint showReGIRCell;
    uint2 pad3;
    
    
    uint2 environmentPdfTextureSize;
    uint2 localLightPdfTextureSize;
};

#endif // SHADER_PARAMETERS_H