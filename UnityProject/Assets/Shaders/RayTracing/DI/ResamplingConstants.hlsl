
#include "Assets/Shaders/Rtxdi/DI/ReSTIRDIParameters.h"

struct ResamplingConstants
{
    RTXDI_RuntimeParameters runtimeParams;
    
    RTXDI_LightBufferParameters lightBufferParams;
    RTXDI_ReservoirBufferParameters restirDIReservoirBufferParams;
    
    ReSTIRDI_Parameters restirDI;
    
    uint frameIndex;
    uint numInitialSamples;
    uint numSpatialSamples;
    uint useAccurateGBufferNormal;

    uint numInitialBRDFSamples;
    float brdfCutoff;
    uint2 pad2;

    uint enableResampling;
    uint unbiasedMode;
    uint inputBufferIndex;
    uint outputBufferIndex;
};

RWStructuredBuffer<ResamplingConstants> ResampleConstants;
#define g_Const ResampleConstants[0]

// RTXDI resources
StructuredBuffer<RAB_LightInfo> t_LightDataBuffer;
Buffer<float2> t_NeighborOffsets;

RWStructuredBuffer<RTXDI_PackedDIReservoir> u_LightReservoirs;

#define RTXDI_LIGHT_RESERVOIR_BUFFER u_LightReservoirs
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER t_NeighborOffsets

#define BACKGROUND_DEPTH 65504.f

#define RTXDI_ENABLE_PRESAMPLING 0