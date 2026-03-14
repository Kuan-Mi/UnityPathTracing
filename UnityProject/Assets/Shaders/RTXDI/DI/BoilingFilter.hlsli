#ifndef RTXDI_DI_BOILING_FILTER_HLSLI
#define RTXDI_DI_BOILING_FILTER_HLSLI

#include "Rtxdi/DI/Reservoir.hlsli"
#include "Rtxdi/Utils/BoilingFilter.hlsli"

#ifdef RTXDI_ENABLE_BOILING_FILTER
// Boiling filter that should be applied at the end of the temporal resampling pass.
// Can be used inside the same shader that does temporal resampling if it's a compute shader,
// or in a separate pass if temporal resampling is a raygen shader.
// The filter analyzes the weights of all reservoirs in a thread group, and discards
// the reservoirs whose weights are very high, i.e. above a certain threshold.
void RTXDI_BoilingFilter(
    uint2 LocalIndex,
    float filterStrength, // (0..1]
    inout RTXDI_DIReservoir reservoir)
{
    if (RTXDI_BoilingFilterInternal(LocalIndex, filterStrength, reservoir.weightSum))
        reservoir = RTXDI_EmptyDIReservoir();
}
#endif // RTXDI_ENABLE_BOILING_FILTER

#endif
