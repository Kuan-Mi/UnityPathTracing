/***************************************************************************
 # Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

#ifndef RTXDI_DI_SPATIOTEMPORAL_RESAMPLIHG_HLSLI
#define RTXDI_DI_SPATIOTEMPORAL_RESAMPLIHG_HLSLI

#include "Rtxdi/DI/PairwiseStreaming.hlsli"
#include "Rtxdi/DI/Reservoir.hlsli"
#include "Rtxdi/Utils/Checkerboard.hlsli"

#ifndef RTXDI_NEIGHBOR_OFFSETS_BUFFER
#error "RTXDI_NEIGHBOR_OFFSETS_BUFFER must be defined to point to a Buffer<float2> type resource"
#endif

// This macro can be defined in the including shader file to reduce code bloat
// and/or remove ray tracing calls from temporal and spatial resampling shaders
// if bias correction is not necessary.
#ifndef RTXDI_ALLOWED_BIAS_CORRECTION
#define RTXDI_ALLOWED_BIAS_CORRECTION RTXDI_BIAS_CORRECTION_RAY_TRACED
#endif

// A structure that groups the application-provided settings for spatio-temporal resampling.
struct RTXDI_DISpatioTemporalResamplingParameters
{
    // Screen-space motion vector, computed as (previousPosition - currentPosition).
    // The X and Y components are measured in pixels.
    // The Z component is in linear depth units.
    float3 screenSpaceMotion;

    // The index of the reservoir buffer to pull the temporal samples from.
    uint sourceBufferIndex;

    // Maximum history length for temporal reuse, measured in frames.
    // Higher values result in more stable and high quality sampling, at the cost of slow reaction to changes.
    uint maxHistoryLength;

    // Controls the bias correction math for temporal reuse. Depending on the setting, it can add
    // some shader cost and one approximate shadow ray per pixel (or per two pixels if checkerboard sampling is enabled).
    // Ideally, these rays should be traced through the previous frame's BVH to get fully unbiased results.
    uint biasCorrectionMode;

    // Surface depth similarity threshold for temporal reuse.
    // If the previous frame surface's depth is within this threshold from the current frame surface's depth,
    // the surfaces are considered similar. The threshold is relative, i.e. 0.1 means 10% of the current depth.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float depthThreshold;

    // Surface normal similarity threshold for temporal reuse.
    // If the dot product of two surfaces' normals is higher than this threshold, the surfaces are considered similar.
    // Otherwise, the pixel is not reused, and the resampling shader will look for a different one.
    float normalThreshold;

    // Number of neighbor pixels considered for resampling (1-32)
    // Some of the may be skipped if they fail the surface similarity test.
    uint numSamples;

    // Number of neighbor pixels considered when there is no temporal surface (1-32)
    // Setting this parameter equal or lower than `numSpatialSamples` effectively
    // disables the disocclusion boost.
    uint numDisocclusionBoostSamples;

    // Screen-space radius for spatial resampling, measured in pixels.
    float samplingRadius;

    // Allows the temporal resampling logic to skip the bias correction ray trace for light samples
    // reused from the previous frame. Only safe to use when invisible light samples are discarded
    // on the previous frame, then any sample coming from the previous frame can be assumed visible.
    bool enableVisibilityShortcut;

    // Enables permuting the pixels sampled from the previous frame in order to add temporal
    // variation to the output signal and make it more denoiser friendly.
    bool enablePermutationSampling;

    // Enables the comparison of surface materials before taking a surface into resampling.
    bool enableMaterialSimilarityTest;

    // Prevents samples which are from the current frame or have no reasonable temporal history merged being spread to neighbors
    bool discountNaiveSamples;

    // Random number for permutation sampling that is the same for all pixels in the frame
    uint uniformRandomNumber;
};

// Fused spatialtemporal resampling pass, using pairwise MIS.  
// Inputs and outputs equivalent to RTXDI_SpatioTemporalResampling(), but only uses pairwise MIS.
// Can call this directly, or call RTXDI_SpatioTemporalResampling() with sparams.biasCorrectionMode 
// set to RTXDI_BIAS_CORRECTION_PAIRWISE, which simply calls this function.
RTXDI_DIReservoir RTXDI_DISpatioTemporalResamplingWithPairwiseMIS(
    uint2 pixelPosition,
    RAB_Surface surface,
    RTXDI_DIReservoir curSample,
    inout RAB_RandomSamplerState rng,
    RTXDI_RuntimeParameters params,
    RTXDI_ReservoirBufferParameters reservoirParams,
    RTXDI_DISpatioTemporalResamplingParameters stparams,
    out int2 temporalSamplePixelPos,
    inout RAB_LightSample selectedLightSample)
{
    uint historyLimit = min(RTXDI_PackedDIReservoir_MaxM, uint(stparams.maxHistoryLength * curSample.M));

    // Backproject this pixel to last frame
    float3 motion = stparams.screenSpaceMotion;
    if (!stparams.enablePermutationSampling)
    {
        motion.xy += float2(RAB_GetNextRandom(rng), RAB_GetNextRandom(rng)) - 0.5;
    }
    int2 prevPos = int2(round(float2(pixelPosition)+motion.xy));
    float expectedPrevLinearDepth = RAB_GetSurfaceLinearDepth(surface) + motion.z;

    // Some default initializations
    temporalSamplePixelPos = int2(-1, -1);
    RAB_Surface temporalSurface = RAB_EmptySurface();
    bool foundTemporalSurface = false;                                                 // Found a valid backprojection?
    const float temporalSearchRadius = (params.activeCheckerboardField == 0) ? 4 : 8;  // How far to search for a match when backprojecting
    int2 temporalSpatialOffset = int2(0, 0);                                           // Offset for the (central) backprojected pixel

    // Try to find a matching surface in the neighborhood of the centrol reprojected pixel
    int i;
    int2 centralIdx;
    for (i = 0; i < 9; i++)
    {
        int2 offset = int2(0, 0);
        offset.x = (i > 0) ? int((RAB_GetNextRandom(rng) - 0.5) * temporalSearchRadius) : 0;
        offset.y = (i > 0) ? int((RAB_GetNextRandom(rng) - 0.5) * temporalSearchRadius) : 0;

        centralIdx = prevPos + offset;
        if (stparams.enablePermutationSampling && i == 0)
        {
            RTXDI_ApplyPermutationSampling(centralIdx, stparams.uniformRandomNumber);
        }

        RTXDI_ActivateCheckerboardPixel(centralIdx, true, params.activeCheckerboardField);

        // Grab shading / g-buffer data from last frame
        temporalSurface = RAB_GetGBufferSurface(centralIdx, true);
        if (!RAB_IsSurfaceValid(temporalSurface))
            continue;

        // Test surface similarity, discard the sample if the surface is too different.
        if (!RTXDI_IsValidNeighbor(
            RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(temporalSurface),
            expectedPrevLinearDepth, RAB_GetSurfaceLinearDepth(temporalSurface),
            stparams.normalThreshold, stparams.depthThreshold))
            continue;

        temporalSpatialOffset = centralIdx - prevPos;
        foundTemporalSurface = true;
        break;
    }

    // How many spatial samples to use?  
    uint numSpatialSamples = (!foundTemporalSurface)
        ? max(stparams.numDisocclusionBoostSamples, stparams.numSamples)
        : uint(int(stparams.numSamples));

    // Count how many of our spatiotemporal samples are valid and streamed via RIS
    int validSamples = 0;

    // Create an empty reservoir we'll use to accumulate into
    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    state.canonicalWeight = 0.0f;    // Important this is 0 for temporal

    // Load the "temporal" reservoir at the temporally backprojected "central" pixel
    RTXDI_DIReservoir prevSample = RTXDI_LoadDIReservoir(reservoirParams,
        RTXDI_PixelPosToReservoirPos(centralIdx, params.activeCheckerboardField), stparams.sourceBufferIndex);
    prevSample.M = min(prevSample.M, historyLimit);
    prevSample.spatialDistance += temporalSpatialOffset;
    prevSample.age += 1;

    // Find the prior frame's light in the current frame
    int mappedLightID = RAB_TranslateLightIndex(RTXDI_GetDIReservoirLightIndex(prevSample), false);

    // Kill the reservoir if it doesn't exist in the current frame, otherwise update its ID for this frame
    prevSample.weightSum = (mappedLightID < 0) ? 0 : prevSample.weightSum;
    prevSample.lightData = (mappedLightID < 0) ? 0 : mappedLightID | RTXDI_DIReservoir_LightValidBit;

    // If we found a valid surface by backprojecting our current pixel, stream it through the reservoir.
    if (foundTemporalSurface && prevSample.M > 0)
    {
        ++validSamples;

        // Pass out the temporal sample location
        temporalSamplePixelPos = (prevSample.age <= 1) ? centralIdx : temporalSamplePixelPos;

        // Stream this light through the reservoir using pairwise MIS
        RTXDI_StreamNeighborWithPairwiseMIS(state, RAB_GetNextRandom(rng),
            prevSample, temporalSurface,    // The temporal neighbor
            curSample, surface,             // The canonical neighbor
            1 + numSpatialSamples);
    }

    // Look for valid (spatiotemporal) neighbors and stream them through the reservoir via pairwise MIS
    uint startIdx = uint(RAB_GetNextRandom(rng) * params.neighborOffsetMask);
    for (i = 1; i < numSpatialSamples; ++i)
    {
        uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;
        int2 spatialOffset = int2(float2(RTXDI_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * stparams.samplingRadius);
        int2 idx = prevPos + spatialOffset;

        if (idx.x < 0 || idx.y < 0)
            continue;

        idx = RAB_ClampSamplePositionIntoView(idx, false);
        
        RTXDI_ActivateCheckerboardPixel(idx, false, params.activeCheckerboardField);

        RAB_Surface neighborSurface = RAB_GetGBufferSurface(idx, true);

        // Check for surface / G-buffer matches between the canonical sample and this neighbor
        if (!RAB_IsSurfaceValid(neighborSurface))
            continue;

        if (!RTXDI_IsValidNeighbor(RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(neighborSurface),
            RAB_GetSurfaceLinearDepth(surface), RAB_GetSurfaceLinearDepth(neighborSurface),
            stparams.normalThreshold, stparams.depthThreshold))
            continue;

        if (stparams.enableMaterialSimilarityTest && !RAB_AreMaterialsSimilar(RAB_GetMaterial(surface), RAB_GetMaterial(neighborSurface)))
            continue;

        // The surfaces are similar enough so we *can* reuse a neighbor from this pixel, so load it.
        RTXDI_DIReservoir neighborSample = RTXDI_LoadDIReservoir(reservoirParams,
            RTXDI_PixelPosToReservoirPos(idx, params.activeCheckerboardField), stparams.sourceBufferIndex);

        if (RTXDI_IsValidDIReservoir(prevSample))
        {
            if (stparams.discountNaiveSamples && neighborSample.M <= RTXDI_NAIVE_SAMPLING_M_THRESHOLD)
                continue;
        }

        neighborSample.M = min(neighborSample.M, historyLimit);
        neighborSample.spatialDistance += spatialOffset;
        neighborSample.age += 1;

        // Find the this neighbors light in the current frame (it may have turned off or moved in the ID list)
        int mappedLightID = RAB_TranslateLightIndex(RTXDI_GetDIReservoirLightIndex(neighborSample), false);

        // Kill the sample if the light doesn't exist in the current frame, otherwise update its ID for this frame
        neighborSample.weightSum = (mappedLightID < 0) ? 0 : neighborSample.weightSum;
        neighborSample.lightData = (mappedLightID < 0) ? 0 : mappedLightID | RTXDI_DIReservoir_LightValidBit;

        if (mappedLightID < 0) continue;

        ++validSamples;

        // If sample has weight 0 due to visibility (or etc), skip the expensive-ish MIS computations
        if (neighborSample.M <= 0) continue;

        // Stream this light through the reservoir using pairwise MIS
        RTXDI_StreamNeighborWithPairwiseMIS(state, RAB_GetNextRandom(rng),
            neighborSample, neighborSurface,   // The spatial neighbor
            curSample, surface,                // The canonical (center) sample
            1 + numSpatialSamples);
    }

    // Stream the canonical sample (i.e., from prior computations at this pixel in this frame) using pairwise MIS.
    RTXDI_StreamCanonicalWithPairwiseStep(state, RAB_GetNextRandom(rng),
        curSample, surface);

    // Renormalize the reservoir so it can be stored in a packed format 
    RTXDI_FinalizeResampling(state, 1.0f, float(max(1, validSamples)));

    // Return the selected light sample.  This is a redundant lookup and could be optimized away by storing
    // the selected sample from the stream steps above.
    selectedLightSample = RAB_SamplePolymorphicLight(
        RAB_LoadLightInfo(RTXDI_GetDIReservoirLightIndex(state), false),
        surface, RTXDI_GetDIReservoirSampleUV(state));

    return state;
}


// Spatio-temporal resampling pass.
// A combination of the temporal and spatial passes that operates only on the previous frame reservoirs.
// The selectedLightSample parameter is used to update and return the selected sample; it's optional,
// and it's safe to pass a null structure there and ignore the result.
RTXDI_DIReservoir RTXDI_DISpatioTemporalResampling(
    uint2 pixelPosition,
    RAB_Surface surface,
    RTXDI_DIReservoir curSample,
    inout RAB_RandomSamplerState rng,
    RTXDI_RuntimeParameters params,
    RTXDI_ReservoirBufferParameters reservoirParams,
    RTXDI_DISpatioTemporalResamplingParameters stparams,
    out int2 temporalSamplePixelPos,
    inout RAB_LightSample selectedLightSample)
{
    if (stparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_PAIRWISE)
    {
        return RTXDI_DISpatioTemporalResamplingWithPairwiseMIS(pixelPosition, surface,
            curSample, rng, params, reservoirParams, stparams, temporalSamplePixelPos, selectedLightSample);
    }

    uint historyLimit = min(RTXDI_PackedDIReservoir_MaxM, uint(stparams.maxHistoryLength * curSample.M));

    int selectedLightPrevID = -1;

    if (RTXDI_IsValidDIReservoir(curSample))
    {
        selectedLightPrevID = RAB_TranslateLightIndex(RTXDI_GetDIReservoirLightIndex(curSample), true);
    }

    temporalSamplePixelPos = int2(-1, -1);

    RTXDI_DIReservoir state = RTXDI_EmptyDIReservoir();
    RTXDI_CombineDIReservoirs(state, curSample, /* random = */ 0.5, curSample.targetPdf);

    uint startIdx = uint(RAB_GetNextRandom(rng) * params.neighborOffsetMask);

    // Backproject this pixel to last frame
    float3 motion = stparams.screenSpaceMotion;

    if (!stparams.enablePermutationSampling)
    {
        motion.xy += float2(RAB_GetNextRandom(rng), RAB_GetNextRandom(rng)) - 0.5;
    }

    float2 reprojectedSamplePosition = float2(pixelPosition) + motion.xy;
    int2 prevPos = int2(round(reprojectedSamplePosition));

    float expectedPrevLinearDepth = RAB_GetSurfaceLinearDepth(surface) + motion.z;

    int i;

    RAB_Surface temporalSurface = RAB_EmptySurface();
    bool foundTemporalSurface = false;
    const float temporalSearchRadius = (params.activeCheckerboardField == 0) ? 4 : 8;
    int2 temporalSpatialOffset = int2(0, 0);

    // Try to find a matching surface in the neighborhood of the reprojected pixel
    for (i = 0; i < 9; i++)
    {
        int2 offset = int2(0, 0);
        if (i > 0)
        {
            offset.x = int((RAB_GetNextRandom(rng) - 0.5) * temporalSearchRadius);
            offset.y = int((RAB_GetNextRandom(rng) - 0.5) * temporalSearchRadius);
        }

        int2 idx = prevPos + offset;

        if (stparams.enablePermutationSampling && i == 0)
        {
            RTXDI_ApplyPermutationSampling(idx, stparams.uniformRandomNumber);
        }

        RTXDI_ActivateCheckerboardPixel(idx, true, params.activeCheckerboardField);

        // Grab shading / g-buffer data from last frame
        temporalSurface = RAB_GetGBufferSurface(idx, true);
        if (!RAB_IsSurfaceValid(temporalSurface))
            continue;
        
        // Test surface similarity, discard the sample if the surface is too different.
        if (!RTXDI_IsValidNeighbor(
            RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(temporalSurface), 
            expectedPrevLinearDepth, RAB_GetSurfaceLinearDepth(temporalSurface), 
            stparams.normalThreshold, stparams.depthThreshold))
            continue;

        temporalSpatialOffset = idx - prevPos;
        foundTemporalSurface = true;
        break;
    }

    // Clamp the sample count at 32 to make sure we can keep the neighbor mask in an uint (cachedResult)
    uint numSamples = clamp(stparams.numSamples, 1, 32);

    // Apply disocclusion boost if there is no temporal surface
    if (!foundTemporalSurface)
        numSamples = clamp(stparams.numDisocclusionBoostSamples, numSamples, 32);

    // We loop through neighbors twice.  Cache the validity / edge-stopping function
    //   results for the 2nd time through.
    uint cachedResult = 0;

    // Since we're using our bias correction scheme, we need to remember which light selection we made
    int selected = -1;

    // Walk the specified number of neighbors, resampling using RIS
    for (i = 0; i < numSamples; ++i)
    {
        int2 spatialOffset, idx;

        // Get screen-space location of neighbor
        if (i == 0 && foundTemporalSurface)
        {
            spatialOffset = temporalSpatialOffset;
            idx = prevPos + spatialOffset;
        }
        else
        {
            uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;
            spatialOffset = int2(float2(RTXDI_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * stparams.samplingRadius);

            idx = prevPos + spatialOffset;

            idx = RAB_ClampSamplePositionIntoView(idx, true);

            RTXDI_ActivateCheckerboardPixel(idx, true, params.activeCheckerboardField);

            temporalSurface = RAB_GetGBufferSurface(idx, true);

            if (!RAB_IsSurfaceValid(temporalSurface))
                continue;

            if (!RTXDI_IsValidNeighbor(RAB_GetSurfaceNormal(surface), RAB_GetSurfaceNormal(temporalSurface), 
                RAB_GetSurfaceLinearDepth(surface), RAB_GetSurfaceLinearDepth(temporalSurface), 
                stparams.normalThreshold, stparams.depthThreshold))
                continue;

            if (stparams.enableMaterialSimilarityTest && !RAB_AreMaterialsSimilar(RAB_GetMaterial(surface), RAB_GetMaterial(temporalSurface)))
                continue;
        }
        
        cachedResult |= (1u << uint(i));

        uint2 neighborReservoirPos = RTXDI_PixelPosToReservoirPos(idx, params.activeCheckerboardField);

        RTXDI_DIReservoir prevSample = RTXDI_LoadDIReservoir(reservoirParams,
            neighborReservoirPos, stparams.sourceBufferIndex);

        if (RTXDI_IsValidDIReservoir(prevSample))
        {
            if (stparams.discountNaiveSamples && prevSample.M <= RTXDI_NAIVE_SAMPLING_M_THRESHOLD)
                continue;
        }

        prevSample.M = min(prevSample.M, historyLimit);
        prevSample.spatialDistance += spatialOffset;
        prevSample.age += 1;

        uint originalPrevLightID = RTXDI_GetDIReservoirLightIndex(prevSample);

        // Map the light ID from the previous frame into the current frame, if it still exists
        if (RTXDI_IsValidDIReservoir(prevSample))
        {   
            if (i == 0 && foundTemporalSurface && prevSample.age <= 1)
            {
                temporalSamplePixelPos = idx;
            }

            int mappedLightID = RAB_TranslateLightIndex(RTXDI_GetDIReservoirLightIndex(prevSample), false);

            if (mappedLightID < 0)
            {
                // Kill the reservoir
                prevSample.weightSum = 0;
                prevSample.lightData = 0;
            }
            else
            {
                // Sample is valid - modify the light ID stored
                prevSample.lightData = mappedLightID | RTXDI_DIReservoir_LightValidBit;
            }
        }

        RAB_LightInfo candidateLight;

        // Load that neighbor's RIS state, do resampling
        float neighborWeight = 0;
        RAB_LightSample candidateLightSample = RAB_EmptyLightSample();
        if (RTXDI_IsValidDIReservoir(prevSample))
        {   
            candidateLight = RAB_LoadLightInfo(RTXDI_GetDIReservoirLightIndex(prevSample), false);
            
            candidateLightSample = RAB_SamplePolymorphicLight(
                candidateLight, surface, RTXDI_GetDIReservoirSampleUV(prevSample));
            
            neighborWeight = RAB_GetLightSampleTargetPdfForSurface(candidateLightSample, surface);
        }

        if (RTXDI_CombineDIReservoirs(state, prevSample, RAB_GetNextRandom(rng), neighborWeight))
        {
            selected = i;
            selectedLightPrevID = int(originalPrevLightID);
            selectedLightSample = candidateLightSample;
        }
    }

    if (RTXDI_IsValidDIReservoir(state))
    {
#if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_BASIC
        if (stparams.biasCorrectionMode >= RTXDI_BIAS_CORRECTION_BASIC)
        {
            // Compute the unbiased normalization term (instead of using 1/M)
            float pi = state.targetPdf;
            float piSum = state.targetPdf * curSample.M;

            if (selectedLightPrevID >= 0)
            {
                const RAB_LightInfo selectedLightPrev = RAB_LoadLightInfo(selectedLightPrevID, true);

                // To do this, we need to walk our neighbors again
                for (i = 0; i < numSamples; ++i)
                {
                    // If we skipped this neighbor above, do so again.
                    if ((cachedResult & (1u << uint(i))) == 0) continue;

                    uint sampleIdx = (startIdx + i) & params.neighborOffsetMask;

                    // Get the screen-space location of our neighbor
                    int2 spatialOffset = (i == 0 && foundTemporalSurface) 
                        ? temporalSpatialOffset 
                        : int2(float2(RTXDI_NEIGHBOR_OFFSETS_BUFFER[sampleIdx].xy) * stparams.samplingRadius);
                    int2 idx = prevPos + spatialOffset;

                    if (!(i == 0 && foundTemporalSurface))
                    {
                        idx = RAB_ClampSamplePositionIntoView(idx, true);
                    }

                    RTXDI_ActivateCheckerboardPixel(idx, true, params.activeCheckerboardField);

                    // Load our neighbor's G-buffer
                    RAB_Surface neighborSurface = RAB_GetGBufferSurface(idx, true);
                    
                    // Get the PDF of the sample RIS selected in the first loop, above, *at this neighbor* 
                    const RAB_LightSample selectedSampleAtNeighbor = RAB_SamplePolymorphicLight(
                        selectedLightPrev, neighborSurface, RTXDI_GetDIReservoirSampleUV(state));

                    float ps = RAB_GetLightSampleTargetPdfForSurface(selectedSampleAtNeighbor, neighborSurface);

#if RTXDI_ALLOWED_BIAS_CORRECTION >= RTXDI_BIAS_CORRECTION_RAY_TRACED
                                                                                                              // TODO:  WHY?
                    if (stparams.biasCorrectionMode == RTXDI_BIAS_CORRECTION_RAY_TRACED && ps > 0 && (selected != i || i != 0 || !stparams.enableVisibilityShortcut))
                    {
                        RAB_Surface fallbackSurface;
                        if (i == 0 && foundTemporalSurface)
                            fallbackSurface = surface;
                        else
                            fallbackSurface = neighborSurface;

                        if (!RAB_GetTemporalConservativeVisibility(fallbackSurface, neighborSurface, selectedSampleAtNeighbor))
                        {
                            ps = 0;
                        }
                    }
#endif

                    uint2 neighborReservoirPos = RTXDI_PixelPosToReservoirPos(idx, params.activeCheckerboardField);

                    RTXDI_DIReservoir prevSample = RTXDI_LoadDIReservoir(reservoirParams,
                        neighborReservoirPos, stparams.sourceBufferIndex);
                    prevSample.M = min(prevSample.M, historyLimit);

                    // Select this sample for the (normalization) numerator if this particular neighbor pixel
                    //     was the one we selected via RIS in the first loop, above.
                    pi = selected == i ? ps : pi;

                    // Add to the sums of weights for the (normalization) denominator
                    piSum += ps * prevSample.M;
                }
            }

            // Use "MIS-like" normalization
            RTXDI_FinalizeResampling(state, pi, piSum);
        }
        else
#endif
        {
            RTXDI_FinalizeResampling(state, 1.0, state.M);
        }
    }

    return state;
}

#endif // RTXDI_DI_SPATIOTEMPORAL_RESAMPLIHG_HLSLI
