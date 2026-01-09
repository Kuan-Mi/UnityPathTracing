public enum ReSTIRDI_LocalLightSamplingMode : uint
{
    Uniform = 0,
    Power_RIS = 1,
    ReGIR_RIS = 2
};

public enum ReSTIRDI_TemporalBiasCorrectionMode : uint
{
    Off = 0,
    Basic = 1,
    Pairwise = 2,
    Raytraced = 3
};

public enum ReSTIRDI_SpatialBiasCorrectionMode : uint
{
    Off = 0,
    Basic = 1,
    Pairwise = 2,
    Raytraced = 3
};


public struct ReSTIRDI_BufferIndices
{
    public uint initialSamplingOutputBufferIndex;
    public uint temporalResamplingInputBufferIndex;
    public uint temporalResamplingOutputBufferIndex;
    public uint spatialResamplingInputBufferIndex;

    public uint spatialResamplingOutputBufferIndex;
    public uint shadingInputBufferIndex;
    public uint pad1;
    public uint pad2;
};


public struct ReSTIRDI_InitialSamplingParameters
{
    public uint numPrimaryLocalLightSamples;
    public uint numPrimaryInfiniteLightSamples;
    public uint numPrimaryEnvironmentSamples;
    public uint numPrimaryBrdfSamples;

    public float brdfCutoff;
    public uint enableInitialVisibility;
    public uint environmentMapImportanceSampling; // Only used in InitialSamplingFunctions.hlsli via RAB_EvaluateEnvironmentMapSamplingPdf
    public ReSTIRDI_LocalLightSamplingMode localLightSamplingMode;
};


public struct ReSTIRDI_TemporalResamplingParameters
{
    public float temporalDepthThreshold;
    public float temporalNormalThreshold;
    public uint maxHistoryLength;
    public ReSTIRDI_TemporalBiasCorrectionMode temporalBiasCorrection;

    public uint enablePermutationSampling;
    public float permutationSamplingThreshold;
    public uint enableBoilingFilter;
    public float boilingFilterStrength;

    public uint discardInvisibleSamples;
    public uint uniformRandomNumber;
    public uint pad2;
    public uint pad3;
};


public struct ReSTIRDI_SpatialResamplingParameters
{
    public float spatialDepthThreshold;
    public float spatialNormalThreshold;
    public ReSTIRDI_SpatialBiasCorrectionMode spatialBiasCorrection;
    public uint numSpatialSamples;

    public uint numDisocclusionBoostSamples;
    public float spatialSamplingRadius;
    public uint neighborOffsetMask;
    public uint discountNaiveSamples;
};

public struct ReSTIRDI_ShadingParameters
{
    public uint enableFinalVisibility;
    public uint reuseFinalVisibility;
    public uint finalVisibilityMaxAge;
    public float finalVisibilityMaxDistance;

    public uint enableDenoiserInputPacking;
    public uint pad1;
    public uint pad2;
    public uint pad3;
};

public struct ReSTIRDI_Parameters
{
    public RTXDI_ReservoirBufferParameters reservoirBufferParams;
    public ReSTIRDI_BufferIndices bufferIndices;
    public ReSTIRDI_InitialSamplingParameters initialSamplingParams;
    public ReSTIRDI_TemporalResamplingParameters temporalResamplingParams;
    public ReSTIRDI_SpatialResamplingParameters spatialResamplingParams;
    public ReSTIRDI_ShadingParameters shadingParams;
};