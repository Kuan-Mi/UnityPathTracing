using Unity.Mathematics;

public struct PrepareLightsConstants
{
    public uint numTasks;
};

public struct PrepareLightsTask
{
    public uint instanceIndex;
    public uint geometryIndex;
    public uint triangleCount;
    public uint lightBufferOffset;
};

public struct ResamplingConstants
{
    // PlanarViewConstants view;
    // PlanarViewConstants prevView;
    public RTXDI_RuntimeParameters runtimeParams;
    public RTXDI_LightBufferParameters lightBufferParams;
    public RTXDI_ReservoirBufferParameters restirDIReservoirBufferParams;

    public uint frameIndex;
    public uint numInitialSamples;
    public uint numSpatialSamples;
    public uint pad1;

    public uint numInitialBRDFSamples;
    public float brdfCutoff;
    public uint2 pad2;

    public uint enableResampling;
    public uint unbiasedMode;
    public uint inputBufferIndex;
    public uint outputBufferIndex;
};

// See TriangleLight.hlsli for encoding format
public struct RAB_LightInfo
{
    // uint4[0]
    public float3 center;
    public uint scalars; // 2x float16

    // uint4[1]
    public uint2 radiance; // fp16x4
    public uint direction1; // oct-encoded
    public uint direction2; // oct-encoded
};