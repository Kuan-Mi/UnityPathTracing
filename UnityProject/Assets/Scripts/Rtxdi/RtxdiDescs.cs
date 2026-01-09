public struct RTXDI_LightBufferRegion
{
    public uint firstLightIndex;
    public uint numLights;
    public uint pad1;
    public uint pad2;
}

public struct RTXDI_EnvironmentLightBufferParameters
{
    public uint lightPresent;
    public uint lightIndex;
    public uint pad1;
    public uint pad2;
}

public struct RTXDI_RuntimeParameters
{
    public uint neighborOffsetMask; // Spatial
    public uint activeCheckerboardField; // 0 - no checkerboard, 1 - odd pixels, 2 - even pixels
    public uint pad1;
    public uint pad2;
}

public struct RTXDI_LightBufferParameters
{
    public RTXDI_LightBufferRegion localLightBufferRegion;
    public RTXDI_LightBufferRegion infiniteLightBufferRegion;
    public RTXDI_EnvironmentLightBufferParameters environmentLightParams;
}

public struct RTXDI_ReservoirBufferParameters
{
    public uint reservoirBlockRowPitch;
    public uint reservoirArrayPitch;
    public uint pad1;
    public uint pad2;
}

public struct RTXDI_PackedDIReservoir
{
    public uint lightData;
    public uint uvData;
    public uint mVisibility;
    public uint distanceAge;
    public float targetPdf;
    public float weight;
}