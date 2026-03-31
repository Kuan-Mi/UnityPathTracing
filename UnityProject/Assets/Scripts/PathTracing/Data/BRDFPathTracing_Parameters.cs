using System.Runtime.InteropServices;
using Rtxdi.DI;

namespace PathTracing
{
    [StructLayout(LayoutKind.Sequential)]
    struct BRDFPathTracing_MaterialOverrideParameters
    {
        float roughnessOverride;
        float metalnessOverride;
        float minSecondaryRoughness;
        uint pad1;
    };

    [StructLayout(LayoutKind.Sequential)]
    struct BRDFPathTracing_SecondarySurfaceReSTIRDIParameters
    {
        ReSTIRDI_InitialSamplingParameters initialSamplingParams;
        ReSTIRDI_SpatialResamplingParameters spatialResamplingParams;
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct BRDFPathTracing_Parameters
    {
        uint enableIndirectEmissiveSurfaces;
        uint enableSecondaryResampling;
        uint enableReSTIRGI;
        uint pad1;

        BRDFPathTracing_MaterialOverrideParameters materialOverrideParams;
        BRDFPathTracing_SecondarySurfaceReSTIRDIParameters secondarySurfaceReSTIRDIParams;
    }
}