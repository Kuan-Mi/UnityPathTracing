using System.Runtime.InteropServices;
using Rtxdi.DI;

namespace PathTracing
{
    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct BRDFPathTracing_MaterialOverrideParameters
    {
        public float roughnessOverride;
        public float metalnessOverride;
        public float minSecondaryRoughness;
        public uint pad1;
    };

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct BRDFPathTracing_SecondarySurfaceReSTIRDIParameters
    {
        public ReSTIRDI_InitialSamplingParameters initialSamplingParams;
        public ReSTIRDI_SpatialResamplingParameters spatialResamplingParams;
    };

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct BRDFPathTracing_Parameters
    {
        public uint enableIndirectEmissiveSurfaces;
        public uint enableSecondaryResampling;
        public uint enableReSTIRGI;
        public uint pad1;

        public BRDFPathTracing_MaterialOverrideParameters materialOverrideParams;
        public BRDFPathTracing_SecondarySurfaceReSTIRDIParameters secondarySurfaceReSTIRDIParams;
    }
}