using System.Runtime.InteropServices;
using Rtxdi.DI;
using UnityEngine;

namespace PathTracing
{
    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct BRDFPathTracing_MaterialOverrideParameters
    {
        [Range(0, 1)]
        public float roughnessOverride;
        [Range(0, 1)]
        public float metalnessOverride;
        [Range(0, 1)]
        public float minSecondaryRoughness;
        [HideInInspector]
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
        [Range(0, 1)]
        public uint enableIndirectEmissiveSurfaces;
        [Range(0, 1)]
        public uint enableSecondaryResampling;
        [Range(0, 1)]
        public uint enableReSTIRGI;
        [HideInInspector]
        public uint pad1;

        public BRDFPathTracing_MaterialOverrideParameters materialOverrideParams;
        public BRDFPathTracing_SecondarySurfaceReSTIRDIParameters secondarySurfaceReSTIRDIParams;
    }
}