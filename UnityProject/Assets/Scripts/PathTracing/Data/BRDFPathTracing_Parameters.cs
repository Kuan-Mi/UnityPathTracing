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
        
        
        public static BRDFPathTracing_MaterialOverrideParameters Default()
        {
            BRDFPathTracing_MaterialOverrideParameters p;
            p.metalnessOverride     = 0.5f;
            p.minSecondaryRoughness = 0.5f;
            p.roughnessOverride     = 0.5f;
            p.pad1                  = 0;
            return p;
        }
    };

    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct BRDFPathTracing_SecondarySurfaceReSTIRDIParameters
    {
        public ReSTIRDI_InitialSamplingParameters initialSamplingParams;
        public ReSTIRDI_SpatialResamplingParameters spatialResamplingParams;

        public static BRDFPathTracing_SecondarySurfaceReSTIRDIParameters Default()
        {
            BRDFPathTracing_SecondarySurfaceReSTIRDIParameters p = default;

            p.initialSamplingParams.localLightSamplingMode         = ReSTIRDI_LocalLightSamplingMode.ReGIR_RIS;
            p.initialSamplingParams.numPrimaryLocalLightSamples    = 2;
            p.initialSamplingParams.numPrimaryInfiniteLightSamples = 1;
            p.initialSamplingParams.numPrimaryEnvironmentSamples   = 1;
            p.initialSamplingParams.numPrimaryBrdfSamples          = 0;
            p.initialSamplingParams.brdfCutoff                     = 0;
            p.initialSamplingParams.enableInitialVisibility        = 0;

            p.spatialResamplingParams.numSpatialSamples           = 1;
            p.spatialResamplingParams.spatialSamplingRadius       = 4.0f;
            p.spatialResamplingParams.spatialBiasCorrection       = ReSTIRDI_SpatialBiasCorrectionMode.Basic;
            p.spatialResamplingParams.numDisocclusionBoostSamples = 0; // Disabled
            p.spatialResamplingParams.spatialDepthThreshold       = 0.1f;
            p.spatialResamplingParams.spatialNormalThreshold      = 0.9f;

            return p;
        }
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
        
        
        
        public static BRDFPathTracing_Parameters Default()
        {
            BRDFPathTracing_Parameters p;
            p.enableIndirectEmissiveSurfaces = 0;
            p.enableSecondaryResampling      = 0;
            p.enableReSTIRGI                 = 0;
            p.pad1                           = 0;
            p.materialOverrideParams         = BRDFPathTracing_MaterialOverrideParameters.Default();
            p.secondarySurfaceReSTIRDIParams = BRDFPathTracing_SecondarySurfaceReSTIRDIParameters.Default();

            return p;
        }
        
    }
}