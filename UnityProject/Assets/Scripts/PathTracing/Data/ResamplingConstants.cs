using Rtxdi;
using Rtxdi.DI;
using Rtxdi.GI;
using Rtxdi.ReGIR;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    public struct ResamplingConstants
    {
        public RTXDI_RuntimeParameters runtimeParams;

        public uint frameIndex;
        public uint enablePreviousTLAS;
        public uint denoiserMode;
        public uint discountNaiveSamples;

        public uint enableBrdfIndirect;
        public uint enableBrdfAdditiveBlend;
        public uint enableAccumulation; // StoreShadingOutput
        public uint showReGIRCell;

        public RTXDI_LightBufferParameters lightBufferParams;
        public RTXDI_RISBufferSegmentParameters localLightsRISBufferSegmentParams;
        public RTXDI_RISBufferSegmentParameters environmentLightRISBufferSegmentParams;

        public ReSTIRDI_Parameters restirDI;
        public ReGIR_Parameters regir;
        public ReSTIRGI_Parameters restirGI;
        public BRDFPathTracing_Parameters brdfPT;

        public uint2 environmentPdfTextureSize;
        public uint2 localLightPdfTextureSize;

        public override string ToString()
        {
            return $"ResamplingConstants: " +
                   $"\n{runtimeParams}" +
                   $"\n{lightBufferParams}" +
                   // $"\n{restirDIReservoirBufferParams}" +
                   $"\n{restirDI}" +
                   $"\nframeIndex: {frameIndex}";
        }
    };
}