using Rtxdi;
using Rtxdi.DI;
using Unity.Mathematics;
using UnityEngine.Rendering;

namespace PathTracing
{
    public struct ResamplingConstants
    {
        public  RTXDI_RuntimeParameters runtimeParams;
        public  RTXDI_LightBufferParameters lightBufferParams;
        public  RTXDI_ReservoirBufferParameters restirDIReservoirBufferParams;

        public ReSTIRDI_Parameters restirDI;
      
        public  uint frameIndex;
        public  uint numInitialSamples;
        public  uint numSpatialSamples;
        public  uint useAccurateGBufferNormal;

        public  uint numInitialBRDFSamples;
        public  float brdfCutoff;
        public  uint2 pad2;

        public  uint enableResampling;
        public  uint unbiasedMode;
    };
}