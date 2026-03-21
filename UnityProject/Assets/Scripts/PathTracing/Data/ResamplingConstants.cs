using Rtxdi;
using Rtxdi.DI;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    public struct ResamplingConstants
    {
        public RTXDI_RuntimeParameters runtimeParams;
        public RTXDI_LightBufferParameters lightBufferParams;
        public RTXDI_ReservoirBufferParameters restirDIReservoirBufferParams;

        public ReSTIRDI_Parameters restirDI;

        public uint frameIndex;
        public uint2 pad2;
        public uint pad3;
        
        public override string ToString()
        {
            return $"ResamplingConstants: " +
                   $"\n{runtimeParams}" +
                   $"\n{lightBufferParams}" +
                   $"\n{restirDIReservoirBufferParams}" +
                   $"\n{restirDI}" +
                   $"\nframeIndex: {frameIndex}";
        }
    };
}