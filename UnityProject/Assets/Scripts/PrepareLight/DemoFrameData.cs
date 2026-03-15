using System;
using System.Runtime.InteropServices;
using Mono.Cecil;
using Nrd;

namespace DefaultNamespace
{
    public struct PrepareLightFrameData
    {
        public IntPtr primitiveBuffer;
        public IntPtr instanceBuffer;
        public IntPtr lightDataBuffer;
        
        public int numPrimitives;
        public int InstanceCount;

        public int instanceId;
    }
    [Serializable]
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct EmissionResourceInput
    {
        public IntPtr texture;
        public NriFormat format;
    }
}