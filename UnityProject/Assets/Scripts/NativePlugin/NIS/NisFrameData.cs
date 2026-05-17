using System;
using System.Runtime.InteropServices;

namespace NIS
{
    [Serializable]
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct NisFrameData
    {
        public IntPtr inputTex;
        public IntPtr outputTex;

        public ushort outputWidth;
        public ushort outputHeight;
        public ushort currentWidth;
        public ushort currentHeight;
        public float  sharpness;   // [0, 1]

        public int instanceId;
    }
}
