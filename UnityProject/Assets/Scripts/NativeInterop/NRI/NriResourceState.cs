using System;
using System.Runtime.InteropServices;

namespace PathTracing.NativeInterop.NRI
{
    [Serializable]
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct NriResourceState
    {
        public AccessBits accessBits;
        public Layout layout;
        public uint stageBits;
    }
}