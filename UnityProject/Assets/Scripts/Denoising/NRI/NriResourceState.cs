using System;
using System.Runtime.InteropServices;

namespace Nri
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