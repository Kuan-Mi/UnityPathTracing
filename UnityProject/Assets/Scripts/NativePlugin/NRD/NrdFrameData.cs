using System;
using System.Runtime.InteropServices;
using Nri;

namespace Nrd
{
    // ===================================================================================
    // FRAME DATA (Packed)
    // ===================================================================================

    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct NrdFrameData
    {
        public CommonSettings commonSettings;
        public SigmaSettings sigmaSettings;
        public ReblurSettings reblurSettings;

        public ushort width;
        public ushort height;

        public int instanceId;

        public static NrdFrameData _default = CreateDefault();

        // -----------------------------------------------------------------------
        // Factory Method for C++ Defaults
        // -----------------------------------------------------------------------
        private static NrdFrameData CreateDefault()
        {
            return new NrdFrameData
            {
                commonSettings = CommonSettings._default,
                sigmaSettings = SigmaSettings._default,
                reblurSettings = ReblurSettings._default,
                width = 0,
                height = 0,
                instanceId = 0
            };
        }
    }



    [Serializable]
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct NrdResourceInput
    {
        public ResourceType type;
        public IntPtr texture;
        public NriResourceState state;
    }
}