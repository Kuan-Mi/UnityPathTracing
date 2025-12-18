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
    public struct FrameData
    {
        public CommonSettings commonSettings;
        public SigmaSettings sigmaSettings;
        public ReblurSettings reblurSettings; // 新增

        public ushort width;
        public ushort height;

        public IntPtr mvPointer;
        public IntPtr normalRoughnessPointer;
        public IntPtr viewZPointer;
        public IntPtr penumbraPointer;
        public IntPtr shadowTranslucencyPointer;
        public IntPtr diffRadiancePointer;
        public IntPtr outDiffRadiancePointer;
        public IntPtr validationPointer;

        public int instanceId;

        public static FrameData _default = CreateDefault();

        // -----------------------------------------------------------------------
        // Factory Method for C++ Defaults
        // -----------------------------------------------------------------------
        private static FrameData CreateDefault()
        {
            return new FrameData
            {
                commonSettings = CommonSettings._default,
                sigmaSettings = SigmaSettings._default,
                reblurSettings = ReblurSettings._default,
                width = 0,
                height = 0,
                mvPointer = IntPtr.Zero,
                normalRoughnessPointer = IntPtr.Zero,
                viewZPointer = IntPtr.Zero,
                penumbraPointer = IntPtr.Zero,
                shadowTranslucencyPointer = IntPtr.Zero,
                diffRadiancePointer = IntPtr.Zero,
                outDiffRadiancePointer = IntPtr.Zero,
                validationPointer = IntPtr.Zero,
                instanceId = 0
            };
        }
    }

    [Serializable]
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct NriResourceState
    {
        public AccessBits accessBits;
        public Layout layout;
        public uint stageBits;
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