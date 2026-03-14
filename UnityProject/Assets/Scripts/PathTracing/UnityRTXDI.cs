using System;
using System.Runtime.InteropServices;

namespace PathTracing
{
    public class UnityRTXDI
    {
        [DllImport("UnityRTXDI")]
        public static extern IntPtr GetRenderEventAndDataFunc();
    }
}