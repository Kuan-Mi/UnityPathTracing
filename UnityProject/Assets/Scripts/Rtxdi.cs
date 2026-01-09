using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace DefaultNamespace
{
    public enum CheckerboardMode : int
    {
        Off = 0,
        Black = 1, // 根据 C++ 实际定义补充
        White = 2 // 根据 C++ 实际定义补充
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ReSTIRDIStaticParameters
    {
        public uint NeighborOffsetCount; // uint32_t -> uint
        public uint RenderWidth; // uint32_t -> uint
        public uint RenderHeight; // uint32_t -> uint

        public CheckerboardMode CheckerboardSamplingMode; // Enum 通常对应 int
    }

    public class Rtxdi : MonoBehaviour
    {
        [DllImport("UnityRtxdi.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern IntPtr CreateReSTIRDIContext(int width, int height);


        [DllImport("UnityRtxdi", CallingConvention = CallingConvention.StdCall)]
        private static extern void DestroyReSTIRDIContext(IntPtr context);

        [DllImport("UnityRtxdi", CallingConvention = CallingConvention.StdCall)]
        private static unsafe extern ReSTIRDIStaticParameters* GetStaticParameters(IntPtr context);

        [ContextMenu("TestReSTIRDI")]
        public void TestReSTIRDI()
        {
            var reStirdiContext = CreateReSTIRDIContext(1920, 1080);

            unsafe
            {
                ReSTIRDIStaticParameters* staticParams = GetStaticParameters(reStirdiContext);

                Debug.Log("NeighborOffsetCount: " + staticParams->NeighborOffsetCount);
                Debug.Log("RenderWidth: " + staticParams->RenderWidth);
                Debug.Log("RenderHeight: " + staticParams->RenderHeight);
                Debug.Log("CheckerboardSamplingMode: " + staticParams->CheckerboardSamplingMode);
            }
        }
    }
}