using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace DefaultNamespace
{
    public class RtxdiNative : MonoBehaviour
    {
        [DllImport("UnityRtxdi.dll", CallingConvention = CallingConvention.StdCall)]
        public static extern void FillNeighborOffsetBuffer(IntPtr buffer, uint neighborOffsetCount);



    }
}