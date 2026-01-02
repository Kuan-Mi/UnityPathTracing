using System;
using System.Runtime.InteropServices;
using Nri;
using Unity.Mathematics;
using UnityEngine;

namespace Nrd
{
    // ===================================================================================
    // FRAME DATA (Packed)
    // ===================================================================================

    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct RRFrameData
    {
        
        public IntPtr inputTex;
        public IntPtr outputTex;
        public IntPtr mvTex;
        public IntPtr depthTex;
        public IntPtr diffuseAlbedoTex;
        public IntPtr specularAlbedoTex;
        public IntPtr normalRoughnessTex;
        public IntPtr specularMvOrHitTex; 
        
        
        public Matrix4x4 worldToViewMatrix;
        public Matrix4x4 viewToClipMatrix;
         
        public ushort width;
        public ushort height;

        public float2 cameraJitter;
        public int instanceId;
    }
 
}