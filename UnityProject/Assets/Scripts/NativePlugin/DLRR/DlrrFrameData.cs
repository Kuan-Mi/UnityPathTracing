using System;
using System.Runtime.InteropServices;
using PathTracing;
using Unity.Mathematics;
using UnityEngine;

namespace DLRR
{
    // ===================================================================================
    // FRAME DATA (Packed)
    // ===================================================================================

    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct DlrrFrameData
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
         
        
        public ushort outputWidth;
        public ushort outputHeight;
        public ushort currentWidth;
        public ushort currentHeight;

        public float2 cameraJitter;
        public int instanceId;
        public bool useSpecularMotionVector; // true: 传入mv，false: 传入hitT

        public UpscalerMode upscalerMode;
        public byte preset; // 0 = default RR network; otherwise a DlssRRPreset value
    }
 
}