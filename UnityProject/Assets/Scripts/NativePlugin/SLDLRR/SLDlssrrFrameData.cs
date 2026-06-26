using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace SLDLRR
{
    // ===================================================================================
    // SLDlssrrFrameData — byte-identical mirror of the native struct in
    // RenderingPlugin/SLDenoiser/include/SLDlssrrFrameData.h (#pragma pack(push,1)).
    // Field order and Pack=1 MUST stay in lock-step with the C++ side.
    //
    // Matrices are Unity Matrix4x4 (column-major). The native side memcpy's their raw bytes
    // into row-major sl::float4x4, which performs the column->row transpose SL expects, so
    // pass them as-is (same convention as StreamlineProbe / NRI DlrrFrameData).
    // ===================================================================================

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct SLDlssrrFrameData
    {
        // shared per-frame Streamline token (sl::FrameToken*, from SL_GetNewFrameToken); Zero in the
        // editor edit-mode game view, where the native side mints its own.
        public IntPtr frameToken;

        // tagged resources (native ID3D12Resource*)
        public IntPtr inputTex;
        public IntPtr outputTex;
        public IntPtr mvTex;
        public IntPtr depthTex;
        public IntPtr diffuseAlbedoTex;
        public IntPtr specularAlbedoTex;
        public IntPtr normalRoughnessTex;
        public IntPtr specularMvOrHitTex;

        // D3D12_RESOURCE_STATES each resource is in when SL reads it
        public uint inputState;
        public uint outputState;
        public uint mvState;
        public uint depthState;
        public uint diffuseAlbedoState;
        public uint specularAlbedoState;
        public uint normalRoughnessState;
        public uint specularMvOrHitState;

        // DLSSDOptions matrices
        public Matrix4x4 worldToViewMatrix; // -> worldToCameraView
        public Matrix4x4 viewToWorldMatrix; // -> cameraViewToWorld

        // sl::Constants camera matrices (no jitter baked in)
        public Matrix4x4 cameraViewToClip;
        public Matrix4x4 clipToCameraView;
        public Matrix4x4 clipToPrevClip;
        public Matrix4x4 prevClipToClip;

        // sl::Constants camera vectors
        public float3 cameraPos;
        public float3 cameraUp;
        public float3 cameraRight;
        public float3 cameraFwd;

        public float2 cameraJitter; // pixel space
        public float2 mvecScale;    // brings mvec into [-1,1]

        public float cameraNear;
        public float cameraFar;
        public float cameraFOV;    // radians
        public float cameraAspect;

        public ushort outputWidth;
        public ushort outputHeight;
        public ushort renderWidth;
        public ushort renderHeight;

        public int instanceId;
        public int depthInverted;
        public int cameraMotionIncluded;
        public int motionVectors3D;
        public int reset;

        public byte useSpecularMotionVector;
        public byte upscalerMode;
        public byte preset;
    }
}
