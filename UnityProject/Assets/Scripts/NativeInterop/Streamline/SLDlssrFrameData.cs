using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace PathTracing.NativeInterop.Streamline
{
    // ===================================================================================
    // SLDlssrFrameData — byte-identical mirror of the native struct in
    // RenderingPlugin/StreamlinePlugin/include/SLDlssrFrameData.h (#pragma pack(push,1)).
    // Field order and Pack=1 MUST stay in lock-step with the C++ side.
    //
    // DLSS Super Resolution counterpart of SLDlssrrFrameData — fewer guides (input color,
    // motion vectors, depth and optional exposure only; no albedo / normal-roughness /
    // specular-hit-distance buffers).
    //
    // Matrices are Unity Matrix4x4 (column-major). The native side memcpy's their raw bytes
    // into row-major sl::float4x4, which performs the column->row transpose SL expects.
    // ===================================================================================

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct SLDlssrFrameData
    {
        // shared per-frame Streamline token (sl::FrameToken*, from SL_GetNewFrameToken); Zero in the
        // editor edit-mode game view, where the native side mints its own.
        public IntPtr frameToken;

        // tagged resources (native ID3D12Resource*)
        public IntPtr inputTex;
        public IntPtr outputTex;
        public IntPtr mvTex;
        public IntPtr depthTex;
        public IntPtr exposureTex; // optional; Zero -> DLSS uses internal auto-exposure

        // D3D12_RESOURCE_STATES each resource is in when SL reads it
        public uint inputState;
        public uint outputState;
        public uint mvState;
        public uint depthState;
        public uint exposureState;

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
        public float2 mvecScale; // brings mvec into [-1,1]

        public float cameraNear;
        public float cameraFar;
        public float cameraFOV; // radians
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

        public byte upscalerMode;
        public byte preset;
    }
}