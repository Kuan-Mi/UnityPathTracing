using System;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace PathTracing.NativeInterop.Streamline
{
    public static class SLDlssg
    {
        public const uint D3D12_STATE_UNORDERED_ACCESS    = 0x08;
        public const uint D3D12_STATE_ALL_SHADER_RESOURCE = 0x40 | 0x80;

        [StructLayout(LayoutKind.Sequential)]
        public struct FrameInputs
        {
            public IntPtr depth;
            public IntPtr motionVectors;
            public IntPtr frameToken;   // this frame's sl::FrameToken* (SL_GetNewFrameToken)
            public uint   depthState, mvecState;
            public uint   mvecDepthW, mvecDepthH;
            public uint   colorW, colorH;

            public Matrix4x4 cameraViewToClip;
            public Matrix4x4 clipToCameraView;
            public Matrix4x4 clipToPrevClip;
            public Matrix4x4 prevClipToClip;

            public Vector3 cameraPos;
            public Vector3 cameraUp;
            public Vector3 cameraRight;
            public Vector3 cameraFwd;

            public float jitterX, jitterY;
            public float mvecScaleX, mvecScaleY;
            public float cameraNear, cameraFar, cameraFOV, cameraAspect;
            public int   depthInverted;
            public int   cameraMotionIncluded;
            public int   motionVectors3D;
            public int   reset;
        }

        private const int RingCount = 3;
        private static NativeArray<FrameInputs> _ring;
        private static IntPtr _frameInputsEventFunc;

        public static IntPtr GetFrameInputsEventFunc()
        {
            if (!SLNative.Available) return IntPtr.Zero;
            if (_frameInputsEventFunc != IntPtr.Zero) return _frameInputsEventFunc;
            try { _frameInputsEventFunc = SLNative.GetSLFGFrameInputsFunc(); }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
            return _frameInputsEventFunc;
        }

        public static void SetFrameGeneration(bool enable)
        {
            if (!SLNative.Available) return;
            try { SLNative.SL_SetFrameGeneration(enable ? 1 : 0); }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
        }

        public static bool IsFrameGenerationOn()
        {
            if (!SLNative.Available) return false;
            try { return SLNative.SL_IsFrameGenerationOn() != 0; }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); return false; }
        }

        public static IntPtr GetInteropDataPtr(in FrameInputs inputs, uint frameIndex)
        {
            if (!_ring.IsCreated)
                _ring = new NativeArray<FrameInputs>(RingCount, Allocator.Persistent);
            int idx = (int)(frameIndex % RingCount);
            _ring[idx] = inputs;
            unsafe { return (IntPtr)_ring.GetUnsafePtr() + idx * sizeof(FrameInputs); }
        }

        internal static void Dispose()
        {
            if (_ring.IsCreated) _ring.Dispose();
        }
    }
}
