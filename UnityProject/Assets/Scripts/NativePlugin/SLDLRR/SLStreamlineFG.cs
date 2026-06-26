using System;
using UnityEngine;

namespace SLDLRR
{
    public static class SLStreamlineFG
    {
        public enum ReflexMode { Off = 0, On = 1, OnPlusBoost = 2 }

        public const uint D3D12_STATE_UNORDERED_ACCESS    = SLDlssg.D3D12_STATE_UNORDERED_ACCESS;
        public const uint D3D12_STATE_ALL_SHADER_RESOURCE = SLDlssg.D3D12_STATE_ALL_SHADER_RESOURCE;

        public struct DlssgInputs
        {
            public IntPtr depth;
            public IntPtr motionVectors;
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

            internal SLDlssg.FrameInputs ToFrameInputs() => new SLDlssg.FrameInputs
            {
                depth                = depth,
                motionVectors        = motionVectors,
                depthState           = depthState,
                mvecState            = mvecState,
                mvecDepthW           = mvecDepthW,
                mvecDepthH           = mvecDepthH,
                colorW               = colorW,
                colorH               = colorH,
                cameraViewToClip     = cameraViewToClip,
                clipToCameraView     = clipToCameraView,
                clipToPrevClip       = clipToPrevClip,
                prevClipToClip       = prevClipToClip,
                cameraPos            = cameraPos,
                cameraUp             = cameraUp,
                cameraRight          = cameraRight,
                cameraFwd            = cameraFwd,
                jitterX              = jitterX,
                jitterY              = jitterY,
                mvecScaleX           = mvecScaleX,
                mvecScaleY           = mvecScaleY,
                cameraNear           = cameraNear,
                cameraFar            = cameraFar,
                cameraFOV            = cameraFOV,
                cameraAspect         = cameraAspect,
                depthInverted        = depthInverted,
                cameraMotionIncluded = cameraMotionIncluded,
                motionVectors3D      = motionVectors3D,
                reset                = reset,
            };
        }

        public static KeyCode ReflexToggleKey
        {
            get => SLReflexRuntime.ReflexToggleKey;
            set => SLReflexRuntime.ReflexToggleKey = value;
        }

        public static KeyCode ReflexBoostKey
        {
            get => SLReflexRuntime.ReflexBoostKey;
            set => SLReflexRuntime.ReflexBoostKey = value;
        }

        public static IntPtr CurrentFrameTokenPtr => SLStreamlineFrameLoop.CurrentFrameTokenPtr;
        public static bool LastFrameHadPclPing => SLPclLatencyPing.LastFrameHadPing;
        public static uint LastFramePclPingCount => SLPclLatencyPing.LastFramePingCount;

        public static IntPtr GetBeginEventFunc() => SLStreamlineFrameLoop.GetBeginEventFunc();
        public static IntPtr GetFrameInputsEventFunc() => SLDlssg.GetFrameInputsEventFunc();
        public static void SetFrameGeneration(bool enable) => SLDlssg.SetFrameGeneration(enable);
        public static bool IsFrameGenerationOn() => SLDlssg.IsFrameGenerationOn();

        public static void SetReflexMode(ReflexMode mode, uint fpsCapUs = 0) =>
            SLReflexRuntime.SetMode((SLReflexRuntime.Mode)(int)mode, fpsCapUs);

        public static int GetReflexMode() => SLReflexRuntime.GetMode();
        public static bool IsReflexLowLatencyAvailable() => SLReflexRuntime.IsLowLatencyAvailable();

        public static IntPtr GetInteropDataPtr(in DlssgInputs inputs, uint frameIndex)
        {
            var frameInputs = inputs.ToFrameInputs();
            return SLDlssg.GetInteropDataPtr(frameInputs, frameIndex);
        }
    }
}
