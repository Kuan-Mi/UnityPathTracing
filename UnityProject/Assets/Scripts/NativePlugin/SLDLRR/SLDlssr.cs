using System;
using System.Runtime.InteropServices;
using Nri;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace SLDLRR
{
    /// <summary>
    /// DLSS Super Resolution through Streamline (SL), the mirror of
    /// <see cref="DLSR.DlsrUpscaler"/> which goes through NRI, and the upscaling counterpart
    /// of <see cref="SLDlssrr"/>. Unlike the NRI path it can run concurrently with DLSS-G
    /// frame generation (both go through the same slInit), so it is the path the features use.
    ///
    /// Backed by the native <c>SLDenoiser</c> DLL. Tags RAW ID3D12Resource* pointers
    /// (<see cref="NriTextureResource.NativePtr"/>) — Streamline evaluates DLSS-SR on Unity's
    /// command list and transitions the tagged resources itself.
    /// </summary>
    public class SLDlssr : IDisposable
    {
        [DllImport("SLDenoiser")]
        private static extern int CreateSLDlssrInstance();

        [DllImport("SLDenoiser")]
        private static extern void DestroySLDlssrInstance(int id);

        [DllImport("SLDenoiser")]
        private static extern bool SLDlssr_QueryOptimalRenderSize(
            uint outputWidth, uint outputHeight, byte mode,
            out uint renderWidth, out uint renderHeight);

        // D3D12_RESOURCE_STATES. Unity pool RTs are enableRandomWrite and last written by
        // compute/ray-tracing, so UNORDERED_ACCESS is the best single guess for the state SL
        // sees them in (same rationale as SLDlssrr).
        public const uint D3D12_STATE_UNORDERED_ACCESS = 0x08;

        /// <summary>
        /// Queries the Streamline/NGX-recommended render resolution for a given output
        /// resolution and DLSS mode (slDLSSGetOptimalSettings). Cached natively.
        /// Returns false (and renderRes = outputRes) if the plugin is unavailable.
        /// </summary>
        public static bool TryGetOptimalRenderSize(int2 outputRes, UpscalerMode mode, out int2 renderRes)
        {
            if (SLDlssr_QueryOptimalRenderSize(
                    (uint)outputRes.x, (uint)outputRes.y, (byte)mode,
                    out uint rw, out uint rh))
            {
                renderRes = new int2((int)rw, (int)rh);
                return true;
            }
            renderRes = outputRes;
            return false;
        }

        private readonly int                         _instanceId;
        private          NativeArray<SLDlssrFrameData> _buffer;
        private const    int                         BufferCount = 3;
        private readonly string                      _cameraName;

        /// <summary>
        /// Per-frame camera data, filled from CameraFrameState by the feature. Matches
        /// <see cref="SLDlssrr.SLDlssrrFrameInput"/> (SL constants need the full camera basis).
        /// </summary>
        public struct SLDlssrFrameInput
        {
            public Matrix4x4 worldToView;     // current frame
            public Matrix4x4 viewToClip;      // current frame, no jitter
            public Matrix4x4 worldToClip;     // current frame
            public Matrix4x4 prevWorldToClip; // previous frame
            public float3    camPos;
            public float2    viewportJitter;  // pixel space
            public int2      renderResolution;
            public int2      outputResolution;
            public uint      frameIndex;
            public float     cameraNear;
            public float     cameraFar;
            public float     cameraFOV;       // radians
            public float     cameraAspect;
            public bool      reset;
        }

        /// <summary>
        /// DLSS-SR guide textures. Native ID3D12Resource* are taken from each NriTextureResource.
        /// </summary>
        public struct DlssrResources
        {
            public NriTextureResource input;    // low-res color (ScalingInputColor)
            public NriTextureResource output;   // upscaled color (ScalingOutputColor)
            public NriTextureResource mv;       // motion vectors
            public NriTextureResource depth;    // depth
            public NriTextureResource exposure; // optional – null -> internal auto-exposure
        }

        public SLDlssr(string camName)
        {
            _instanceId = CreateSLDlssrInstance();
            _cameraName = camName;
            _buffer     = new NativeArray<SLDlssrFrameData>(BufferCount, Allocator.Persistent);
            Debug.Log($"[SL DLSS SR] Created instance {_instanceId} for camera {_cameraName}");
        }

        private SLDlssrFrameData GetData(SLDlssrFrameInput fi, DlssrResources res,
                                         UpscalerMode upscalerMode, byte preset)
        {
            // Streamline common constants (row-major; SL receives Unity column-major bytes and
            // transposes). Matches SLDlssrr's derivation.
            Matrix4x4 viewToWorld        = fi.worldToView.inverse;
            Matrix4x4 clipToCameraView   = fi.viewToClip.inverse;
            Matrix4x4 worldToClipInv     = fi.worldToClip.inverse;
            Matrix4x4 prevWorldToClipInv = fi.prevWorldToClip.inverse;
            Matrix4x4 clipToPrevClip     = fi.prevWorldToClip * worldToClipInv;
            Matrix4x4 prevClipToClip     = fi.worldToClip * prevWorldToClipInv;

            // Camera basis (world space) from viewToWorld columns. Unity view space looks down -Z.
            float3 camRight =  new float3(viewToWorld.m00, viewToWorld.m10, viewToWorld.m20);
            float3 camUp    =  new float3(viewToWorld.m01, viewToWorld.m11, viewToWorld.m21);
            float3 camFwd   = -new float3(viewToWorld.m02, viewToWorld.m12, viewToWorld.m22);

            float renderW = math.max(1, fi.renderResolution.x);
            float renderH = math.max(1, fi.renderResolution.y);

            bool hasExposure = res.exposure != null && res.exposure.NativePtr != IntPtr.Zero;

            return new SLDlssrFrameData
            {
                // This frame's shared SL token (minted on the main thread by SL_GetNewFrameToken and
                // forwarded to the present latch). Zero in editor edit-mode → native self-mints.
                frameToken = SLStreamlineFrameLoop.CurrentFrameTokenPtr,

                inputTex    = res.input.NativePtr,
                outputTex   = res.output.NativePtr,
                mvTex       = res.mv.NativePtr,
                depthTex    = res.depth.NativePtr,
                exposureTex = hasExposure ? res.exposure.NativePtr : IntPtr.Zero,

                inputState    = D3D12_STATE_UNORDERED_ACCESS,
                outputState   = D3D12_STATE_UNORDERED_ACCESS,
                mvState       = D3D12_STATE_UNORDERED_ACCESS,
                depthState    = D3D12_STATE_UNORDERED_ACCESS,
                exposureState = hasExposure ? D3D12_STATE_UNORDERED_ACCESS : 0,

                cameraViewToClip = fi.viewToClip,
                clipToCameraView = clipToCameraView,
                clipToPrevClip   = clipToPrevClip,
                prevClipToClip   = prevClipToClip,

                cameraPos   = fi.camPos,
                cameraUp    = camUp,
                cameraRight = camRight,
                cameraFwd   = camFwd,

                cameraJitter = fi.viewportJitter,
                mvecScale    = new float2(1.0f / renderW, 1.0f / renderH),

                cameraNear   = fi.cameraNear,
                cameraFar    = fi.cameraFar,
                cameraFOV    = fi.cameraFOV,
                cameraAspect = fi.cameraAspect,

                outputWidth  = (ushort)fi.outputResolution.x,
                outputHeight = (ushort)fi.outputResolution.y,
                renderWidth  = (ushort)fi.renderResolution.x,
                renderHeight = (ushort)fi.renderResolution.y,

                instanceId           = _instanceId,
                depthInverted        = 1, // Unity uses reversed-Z
                cameraMotionIncluded = 1, // dense motion vectors include camera motion
                motionVectors3D      = 0,
                reset                = (fi.reset || fi.frameIndex < 2) ? 1 : 0,

                upscalerMode = (byte)upscalerMode,
                preset       = preset,
            };
        }

        public IntPtr GetInteropDataPtr(SLDlssrFrameInput fi, DlssrResources res,
                                        UpscalerMode upscalerMode, byte preset = 0)
        {
            var index = (int)(fi.frameIndex % BufferCount);
            _buffer[index] = GetData(fi, res, upscalerMode, preset);
            unsafe
            {
                return (IntPtr)_buffer.GetUnsafePtr() + index * sizeof(SLDlssrFrameData);
            }
        }

        public void Dispose()
        {
            if (_buffer.IsCreated)
                _buffer.Dispose();

            DestroySLDlssrInstance(_instanceId);
            Debug.Log($"[SL DLSS SR] Destroyed instance {_instanceId} for camera {_cameraName}");
        }
    }
}
