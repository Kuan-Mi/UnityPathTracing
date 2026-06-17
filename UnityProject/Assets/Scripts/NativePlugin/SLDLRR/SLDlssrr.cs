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
    /// DLSS Ray Reconstruction through Streamline (SL), the mirror of
    /// <see cref="DLRR.DlrrDenoiser"/> which goes through NRI. Use this to A/B compare the
    /// SL route against the NRI route inside NativeNrdFeature: swap the
    /// <c>DlrrDenoiser</c>/<c>DlssRRPass</c> pair for <c>SLDlssrr</c>/<c>SLDlssrrPass</c>.
    ///
    /// Backed by the native <c>SLDenoiser</c> DLL. Unlike the NRI path this tags RAW
    /// ID3D12Resource* pointers (<see cref="NriTextureResource.NativePtr"/>) — Streamline
    /// evaluates DLSS-RR on Unity's command list and transitions the tagged resources itself.
    ///
    /// IMPORTANT: keep this mutually exclusive with the DLSS-G StreamlineProbePlugin at
    /// runtime (Streamline allows one slInit per process).
    /// </summary>
    public class SLDlssrr : IDisposable
    {
        [DllImport("SLDenoiser")]
        private static extern int CreateSLDlssrrInstance();

        [DllImport("SLDenoiser")]
        private static extern void DestroySLDlssrrInstance(int id);

        [DllImport("SLDenoiser")]
        private static extern bool SLDlssrr_QueryOptimalRenderSize(
            uint outputWidth, uint outputHeight, byte mode,
            out uint renderWidth, out uint renderHeight);

        // D3D12_RESOURCE_STATES. Unity pool RTs are enableRandomWrite and last written by
        // compute/ray-tracing, so UNORDERED_ACCESS is the best single guess for the state SL
        // sees them in. If the D3D12 debug layer complains about a barrier, this is the first
        // knob to adjust (per-resource states are exposed in DlssrrResources).
        public const uint D3D12_STATE_UNORDERED_ACCESS    = 0x08;
        public const uint D3D12_STATE_ALL_SHADER_RESOURCE = 0x40 | 0x80; // NON_PIXEL | PIXEL

        /// <summary>
        /// Queries the Streamline/NGX-recommended render resolution for a given output
        /// resolution and DLSS mode (slDLSSDGetOptimalSettings). Cached natively.
        /// Returns false (and renderRes = outputRes) if the plugin is unavailable.
        /// </summary>
        public static bool TryGetOptimalRenderSize(int2 outputRes, UpscalerMode mode, out int2 renderRes)
        {
            if (SLDlssrr_QueryOptimalRenderSize(
                    (uint)outputRes.x, (uint)outputRes.y, (byte)mode,
                    out uint rw, out uint rh))
            {
                renderRes = new int2((int)rw, (int)rh);
                return true;
            }
            renderRes = outputRes;
            return false;
        }

        private readonly int                          _instanceId;
        private          NativeArray<SLDlssrrFrameData> _buffer;
        private const    int                          BufferCount = 3;
        private readonly string                       _cameraName;

        /// <summary>
        /// Per-frame camera data, filled from CameraFrameState by the feature.
        /// </summary>
        public struct SLDlssrrFrameInput
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
            public bool      useSpecularMotionVector; // false: pass hit distance (matches NRI default)
            public bool      reset;
        }

        /// <summary>
        /// DLSS-RR guide textures. Native ID3D12Resource* are taken from each NriTextureResource.
        /// Optional per-resource D3D12 state overrides (0 = use the UNORDERED_ACCESS default).
        /// </summary>
        public struct DlssrrResources
        {
            public NriTextureResource input;              // noisy color (ScalingInputColor)
            public NriTextureResource output;             // denoised/upscaled (ScalingOutputColor)
            public NriTextureResource mv;                 // motion vectors
            public NriTextureResource depth;              // linear view-Z
            public NriTextureResource diffAlbedo;         // Albedo
            public NriTextureResource specAlbedo;         // SpecularAlbedo
            public NriTextureResource normalRoughness;    // NormalRoughness (roughness packed in .w)
            public NriTextureResource specularMvOrHitTex; // SpecularHitDistance or SpecularMotionVectors
        }

        public SLDlssrr(string camName)
        {
            _instanceId = CreateSLDlssrrInstance();
            _cameraName = camName;
            _buffer     = new NativeArray<SLDlssrrFrameData>(BufferCount, Allocator.Persistent);
            Debug.Log($"[SL DLSS RR] Created instance {_instanceId} for camera {_cameraName}");
        }

        private SLDlssrrFrameData GetData(SLDlssrrFrameInput fi, DlssrrResources res,
                                          UpscalerMode upscalerMode, DlssRRPreset preset)
        {
            // Streamline common constants (row-major; SL receives Unity column-major bytes and
            // transposes). Matches StreamlineProbe's DLSS-G constants derivation.
            Matrix4x4 viewToWorld     = fi.worldToView.inverse;
            Matrix4x4 clipToCameraView = fi.viewToClip.inverse;
            Matrix4x4 worldToClipInv   = fi.worldToClip.inverse;
            Matrix4x4 prevWorldToClipInv = fi.prevWorldToClip.inverse;
            Matrix4x4 clipToPrevClip   = fi.prevWorldToClip * worldToClipInv;
            Matrix4x4 prevClipToClip   = fi.worldToClip * prevWorldToClipInv;

            // Camera basis (world space) from viewToWorld columns. Unity view space looks down -Z.
            float3 camRight =  new float3(viewToWorld.m00, viewToWorld.m10, viewToWorld.m20);
            float3 camUp    =  new float3(viewToWorld.m01, viewToWorld.m11, viewToWorld.m21);
            float3 camFwd   = -new float3(viewToWorld.m02, viewToWorld.m12, viewToWorld.m22);

            float renderW = math.max(1, fi.renderResolution.x);
            float renderH = math.max(1, fi.renderResolution.y);

            return new SLDlssrrFrameData
            {
                inputTex            = res.input.NativePtr,
                outputTex           = res.output.NativePtr,
                mvTex               = res.mv.NativePtr,
                depthTex            = res.depth.NativePtr,
                diffuseAlbedoTex    = res.diffAlbedo.NativePtr,
                specularAlbedoTex   = res.specAlbedo.NativePtr,
                normalRoughnessTex  = res.normalRoughness.NativePtr,
                specularMvOrHitTex  = res.specularMvOrHitTex.NativePtr,

                inputState           = D3D12_STATE_UNORDERED_ACCESS,
                outputState          = D3D12_STATE_UNORDERED_ACCESS,
                mvState              = D3D12_STATE_UNORDERED_ACCESS,
                depthState           = D3D12_STATE_UNORDERED_ACCESS,
                diffuseAlbedoState   = D3D12_STATE_UNORDERED_ACCESS,
                specularAlbedoState  = D3D12_STATE_UNORDERED_ACCESS,
                normalRoughnessState = D3D12_STATE_UNORDERED_ACCESS,
                specularMvOrHitState = D3D12_STATE_UNORDERED_ACCESS,

                worldToViewMatrix = fi.worldToView,
                viewToWorldMatrix = viewToWorld,

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

                useSpecularMotionVector = (byte)(fi.useSpecularMotionVector ? 1 : 0),
                upscalerMode            = (byte)upscalerMode,
                preset                  = (byte)preset,
            };
        }

        public IntPtr GetInteropDataPtr(SLDlssrrFrameInput fi, DlssrrResources res,
                                        UpscalerMode upscalerMode, DlssRRPreset preset = DlssRRPreset.Default)
        {
            var index = (int)(fi.frameIndex % BufferCount);
            _buffer[index] = GetData(fi, res, upscalerMode, preset);
            unsafe
            {
                return (IntPtr)_buffer.GetUnsafePtr() + index * sizeof(SLDlssrrFrameData);
            }
        }

        public void Dispose()
        {
            if (_buffer.IsCreated)
                _buffer.Dispose();

            DestroySLDlssrrInstance(_instanceId);
            Debug.Log($"[SL DLSS RR] Destroyed instance {_instanceId} for camera {_cameraName}");
        }
    }
}
