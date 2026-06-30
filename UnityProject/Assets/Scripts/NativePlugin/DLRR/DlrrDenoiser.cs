using System;
using System.Runtime.InteropServices;
using Nri;
using PathTracing;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace DLRR
{
    /// <summary>
    /// DEPRECATED — NRI/NGX DLSS Ray Reconstruction path. Cannot run concurrently with
    /// DLSS-G frame generation (it drives a separate NGX upscaler outside the Streamline
    /// slInit). Use <see cref="SLDLRR.SLDlssrr"/> instead; retained for offline A/B only.
    /// </summary>
    [Obsolete("Use SLDLRR.SLDlssrr (Streamline) — the NRI DLSS-RR path cannot run concurrently with DLSS-G frame generation.")]
    public class DlrrDenoiser : IDisposable
    {
        [DllImport("Denoiser")]
        private static extern int CreateDLRRInstance();

        [DllImport("Denoiser")]
        private static extern void DestroyDLRRInstance(int id);

        [DllImport("Denoiser")]
        private static extern bool DLRR_QueryOptimalRenderSize(
            uint outputWidth, uint outputHeight, byte mode,
            out uint renderWidth, out uint renderHeight);

        /// <summary>
        /// Queries the NGX/Streamline-recommended render resolution for a given output
        /// resolution and DLSS mode, matching RTXPT's QueryDLSSOptimalSettings path.
        /// Results are cached in the native plugin; the first call per unique
        /// (outputRes, mode) may stall briefly for GPU idle.
        /// Returns false if the native plugin is unavailable, in which case
        /// <paramref name="renderRes"/> is set to <paramref name="outputRes"/>.
        /// </summary>
        public static bool TryGetOptimalRenderSize(int2 outputRes, UpscalerMode mode, out int2 renderRes)
        {
            if (DLRR_QueryOptimalRenderSize(
                    (uint)outputRes.x, (uint)outputRes.y, (byte)mode,
                    out uint rw, out uint rh))
            {
                renderRes = new int2((int)rw, (int)rh);
                return true;
            }
            renderRes = outputRes;
            return false;
        }

        private readonly int                        _instanceId;
        private          NativeArray<DlrrFrameData> _buffer;
        private const    int                        BufferCount = 3;
        private readonly string                     _cameraName;

        /// <summary>
        /// Per-frame camera data filled by PathTracingFeature from CameraFrameState.
        /// DLRRDenoiser does not depend on CameraFrameState directly.
        /// </summary>
        public struct DlrrFrameInput
        {
            public Matrix4x4 worldToView;
            public Matrix4x4 viewToClip;
            public float2    viewportJitter;
            public int2      renderResolution;
            public uint      frameIndex;
            public ushort    outputWidth;
            public ushort    outputHeight;
            public bool      useMv;
        }

        /// <summary>
        /// DLSS-RR textures packed by PathTracingFeature and passed each frame.
        /// </summary>
        public struct DlrrResources
        {
            public NriTextureResource input; // Composed
            public NriTextureResource output; // DlssOutput
            public NriTextureResource mv; // IN_MV
            public NriTextureResource depth; // IN_VIEWZ
            public NriTextureResource diffAlbedo; // RRGuide_DiffAlbedo
            public NriTextureResource specAlbedo; // RRGuide_SpecAlbedo
            public NriTextureResource normalRoughness; // RRGuide_Normal_Roughness
            public NriTextureResource specularMvOrHitTex; // RRGuide_SpecHitDistance
        }

        public DlrrDenoiser(string camName)
        {
            _instanceId = CreateDLRRInstance();
            _cameraName = camName;
            _buffer     = new NativeArray<DlrrFrameData>(BufferCount, Allocator.Persistent);
            Debug.Log($"[DLSS RR] Created Denoiser Instance {_instanceId} for Camera {_cameraName}");
        }

        private DlrrFrameData GetData(DlrrFrameInput fi, DlrrResources res, float resolutionScale, UpscalerMode upscalerMode, DlssRRPreset preset)
        {
            ushort rectW = (ushort)(fi.renderResolution.x * resolutionScale + 0.5f);
            ushort rectH = (ushort)(fi.renderResolution.y * resolutionScale + 0.5f);

            var data = new DlrrFrameData
            {
                inputTex                = res.input.NriPtr,
                outputTex               = res.output.NriPtr,
                mvTex                   = res.mv.NriPtr,
                depthTex                = res.depth.NriPtr,
                diffuseAlbedoTex        = res.diffAlbedo.NriPtr,
                specularAlbedoTex       = res.specAlbedo.NriPtr,
                normalRoughnessTex      = res.normalRoughness.NriPtr,
                specularMvOrHitTex      = res.specularMvOrHitTex.NriPtr, // 默认传hitT，后续可根据外部参数切换
                useSpecularMotionVector = fi.useMv,
                worldToViewMatrix       = fi.worldToView,
                viewToClipMatrix        = fi.viewToClip,
                outputWidth             = fi.outputWidth,
                outputHeight            = fi.outputHeight,
                currentWidth            = rectW,
                currentHeight           = rectH,
                upscalerMode            = upscalerMode,
                preset                  = (byte)preset,
                cameraJitter            = fi.viewportJitter,
                instanceId              = _instanceId,
            };

            return data;
        }

        public IntPtr GetInteropDataPtr(DlrrFrameInput fi, DlrrResources res, float resolutionScale, UpscalerMode upscalerMode, DlssRRPreset preset = DlssRRPreset.Default)
        {
            var index = (int)(fi.frameIndex % BufferCount);
            _buffer[index] = GetData(fi, res, resolutionScale, upscalerMode, preset);
            unsafe
            {
                return (IntPtr)_buffer.GetUnsafePtr() + index * sizeof(DlrrFrameData);
            }
        }

        public void Dispose()
        {
            if (_buffer.IsCreated)
            {
                _buffer.Dispose();
            }

            DestroyDLRRInstance(_instanceId);


            Debug.Log($"[DLSS RR] Destroyed Denoiser Instance {_instanceId} for Camera {_cameraName} - Dispose Complete");
        }
    }
}