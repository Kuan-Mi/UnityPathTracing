using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering.Universal;
using static PathTracing.PathTracingUtils;

namespace PathTracing
{
    /// <summary>
    /// Per-camera temporal state: current and previous frame matrices, jitter, resolution scale.
    /// PathTracingFeature owns one instance per camera key and calls Update() once per frame
    /// before passing the data to NRD/DLRR via their own input structs.
    /// </summary>
    public class CameraFrameState
    {
        // ── current frame ──────────────────────────────────────────────
        public Matrix4x4 worldToView;
        public Matrix4x4 worldToClip;
        public Matrix4x4 viewToClip;
        public float3    camPos;
        public float2    ViewportJitter;
        public float     resolutionScale;
        public int2      renderResolution;
        public uint      FrameIndex;

        // ── previous frame ─────────────────────────────────────────────
        public Matrix4x4 prevWorldToView;
        public Matrix4x4 prevWorldToClip;
        public Matrix4x4 prevViewToClip;
        public float3    prevCamPos;
        public float2    PrevViewportJitter;
        public float     prevResolutionScale;

        public CameraFrameState(float initialResolutionScale)
        {
            resolutionScale     = initialResolutionScale;
            prevResolutionScale = initialResolutionScale;
        }

        /// <summary>
        /// Must be called once per frame before GetInteropDataPtr on NRD / DLRR.
        /// Saves current values to prev*, then refreshes from the camera.
        /// </summary>
        public void Update(RenderingData renderingData, PathTracingSetting setting)
        {
            // 1. save prev
            prevWorldToView     = worldToView;
            prevWorldToClip     = worldToClip;
            prevViewToClip      = viewToClip;
            prevCamPos          = camPos;
            PrevViewportJitter  = ViewportJitter;
            prevResolutionScale = resolutionScale;

            // 2. refresh from camera
            var cameraData = renderingData.cameraData;
            var xrPass     = cameraData.xr;
            if (xrPass.enabled)
            {
                worldToView = xrPass.GetViewMatrix();
                var proj    = GL.GetGPUProjectionMatrix(xrPass.GetProjMatrix(), false);
                worldToClip = proj * worldToView;
                viewToClip  = proj;
                var invView = worldToView.inverse;
                camPos      = new float3(invView.m03, invView.m13, invView.m23);
            }
            else
            {
                var cam     = cameraData.camera;
                camPos      = new float3(cam.transform.position.x, cam.transform.position.y, cam.transform.position.z);
                worldToView = cam.worldToCameraMatrix;
                worldToClip = GetWorldToClipMatrix(cam);
                viewToClip  = GL.GetGPUProjectionMatrix(cam.projectionMatrix, false);
            }

            // 3. resolution scale (RR forces 1.0)
            resolutionScale = setting.RR ? 1.0f : setting.resolutionScale;

            // 4. jitter
            PrevViewportJitter = ViewportJitter;
            ViewportJitter     = Halton2D(FrameIndex + 1) - new float2(0.5f, 0.5f);

            // 5. advance frame counter
            FrameIndex++;
        }

        // ── Halton helpers (moved from NRDDenoiser) ────────────────────

        public static float Halton(uint n, uint @base)
        {
            float a       = 1.0f;
            float b       = 0.0f;
            float baseInv = 1.0f / @base;
            while (n != 0)
            {
                a    *= baseInv;
                b    += a * (n % @base);
                n     = (uint)(n * baseInv);
            }
            return b;
        }

        public static uint ReverseBits32(uint v)
        {
            v = ((v & 0x55555555u) << 1)  | ((v >> 1)  & 0x55555555u);
            v = ((v & 0x33333333u) << 2)  | ((v >> 2)  & 0x33333333u);
            v = ((v & 0x0F0F0F0Fu) << 4)  | ((v >> 4)  & 0x0F0F0F0Fu);
            v = ((v & 0x00FF00FFu) << 8)  | ((v >> 8)  & 0x00FF00FFu);
            v = (v << 16) | (v >> 16);
            return v;
        }

        public static float Halton2(uint n)   => ReverseBits32(n) * 2.3283064365386963e-10f;
        public static float Halton1D(uint n)  => Halton2(n);

        public static float2 Halton2D(uint n) => new float2(Halton2(n), Halton(n, 3));
    }
}
