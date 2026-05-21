using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Mirrors <c>PlanarViewConstants</c> from
    /// <c>RenderingPlugin/External/donut/include/donut/shaders/view_cb.h</c>.
    ///
    /// Memory layout must exactly match the HLSL struct so that raw
    /// <see cref="GraphicsBuffer.SetData"/> uploads work without any transposition.
    ///
    /// Convention: Unity stores <c>Matrix4x4</c> column-major. HLSL shaders in this project
    /// use <c>#pragma pack_matrix(row_major)</c> and post-multiply (<c>mul(v, M)</c>).
    /// Uploading Unity column-major data directly into a row-major HLSL <c>float4x4</c>
    /// field is equivalent to the HLSL seeing the transposed matrix, so
    /// <c>mul(v, M_hlsl)</c> = <c>UnityMatrix * v</c> in standard column-vector math.
    /// Therefore matrices are assigned from Unity as-is, with no extra transpose step.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    [Serializable]
    public struct NativePlanarViewConstants
    {
        public float4x4 matWorldToView;
        public float4x4 matViewToClip;
        public float4x4 matWorldToClip;
        public float4x4 matClipToView;
        public float4x4 matViewToWorld;
        public float4x4 matClipToWorld;
        public float4x4 matViewToClipNoOffset;
        public float4x4 matWorldToClipNoOffset;
        public float4x4 matClipToViewNoOffset;
        public float4x4 matClipToWorldNoOffset;
        public float2   viewportOrigin;
        public float2   viewportSize;
        public float2   viewportSizeInv;
        public float2   pixelOffset;
        public float2   clipToWindowScale;
        public float2   clipToWindowBias;
        public float2   windowToClipScale;
        public float2   windowToClipBias;
        public float4   cameraDirectionOrPosition;
    }

    /// <summary>
    /// Mirrors <c>GBufferConstants</c> from
    /// <c>RTXDI/Samples/FullSample/Shaders/SharedShaderInclude/ShaderParameters.h</c>.
    ///
    /// Used as <c>ConstantBuffer&lt;GBufferConstants&gt; g_Const : register(b0)</c> in
    /// <c>RaytracedGBuffer.hlsl</c>.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct NativeGBufferConstants
    {
        public NativePlanarViewConstants view;
        public NativePlanarViewConstants viewPrev;
        public float                     roughnessOverride; // negative = disabled
        public float                     metalnessOverride; // negative = disabled
        public float                     normalMapScale;
        public uint                      enableAlphaTestedGeometry;
        public int2                      materialReadbackPosition; // (-1,-1) = disabled
        public uint                      materialReadbackBufferIndex;
        public uint                      enableTransparentGeometry;
        public float                     textureLodBias;
        public float                     textureGradientScale;
    }

    /// <summary>
    /// Builds a <see cref="NativeGBufferConstants"/> from a <see cref="CameraFrameState"/>
    /// each frame before uploading to GPU.
    /// </summary>
    public static class NativeGBufferConstantsBuilder
    {
        public static NativeGBufferConstants Build(CameraFrameState fs, int2 renderResolution, float resolutionScale)
        {
            return new NativeGBufferConstants
            {
                view     = BuildView(fs.worldToView, fs.viewToClip, fs.worldToClip, fs.camPos, renderResolution, resolutionScale, fs.viewportJitter),
                viewPrev = BuildView(fs.prevWorldToView, fs.prevViewToClip, fs.prevWorldToClip, fs.prevCamPos, renderResolution, resolutionScale, fs.prevViewportJitter),

                roughnessOverride           = -1f, // disabled — use material value
                metalnessOverride           = -1f, // disabled — use material value
                normalMapScale              = 1f,
                enableAlphaTestedGeometry   = 1u,
                materialReadbackPosition    = new int2(-1, -1), // disabled
                materialReadbackBufferIndex = 0u,
                enableTransparentGeometry   = 0u,
                textureLodBias              = 0f,
                textureGradientScale        = 1f,
            };
        }

        public static NativePlanarViewConstants BuildViewPublic(
            Matrix4x4 worldToView, Matrix4x4 viewToClip, Matrix4x4 worldToClip,
            float3 cameraPos, int2 renderResolution, float resolutionScale, float2 jitter)
            => BuildView(worldToView, viewToClip, worldToClip, cameraPos, renderResolution, resolutionScale, jitter);

        static NativePlanarViewConstants BuildView(
            Matrix4x4 worldToView,
            Matrix4x4 viewToClipNoOffset,
            Matrix4x4 worldToClipNoOffset,
            float3 cameraPos,
            int2 renderResolution,
            float resolutionScale,
            float2 jitter)
        {
            var w     = renderResolution.x * resolutionScale;
            var h     = renderResolution.y * resolutionScale;
            var vSize = new float2(w, h);

            // 1. 计算偏移矩阵 (NDC 空间平移)
            float offsetX = 2f * jitter.x / w;
            float offsetY = -2f * jitter.y / h;

            // Unity Matrix4x4.Translate 创建的是列主序平移矩阵
            Matrix4x4 pixelOffsetMatrix    = Matrix4x4.Translate(new Vector3(offsetX, offsetY, 0));
            Matrix4x4 pixelOffsetMatrixInv = Matrix4x4.Translate(new Vector3(-offsetX, -offsetY, 0));

            // 2. 【关键修复】在 Unity 中，应用 NDC 偏移需要左乘 (Pre-multiply)
            // Clip_jittered = T_jitter * Clip_base
            var viewToClip  = pixelOffsetMatrix * viewToClipNoOffset;
            var worldToClip = pixelOffsetMatrix * worldToClipNoOffset;

            // 3. 计算逆矩阵
            // (T * P * V)^-1 = V^-1 * P^-1 * T^-1
            // 在 Unity 中 A * B 的逆是 B.inv * A.inv
            var clipToViewNoOffset  = viewToClipNoOffset.inverse;
            var clipToWorldNoOffset = worldToClipNoOffset.inverse;

            // 注意逆矩阵的顺序也要反过来
            var clipToView  = clipToViewNoOffset * pixelOffsetMatrixInv;
            var clipToWorld = clipToWorldNoOffset * pixelOffsetMatrixInv;

            var viewToWorld = worldToView.inverse;

            // ... 其他逻辑保持不变 ...
            var ctw_scale = new float2(0.5f * w, -0.5f * h);
            var ctw_bias  = new float2(0.5f * w, 0.5f * h);
            var wtc_scale = 1f / ctw_scale;
            var wtc_bias  = -ctw_bias * wtc_scale;

            return new NativePlanarViewConstants
            {
                matWorldToView            = worldToView,
                matViewToClip             = viewToClip,
                matWorldToClip            = worldToClip,
                matClipToView             = clipToView,
                matViewToWorld            = viewToWorld,
                matClipToWorld            = clipToWorld,
                matViewToClipNoOffset     = viewToClipNoOffset,
                matWorldToClipNoOffset    = worldToClipNoOffset,
                matClipToViewNoOffset     = clipToViewNoOffset,
                matClipToWorldNoOffset    = clipToWorldNoOffset,
                viewportOrigin            = float2.zero,
                viewportSize              = vSize,
                viewportSizeInv           = math.rcp(vSize),
                pixelOffset               = jitter,
                clipToWindowScale         = ctw_scale,
                clipToWindowBias          = ctw_bias,
                windowToClipScale         = wtc_scale,
                windowToClipBias          = wtc_bias,
                cameraDirectionOrPosition = new float4(cameraPos, 1f),
            };
        }
    }
}