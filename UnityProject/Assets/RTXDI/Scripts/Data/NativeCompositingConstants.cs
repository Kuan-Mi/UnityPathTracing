using System;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Mirrors <c>CompositingConstants</c> from
    /// <c>RTXDI/Samples/FullSample/Shaders/SharedShaderInclude/ShaderParameters.h</c>.
    ///
    /// Used as <c>ConstantBuffer&lt;CompositingConstants&gt; g_Const : register(b0)</c> in
    /// <c>CompositingPass.hlsl</c>.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct NativeCompositingConstants
    {
        public NativePlanarViewConstants view;
        public NativePlanarViewConstants viewPrev;

        public uint  enableTextures;
        public uint  denoiserMode;
        public uint  enableEnvironmentMap;
        public uint  environmentMapTextureIndex;

        public float environmentScale;
        public float environmentRotation;
        public float noiseClampLow;
        public float noiseClampHigh;

        public uint  checkerboard;
        // 3 padding floats to keep 16-byte alignment
        public uint  _pad0;
        public uint  _pad1;
        public uint  _pad2;
    }

    public static class NativeCompositingConstantsBuilder
    {
        public static NativeCompositingConstants Build(
            CameraFrameState fs,
            int2             renderResolution,
            float            resolutionScale,
            uint             denoiserMode)
        {
            return new NativeCompositingConstants
            {
                view     = NativeGBufferConstantsBuilder.BuildViewPublic(fs.worldToView,     fs.viewToClip,     fs.worldToClip,     fs.camPos,     renderResolution, resolutionScale, fs.viewportJitter),
                viewPrev = NativeGBufferConstantsBuilder.BuildViewPublic(fs.prevWorldToView, fs.prevViewToClip, fs.prevWorldToClip, fs.prevCamPos, renderResolution, resolutionScale, fs.prevViewportJitter),

                enableTextures         = 1u,
                denoiserMode           = denoiserMode,
                enableEnvironmentMap   = 0u,
                environmentMapTextureIndex = 0u,

                environmentScale    = 1f,
                environmentRotation = 0f,
                noiseClampLow       = 0f,
                noiseClampHigh      = 1e10f,

                checkerboard = 0u,
            };
        }
    }
}
