using System;
using System.Runtime.InteropServices;

namespace NativeRender
{
    public static class OMMBakerPlugin
    {
        
        private const string BakerDllName  = "OMMBakerPlugin";
        
        // ================================================================
        // OMM CPU Bake API (Editor-time; does not require D3D12 device)
        // ================================================================

        /// <summary>
        /// Runs ommCpuBake synchronously. Returns 1 on success.
        /// Result blobs are held in static plugin storage until NR_FreeBakeResult is called.
        /// alphaPixels: pinned byte[] R8_UNORM (texW * texH bytes).
        /// uvs: pinned float[] of indexCount float2 pairs packed.
        /// indices: pinned byte[] CPU index buffer (indexCount * indexStride bytes).
        /// </summary>
        [DllImport(BakerDllName)]
        public static extern int NR_BakeOMMCPU(
            IntPtr alphaPixels, uint texW, uint texH,
            IntPtr uvs,
            IntPtr indices, uint indexCount, uint indexStride,
            float alphaCutoff,
            byte maxSubdivisionLevel,
            float dynamicSubdivisionScale,
            byte format);

        /// <summary>
        /// Mirrors the native NR_BakeResultDesc struct (Pack=4).
        /// All IntPtr fields point directly into plugin-owned static storage.
        /// Copy the data before calling NR_FreeBakeResult.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        public struct NR_BakeResultDesc
        {
            public IntPtr  arrayData;          // OMM array data blob
            public uint    arrayDataSize;       // bytes
            public IntPtr  descArray;           // ommCpuOpacityMicromapDesc[] blob
            public uint    descArrayByteCount;  // bytes (descArrayCount * 8)
            public uint    descArrayCount;      // number of desc entries
            public IntPtr  indexBuffer;         // per-triangle OMM index blob
            public uint    indexCount;          // number of indices
            public uint    indexStride;         // 1, 2, or 4
            public IntPtr  histogramFlat;       // uint32[histogramCount * 3]: [count, subdivLevel, format]
            public uint    histogramCount;      // number of histogram entries
        }

        /// <summary>
        /// Fills <paramref name="result"/> with pointers into plugin static storage.
        /// Returns 1 if a valid result is available, 0 otherwise.
        /// All data must be copied before calling NR_FreeBakeResult.
        /// </summary>
        [DllImport(BakerDllName)]
        public static extern int NR_GetBakeResult(out NR_BakeResultDesc result);

        /// <summary>Frees the static bake result storage in the plugin.</summary>
        [DllImport(BakerDllName)] public static extern void NR_FreeBakeResult();
    }
}