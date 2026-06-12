using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NativeRender
{
    /// <summary>
    /// Records a <c>ClearUnorderedAccessViewFloat</c> on a UAV-capable D3D12 texture owned by
    /// Unity (a RenderTexture created with enableRandomWrite, passed by GetNativeTexturePtr).
    ///
    /// This replicates nvrhi's <c>CommandList::clearTextureFloat</c> UAV path: one clear per mip,
    /// no pipeline and no draw — unlike a fullscreen "clear draw", it appears in captures exactly
    /// as the original donut/RTXPT apps record their texture clears. State transitions go through
    /// Unity's resource-state tracker inside the plugin callback.
    /// </summary>
    public static class NativeTextureClear
    {
        // Blob layout must match ClearTextureUavFloatEventData in Plugin.cpp (Pack=4):
        // uint64 resource + uint32 dxgiFormat + float4 color.
        private const int BlobSize = 28;

        /// <summary>
        /// Records the clear on <paramref name="cmd"/> (deferred to the render thread).
        /// <paramref name="dxgiFormat"/> is the UAV view format; pass 0 (UNKNOWN) to derive it
        /// from the resource desc — supply it explicitly for typeless resources.
        /// </summary>
        public static unsafe void ClearUavFloat(CommandBuffer cmd, IntPtr textureResource, uint dxgiFormat, Color color)
        {
            if (textureResource == IntPtr.Zero) return;

            IntPtr blob = NativeRenderPlugin.NR_NSB_AllocFlushBuffer(BlobSize);
            if (blob == IntPtr.Zero) return;

            byte* p = (byte*)blob;
            *(ulong*)(p + 0) = (ulong)textureResource.ToInt64();
            *(uint*)(p + 8)  = dxgiFormat;
            float* c = (float*)(p + 12);
            c[0] = color.r; c[1] = color.g; c[2] = color.b; c[3] = color.a;

            cmd.IssuePluginEventAndData(NativeRenderPlugin.NR_GetClearTextureUavFloatCallbackPtr(), 1, blob);
        }
    }
}
