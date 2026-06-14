using System;
using NativeRender;
using UnityEngine;

namespace Rendering.Resources
{
    // ----- Merged BLAS resources -----
    public class MergedBlas : IDisposable
    {
        public GraphicsBuffer vb; // float3 world-space positions, stride = 12
        public GraphicsBuffer ib; // uint32 indices,               stride = 4

        public uint vertexCount;

        // Per-submesh records (one entry per submesh of every target in the BLAS).
        public NativeRenderPlugin.SubmeshDesc[] submeshDescs;

        // Parallel to submeshDescs; null element means no OMM for that submesh.
        public OMMCache[] ommCaches;

        public void Dispose()
        {
            vb?.Release();
            vb = null;
            ib?.Release();
            ib = null;
        }
    }
}