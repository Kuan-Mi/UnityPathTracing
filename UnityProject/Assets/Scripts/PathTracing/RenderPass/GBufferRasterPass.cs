using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RendererUtils;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    /// <summary>
    /// Rasterization-based G-Buffer fill pass.
    ///
    /// Renders all opaque objects tagged with the "GBufferRaster" shader pass and writes the
    /// same seven G-Buffer textures that GBuffer.hlsl (RT) does, using Multiple Render Targets.
    ///
    /// MRT binding order (matches SV_TargetN in GBufferRaster.hlsl):
    ///   color[0] → ViewDepth       R32_Float
    ///   color[1] → DiffuseAlbedo   R32_UInt
    ///   color[2] → SpecularRough   R32_UInt
    ///   color[3] → Normals         R32_UInt
    ///   color[4] → GeoNormals      R32_UInt
    ///   color[5] → Emissive        R16G16B16A16_SFloat
    ///   color[6] → MotionVectors   R16G16B16A16_SFloat
    ///   depth    → DepthBuffer     D32
    ///
    /// Integration with PathTracingFeature:
    ///   1. Add a GBufferRasterPass field and instantiate it in Create().
    ///   2. Call pass.Setup(gBufferResource, rasterResource, settings) each frame.
    ///   3. Call renderer.EnqueuePass(_gBufferRasterPass) in AddRenderPasses().
    ///   The pass reuses the same GBufferPass.Resource RTHandles so it writes
    ///   directly into the buffers that NRD / path tracing already reads from.
    /// </summary>
    public class GBufferRasterPass : ScriptableRenderPass
    {
        // ── Shader tag ─────────────────────────────────────────────────────────
        private static readonly ShaderTagId k_ShaderTag = new ShaderTagId("GBufferRaster");

        // ── Resources ──────────────────────────────────────────────────────────
        private GBufferPass.Resource _gBufferResource;
        private Resource             _rasterResource;
        private GBufferPass.Settings _settings;

        public GBufferRasterPass()
        {
        }

        public void Setup(
            GBufferPass.Resource gBufferResource,
            Resource             rasterResource,
            GBufferPass.Settings settings)
        {
            _gBufferResource = gBufferResource;
            _rasterResource  = rasterResource;
            _settings        = settings;
        }

        // ── Per-pass resources ────────────────────────────────────────────────
        /// <summary>
        /// Resources owned by GBufferRasterPass (separate from GBufferPass.Resource).
        /// </summary>
        public class Resource
        {
            /// <summary>
            /// Full-resolution depth buffer used for hardware depth testing.
            /// Format: Depth32 (or Depth24Stencil8).
            /// Allocated and freed by the owner (e.g., PathTracingFeature).
            /// </summary>
            internal RTHandle DepthBuffer;

            /// <summary>
            /// Allocates / re-allocates the depth buffer to match <paramref name="renderResolution"/>.
            /// </summary>
            /// <returns>True if the allocation changed.</returns>
            public bool EnsureResources(int2 renderResolution)
            {
                int w = renderResolution.x;
                int h = renderResolution.y;

                if (DepthBuffer == null
                    || DepthBuffer.rt == null
                    || DepthBuffer.rt.width  != w
                    || DepthBuffer.rt.height != h)
                {
                    DepthBuffer?.Release();
                    DepthBuffer = RTHandles.Alloc(
                        w, h,
                        depthBufferBits: DepthBits.Depth32,
                        colorFormat: GraphicsFormat.None,
                        dimension: TextureDimension.Tex2D,
                        name: "GBufferRaster_Depth");
                    return true;
                }

                return false;
            }

            public void Dispose()
            {
                DepthBuffer?.Release();
                DepthBuffer = null;
            }
        }

        // ── RenderGraph pass data ─────────────────────────────────────────────
        private class PassData
        {
            internal GraphicsBuffer     ConstantBuffer;
            internal RendererListHandle RendererList;
        }

        // ── RecordRenderGraph ─────────────────────────────────────────────────
        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var renderingData = frameData.Get<UniversalRenderingData>();
            var cameraData    = frameData.Get<UniversalCameraData>();

            if (!frameData.Contains<PTContextItem>())
                frameData.Create<PTContextItem>();

            var rendererListDesc = new RendererListDesc(k_ShaderTag, renderingData.cullResults, cameraData.camera)
            {
                sortingCriteria  = SortingCriteria.CommonOpaque,
                renderQueueRange = RenderQueueRange.opaque,
                layerMask        = cameraData.camera.cullingMask,
            };

            using var builder = renderGraph.AddRasterRenderPass<PassData>("GBufferRaster", out var passData);

            passData.ConstantBuffer = _gBufferResource.ConstantBuffer;
            passData.RendererList   = renderGraph.CreateRendererList(rendererListDesc);

            var gb = _gBufferResource;
            var rs = _rasterResource;

            // Import external RTHandles and bind as MRT (order must match SV_TargetN in GBufferRaster.hlsl)
            builder.SetRenderAttachment(renderGraph.ImportTexture(gb.ViewDepth),     0, AccessFlags.Write);
            builder.SetRenderAttachment(renderGraph.ImportTexture(gb.DiffuseAlbedo), 1, AccessFlags.Write);
            builder.SetRenderAttachment(renderGraph.ImportTexture(gb.SpecularRough), 2, AccessFlags.Write);
            builder.SetRenderAttachment(renderGraph.ImportTexture(gb.Normals),       3, AccessFlags.Write);
            builder.SetRenderAttachment(renderGraph.ImportTexture(gb.GeoNormals),    4, AccessFlags.Write);
            builder.SetRenderAttachment(renderGraph.ImportTexture(gb.Emissive),      5, AccessFlags.Write);
            builder.SetRenderAttachment(renderGraph.ImportTexture(gb.MotionVectors), 6, AccessFlags.Write);
            builder.SetRenderAttachmentDepth(renderGraph.ImportTexture(rs.DepthBuffer), AccessFlags.Write);

            builder.UseRendererList(passData.RendererList);
            builder.AllowPassCulling(false);
            builder.AllowGlobalStateModification(true);
            builder.SetRenderFunc((PassData data, RasterGraphContext context) => ExecutePass(data, context));
        }

        // ── ExecutePass ───────────────────────────────────────────────────────
        private static void ExecutePass(PassData data, RasterGraphContext context)
        {
            // Bind GlobalConstants so Shared.hlsl globals are available.
            context.cmd.SetGlobalConstantBuffer(
                data.ConstantBuffer, paramsID,
                0, data.ConstantBuffer.stride);

            // Clear only depth (MRT color targets are written by geometry below).
            context.cmd.ClearRenderTarget(true, false, Color.clear, 1.0f);

            // Draw opaque objects using the "GBufferRaster" shader pass.
            context.cmd.DrawRendererList(data.RendererList);
        }
    }
}
