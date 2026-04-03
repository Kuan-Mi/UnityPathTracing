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
    ///   The pass reuses the same RtxdiPassContext RTHandles so it writes
    ///   directly into the buffers that NRD / path tracing already reads from.
    /// </summary>
    public class GBufferRasterPass : ScriptableRenderPass
    {
        // ── Shader tag ─────────────────────────────────────────────────────────
        private static readonly ShaderTagId k_ShaderTag = new ShaderTagId("GBufferRaster");

        // ── Resources ──────────────────────────────────────────────────────────
        private RtxdiPassContext _context;
        private Resource         _rasterResource;

        public GBufferRasterPass()
        {
        }

        public void Setup(RtxdiPassContext ctx, Resource rasterResource)
        {
            _context        = ctx;
            _rasterResource = rasterResource;
        }

        // ── Per-pass resources ────────────────────────────────────────────────
        /// <summary>
        /// Resources owned by GBufferRasterPass (separate from RtxdiPassContext).
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
                sortingCriteria       = SortingCriteria.CommonOpaque,
                renderQueueRange      = RenderQueueRange.opaque,
                layerMask             = cameraData.camera.cullingMask,
                rendererConfiguration = PerObjectData.MotionVectors,
            };

            using var builder = renderGraph.AddRasterRenderPass<PassData>("GBufferRaster", out var passData);

            passData.ConstantBuffer = _context.ConstantBuffer;
            passData.RendererList   = renderGraph.CreateRendererList(rendererListDesc);

            var ctx = _context;
            var rs  = _rasterResource;

            // Import external RTHandles and bind as MRT (order must match SV_TargetN in GBufferRaster.hlsl)
            builder.SetRenderAttachment(renderGraph.ImportTexture(ctx.ViewDepth),     0, AccessFlags.Write);
            builder.SetRenderAttachment(renderGraph.ImportTexture(ctx.DiffuseAlbedo), 1, AccessFlags.Write);
            builder.SetRenderAttachment(renderGraph.ImportTexture(ctx.SpecularRough), 2, AccessFlags.Write);
            builder.SetRenderAttachment(renderGraph.ImportTexture(ctx.Normals),       3, AccessFlags.Write);
            builder.SetRenderAttachment(renderGraph.ImportTexture(ctx.GeoNormals),    4, AccessFlags.Write);
            builder.SetRenderAttachment(renderGraph.ImportTexture(ctx.Emissive),      5, AccessFlags.Write);
            builder.SetRenderAttachment(renderGraph.ImportTexture(ctx.MotionVectors), 6, AccessFlags.Write);
            builder.SetRenderAttachmentDepth(renderGraph.ImportTexture(rs.DepthBuffer), AccessFlags.Write);

            builder.UseRendererList(passData.RendererList);
            builder.AllowPassCulling(false);
            builder.AllowGlobalStateModification(true);
            builder.SetRenderFunc((PassData data, RasterGraphContext context) => ExecutePass(data, context));
        }

        // ── ExecutePass ───────────────────────────────────────────────────────
        private static void ExecutePass(PassData data, RasterGraphContext context)
        {
            var marker = RenderPassMarkers.GBufferRaster;
            context.cmd.BeginSample(marker);
            // Bind GlobalConstants so Shared.hlsl globals are available.
            context.cmd.SetGlobalConstantBuffer(
                data.ConstantBuffer, paramsID,
                0, data.ConstantBuffer.stride);

            // Clear ViewDepth (color0) to BACKGROUND_DEPTH (1e5) and reset hardware depth to 1.0.
            context.cmd.ClearRenderTarget(RTClearFlags.Color0 | RTClearFlags.Depth, new Color(-1e5f, 0, 0, 0), 1.0f, 0);
            

            // Draw opaque objects using the "GBufferRaster" shader pass.
            context.cmd.DrawRendererList(data.RendererList);
            
            context.cmd.EndSample(marker);
        }
    }
}
