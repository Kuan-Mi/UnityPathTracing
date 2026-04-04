using mini;
using Unity.Mathematics;
using Unity.Profiling;
using Unity.Profiling.LowLevel;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using static PathTracing.ShaderIDs;

namespace PathTracing
{
    public class GBufferPass : ScriptableRenderPass
    {
        private readonly RayTracingShader _gBufferTs;
        private RtxdiPassContext _context;

        public GBufferPass(RayTracingShader gBufferTs)
        {
            _gBufferTs = gBufferTs;
        }

        public void Setup(RtxdiPassContext ctx)
        {
            _context = ctx;
        }

        class PassData
        {
            internal RayTracingShader gBufferTs;
            internal RtxdiPassContext Context;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var gBufferTracingMarker = RenderPassMarkers.GBufferRay;

            natCmd.BeginSample(gBufferTracingMarker);

            var ctx = data.Context;

            natCmd.SetRayTracingShaderPass(data.gBufferTs, "Test2");
            natCmd.SetRayTracingConstantBufferParam(data.gBufferTs, paramsID, ctx.ConstantBuffer, 0, ctx.ConstantBuffer.stride);

            natCmd.SetRayTracingTextureParam(data.gBufferTs, u_ViewDepthID, ctx.ViewDepth);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, u_DiffuseAlbedoID, ctx.DiffuseAlbedo);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, u_SpecularRoughID, ctx.SpecularRough);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, u_NormalsID, ctx.Normals);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, u_GeoNormalsID, ctx.GeoNormals);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, u_EmissiveID, ctx.Emissive);
            natCmd.SetRayTracingTextureParam(data.gBufferTs, u_MotionVectorsID, ctx.MotionVectors);

            uint rectWmod = (uint)(ctx.RenderResolution.x * ctx.ResolutionScale + 0.5f);
            uint rectHmod = (uint)(ctx.RenderResolution.y * ctx.ResolutionScale + 0.5f);

            // Debug.Log($"Dispatch Rays Size: {rectWmod} x {rectHmod}");


            natCmd.DispatchRays(data.gBufferTs, "MainRayGenShader", rectWmod, rectHmod, 1);

            natCmd.EndSample(gBufferTracingMarker);
        }


        private TextureHandle CreateTex(TextureDesc textureDesc, RenderGraph renderGraph, string name, GraphicsFormat format)
        {
            textureDesc.format = format;
            textureDesc.name = name;
            return renderGraph.CreateTexture(textureDesc);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("GBuffer", out var passData);

            passData.gBufferTs = _gBufferTs;
            passData.Context = _context;

            var resourceData = frameData.Get<UniversalResourceData>();


            if (!frameData.Contains<PTContextItem>())
            {
                var ptContextItem = frameData.Create<PTContextItem>();
            }

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}