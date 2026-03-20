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
    public class TemporalResamplingPass: ScriptableRenderPass
    {
        private readonly RayTracingShader _opaqueTs;
        private Resource _resource;
        private Settings _settings;


        public TemporalResamplingPass(RayTracingShader opaqueTs)
        {
            _opaqueTs = opaqueTs;
        }

        public void Setup(Resource sharcResource, Settings sharcSettings)
        {
            _resource = sharcResource;
            _settings = sharcSettings;
        }

        public class Resource
        {
            internal GraphicsBuffer ConstantBuffer;
            internal GraphicsBuffer ResamplingConstantBuffer;


            internal RTHandle Mv;
            internal RTHandle ViewZ;
            internal RTHandle NormalRoughness;
            internal RTHandle BaseColorMetalness;
            internal RTHandle GeoNormal;



            internal RTHandle PrevViewZ;
            internal RTHandle PrevNormalRoughness;
            internal RTHandle PrevBaseColorMetalness;
            internal RTHandle PrevGeoNormal;

            internal RtxdiResources RtxdiResources;
        }

        public class Settings
        {
            internal int2 m_RenderResolution;
            internal float resolutionScale;     
        }

        class PassData
        {
            internal RayTracingShader OpaqueTs;
            internal Resource Resource;
            internal Settings Settings;

            internal TextureHandle DirectLighting;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var opaqueTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "TemporalResampling", MarkerFlags.SampleGPU);

            natCmd.BeginSample(opaqueTracingMarker);

            var resource = data.Resource;
            var settings = data.Settings;

            natCmd.SetRayTracingShaderPass(data.OpaqueTs, "Test2");
            natCmd.SetRayTracingConstantBufferParam(data.OpaqueTs, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, "ResampleConstants", resource.ResamplingConstantBuffer);

            natCmd.SetRayTracingBufferParam(data.OpaqueTs, t_LightDataBufferID, resource.RtxdiResources.LightDataBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, t_NeighborOffsetsID, resource.RtxdiResources.NeighborOffsetsBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, u_LightReservoirsID, resource.RtxdiResources.LightReservoirBuffer);


            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_MvID, resource.Mv);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_ViewZID, resource.ViewZ);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_Normal_RoughnessID, resource.NormalRoughness);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_BaseColor_MetalnessID, resource.BaseColorMetalness);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_DirectLightingID, data.DirectLighting);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevViewZID, resource.PrevViewZ);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevNormalRoughnessID, resource.PrevNormalRoughness);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevBaseColorMetalnessID, resource.PrevBaseColorMetalness);
            
            
            natCmd.SetRayTracingTextureParam(data.OpaqueTs,"gOut_GeoNormal", resource.GeoNormal);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs,"gIn_PrevGeoNormal", resource.PrevGeoNormal);
            

            uint rectWmod = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint rectHmod = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);

            // Debug.Log($"Dispatch Rays Size: {rectWmod} x {rectHmod}");


            natCmd.DispatchRays(data.OpaqueTs, "MainRayGenShader", rectWmod, rectHmod, 1);

            natCmd.EndSample(opaqueTracingMarker);

        }


        private TextureHandle CreateTex(TextureDesc textureDesc, RenderGraph renderGraph, string name, GraphicsFormat format)
        {
            textureDesc.format = format;
            textureDesc.name = name;
            return renderGraph.CreateTexture(textureDesc);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("TemporalResampling", out var passData);

            passData.OpaqueTs = _opaqueTs;

            passData.Resource = _resource;
            passData.Settings = _settings;

            var resourceData = frameData.Get<UniversalResourceData>();

            var textureDesc = resourceData.activeColorTexture.GetDescriptor(renderGraph);
            textureDesc.enableRandomWrite = true;
            textureDesc.depthBufferBits = 0;
            textureDesc.clearBuffer = false;
            textureDesc.discardBuffer = false;
            textureDesc.width = _settings.m_RenderResolution.x;
            textureDesc.height = _settings.m_RenderResolution.y;

            
            var ptContextItem = frameData.Get<PTContextItem>();
            
            passData.DirectLighting = ptContextItem.DirectLighting;
            
            builder.UseTexture(passData.DirectLighting,  AccessFlags.ReadWrite);

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}