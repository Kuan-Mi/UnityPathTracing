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
    public class ShadeSamplesPass: ScriptableRenderPass
    {
        private readonly RayTracingShader _opaqueTs;
        private Resource _resource;
        private Settings _settings;


        public ShadeSamplesPass(RayTracingShader opaqueTs)
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
            internal GraphicsBuffer t_GeometryInstanceToLight;


            internal RTHandle Mv;
            internal RTHandle ViewZ;
            internal RTHandle NormalRoughness;
            internal RTHandle BaseColorMetalness;
            internal RTHandle GeoNormal;
            internal RTHandle DirectLighting;



            internal RTHandle PrevViewZ;
            internal RTHandle PrevNormalRoughness;
            internal RTHandle PrevBaseColorMetalness;
            internal RTHandle PrevGeoNormal;

            internal RtxdiResources RtxdiResources;
            internal Texture2D envTexture;
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
            internal TextureHandle gIn_EmissiveLighting;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var opaqueTracingMarker = new ProfilerMarker(ProfilerCategory.Render, "ShadeSamples", MarkerFlags.SampleGPU);
            var copyGBufferMarker = new ProfilerMarker(ProfilerCategory.Render, "Copy GBuffer to Prev", MarkerFlags.SampleGPU);

            natCmd.BeginSample(opaqueTracingMarker);

            var resource = data.Resource;
            var settings = data.Settings;

            natCmd.SetRayTracingShaderPass(data.OpaqueTs, "Test2");
            natCmd.SetRayTracingConstantBufferParam(data.OpaqueTs, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, "ResampleConstants", resource.ResamplingConstantBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, "t_GeometryInstanceToLight", resource.t_GeometryInstanceToLight);

            natCmd.SetRayTracingBufferParam(data.OpaqueTs, t_LightDataBufferID, resource.RtxdiResources.LightDataBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, t_NeighborOffsetsID, resource.RtxdiResources.NeighborOffsetsBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, u_LightReservoirsID, resource.RtxdiResources.LightReservoirBuffer);


            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_MvID, resource.Mv);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_ViewZID, resource.ViewZ);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_Normal_RoughnessID, resource.NormalRoughness);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_BaseColor_MetalnessID, resource.BaseColorMetalness);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_DirectLightingID, resource.DirectLighting);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs,"gIn_EmissiveLighting", data.gIn_EmissiveLighting);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs,"_environmentMap", resource.envTexture);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevViewZID, resource.PrevViewZ);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevNormalRoughnessID, resource.PrevNormalRoughness);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevBaseColorMetalnessID, resource.PrevBaseColorMetalness);
            
            
            natCmd.SetRayTracingTextureParam(data.OpaqueTs,"gOut_GeoNormal", resource.GeoNormal);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs,"gIn_PrevGeoNormal", resource.PrevGeoNormal);
            

            natCmd.SetRayTracingTextureParam(data.OpaqueTs,"t_LocalLightPdfTexture", resource.RtxdiResources.Scene.localLightPdfTexture);

            natCmd.SetRayTracingBufferParam(data.OpaqueTs, "u_RisBuffer", resource.RtxdiResources.RisBuffer);

            
            uint rectWmod = (uint)(settings.m_RenderResolution.x * settings.resolutionScale + 0.5f);
            uint rectHmod = (uint)(settings.m_RenderResolution.y * settings.resolutionScale + 0.5f);

            // Debug.Log($"Dispatch Rays Size: {rectWmod} x {rectHmod}");


            natCmd.DispatchRays(data.OpaqueTs, "MainRayGenShader", rectWmod, rectHmod, 1);

            natCmd.EndSample(opaqueTracingMarker);

            // 保存当帧 GBuffer 到 prev 纹理，供下一帧 RTXDI 时间复用读取
            natCmd.BeginSample(copyGBufferMarker);
            natCmd.CopyTexture(resource.ViewZ, resource.PrevViewZ);
            natCmd.CopyTexture(resource.NormalRoughness, resource.PrevNormalRoughness);
            natCmd.CopyTexture(resource.BaseColorMetalness, resource.PrevBaseColorMetalness);
            natCmd.CopyTexture(resource.GeoNormal, resource.PrevGeoNormal);
            natCmd.EndSample(copyGBufferMarker);
        }


        private TextureHandle CreateTex(TextureDesc textureDesc, RenderGraph renderGraph, string name, GraphicsFormat format)
        {
            textureDesc.format = format;
            textureDesc.name = name;
            return renderGraph.CreateTexture(textureDesc);
        }


        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("ShadeSamples", out var passData);

            passData.OpaqueTs = _opaqueTs;

            passData.Resource = _resource;
            passData.Settings = _settings;

            var resourceData = frameData.Get<PTContextItem>();

            passData.gIn_EmissiveLighting = resourceData.DirectEmission;

            

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}