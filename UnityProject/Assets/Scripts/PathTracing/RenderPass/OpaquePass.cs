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
    public class OpaquePass : ScriptableRenderPass
    {
        private readonly RayTracingShader _opaqueTs;
        private Resource _resource;
        private Settings _settings;


        public OpaquePass(RayTracingShader opaqueTs)
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

            internal GraphicsBuffer HashEntriesBuffer;
            internal GraphicsBuffer AccumulationBuffer;
            internal GraphicsBuffer ResolvedBuffer;

            internal GraphicsBuffer SpotLightBuffer;
            internal GraphicsBuffer AreaLightBuffer;
            internal GraphicsBuffer PointLightBuffer;

            internal GraphicsBuffer ScramblingRanking;
            internal GraphicsBuffer Sobol;


            internal RTHandle Mv;
            internal RTHandle ViewZ;
            internal RTHandle NormalRoughness;
            internal RTHandle BaseColorMetalness;
            internal RTHandle GeoNormal;
            internal RTHandle DirectLighting;


            internal RTHandle Penumbra;
            internal RTHandle Diff;
            internal RTHandle Spec;


            internal RTHandle PrevViewZ;
            internal RTHandle PrevNormalRoughness;
            internal RTHandle PrevBaseColorMetalness;
            internal RTHandle PrevGeoNormal;

            internal RTHandle PsrThroughput;
        }

        public class Settings
        {
            internal int2 m_RenderResolution;
            internal float resolutionScale;     
            internal int convergenceStep;
        }

        class PassData
        {
            internal RayTracingShader OpaqueTs;
            internal Resource Resource;
            internal Settings Settings;

            internal TextureHandle OutputTexture;
            internal TextureHandle DirectEmission;
            internal TextureHandle ComposedDiff;
            internal TextureHandle ComposedSpecViewZ;
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var natCmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            var opaqueTracingMarker = RenderPassMarkers.OpaqueTracing;

            natCmd.BeginSample(opaqueTracingMarker);

            var resource = data.Resource;
            var settings = data.Settings;

            natCmd.SetRayTracingShaderPass(data.OpaqueTs, "Test2");
            natCmd.SetRayTracingConstantBufferParam(data.OpaqueTs, paramsID, resource.ConstantBuffer, 0, resource.ConstantBuffer.stride);

            natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_ScramblingRankingID, resource.ScramblingRanking);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_SobolID, resource.Sobol);

            natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_HashEntriesID, resource.HashEntriesBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_AccumulationBufferID, resource.AccumulationBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, g_ResolvedBufferID, resource.ResolvedBuffer);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_OutputID, data.OutputTexture);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_MvID, resource.Mv);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_ViewZID, resource.ViewZ);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_Normal_RoughnessID, resource.NormalRoughness);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_BaseColor_MetalnessID, resource.BaseColorMetalness);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_DirectLightingID, resource.DirectLighting);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_DirectEmissionID, data.DirectEmission);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_PsrThroughputID, resource.PsrThroughput);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_ShadowDataID, resource.Penumbra);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_DiffID, resource.Diff);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, g_SpecID, resource.Spec);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevComposedDiffID, data.ComposedDiff);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevComposedSpec_PrevViewZID, data.ComposedSpecViewZ);

            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevViewZID, resource.PrevViewZ);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevNormalRoughnessID, resource.PrevNormalRoughness);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevBaseColorMetalnessID, resource.PrevBaseColorMetalness);
            
            
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gOut_GeoNormalID, resource.GeoNormal);
            natCmd.SetRayTracingTextureParam(data.OpaqueTs, gIn_PrevGeoNormalID, resource.PrevGeoNormal);
            
            
            

            natCmd.SetRayTracingBufferParam(data.OpaqueTs, gIn_SpotLightsID, resource.SpotLightBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, gIn_AreaLightsID, resource.AreaLightBuffer);
            natCmd.SetRayTracingBufferParam(data.OpaqueTs, gIn_PointLightsID, resource.PointLightBuffer);


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
            using var builder = renderGraph.AddUnsafePass<PassData>("Opaque", out var passData);

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

            
            var ptContextItem = frameData.Create<PTContextItem>();
            
            ptContextItem.OutputTexture = CreateTex(textureDesc, renderGraph, "PathTracingOutput", GraphicsFormat.R16G16B16A16_SFloat);
            ptContextItem.DirectEmission = CreateTex(textureDesc, renderGraph, "DirectEmission", GraphicsFormat.B10G11R11_UFloatPack32);
            ptContextItem.ComposedDiff = CreateTex(textureDesc, renderGraph, "ComposedDiff", GraphicsFormat.R16G16B16A16_SFloat);
            ptContextItem.ComposedSpecViewZ = CreateTex(textureDesc, renderGraph, "ComposedSpec_ViewZ", GraphicsFormat.R16G16B16A16_SFloat);
            
            passData.OutputTexture = ptContextItem.OutputTexture;
            passData.DirectEmission = ptContextItem.DirectEmission;
            passData.ComposedDiff = ptContextItem.ComposedDiff;
            passData.ComposedSpecViewZ = ptContextItem.ComposedSpecViewZ;
            
            builder.UseTexture(passData.OutputTexture,  AccessFlags.ReadWrite);
            builder.UseTexture(passData.DirectEmission,  AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedDiff,  AccessFlags.ReadWrite);
            builder.UseTexture(passData.ComposedSpecViewZ,  AccessFlags.ReadWrite);
            

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => { ExecutePass(data, context); });
        }
    }
}