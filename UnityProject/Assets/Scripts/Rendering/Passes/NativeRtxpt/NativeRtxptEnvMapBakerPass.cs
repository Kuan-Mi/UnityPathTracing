using System;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using Object = UnityEngine.Object;

namespace PathTracing
{
    public class NativeRtxptEnvMapBakerPass : ScriptableRenderPass, IDisposable
    {
        private const int CubeDim                 = 256;
        private const int CubeDimLowRes           = 32;
        private const int ImportanceMapDim        = 1024;
        private const int ImportanceSamples       = 16;
        private const int ImportanceSamplesX      = 4;
        private const int ImportanceSamplesY      = 4;
        private const int BaseLayerGroupsXY       = (CubeDim / 2 + 7) / 8;
        private const int ImportanceBakerGroupsXY = (ImportanceMapDim + 15) / 16;

        private readonly NativeComputePipeline      _baseLayerCs;
        private readonly NativeComputeDescriptorSet _baseLayerDs;
        private readonly NativeComputePipeline      _importanceBakerCs;
        private readonly NativeComputeDescriptorSet _importanceBakerDs;

        private static readonly uint[] s_envBakerWords   = new uint[704 / 4];
        private static readonly uint[] s_importanceWords = new uint[48 / 4];

        private NativeRtxptPassContext _ctx;

        public NativeRtxptEnvMapBakerPass(NativeComputeShader baseLayerCs, NativeComputeShader importanceBakerCs)
        {
            _baseLayerCs       = new NativeComputePipeline(baseLayerCs);
            _baseLayerDs       = new NativeComputeDescriptorSet(_baseLayerCs);
            _importanceBakerCs = new NativeComputePipeline(importanceBakerCs);
            _importanceBakerDs = new NativeComputeDescriptorSet(_importanceBakerCs);
        }

        public void Dispose()
        {
            _baseLayerDs?.Dispose();
            _baseLayerCs?.Dispose();
            _importanceBakerDs?.Dispose();
            _importanceBakerCs?.Dispose();
        }

        public void Setup(NativeRtxptPassContext ctx)
        {
            _ctx = ctx;
            ctx.Textures.EnsureEnvMapResources();
            ctx.Buffers.EnsureEnvBakerBuffers();

            FillEnvBakerConstants(ctx.Setting);
            ctx.Buffers.EnvBakerCb.SetData(s_envBakerWords);
            FillImportanceBakerConstants();
            ctx.Buffers.ImportanceBakerCb.SetData(s_importanceWords);
            ctx.Buffers.RefreshEnvBakerBufferPtrs();

            ctx.BakedEnvCubePtr                = ctx.Textures.EnvCubeMip0.IsCreated ? ctx.Textures.EnvCubeMip0.NativePtr : IntPtr.Zero;
            ctx.EnvImportanceMapPtr            = ctx.Textures.EnvImportanceMap.IsCreated ? ctx.Textures.EnvImportanceMap.NativePtr : IntPtr.Zero;
            ctx.EnvRadianceAndImportanceMapPtr = ctx.Textures.EnvRadianceMap.IsCreated ? ctx.Textures.EnvRadianceMap.NativePtr : IntPtr.Zero;
        }

        private class PassData
        {
            internal NativeComputePipeline      BaseLayerCs;
            internal NativeComputeDescriptorSet BaseLayerDs;
            internal NativeComputePipeline      ImportanceBakerCs;
            internal NativeComputeDescriptorSet ImportanceBakerDs;
            internal IntPtr                     EnvBakerCbPtr;
            internal IntPtr                     ImportanceBakerCbPtr;
            internal IntPtr                     SkyTexturePtr;
            internal IntPtr                     EnvCubeMip0Ptr;
            internal IntPtr                     EnvCubeMip1Ptr;
            internal IntPtr                     ImportanceMapPtr;
            internal IntPtr                     RadianceMapPtr;
            internal RenderTexture              ImportanceMapRt;
            internal RenderTexture              RadianceMapRt;
            internal IntPtr                     DummyCubePtr;
            internal IntPtr                     DummyTex2DPtr;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.EnvMapBaker", out var pd);

            pd.BaseLayerCs          = _baseLayerCs;
            pd.BaseLayerDs          = _baseLayerDs;
            pd.ImportanceBakerCs    = _importanceBakerCs;
            pd.ImportanceBakerDs    = _importanceBakerDs;
            pd.EnvBakerCbPtr        = _ctx.Buffers.EnvBakerCbPtr;
            pd.ImportanceBakerCbPtr = _ctx.Buffers.ImportanceBakerCbPtr;
            var skyTex = _ctx.Setting?.environmentMap;
            pd.SkyTexturePtr    = skyTex != null ? skyTex.GetNativeTexturePtr() : Texture2D.blackTexture.GetNativeTexturePtr();
            pd.EnvCubeMip0Ptr   = _ctx.Textures.EnvCubeMip0.NativePtr;
            pd.EnvCubeMip1Ptr   = _ctx.Textures.EnvCubeMip1.NativePtr;
            pd.ImportanceMapPtr = _ctx.Textures.EnvImportanceMap.NativePtr;
            pd.RadianceMapPtr   = _ctx.Textures.EnvRadianceMap.NativePtr;
            pd.ImportanceMapRt  = _ctx.Textures.EnvImportanceMap.rt;
            pd.RadianceMapRt    = _ctx.Textures.EnvRadianceMap.rt;
            pd.DummyCubePtr     = _ctx.Textures.EnvDummyCube.NativePtr;
            pd.DummyTex2DPtr    = Texture2D.blackTexture.GetNativeTexturePtr();

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.RtxptEnvMapBaker);

            {
                var ds = data.BaseLayerDs;
                ds.SetConstantBuffer("g_Const", data.EnvBakerCbPtr);
                ds.SetTexture("t_SrcEquirectangularEnvMap", data.SkyTexturePtr);
                ds.SetTexture("t_SrcCubemapEnvMap", data.DummyCubePtr);
                ds.SetTexture("t_LowResPrePassCube", data.DummyCubePtr);
                ds.SetTexture("t_ProcSkyTransmittance", data.DummyTex2DPtr);
                ds.SetTexture("t_ProcSkyScatter", data.DummyTex2DPtr);
                ds.SetRWTexture("u_EnvMapCubeFacesDst0", data.EnvCubeMip0Ptr);
                ds.SetRWTexture("u_EnvMapCubeFacesDst1", data.EnvCubeMip1Ptr);
                cmd.BeginSample(RenderPassMarkers.RtxptEnvMapBaseLayer);
                data.BaseLayerCs.Dispatch(cmd, ds, BaseLayerGroupsXY, BaseLayerGroupsXY, 6);
                cmd.EndSample(RenderPassMarkers.RtxptEnvMapBaseLayer);
            }

            {
                var ds = data.ImportanceBakerDs;
                ds.SetConstantBuffer("g_BuilderConsts", data.ImportanceBakerCbPtr);
                ds.SetTexture("t_EnvMapCube", data.EnvCubeMip0Ptr);
                ds.SetRWTexture("u_ImportanceMap", data.ImportanceMapPtr);
                ds.SetRWTexture("u_RadianceMap", data.RadianceMapPtr);
                cmd.BeginSample(RenderPassMarkers.RtxptEnvMapImportanceBaker);
                data.ImportanceBakerCs.Dispatch(cmd, ds, ImportanceBakerGroupsXY, ImportanceBakerGroupsXY, 1);
                cmd.EndSample(RenderPassMarkers.RtxptEnvMapImportanceBaker);
            }

            cmd.GenerateMips(data.ImportanceMapRt);
            cmd.GenerateMips(data.RadianceMapRt);

            cmd.EndSample(RenderPassMarkers.RtxptEnvMapBaker);
        }

        private static void FillEnvBakerConstants(NativeRtxptSetting setting)
        {
            Array.Clear(s_envBakerWords, 0, s_envBakerWords.Length);
            int lightCount = 0;

            foreach (var light in Object.FindObjectsByType<Light>(FindObjectsSortMode.None))
            {
                if (!light.enabled || !light.gameObject.activeInHierarchy) continue;
                if (light.type != LightType.Directional) continue;
                if (lightCount >= 16) break;

                Color linear    = light.color.linear;
                float intensity = light.intensity;
                int   offset    = lightCount * 32;

                WriteF32(s_envBakerWords, offset + 0, linear.r);
                WriteF32(s_envBakerWords, offset + 4, linear.g);
                WriteF32(s_envBakerWords, offset + 8, linear.b);
                WriteF32(s_envBakerWords, offset + 12, intensity);

                Vector3 fwd = light.transform.forward;
                WriteF32(s_envBakerWords, offset + 16, fwd.x);
                WriteF32(s_envBakerWords, offset + 20, fwd.y);
                WriteF32(s_envBakerWords, offset + 24, fwd.z);
                WriteF32(s_envBakerWords, offset + 28, 0.1f);

                lightCount++;
            }

            float envIntensity = setting?.environmentMapIntensity ?? 1.0f;
            Color tint         = setting?.environmentMapTint ?? Color.white;
            bool  hasSky       = setting?.environmentMap != null;
            int   o            = 672;

            WriteF32(s_envBakerWords, o + 0, tint.linear.r * envIntensity);
            WriteF32(s_envBakerWords, o + 4, tint.linear.g * envIntensity);
            WriteF32(s_envBakerWords, o + 8, tint.linear.b * envIntensity);
            WriteU32(s_envBakerWords, o + 12, (uint)lightCount);
            WriteU32(s_envBakerWords, o + 16, (uint)CubeDim);
            WriteU32(s_envBakerWords, o + 20, (uint)CubeDimLowRes);
            WriteU32(s_envBakerWords, o + 24, 0u);
            WriteU32(s_envBakerWords, o + 28, hasSky ? 1u : 0u);
        }

        private static void FillImportanceBakerConstants()
        {
            Array.Clear(s_importanceWords, 0, s_importanceWords.Length);
            WriteU32(s_importanceWords, 0, (uint)CubeDim);
            WriteU32(s_importanceWords, 4, 1u);
            WriteU32(s_importanceWords, 8, 0u);
            WriteU32(s_importanceWords, 12, 0u);
            WriteU32(s_importanceWords, 16, (uint)ImportanceMapDim);
            WriteU32(s_importanceWords, 20, (uint)ImportanceMapDim);
            WriteU32(s_importanceWords, 24, (uint)(ImportanceMapDim * ImportanceSamplesX));
            WriteU32(s_importanceWords, 28, (uint)(ImportanceMapDim * ImportanceSamplesY));
            WriteU32(s_importanceWords, 32, (uint)ImportanceSamplesX);
            WriteU32(s_importanceWords, 36, (uint)ImportanceSamplesY);
            WriteF32(s_importanceWords, 40, 1.0f / ImportanceSamples);
            WriteU32(s_importanceWords, 44, 10u);
        }

        private static void WriteF32(uint[] buf, int offset, float v)
        {
            buf[offset / 4] = unchecked((uint)BitConverter.ToInt32(BitConverter.GetBytes(v), 0));
        }

        private static void WriteU32(uint[] buf, int offset, uint v)
        {
            buf[offset / 4] = v;
        }
    }
}
