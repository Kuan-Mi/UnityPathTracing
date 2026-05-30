using System;
using System.Runtime.InteropServices;
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

        // Mirrors HLSL EnvMapBakerConstants (EnvMapBaker.hlsl):
        //   EMB_DirectionalLight DirectionalLights[16]  — 512 bytes (16 × float4 ColorIntensity + float3 Direction + float AngularSize)
        //   ProceduralSkyConstants ProcSkyConsts        — 160 bytes (always zeroed in Unity)
        //   float3 ScaleColor; uint DirectionalLightCount;
        //   uint CubeDim, CubeDimLowRes, ProcSkyEnabled, BackgroundSourceType;
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private unsafe struct EnvMapBakerCB
        {
            public fixed float DirectionalLights[16 * 8]; // EMB_DirectionalLight[16]
            public fixed uint  ProcSkyConstsPad[160 / 4]; // ProceduralSkyConstants (zeroed)
            public       float ScaleColorR, ScaleColorG, ScaleColorB;
            public       uint  DirectionalLightCount;
            public       uint  CubeDim, CubeDimLowRes, ProcSkyEnabled, BackgroundSourceType;
        }

        // Mirrors HLSL EnvMapImportanceSamplingBakerConstants (EnvMapImportanceSamplingBaker.hlsl)
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct ImportanceBakerCB
        {
            public uint  SourceCubeDim,              SourceCubeMIPCount, SampleIndex, Padding1;
            public uint  ImportanceMapDimX,          ImportanceMapDimY;
            public uint  ImportanceMapDimInSamplesX, ImportanceMapDimInSamplesY;
            public uint  ImportanceMapNumSamplesX,   ImportanceMapNumSamplesY;
            public float ImportanceMapInvSamples;
            public uint  ImportanceMapBaseMip;
        }

        internal static unsafe int EnvBakerCbSize => sizeof(EnvMapBakerCB);
        internal static int ImportanceBakerCbSize => System.Runtime.InteropServices.Marshal.SizeOf<ImportanceBakerCB>();

        private readonly NativeComputePipeline      _baseLayerCs;
        private readonly NativeComputeDescriptorSet _baseLayerDs;
        private readonly NativeComputePipeline      _importanceBakerCs;
        private readonly NativeComputeDescriptorSet _importanceBakerDs;

        private static EnvMapBakerCB     s_envBakerCb;
        private static ImportanceBakerCB s_importanceCb;

        private NativeRtxptPassContext _ctx;

        // When true the env contents are identical to what is already baked into this camera's
        // cube/importance maps, so the whole bake (base layer + importance map + mip chains) is
        // skipped this frame. Mirrors the original EnvMapBaker::Update contentsChanged early-out.
        private bool _skipBake;

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
            FillEnvBakerConstants(ctx.Setting);
            FillImportanceBakerConstants();

            // Decide whether anything that affects the baked cube/importance map changed since
            // this camera's last bake. If not, skip the whole pass (the textures persist across
            // frames). This matches the original EnvMapBaker, which only re-bakes on change.
            var tex = ctx.Textures;
            var skyTex = ctx.Setting?.environmentMap;
            int skyId = skyTex != null ? skyTex.GetInstanceID() : 0;
            ulong signature = ComputeEnvSignature(skyId);

            _skipBake = tex.EnvBaked && tex.EnvBakeSignature == signature;
            if (!_skipBake)
            {
                // The bake is guaranteed to be recorded this frame, so mark it done now.
                tex.EnvBaked         = true;
                tex.EnvBakeSignature = signature;
            }
        }

        // FNV-1a hash over the env baker constants (directional lights, scale color, counts,
        // background type) plus the sky texture identity — the same inputs the original baker
        // compares to detect changes.
        private static unsafe ulong ComputeEnvSignature(int skyTextureId)
        {
            ulong h = 14695981039346656037UL;
            fixed (EnvMapBakerCB* p = &s_envBakerCb)
            {
                byte* b = (byte*)p;
                int   n = sizeof(EnvMapBakerCB);
                for (int i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211UL; }
            }
            h ^= (uint)skyTextureId;
            h *= 1099511628211UL;
            return h;
        }

        private class PassData
        {
            internal NativeComputePipeline      BaseLayerCs;
            internal NativeComputeDescriptorSet BaseLayerDs;
            internal NativeComputePipeline      ImportanceBakerCs;
            internal NativeComputeDescriptorSet ImportanceBakerDs;
            internal VolatileConstantBuffer     EnvBakerCb;
            internal VolatileConstantBuffer     ImportanceBakerCb;
            internal EnvMapBakerCB              EnvBakerCbData;
            internal ImportanceBakerCB          ImportanceCbData;
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
            // Env contents unchanged since last bake — the cube/importance maps already hold valid
            // data, so don't enqueue any GPU work this frame (the expensive part of the regression).
            if (_skipBake) return;

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxpt.EnvMapBaker", out var pd);

            pd.BaseLayerCs       = _baseLayerCs;
            pd.BaseLayerDs       = _baseLayerDs;
            pd.ImportanceBakerCs = _importanceBakerCs;
            pd.ImportanceBakerDs = _importanceBakerDs;
            pd.EnvBakerCb        = _ctx.Buffers.EnvBakerCb;
            pd.ImportanceBakerCb = _ctx.Buffers.ImportanceBakerCb;
            pd.EnvBakerCbData    = s_envBakerCb;
            pd.ImportanceCbData  = s_importanceCb;
            var skyTex = _ctx.Setting?.environmentMap;
            pd.SkyTexturePtr    = skyTex != null ? skyTex.GetNativeTexturePtr() : _ctx.blackTexturePtr;
            pd.EnvCubeMip0Ptr   = _ctx.Textures.EnvCubeMip0.NativePtr;
            pd.EnvCubeMip1Ptr   = _ctx.Textures.EnvCubeMip1.NativePtr;
            pd.ImportanceMapPtr = _ctx.Textures.EnvImportanceMap.NativePtr;
            pd.RadianceMapPtr   = _ctx.Textures.EnvRadianceMap.NativePtr;
            pd.ImportanceMapRt  = _ctx.Textures.EnvImportanceMap.rt;
            pd.RadianceMapRt    = _ctx.Textures.EnvRadianceMap.rt;
            pd.DummyCubePtr     = _ctx.Textures.EnvDummyCube.NativePtr;
            pd.DummyTex2DPtr    = _ctx.blackTexturePtr;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData d, UnsafeGraphContext c) => ExecutePass(d, c));
        }

        private static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            cmd.BeginSample(RenderPassMarkers.RtxptEnvMapBaker);

            {
                var ds = data.BaseLayerDs;
                data.EnvBakerCb.UploadDirect(context.cmd, data.EnvBakerCbData); // single-value overload
                ds.SetConstantBuffer("g_Const", data.EnvBakerCb);
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
                data.ImportanceBakerCb.UploadDirect(context.cmd, data.ImportanceCbData); // single-value overload
                ds.SetConstantBuffer("g_BuilderConsts", data.ImportanceBakerCb);
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

        private static unsafe void FillEnvBakerConstants(NativeRtxptSetting setting)
        {
            s_envBakerCb = default;
            ref var cb = ref s_envBakerCb;

            int lightCount = 0;
            foreach (var light in Object.FindObjectsByType<Light>(FindObjectsSortMode.None))
            {
                if (!light.enabled || !light.gameObject.activeInHierarchy) continue;
                if (light.type != LightType.Directional) continue;
                if (lightCount >= 16) break;

                Color   linear = light.color.linear;
                Vector3 fwd    = light.transform.forward;
                int     f      = lightCount * 8; // 8 floats per EMB_DirectionalLight
                cb.DirectionalLights[f + 0] = linear.r;
                cb.DirectionalLights[f + 1] = linear.g;
                cb.DirectionalLights[f + 2] = linear.b;
                cb.DirectionalLights[f + 3] = light.intensity;
                cb.DirectionalLights[f + 4] = fwd.x;
                cb.DirectionalLights[f + 5] = fwd.y;
                cb.DirectionalLights[f + 6] = fwd.z;
                cb.DirectionalLights[f + 7] = 0.1f; // AngularSize
                lightCount++;
            }

            float envIntensity = setting?.environmentMapIntensity ?? 1.0f;
            Color tint         = setting?.environmentMapTint ?? Color.white;
            bool  hasSky       = setting?.environmentMap != null;

            cb.ScaleColorR           = tint.linear.r * envIntensity;
            cb.ScaleColorG           = tint.linear.g * envIntensity;
            cb.ScaleColorB           = tint.linear.b * envIntensity;
            cb.DirectionalLightCount = (uint)lightCount;
            cb.CubeDim               = CubeDim;
            cb.CubeDimLowRes         = CubeDimLowRes;
            cb.ProcSkyEnabled        = 0;
            cb.BackgroundSourceType  = hasSky ? 1u : 0u;
        }

        private static void FillImportanceBakerConstants()
        {
            s_importanceCb = default;
            ref var cb = ref s_importanceCb;
            cb.SourceCubeDim              = (uint)CubeDim;
            cb.SourceCubeMIPCount         = 1;
            cb.SampleIndex                = 0;
            cb.Padding1                   = 0;
            cb.ImportanceMapDimX          = (uint)ImportanceMapDim;
            cb.ImportanceMapDimY          = (uint)ImportanceMapDim;
            cb.ImportanceMapDimInSamplesX = (uint)(ImportanceMapDim * ImportanceSamplesX);
            cb.ImportanceMapDimInSamplesY = (uint)(ImportanceMapDim * ImportanceSamplesY);
            cb.ImportanceMapNumSamplesX   = (uint)ImportanceSamplesX;
            cb.ImportanceMapNumSamplesY   = (uint)ImportanceSamplesY;
            cb.ImportanceMapInvSamples    = 1.0f / ImportanceSamples;
            cb.ImportanceMapBaseMip       = 10;
        }
    }
}