using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using NativeRender;
using Nri;
using RTXDI;
using Rtxdi;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    // -------------------------------------------------------------------------
    // Internal GPU structs (mirror SharedShaderInclude/ShaderParameters.h)
    // -------------------------------------------------------------------------

    /// <summary>
    /// Mirrors <c>PrepareLightsTask</c> from <c>ShaderParameters.h</c>.
    /// One entry per emissive sub-mesh (or per analytic primitive light).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct PrepareLightsTask
    {
        /// <summary>
        /// Bits [31..12] = flat instance index in t_InstanceData.
        /// Bits [11..0]  = sub-geometry index within the instance.
        /// Bit  31 (TASK_PRIMITIVE_LIGHT_BIT) = 1 for analytic lights.
        /// </summary>
        public uint instanceAndGeometryIndex;

        /// <summary>Number of triangles this task covers (1 for primitive lights).</summary>
        public uint triangleCount;

        /// <summary>First slot in LightDataBuffer[currentFrameOffset + lightBufferOffset].</summary>
        public uint lightBufferOffset;

        /// <summary>Frame-relative slot from the previous frame, or -1 if new.</summary>
        public int previousLightBufferOffset;
    }

    /// <summary>
    /// Mirrors <c>PrepareLightsConstants</c> from <c>ShaderParameters.h</c>.
    /// Bound as a constant buffer at <c>b0 / g_Const</c> (VK_PUSH_CONSTANT maps to a
    /// regular CBV in the compiled DX12 DXIL).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct NativePrepareLightsConstants
    {
        public uint numTasks;
        public uint currentFrameLightOffset;
        public uint previousFrameLightOffset;
        public uint _pad; // align to 16 bytes
    }

    // =========================================================================
    /// <summary>
    /// Native-RTXDI compute pass that mirrors
    /// <c>RTXDI/Samples/FullSample/Source/RenderPasses/PrepareLightsPass.{h,cpp}</c>.
    ///
    /// On the CPU side (<see cref="BuildTasksOnCpu"/>, called from
    /// <see cref="NativeRtxdiFeature.AddRenderPasses"/>), it:
    /// <list type="number">
    ///   <item>Iterates emissive geometries reported by <see cref="NativeRtxdiGPUScene"/>.</item>
    ///   <item>Builds the <c>TaskBuffer</c> and <c>GeometryInstanceToLight</c> arrays.</item>
    ///   <item>Uploads them to GPU via <c>GraphicsBuffer.SetData</c>.</item>
    ///   <item>Clears <c>LightIndexMappingBuffer</c> to zero.</item>
    ///   <item>Returns the <see cref="RTXDI_LightBufferParameters"/> for use in
    ///         <see cref="NativeResamplingConstants.lightBufferParams"/>.</item>
    /// </list>
    ///
    /// On the GPU side (<see cref="RecordRenderGraph"/>), it dispatches
    /// <c>PrepareLights.computeshader</c> with 256 threads per group.
    ///
    /// <para><b>Emissive meshes only for now</b> — analytic Unity lights (point / spot /
    /// directional) are not yet converted to <see cref="RTXDI.PolymorphicLightInfo"/> and are
    /// left for a follow-up.</para>
    /// </summary>
    // =========================================================================
    // === Shader Reflection: PrepareLights.computeshader ===
    // numthreads  [256, 1, 1]
    //
    // -- CBV (1) --
    //   g_Const                    ConstantBuffer<PrepareLightsConstants>  space0:b0
    //
    // -- Sampler (1) --
    //   s_MaterialSampler          SamplerState                            space0:s0
    //
    // -- SRV (7) --
    //   t_BindlessBuffers          ByteAddressBuffer[]                     space1:t0
    //   t_BindlessTextures         Texture2D[]                             space2:t0
    //   t_TaskBuffer               StructuredBuffer<PrepareLightsTask>     space0:t0
    //   t_PrimitiveLights          StructuredBuffer<PolymorphicLightInfo>  space0:t1
    //   t_InstanceData             StructuredBuffer<InstanceData>          space0:t2
    //   t_GeometryData             StructuredBuffer<GeometryData>          space0:t3
    //   t_MaterialConstants        StructuredBuffer<MaterialConstants>     space0:t4
    //
    // -- UAV (3) --
    //   u_LightDataBuffer          RWStructuredBuffer<PolymorphicLightInfo> space0:u0
    //   u_LightIndexMappingBuffer  RWBuffer<uint>                           space0:u1
    //   u_LocalLightPdfTexture     RWTexture2D<float>                       space0:u2
    public class NativeRtxdiPrepareLightsPass : ScriptableRenderPass, IDisposable
    {
        private const uint GroupSize = 256u;

        // ── GPU pipeline ──────────────────────────────────────────────────────
        private readonly NativeComputePipeline      _cs;
        private readonly NativeComputeDescriptorSet _ds;

        // b0 / g_Const: per-dispatch PrepareLightsConstants.
        private readonly GraphicsBuffer                 _prepareLightsConstantBuffer;
        private readonly NativePrepareLightsConstants[] _constantsArr = new NativePrepareLightsConstants[1];

        // ── Per-frame CPU state (computed in BuildTasksOnCpu) ─────────────────
        private NativeRtxdiPassContext _context;
        private int                    _totalLightCount; // total threads to dispatch
        private uint                   _numTasks;
        private uint                   _currentFrameOffset;
        private uint                   _previousFrameOffset;

        /// <summary>First slot in LightDataBuffer that belongs to the current frame.</summary>
        public uint CurrentFrameOffset => _currentFrameOffset;

        /// <summary>Number of valid light entries written this frame.</summary>
        public int TotalLightCount => _totalLightCount;

        /// <summary>
        /// Bindless texture array index of the environment map registered this frame.
        /// -1 when no environment map is active. Set by BuildTasksOnCpu.
        /// </summary>
        public int EnvMapBindlessTextureIndex { get; private set; } = -1;

        // ── Temporal tracking (frame-relative light buffer offsets per geometry) ──
        // Key = (instanceIndex << 12 | geometrySubIndex).  Value = frame-relative offset.
        private readonly Dictionary<long, int> _prevEmissiveOffsets = new();
        private          bool                  _oddFrame;

        // ── Working arrays (reused across frames to reduce GC pressure) ───────
        private PrepareLightsTask[]    _taskScratch           = Array.Empty<PrepareLightsTask>();
        private uint[]                 _geomToLightScratch    = Array.Empty<uint>();
        private PolymorphicLightInfo[] _primitiveLightScratch = Array.Empty<PolymorphicLightInfo>();

        // ── Temporal tracking for analytic (primitive) lights ────────────────────
        // Key = Unity Light.GetInstanceID().  Value = frame-relative offset in PrimitiveLightBuffer.
        private readonly Dictionary<int, int> _prevPrimitiveLightOffsets = new();

        // ── RTXDI_INVALID_LIGHT_INDEX (0xFFFFFFFF) ────────────────────────────
        private const uint InvalidLightIndex     = 0xFFFF_FFFFu;
        private const uint TaskPrimitiveLightBit = 0x8000_0000u;

        // ─────────────────────────────────────────────────────────────────────

        public NativeRtxdiPrepareLightsPass(NativeComputeShader shader)
        {
            _cs = new NativeComputePipeline(shader);
            _ds = new NativeComputeDescriptorSet(_cs);

            // Minimum constant-buffer size is 256 bytes on D3D12; allocate one element
            // padded to 16 bytes via the struct (NativePrepareLightsConstants = 16 B).
            _prepareLightsConstantBuffer = new GraphicsBuffer(
                    GraphicsBuffer.Target.Constant,
                    1,
                    Marshal.SizeOf<NativePrepareLightsConstants>())
                { name = "PrepareLightsConstants" };
        }

        public void Dispose()
        {
            _ds?.Dispose();
            _cs?.Dispose();
            _prepareLightsConstantBuffer?.Release();
        }

        // =====================================================================
        // CPU work — call from AddRenderPasses on the main thread.
        // =====================================================================

        /// <summary>
        /// Iterates emissive scene geometries, builds the <c>TaskBuffer</c> and
        /// <c>GeometryInstanceToLight</c> mappings, uploads them to GPU, and returns
        /// the <see cref="RTXDI_LightBufferParameters"/> to feed into
        /// <c>ResamplingConstants.lightBufferParams</c>.
        /// </summary>
        public RTXDI_LightBufferParameters BuildTasksOnCpu(
            NativeRtxdiPassContext ctx,
            Texture2D environmentMap     ,
            float     environmentRotDeg  ,
            float     environmentScale   )
        {
            var resources = ctx.Resources;
            var gpuScene  = ctx.RtxdiGpuScene;

            // ---- Double-buffer offset for LightDataBuffer ping-pong ----
            // Mirror: constants.currentFrameLightOffset = m_maxLightsInBuffer * m_oddFrame;
            uint maxLights = resources.MaxEmissiveTriangles + resources.MaxPrimitiveLights;
            _currentFrameOffset  = maxLights * (_oddFrame ? 1u : 0u);
            _previousFrameOffset = maxLights * (_oddFrame ? 0u : 1u);

            // ---- Register environment map in GPUScene bindless array ----
            Texture2D envMapTex = environmentMap;

            gpuScene.SetEnvironmentMap(envMapTex); // no-op if unchanged

            // ---- Collect analytic Unity lights ----
            // Gather all enabled Directional / Point / Spot lights in the scene.
            var sceneLights = GameObject.FindObjectsByType<Light>(FindObjectsSortMode.None);

            // Separate into finite (point/spot) and infinite (directional) for RTXDI region layout.
            var finiteLights   = new List<Light>(sceneLights.Length);
            var infiniteLights = new List<Light>(sceneLights.Length);
            foreach (var l in sceneLights)
            {
                if (!l.enabled || !l.gameObject.activeInHierarchy) continue;
                if (l.type == LightType.Directional)
                    infiniteLights.Add(l);
                else if (l.type == LightType.Point || l.type == LightType.Spot)
                    finiteLights.Add(l);
                // Rectangle / Disc lights handled by LightScene; skip here.
            }

            bool hasEnvLight = envMapTex != null;
            // ---- Collect emissive geometries from GPUScene ----
            var emissiveGeos = gpuScene.GetEmissiveGeometries();
            int numAnalytic  = finiteLights.Count + infiniteLights.Count + (hasEnvLight ? 1 : 0);
            int numTasks     = emissiveGeos.Count + numAnalytic;

            // Resize scratch arrays if needed
            if (_taskScratch.Length < numTasks)
                _taskScratch = new PrepareLightsTask[numTasks];

            int totalGeomInstances = gpuScene.TotalGeometryInstanceCount;
            int geomToLightLen     = Mathf.Max(totalGeomInstances, 1);
            if (_geomToLightScratch.Length < geomToLightLen)
                _geomToLightScratch = new uint[geomToLightLen];

            // Fill with RTXDI_INVALID_LIGHT_INDEX
            for (int i = 0; i < geomToLightLen; i++)
                _geomToLightScratch[i] = InvalidLightIndex;

            // ---- Build tasks ----
            uint lightBufferOffset = 0u;
            int  validTasks        = 0;

            foreach (var e in emissiveGeos)
            {
                if (lightBufferOffset + e.TriangleCount > maxLights)
                    break; // overflow guard

                long hash       = ((long)e.InstanceIndex << 12) | (long)e.GeometrySubIndex;
                int  prevOffset = _prevEmissiveOffsets.TryGetValue(hash, out int po) ? po : -1;

                _taskScratch[validTasks++] = new PrepareLightsTask
                {
                    instanceAndGeometryIndex  = ((uint)e.InstanceIndex << 12) | ((uint)e.GeometrySubIndex & 0xFFFu),
                    triangleCount             = e.TriangleCount,
                    lightBufferOffset         = lightBufferOffset,
                    previousLightBufferOffset = prevOffset, // frame-relative, shader adds _previousFrameOffset
                };

                _prevEmissiveOffsets[hash] = (int)lightBufferOffset; // record for next frame

                // Fill GeometryInstanceToLight  (firstGeometryInstanceIndex + subIndex)
                // Store absolute offset (including double-buffer ping-pong base) so the shader
                // can use t_GeometryInstanceToLight[instanceID] + primitiveIndex directly.
                int giIdx = (int)(e.FirstGeometryInstanceIndex + (uint)e.GeometrySubIndex);
                if (giIdx < geomToLightLen)
                    _geomToLightScratch[giIdx] = _currentFrameOffset + lightBufferOffset;

                lightBufferOffset += e.TriangleCount;
            }

            // ---- Build tasks for finite analytic lights (point / spot) ----
            // These go into the local light region, so they are appended right after emissive triangles.
            if (_primitiveLightScratch.Length < numAnalytic)
                _primitiveLightScratch = new PolymorphicLightInfo[numAnalytic];

            uint numFinitePrimLights   = 0u;
            uint numInfinitePrimLights = 0u;
            int  primitiveWriteIdx     = 0;

            // Finite lights (point / spot) — appended to local region
            foreach (var l in finiteLights)
            {
                if (!ConvertUnityLight(l, out var pli)) continue;
                if (lightBufferOffset >= maxLights) break;

                int id      = l.GetInstanceID();
                int prevOff = _prevPrimitiveLightOffsets.TryGetValue(id, out int po) ? po : -1;

                _taskScratch[validTasks++] = new PrepareLightsTask
                {
                    instanceAndGeometryIndex  = TaskPrimitiveLightBit | (uint)primitiveWriteIdx,
                    triangleCount             = 1u,
                    lightBufferOffset         = lightBufferOffset,
                    previousLightBufferOffset = prevOff,
                };
                _prevPrimitiveLightOffsets[id]              = (int)lightBufferOffset;
                _primitiveLightScratch[primitiveWriteIdx++] = pli;
                lightBufferOffset++;
                numFinitePrimLights++;
            }

            uint emissivePlusFiniteCount = lightBufferOffset; // boundary between local and infinite

            // Infinite lights (directional) — appended after the local region
            foreach (var l in infiniteLights)
            {
                if (!ConvertUnityLight(l, out var pli)) continue;
                if (lightBufferOffset >= maxLights) break;

                int id      = l.GetInstanceID();
                int prevOff = _prevPrimitiveLightOffsets.TryGetValue(id, out int po2) ? po2 : -1;

                _taskScratch[validTasks++] = new PrepareLightsTask
                {
                    instanceAndGeometryIndex  = TaskPrimitiveLightBit | (uint)primitiveWriteIdx,
                    triangleCount             = 1u,
                    lightBufferOffset         = lightBufferOffset,
                    previousLightBufferOffset = prevOff,
                };
                _prevPrimitiveLightOffsets[id]              = (int)lightBufferOffset;
                _primitiveLightScratch[primitiveWriteIdx++] = pli;
                lightBufferOffset++;
                numInfinitePrimLights++;
            }

            // Environment light — placed after infinite lights, treated as another infinite light
            // so it is sampled via infiniteLightBufferRegion (no presampled RIS required).
            // direction1 holds the bindless texture index; direction2 holds the texture dimensions.
            EnvMapBindlessTextureIndex = -1;
            bool hasActiveEnvLight = false;
            int  envTexIdx         = gpuScene.EnvironmentMapTextureIndex;
            if (hasEnvLight && envTexIdx >= 0 && lightBufferOffset < maxLights)
            {
                var envPli = new PolymorphicLightInfo();
                envPli.SetColorAndType( Color.white * environmentScale, PolymorphicLightType.kEnvironment);
                envPli.direction1 = (uint)envTexIdx;
                envPli.direction2 = (uint)(envMapTex.width | (envMapTex.height << 16));

                float rotRad  = environmentRotDeg * Mathf.Deg2Rad;
                envPli.scalars = LightScene.Fp32ToFp16(rotRad);

                int prevOff = _prevPrimitiveLightOffsets.TryGetValue(0 /*env key = 0*/, out int envPo) ? envPo : -1;
                _taskScratch[validTasks++] = new PrepareLightsTask
                {
                    instanceAndGeometryIndex  = TaskPrimitiveLightBit | (uint)primitiveWriteIdx,
                    triangleCount             = 1u,
                    lightBufferOffset         = lightBufferOffset,
                    previousLightBufferOffset = prevOff,
                };
                _prevPrimitiveLightOffsets[0]               = (int)lightBufferOffset;
                _primitiveLightScratch[primitiveWriteIdx++] = envPli;
                lightBufferOffset++;
                numInfinitePrimLights++;
                EnvMapBindlessTextureIndex = envTexIdx;
                hasActiveEnvLight = true;
            }

            _numTasks        = (uint)validTasks;
            _totalLightCount = (int)lightBufferOffset;

            // ---- Upload GPU buffers ----
            if (validTasks > 0 && resources.TaskBuffer != null)
                resources.TaskBuffer.SetData(_taskScratch, 0, 0, validTasks);

            if (resources.GeometryInstanceToLight != null)
            {
                int uploadLen = Mathf.Min(geomToLightLen, resources.GeometryInstanceToLight.count);
                resources.GeometryInstanceToLight.SetData(_geomToLightScratch, 0, 0, uploadLen);
            }

            // Upload primitive light data (analytic lights)
            if (primitiveWriteIdx > 0 && resources.PrimitiveLightBuffer != null)
                resources.PrimitiveLightBuffer.SetData(_primitiveLightScratch, 0, 0,
                    Mathf.Min(primitiveWriteIdx, resources.PrimitiveLightBuffer.count));

            // ---- Update constant buffer ----
            _constantsArr[0] = new NativePrepareLightsConstants
            {
                numTasks                 = _numTasks,
                currentFrameLightOffset  = _currentFrameOffset,
                previousFrameLightOffset = _previousFrameOffset,
                _pad                     = 0u,
            };
            _prepareLightsConstantBuffer.SetData(_constantsArr);

            // Flip ping-pong for next frame
            _oddFrame = !_oddFrame;

            // ---- Compose RTXDI_LightBufferParameters ----
            // Layout (frame-relative, shader adds _currentFrameOffset at runtime):
            //   [0 .. emissivePlusFiniteCount)                       — local lights (emissive + point/spot)
            //   [emissivePlusFiniteCount .. +numInfinite)            — infinite (directional + env)
            // environmentLightParams is intentionally left empty: env map goes into the infinite
            // region so it is sampled without a presampled-RIS pipeline.
            var p = new RTXDI_LightBufferParameters();
            p.localLightBufferRegion.firstLightIndex    = _currentFrameOffset;
            p.localLightBufferRegion.numLights          = emissivePlusFiniteCount;
            p.infiniteLightBufferRegion.firstLightIndex = _currentFrameOffset + emissivePlusFiniteCount;
            p.infiniteLightBufferRegion.numLights       = numInfinitePrimLights;
            p.environmentLightParams.lightPresent       = 0;
            p.environmentLightParams.lightIndex         = 0xFFFF_FFFFu;

            return p;
        }

        // =====================================================================
        // Setup — call after BuildTasksOnCpu.
        // =====================================================================

        public void Setup(NativeRtxdiPassContext ctx)
        {
            _context = ctx;
        }

        // =====================================================================
        // Static light conversion helpers
        // =====================================================================

        /// <summary>
        /// Converts a Unity <see cref="Light"/> (Directional / Point / Spot) to a
        /// <see cref="PolymorphicLightInfo"/> ready to be uploaded in the PrimitiveLightBuffer.
        /// Returns <c>false</c> for unsupported or degenerate lights.
        /// Mirrors <c>PrepareLightsPass.cpp : ConvertLight()</c> from RTXDI FullSample.
        /// </summary>
        private static bool ConvertUnityLight(Light light, out PolymorphicLightInfo info)
        {
            info = new PolymorphicLightInfo();
            switch (light.type)
            {
                case LightType.Directional:
                {
                    // Use bounceIntensity as angular diameter (degrees), matching LightScene.cs convention.
                    float angularDeg                   = light.bounceIntensity;
                    float halfAngRad                   = 0.5f * angularDeg * Mathf.Deg2Rad;
                    float solidAngle                   = 2f * Mathf.PI * (1f - Mathf.Cos(halfAngRad));
                    if (solidAngle < 1e-7f) solidAngle = 1e-7f;
                    Color radiance                     = light.color * (light.intensity / solidAngle);
                    info.SetColorAndType(radiance, PolymorphicLightType.kDirectional);
                    info.direction1 = LightScene.PackNormalizedVector(light.transform.forward);
                    info.scalars = (uint)(LightScene.Fp32ToFp16(halfAngRad)
                                          | (LightScene.Fp32ToFp16(solidAngle) << 16));
                    return true;
                }
                case LightType.Point:
                {
                    light.TryGetComponent<PathTracingAdditionalLightData>(out var ad);
                    float radius = ad != null ? ad.radius : 0f;
                    if (radius <= 0f)
                    {
                        info.SetColorAndType(light.color * light.intensity, PolymorphicLightType.kPoint);
                        info.center = (float3)light.transform.position;
                    }
                    else
                    {
                        float projArea = Mathf.PI * radius * radius;
                        info.SetColorAndType(light.color * (light.intensity / projArea), PolymorphicLightType.kSphere);
                        info.center  = (float3)light.transform.position;
                        info.scalars = LightScene.Fp32ToFp16(radius);
                    }

                    return true;
                }
                case LightType.Spot:
                {
                    light.TryGetComponent<PathTracingAdditionalLightData>(out var ad);
                    float      radius            = ad != null ? ad.radius : 0f;
                    float      softness          = Mathf.Clamp01(1f - light.innerSpotAngle / light.spotAngle);
                    const uint kShapingEnableBit = 1u << 28;
                    if (radius <= 0f)
                    {
                        info.SetColorAndType(light.color * light.intensity, PolymorphicLightType.kSphere);
                        info.scalars = 0;
                    }
                    else
                    {
                        float projArea = Mathf.PI * radius * radius;
                        info.SetColorAndType(light.color * (light.intensity / projArea), PolymorphicLightType.kSphere);
                        info.scalars = LightScene.Fp32ToFp16(radius);
                    }

                    info.colorTypeAndFlags |= kShapingEnableBit;
                    info.center            =  (float3)light.transform.position;
                    info.primaryAxis       =  LightScene.PackNormalizedVector(light.transform.forward);
                    info.cosConeAngleAndSoftness = (uint)(
                        LightScene.Fp32ToFp16(Mathf.Cos(light.spotAngle * 0.5f * Mathf.Deg2Rad))
                        | (LightScene.Fp32ToFp16(softness) << 16));
                    return true;
                }
                default:
                    return false;
            }
        }

        // =====================================================================
        // RenderGraph
        // =====================================================================

        class PassData
        {
            internal NativeComputePipeline      Cs;
            internal NativeComputeDescriptorSet Ds;
            internal NativeRtxdiPassContext     Context;
            internal int                        TotalLightCount;
            internal GraphicsBuffer             ConstantBuffer;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            // Nothing to do if no emissive lights exist this frame
            if (_totalLightCount == 0 || _numTasks == 0)
                return;

            using var builder = renderGraph.AddUnsafePass<PassData>("NativeRtxdi.PrepareLights", out var passData);

            passData.Cs              = _cs;
            passData.Ds              = _ds;
            passData.Context         = _context;
            passData.TotalLightCount = _totalLightCount;
            passData.ConstantBuffer  = _prepareLightsConstantBuffer;

            builder.AllowPassCulling(false);
            builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => ExecutePass(data, context));
        }

        static void ExecutePass(PassData data, UnsafeGraphContext context)
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);
            cmd.BeginSample(RenderPassMarkers.PrepareLightsCompute);

            var cs  = data.Cs;
            var ds  = data.Ds;
            var ctx = data.Context;
            var rtx = ctx.Resources;

            // ---- g_Const (b0): PrepareLightsConstants ----
            ds.SetConstantBuffer("g_Const", data.ConstantBuffer.GetNativeBufferPtr());

            // ---- SRV inputs ----
            if (rtx.TaskBuffer != null)
                ds.SetStructuredBuffer("t_TaskBuffer",
                    rtx.TaskBuffer.GetNativeBufferPtr(),
                    rtx.TaskBuffer.count, rtx.TaskBuffer.stride);

            if (rtx.PrimitiveLightBuffer != null)
                ds.SetStructuredBuffer("t_PrimitiveLights",
                    rtx.PrimitiveLightBuffer.GetNativeBufferPtr(),
                    rtx.PrimitiveLightBuffer.count, rtx.PrimitiveLightBuffer.stride);

            // t_InstanceData / t_GeometryData / t_MaterialConstants / bindless
            ctx.RtxdiGpuScene?.BindToShader(ds);

            // ---- UAV outputs ----
            if (rtx.LightDataBuffer != null)
                ds.SetRWStructuredBuffer("u_LightDataBuffer",
                    rtx.LightDataBuffer.GetNativeBufferPtr(),
                    rtx.LightDataBuffer.count, rtx.LightDataBuffer.stride);

            if (rtx.LightIndexMappingBuffer != null)
                // RWBuffer<uint> requires a typed UAV: DXGI_FORMAT_R32_UINT
                ds.SetRWTypedBuffer("u_LightIndexMappingBuffer",
                    rtx.LightIndexMappingBuffer.GetNativeBufferPtr(),
                    rtx.LightIndexMappingBuffer.count,
                    (uint)Nri.DXGI_FORMAT.DXGI_FORMAT_R32_UINT);

            if (rtx.LocalLightPdfTexture?.rt != null)
                ds.SetRWTexture("u_LocalLightPdfTexture",
                    rtx.LocalLightPdfTexture.rt.GetNativeTexturePtr());

            // ---- Dispatch: ceil(totalLightCount / 256) groups of 256 threads ----
            uint groups = ((uint)data.TotalLightCount + GroupSize - 1u) / GroupSize;

            cs.Dispatch(cmd, ds, groups, 1u, 1u);

            cmd.EndSample(RenderPassMarkers.PrepareLightsCompute);
        }
    }
}