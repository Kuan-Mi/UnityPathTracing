using System;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace NativeRender
{
    /// <summary>
    /// The actual render pass that calls into the native plugin.
    /// </summary>
    public class NativeRayTracingPass : ScriptableRenderPass
    {
        private const string ProfilerTag = "Native Ray Tracing";

        // Tracks which targets are currently registered with the native plugin.
        // NOTE: Scene registration state is now owned by GPUScene.

        // Shared GPU scene — owned by NativeRayTracingFeature, injected each frame.
        private GPUScene _gpuScene;

        // Persistent RenderTexture so we can call GetNativeTexturePtr() before recording.
        private RenderTexture m_PersistentRT;
        private RTHandle      m_PersistentRTHandle;
        private IntPtr        m_PersistentNativePtr;

        // Scene constants uploaded to GPU each frame and bound as ConstantBuffer.
        private ComputeBuffer m_SceneConstantsCB;

        [StructLayout(LayoutKind.Sequential)]
        private struct SceneConstants
        {
            public Matrix4x4 viewProjInv;
            public Vector3   cameraPos;
            public float     _pad;
        }

        // Persistent unmanaged memory for the RenderEventData struct.
        private IntPtr m_EventDataPtr = IntPtr.Zero;

        // RayTraceShader asset — lifetime managed by the asset itself.
        private RayTraceShader _shader;

        // Constant buffer for TestConstants (b1)
        private ComputeBuffer _testConstantsCB;

        /// <summary>Injects the shared GPU scene resource container for this frame.</summary>
        public void SetGPUScene(GPUScene scene)
        {
            _gpuScene = scene;
        }

        /// <summary>Assigns the RayTraceShader asset. The asset manages its own native handle.</summary>
        public void SetShaderAsset(RayTraceShader asset)
        {
            _shader = asset;
        }

        /// <summary>Syncs <paramref name="textures"/> into a <see cref="BindlessTexture"/> and binds it to the shader as t_TestBindless.</summary>
        public void SetTestBindless(Texture[] textures, ref BindlessTexture bt)
        {
            if (_shader == null || !_shader.IsValid) return;

            int count = textures != null ? textures.Length : 0;
            if (count == 0) return;

            if (bt == null || bt.Capacity < count)
            {
                bt?.Dispose();
                bt = new BindlessTexture(count);
            }

            for (int i = 0; i < count; i++)
                bt[i] = textures[i];

            _shader.SetBindlessTexture("t_TestBindless", bt);
        }

        /// <summary>Uploads a Vector4 to testConstants (b1) and binds it to the shader.</summary>
        public void SetTestConstants(Vector4 dummy)
        {
            if (_shader == null || !_shader.IsValid) return;

            if (_testConstantsCB == null)
                _testConstantsCB = new ComputeBuffer(1, 16, ComputeBufferType.Constant);

            _testConstantsCB.SetData(new[] { dummy });
            _shader.SetConstantBuffer("testConstants", _testConstantsCB);
        }

        public NativeRayTracingPass(RenderPassEvent evt)
        {
            renderPassEvent = evt;
        }

        public void Dispose()
        {
            if (m_PersistentRTHandle != null) { m_PersistentRTHandle.Release(); m_PersistentRTHandle = null; }
            if (m_PersistentRT != null) { m_PersistentRT.Release(); m_PersistentRT = null; }
            if (m_EventDataPtr != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(m_EventDataPtr);
                m_EventDataPtr = IntPtr.Zero;
            }

            // _shader is a ScriptableObject asset — do not destroy it here.
            _shader = null;

            _testConstantsCB?.Release();
            _testConstantsCB = null;

            m_SceneConstantsCB?.Release();
            m_SceneConstantsCB = null;

            // GPUScene lifetime is managed by NativeRayTracingFeature — do not dispose here.
            _gpuScene = null;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var cameraData   = frameData.Get<UniversalCameraData>();
            var resourceData = frameData.Get<UniversalResourceData>();

            int width  = cameraData.cameraTargetDescriptor.width;
            int height = cameraData.cameraTargetDescriptor.height;

            if (m_PersistentRT == null || m_PersistentRT.width != width || m_PersistentRT.height != height)
            {
                if (m_PersistentRTHandle != null) { m_PersistentRTHandle.Release(); m_PersistentRTHandle = null; }
                if (m_PersistentRT != null) { m_PersistentRT.Release(); m_PersistentRT = null; }

                m_PersistentRT = new RenderTexture(width, height, 0,
                    RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
                {
                    enableRandomWrite = true,
                    name              = "_NativeRayTracingOutput",
                };
                m_PersistentRT.Create();
                m_PersistentRTHandle  = RTHandles.Alloc(m_PersistentRT);
                m_PersistentNativePtr = m_PersistentRT.GetNativeTexturePtr();
            }

            var rtInfo = new RenderTargetInfo
            {
                width          = width,
                height         = height,
                volumeDepth    = 1,
                msaaSamples    = 1,
                format         = m_PersistentRT.graphicsFormat,
            };
            TextureHandle outputTex = renderGraph.ImportTexture(m_PersistentRTHandle, rtInfo);

            // --- Unsafe pass: invoke native DXR rendering ---
            using (var builder = renderGraph.AddUnsafePass<PassData>(ProfilerTag, out var passData))
            {
                passData.outputTexture               = outputTex;
                passData.nativeTexturePtr            = m_PersistentNativePtr;
                passData.accelerationStructureHandle = _gpuScene?.AccelerationStructure?.Handle ?? 0;
                passData.viewProjInv                 = (cameraData.GetProjectionMatrix() * cameraData.GetViewMatrix()).inverse;
                passData.cameraPos                   = cameraData.worldSpaceCameraPos;

                builder.UseTexture(outputTex, AccessFlags.Write);
                builder.AllowPassCulling(false);

                builder.SetRenderFunc<PassData>((PassData data, UnsafeGraphContext context) => { ExecuteNativeRender(data, context); });
            }

            // --- Raster pass: blit result to camera color ---
            using (var builder = renderGraph.AddRasterRenderPass<BlitPassData>("Blit Native RT", out var blitData))
            {
                blitData.source      = outputTex;
                blitData.destination = resourceData.activeColorTexture;

                builder.UseTexture(blitData.source, AccessFlags.Read);
                builder.SetRenderAttachment(blitData.destination, 0, AccessFlags.Write);

                builder.SetRenderFunc<BlitPassData>((BlitPassData data, RasterGraphContext context) => { Blitter.BlitTexture(context.cmd, data.source, new Vector4(1, 1, 0, 0), 0, false); });
            }
        }

        private void ExecuteNativeRender(PassData data, UnsafeGraphContext context)
        {
            CommandBuffer cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

            if (_shader == null || !_shader.IsValid)
            {
                Debug.LogWarning("[NativeRayTracing] No valid RayTraceShader — assign a shader asset in the Renderer Feature.");
                return;
            }

            if (data.nativeTexturePtr == IntPtr.Zero)
            {
                Debug.LogError("[NativeRayTracing] Output texture native pointer is null");
                return;
            }

            if (data.accelerationStructureHandle == 0)
            {
                Debug.LogError("[NativeRayTracing] Acceleration structure handle is null");
                return;
            }

            // Build / update the acceleration structure before dispatch
            _gpuScene.BuildAccelerationStructure(cmd);

            // Bind TLAS – after BuildOrUpdate the TLAS pointer is valid
            _shader.SetAccelerationStructure("SceneBVH", _gpuScene.AccelerationStructure);

            // Bind output RenderTexture (UAV) — C# owns the resource lifetime
            _shader.SetRWTexture("OutputTexture", m_PersistentRT);

            // Upload and bind SceneConstants (viewProjInv, cameraPos)
            if (m_SceneConstantsCB == null)
                m_SceneConstantsCB = new ComputeBuffer(1, Marshal.SizeOf<SceneConstants>(),
                    ComputeBufferType.Constant);
            m_SceneConstantsCB.SetData(new[]
            {
                new SceneConstants
                {
                    viewProjInv = data.viewProjInv,
                    cameraPos   = data.cameraPos,
                    _pad        = 0f,
                }
            });
            _shader.SetConstantBuffer("SceneConstants", m_SceneConstantsCB);

            _gpuScene.BindToShader(_shader);

            _shader.Dispatch(cmd, (uint)m_PersistentRT.width, (uint)m_PersistentRT.height);
        }

        private class PassData
        {
            public TextureHandle outputTexture;
            public IntPtr        nativeTexturePtr;
            public ulong         accelerationStructureHandle;
            public Matrix4x4     viewProjInv;
            public Vector3       cameraPos;
        }

        private class BlitPassData
        {
            public TextureHandle source;
            public TextureHandle destination;
        }
    }
}