using System;
using NativeRender;
using PathTracing;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using NativeRayTracingAccelerationStructure = NativeRender.RayTracingAccelerationStructure;

namespace SER
{
    public sealed class HelloSERFeature : ScriptableRendererFeature
    {
        public enum ShaderMode
        {
            SEROn,
            SEROff
        }

        public bool enabledPass = true;
        public ShaderMode mode = ShaderMode.SEROn;
        public RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;
        public RayTraceShader serOnShader;
        public RayTraceShader serOffShader;
        public Material blitMaterial;
        public float resolutionScale = 1.0f;
        public bool blitToCamera = true;

        private HelloSERPass _pass;

        public override void Create()
        {
            _pass ??= new HelloSERPass();
            _pass.renderPassEvent = renderPassEvent;
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (!enabledPass) return;

            var cam = renderingData.cameraData.camera;
            if (cam.cameraType is CameraType.Preview or CameraType.Reflection) return;
            if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView) return;

            _pass ??= new HelloSERPass();
            _pass.renderPassEvent = renderPassEvent;
            _pass.Setup(serOnShader, serOffShader, mode, blitMaterial, Mathf.Clamp(resolutionScale, 0.1f, 1.0f), blitToCamera);
            renderer.EnqueuePass(_pass);
        }

        protected override void Dispose(bool disposing)
        {
            _pass?.Dispose();
            _pass = null;
        }

#if UNITY_EDITOR
        [ContextMenu("Auto Fill Hello SER Assets")]
        public void AutoFillAssets()
        {
            serOnShader = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTraceShader>("Assets/SER/Shaders/HelloSER_ON.rayshader");
            serOffShader = UnityEditor.AssetDatabase.LoadAssetAtPath<RayTraceShader>("Assets/SER/Shaders/HelloSER_OFF.rayshader");
            blitMaterial = UnityEditor.AssetDatabase.LoadAssetAtPath<Material>("Assets/Shaders/Mat/KM_Final.mat");
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif

        private sealed class HelloSERPass : ScriptableRenderPass, IDisposable
        {
            private RayTraceShader _serOnShader;
            private RayTraceShader _serOffShader;
            private ShaderMode _mode;
            private Material _blitMaterial;
            private float _resolutionScale = 1.0f;
            private bool _blitToCamera = true;

            private RayTracePipeline _serOnPipeline;
            private RayTracePipeline _serOffPipeline;
            private NativeRayTraceDescriptorSet _serOnDescriptorSet;
            private NativeRayTraceDescriptorSet _serOffDescriptorSet;
            private VolatileConstantBuffer _rayGenConstantBuffer;

            private NativeRayTracingAccelerationStructure _accel;
            private NativeArray<uint> _singleHitGroupVariant;
            private Mesh _quadMesh;

            private RenderTexture _outputTexture;
            private RTHandle _outputHandle;
            private int _outputWidth;
            private int _outputHeight;
            private bool _sbtBuiltOn;
            private bool _sbtBuiltOff;

            public void Setup(
                RayTraceShader serOnShader,
                RayTraceShader serOffShader,
                ShaderMode mode,
                Material blitMaterial,
                float resolutionScale,
                bool blitToCamera)
            {
                _serOnShader = serOnShader;
                _serOffShader = serOffShader;
                _mode = mode;
                _blitMaterial = blitMaterial;
                _resolutionScale = resolutionScale;
                _blitToCamera = blitToCamera;
            }

            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                var shader = _mode == ShaderMode.SEROn ? _serOnShader : _serOffShader;
                if (shader == null) return;

                EnsureScene();
                EnsurePipeline(_mode);

                var pipeline = _mode == ShaderMode.SEROn ? _serOnPipeline : _serOffPipeline;
                var ds = _mode == ShaderMode.SEROn ? _serOnDescriptorSet : _serOffDescriptorSet;
                if (pipeline == null || ds == null || !pipeline.IsValid || _accel == null) return;

                var cameraData = frameData.Get<UniversalCameraData>();
                int width = Mathf.Max(1, Mathf.RoundToInt(cameraData.cameraTargetDescriptor.width * _resolutionScale));
                int height = Mathf.Max(1, Mathf.RoundToInt(cameraData.cameraTargetDescriptor.height * _resolutionScale));
                EnsureOutput(width, height);

                var resourceData = frameData.Get<UniversalResourceData>();
                using var builder = renderGraph.AddUnsafePass<PassData>("Hello SER", out var passData);

                passData.Pass = this;
                passData.Pipeline = pipeline;
                passData.DescriptorSet = ds;
                passData.Accel = _accel;
                passData.RayGenConstantBuffer = _rayGenConstantBuffer;
                passData.Mode = _mode;
                passData.Output = _outputHandle;
                passData.OutputTexture = _outputTexture;
                passData.Width = (uint)width;
                passData.Height = (uint)height;
                passData.BlitMaterial = _blitMaterial;
                passData.BlitToCamera = _blitToCamera;
                passData.CameraTexture = resourceData.activeColorTexture;

                var outputHandle = renderGraph.ImportTexture(_outputHandle);
                builder.UseTexture(outputHandle, AccessFlags.ReadWrite);
                if (_blitToCamera)
                    builder.UseTexture(passData.CameraTexture, AccessFlags.Write);

                builder.AllowPassCulling(false);
                builder.SetRenderFunc((PassData data, UnsafeGraphContext context) => data.Pass.ExecutePass(data, context));
            }

            private void ExecutePass(PassData data, UnsafeGraphContext context)
            {
                var cmd = CommandBufferHelpers.GetNativeCommandBuffer(context.cmd);

                cmd.BeginSample(data.Mode == ShaderMode.SEROn ? "HelloSER.Trace.ON" : "HelloSER.Trace.OFF");

                data.Accel.BuildOrUpdate(cmd);

                if (data.Mode == ShaderMode.SEROn)
                {
                    if (!_sbtBuiltOn)
                    {
                        data.Pipeline.RebuildHitGroupTable(cmd, _singleHitGroupVariant);
                        _sbtBuiltOn = true;
                    }
                }
                else if (!_sbtBuiltOff)
                {
                    data.Pipeline.RebuildHitGroupTable(cmd, _singleHitGroupVariant);
                    _sbtBuiltOff = true;
                }

                data.DescriptorSet.SetAccelerationStructure("Scene", data.Accel);
                data.DescriptorSet.SetRWTexture("RenderTarget", data.OutputTexture.GetNativeTexturePtr());
                data.RayGenConstantBuffer.UploadDirect(context.cmd, RayGenConstants.Fullscreen);
                data.DescriptorSet.SetConstantBuffer("g_rayGenCB", data.RayGenConstantBuffer);
                data.Pipeline.Dispatch(cmd, data.DescriptorSet, data.Width, data.Height);

                cmd.EndSample(data.Mode == ShaderMode.SEROn ? "HelloSER.Trace.ON" : "HelloSER.Trace.OFF");

                if (data.BlitToCamera && data.BlitMaterial != null)
                {
                    cmd.SetRenderTarget(data.CameraTexture);
                    Blitter.BlitTexture(cmd, data.Output, new Vector4(1, 1, 0, 0), data.BlitMaterial, (int)ShowPass.Out);
                }
            }

            private void EnsurePipeline(ShaderMode mode)
            {
                if (mode == ShaderMode.SEROn)
                {
                    if (_serOnPipeline != null || _serOnShader == null) return;
                    _serOnPipeline = new RayTracePipeline(_serOnShader);
                    _serOnDescriptorSet = new NativeRayTraceDescriptorSet(_serOnPipeline);
                    _sbtBuiltOn = false;
                    return;
                }

                if (_serOffPipeline != null || _serOffShader == null) return;
                _serOffPipeline = new RayTracePipeline(_serOffShader);
                _serOffDescriptorSet = new NativeRayTraceDescriptorSet(_serOffPipeline);
                _sbtBuiltOff = false;
            }

            private void EnsureScene()
            {
                if (_accel != null) return;

                _accel = new NativeRayTracingAccelerationStructure(new RayTracingAccelerationStructureOptions
                {
                    UseRtxmu = true,
                    UseCompaction = false
                });

                _quadMesh = new Mesh
                {
                    name = "HelloSER_Quad",
                    hideFlags = HideFlags.HideAndDontSave,
                    vertices = new[]
                    {
                        new Vector3(-1, -1, 1),
                        new Vector3(-1, 1, 1),
                        new Vector3(1, 1, 1),
                        new Vector3(1, -1, 1),
                    },
                    triangles = new[] { 0, 1, 2, 0, 3, 2 },
                    bounds = new Bounds(new Vector3(0, 0, 1), new Vector3(2, 2, 0.01f))
                };
                _quadMesh.UploadMeshData(false);

                var submeshDescs = new[]
                {
                    new NativeRenderPlugin.SubmeshDesc
                    {
                        indexCount = 6,
                        indexByteOffset = 0,
                        baseVertex = 0,
                        flags = NativeRenderPlugin.SUBMESH_FLAG_GEOMETRY_OPAQUE
                    }
                };

                const uint quadHandle = 1;
                _accel.AddInstanceGroup(_quadMesh, submeshDescs, quadHandle);
                _accel.SetInstanceTransform(quadHandle, Matrix4x4.identity);
                _accel.SetInstanceMask(quadHandle, 0xff);
                _accel.SetInstanceHitGroupContribution(quadHandle, 0);

                _singleHitGroupVariant = new NativeArray<uint>(1, Allocator.Persistent);
                _singleHitGroupVariant[0] = 0;

                _rayGenConstantBuffer = new VolatileConstantBuffer(32, "HelloSER_RayGenCB");
            }

            private void EnsureOutput(int width, int height)
            {
                if (_outputTexture != null && _outputWidth == width && _outputHeight == height) return;

                ReleaseOutput();

                var desc = new RenderTextureDescriptor(width, height, GraphicsFormat.R16G16B16A16_SFloat, 0)
                {
                    enableRandomWrite = true,
                    useMipMap = false,
                    autoGenerateMips = false,
                    msaaSamples = 1,
                    sRGB = false
                };

                _outputTexture = new RenderTexture(desc)
                {
                    name = "HelloSER_Output",
                    hideFlags = HideFlags.HideAndDontSave
                };
                _outputTexture.Create();
                _outputHandle = RTHandles.Alloc(_outputTexture);
                _outputWidth = width;
                _outputHeight = height;
            }

            private void ReleaseOutput()
            {
                _outputHandle?.Release();
                _outputHandle = null;

                if (_outputTexture != null)
                {
                    _outputTexture.Release();
                    DestroyObject(_outputTexture);
                    _outputTexture = null;
                }

                _outputWidth = 0;
                _outputHeight = 0;
            }

            public void Dispose()
            {
                ReleaseOutput();

                _serOnDescriptorSet?.Dispose();
                _serOffDescriptorSet?.Dispose();
                _serOnPipeline?.Dispose();
                _serOffPipeline?.Dispose();
                _accel?.Dispose();
                _rayGenConstantBuffer?.Dispose();

                _serOnDescriptorSet = null;
                _serOffDescriptorSet = null;
                _serOnPipeline = null;
                _serOffPipeline = null;
                _accel = null;
                _rayGenConstantBuffer = null;

                if (_singleHitGroupVariant.IsCreated)
                    _singleHitGroupVariant.Dispose();

                DestroyObject(_quadMesh);

                _quadMesh = null;
            }

            private static void DestroyObject(UnityEngine.Object obj)
            {
                if (obj == null) return;

                if (Application.isPlaying)
                    UnityEngine.Object.Destroy(obj);
                else
                    UnityEngine.Object.DestroyImmediate(obj);
            }

            private sealed class PassData
            {
                public HelloSERPass Pass;
                public RayTracePipeline Pipeline;
                public NativeRayTraceDescriptorSet DescriptorSet;
                public NativeRayTracingAccelerationStructure Accel;
                public VolatileConstantBuffer RayGenConstantBuffer;
                public ShaderMode Mode;
                public RTHandle Output;
                public RenderTexture OutputTexture;
                public uint Width;
                public uint Height;
                public Material BlitMaterial;
                public bool BlitToCamera;
                public TextureHandle CameraTexture;
            }

            private struct RayGenConstants
            {
                public float ViewportLeft;
                public float ViewportTop;
                public float ViewportRight;
                public float ViewportBottom;
                public float StencilLeft;
                public float StencilTop;
                public float StencilRight;
                public float StencilBottom;

                public static readonly RayGenConstants Fullscreen = new RayGenConstants
                {
                    ViewportLeft = -1.0f,
                    ViewportTop = -1.0f,
                    ViewportRight = 1.0f,
                    ViewportBottom = 1.0f,
                    StencilLeft = -1.0f,
                    StencilTop = -1.0f,
                    StencilRight = 1.0f,
                    StencilBottom = 1.0f
                };
            }
        }
    }
}
