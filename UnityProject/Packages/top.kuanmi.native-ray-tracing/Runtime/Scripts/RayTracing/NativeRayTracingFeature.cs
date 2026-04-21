using UnityEngine;
using UnityEngine.Rendering.Universal;
using System;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace NativeRender
{
    /// <summary>
    /// ScriptableRendererFeature that injects native DX12 ray tracing into URP.
    /// </summary>
    public class NativeRayTracingFeature : ScriptableRendererFeature
    {
        [SerializeField]
        private RenderPassEvent renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing;

        [Tooltip("RayTraceShader asset (.rayshader file) to use for ray tracing.")]
        [SerializeField]
        private RayTraceShader shaderSourceAsset;

        [Serializable]
        public struct TestConstants
        {
            public Vector4 dummy;
        }

        [Tooltip("Bound to testConstants (b1) in RayTracing.hlsl.")]
        [SerializeField]
        private TestConstants testConstants = new TestConstants { dummy = Vector4.one };

        [Tooltip("Textures bound to t_TestBindless (t0) in RayTracing.hlsl.")]
        [SerializeField]
        private Texture[] testBindlessTextures = Array.Empty<Texture>();

        private BindlessTexture _testBT;

        private NativeRayTracingPass m_RenderPass;
        private RayTracePipeline     _pipeline;
        private GPUScene             _gpuScene;
        private NRDSampleResource    _sampleResource;

        public override void Create()
        {
            if (m_RenderPass == null)
                m_RenderPass = new NativeRayTracingPass(renderPassEvent);
            if (_gpuScene == null)
                _gpuScene = new GPUScene();
            if (_sampleResource == null)
                _sampleResource = new NRDSampleResource();
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            _gpuScene?.UpdateForFrame();
            _sampleResource.UpdateForFrame();

            // (Re-)create the pipeline if the asset changed or hasn't been built yet.
            if (_pipeline == null && shaderSourceAsset != null)
            {
                try { _pipeline = new RayTracePipeline(shaderSourceAsset); }
                catch (Exception e) { Debug.LogError(e.Message); }
            }

            m_RenderPass.SetPipeline(_pipeline);
            m_RenderPass.SetGPUScene(_gpuScene);
            m_RenderPass.SetSampleResource(_sampleResource);

            var camera = renderingData.cameraData.camera;
            if (camera == null)
                return;

            if (camera.cameraType == CameraType.Preview || camera.cameraType == CameraType.Reflection)
                return;

            m_RenderPass.SetTestConstants(testConstants.dummy);
            m_RenderPass.SetTestBindless(testBindlessTextures, ref _testBT);
            renderer.EnqueuePass(m_RenderPass);
        }

        protected override void Dispose(bool disposing)
        {
            _testBT?.Dispose();
            _testBT = null;
            _pipeline?.Dispose();
            _pipeline = null;
            m_RenderPass?.Dispose();
            _gpuScene?.Dispose();
            _gpuScene = null;
            _sampleResource?.Dispose();
            _sampleResource = null;
        }

#if UNITY_EDITOR
        private void OnEnable()  => ObjectChangeEvents.changesPublished += OnObjectChangesPublished;
        private void OnDisable() => ObjectChangeEvents.changesPublished -= OnObjectChangesPublished;

        private void OnObjectChangesPublished(ref ObjectChangeEventStream stream)
        {
            if (m_RenderPass == null) return;

            for (int i = 0; i < stream.length; i++)
            {
                var kind = stream.GetEventType(i);

                if (kind == ObjectChangeKind.ChangeAssetObjectProperties)
                {
                    stream.GetChangeAssetObjectPropertiesEvent(i, out var e);
                    var obj = EditorUtility.EntityIdToObject(e.instanceId);
                    if (obj is Material mat && IsMaterialUsedByTargets(mat))
                        _gpuScene?.MarkMaterialDirty(mat);
                }
                else if (kind == ObjectChangeKind.ChangeGameObjectOrComponentProperties)
                {
                    stream.GetChangeGameObjectOrComponentPropertiesEvent(i, out var e);
                    var obj = EditorUtility.EntityIdToObject(e.instanceId);
                    // If a MeshRenderer on a tracked target changed, its material slots may have been reassigned.
                    if (obj is MeshRenderer mr && IsRendererOnTarget(mr))
                        _gpuScene?.MarkRebuildDirty();
                }
            }
        }

        private static bool IsMaterialUsedByTargets(Material mat)
        {
            foreach (var target in NativeRayTracingTarget.All)
            {
                if (target == null) continue;
                var renderer = target.GetComponent<Renderer>();
                if (renderer == null) continue;
                foreach (var m in renderer.sharedMaterials)
                {
                    if (m == mat) return true;
                }
            }

            return false;
        }

        private static bool IsRendererOnTarget(MeshRenderer mr)
        {
            foreach (var target in NativeRayTracingTarget.All)
            {
                if (target == null) continue;
                if (target.GetComponent<MeshRenderer>() == mr) return true;
            }

            return false;
        }

#endif
    }
}