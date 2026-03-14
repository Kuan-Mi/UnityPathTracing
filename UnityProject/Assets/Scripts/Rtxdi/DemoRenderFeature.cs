using System.Collections.Generic;
using RTXDI;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using static UnityEngine.Rendering.RayTracingAccelerationStructure;

namespace DefaultNamespace
{
    public class DemoRenderFeature : ScriptableRendererFeature
    {
        public Texture2D inputTexture;
        public DemoRenderPass demoRenderPass;

        public RayTracingAccelerationStructure accelerationStructure;

        public Settings settings;

        private Dictionary<long, DemoResource> _resources = new Dictionary<long, DemoResource>();

        public GPUScene gpuScene = new GPUScene();

        public override void Create()
        {
            if (accelerationStructure == null)
            {
                settings = new Settings
                {
                    managementMode = ManagementMode.Automatic,
                    rayTracingModeMask = RayTracingModeMask.Everything
                };
                accelerationStructure = new RayTracingAccelerationStructure(settings);

                accelerationStructure.Build();

                // SetMask();
            }

            if (gpuScene.IsEmpty())
            {
                gpuScene.Build();
            }

            demoRenderPass = new DemoRenderPass();
        }
 
        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            Camera cam = renderingData.cameraData.camera;
            if (cam.cameraType is CameraType.Preview or CameraType.Reflection)
                return;

            int eyeIndex = renderingData.cameraData.xr.enabled ? renderingData.cameraData.xr.multipassId : 0;


            long uniqueKey = cam.GetInstanceID() + (eyeIndex * 100000L);


            if (!_resources.TryGetValue(uniqueKey, out var demoResource))
            {
                demoResource = new DemoResource();
                _resources.Add(uniqueKey, demoResource);
                demoResource.SendTexture(gpuScene.globalTexturePool);
                demoResource.SetBuffer(gpuScene);
            }

            demoRenderPass.demoResource = demoResource;
            renderer.EnqueuePass(demoRenderPass);
        }

        protected override void Dispose(bool disposing)
        {
            foreach (var resource in _resources.Values)
            {
                resource.Dispose();
            }

            _resources.Clear();
        }

        public void Test()
        {
             gpuScene.DebugReadback();
        }
    }
}