using System.Collections.Generic;
using Nrd;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using static UnityEngine.Rendering.RayTracingAccelerationStructure;

namespace PathTracing
{
    public class PathTracingFeature : ScriptableRendererFeature
    {
        public Material showShadowMaterial;
        public RayTracingShader rayTracingShader;
        public PathTracingSetting pathTracingSetting;

        private PathTracingPassSingle _pathTracingPass;

        public RayTracingAccelerationStructure accelerationStructure;
        public Settings settings;

        public Texture2D gIn_ScramblingRanking;
        public Texture2D gIn_Sobol;

        private Dictionary<int, NRDDenoiser> m_HelperDic = new();

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
            }

            
            rayTracingShader.SetShaderPass("Test2");

            _pathTracingPass = new PathTracingPassSingle(pathTracingSetting)
            {
                renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing,
                rayTracingShader = rayTracingShader,
                accelerationStructure = accelerationStructure,
                scramblingRanking = RTHandles.Alloc(gIn_ScramblingRanking),
                sobol = RTHandles.Alloc(gIn_Sobol)
            };

            if (showShadowMaterial == null)
            {
                var shader = Shader.Find("KM/ShowShadow");
                showShadowMaterial = new Material(shader);
            }

            _pathTracingPass.biltMaterial = showShadowMaterial;
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            Camera cam = renderingData.cameraData.camera;
            if (cam.cameraType == CameraType.Preview || cam.cameraType == CameraType.Reflection)
                return;

            int camID = cam.GetInstanceID();

            if (!m_HelperDic.TryGetValue(camID, out var nrd))
            {
                nrd = new NRDDenoiser(pathTracingSetting);
                m_HelperDic.Add(camID, nrd);
            }

            _pathTracingPass.NrdDenoiser = nrd;
            renderer.EnqueuePass(_pathTracingPass);
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
            accelerationStructure.Dispose();
            accelerationStructure.Release();
            accelerationStructure = null;
            _pathTracingPass.Dispose();

            foreach (var helper in m_HelperDic.Values)
            {
                helper.Dispose();
            }

            m_HelperDic.Clear();
        }
    }
}