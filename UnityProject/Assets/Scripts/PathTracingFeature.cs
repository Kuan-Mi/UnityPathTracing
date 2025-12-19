using System.Collections.Generic;
using Nrd;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
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
        public ComputeBuffer gIn_ScramblingRankingUint;
        public ComputeBuffer gIn_SobolUint;

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


            if (gIn_ScramblingRankingUint == null)
            {
                Debug.Log(
                    $"gIn_ScramblingRanking {gIn_ScramblingRanking.format} width:{gIn_ScramblingRanking.width} height:{gIn_ScramblingRanking.height}");
                Debug.Log($"gIn_Sobol {gIn_Sobol.format} width:{gIn_Sobol.width} height:{gIn_Sobol.height}");

                gIn_ScramblingRankingUint =
                    new ComputeBuffer(gIn_ScramblingRanking.width * gIn_ScramblingRanking.height, 16);
                var scramblingRankingData = new uint4[gIn_ScramblingRanking.width * gIn_ScramblingRanking.height];
                byte[] rawData = gIn_ScramblingRanking.GetRawTextureData();
                
                Color32[] colors = gIn_ScramblingRanking.GetPixels32();


                Debug.Log($"gIn_ScramblingRanking rawData Length: {rawData.Length}");
                Debug.Log($"gIn_ScramblingRanking colors Length: {colors.Length}");
                
                
                int count = scramblingRankingData.Length;
                for (int i = 0; i < count; i++)
                {
                    scramblingRankingData[i] = new uint4(
                        rawData[i * 4 + 0],
                        rawData[i * 4 + 1],
                        rawData[i * 4 + 2],
                        rawData[i * 4 + 3]);                   
                    
                    // scramblingRankingData[i] = new uint4(
                    //     colors[i].r,
                    //     colors[i].g,
                    //     colors[i].b,
                    //     colors[i].a);
                }

                gIn_ScramblingRankingUint.SetData(scramblingRankingData);


                gIn_SobolUint = new ComputeBuffer(gIn_Sobol.width * gIn_Sobol.height, 16);
                var sobolData = new uint4[gIn_Sobol.width * gIn_Sobol.height];
                rawData = gIn_Sobol.GetRawTextureData();
                colors = gIn_Sobol.GetPixels32();

                Debug.Log($"gIn_Sobol rawData Length: {rawData.Length}");
                Debug.Log($"gIn_Sobol colors Length: {colors.Length}");

                count = sobolData.Length;
                for (int i = 0; i < count; i++)
                {
                    sobolData[i] = new uint4(
                        rawData[i * 4 + 0],
                        rawData[i * 4 + 1],
                        rawData[i * 4 + 2],
                        rawData[i * 4 + 3]);
                    
                    // sobolData[i] = new uint4(
                    //     colors[i].r,
                    //     colors[i].g,
                    //     colors[i].b,
                    //     colors[i].a);
                }
 
                gIn_SobolUint.SetData(sobolData);
            }

            rayTracingShader.SetShaderPass("Test2");

            _pathTracingPass = new PathTracingPassSingle(pathTracingSetting)
            {
                renderPassEvent = RenderPassEvent.BeforeRenderingPostProcessing,
                rayTracingShader = rayTracingShader,
                accelerationStructure = accelerationStructure,
                scramblingRanking = gIn_ScramblingRankingUint,
                sobol = gIn_SobolUint
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