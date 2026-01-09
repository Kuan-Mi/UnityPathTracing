using System.Collections.Generic;
using UnityEngine;

namespace DefaultNamespace
{
    public class PrepareLightsPass
    {
        const uint RTXDI_INVALID_LIGHT_INDEX = 0xFFFFFFFF;
        
        
        public RTXDI_LightBufferParameters Process()
        {
            
            RTXDI_LightBufferParameters outLightBufferParams = new RTXDI_LightBufferParameters();
            
            List<PrepareLightsTask>tasks;
            
            uint lightBufferOffset = 0;
            
            var allMeshRenderers = GameObject.FindObjectsOfType<MeshRenderer>();
            int geometryInstanceCount =  allMeshRenderers.Length;
            
            uint[] geometryInstanceToLight = new uint[geometryInstanceCount];
            
            System.Array.Fill(geometryInstanceToLight, RTXDI_INVALID_LIGHT_INDEX);
            
             
            
            
            foreach (var instance in allMeshRenderers)
            {
                
            }
            
            return outLightBufferParams;
        }
    }
}