using System;
using UnityEngine;

namespace DefaultNamespace
{
    public class Rtxdi : MonoBehaviour
    {
        ReSTIRDIContext restirdiContext; 
        RtxdiResources resources;
        
        
        [ContextMenu("TestReSTIRDI")]
        public void TestReSTIRDI()
        {
            restirdiContext = new ReSTIRDIContext(1920, 1080);
            resources = new RtxdiResources(restirdiContext, 10, 100, 5);
        }



        public unsafe void Render()
        {
            resources.InitializeNeighborOffsets(restirdiContext.GetStaticParameters()->NeighborOffsetCount);
            restirdiContext.SetFrameIndex((uint)Time.frameCount);
            
            
            
        }
        
    }
}