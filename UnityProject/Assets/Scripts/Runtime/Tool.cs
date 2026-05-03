using UnityEngine;

namespace Runtime
{
    public class Tool : MonoBehaviour
    {
        [ContextMenu("fun")]
        public void fun()
        {
            var allMeshRenderers = GetComponentsInChildren<MeshRenderer>();
            var meshCount        = 0;
            
            
            var m_OpaqueObjectsNum      = 0;
            var m_TransparentObjectsNum = 0;
            var m_EmissiveObjectsNum    = 0;
            
            foreach (var allMeshRenderer in allMeshRenderers)
            {
                meshCount += allMeshRenderer.GetComponent<MeshFilter>().sharedMesh.subMeshCount;

                foreach (var material in allMeshRenderer.sharedMaterials)
                {
                    var IsTransparent = material.IsKeywordEnabled("_SURFACE_TYPE_TRANSPARENT");
                    var isEmissive    = material.IsKeywordEnabled("_EMISSIVE");

                    if (IsTransparent)
                    {
                        m_TransparentObjectsNum++;
                    }else
                    {
                        m_OpaqueObjectsNum++;
                    }
                    
                    if (isEmissive)
                    {
                        m_EmissiveObjectsNum++;
                    }
                    
                }
            }
            
            Debug.Log(meshCount);
            Debug.Log($"Opaque: {m_OpaqueObjectsNum}, Transparent: {m_TransparentObjectsNum}, Emissive: {m_EmissiveObjectsNum}");
            Debug.Log($"All {m_OpaqueObjectsNum + m_TransparentObjectsNum + m_EmissiveObjectsNum} objects");
        }
        
    }
}