using System;
using UnityEngine;

namespace Runtime
{
    public class CameraHelper : MonoBehaviour
    {
        [SerializeField]
        private Camera _camera;

        public Vector3 dir;
        public Bounds  aabb;

        [ContextMenu("Set Camera")]
        public void SetCamera()
        {
            var allMeshRenderers = FindObjectsByType<MeshRenderer>(FindObjectsSortMode.None);

             aabb = allMeshRenderers[0].bounds;
            foreach (var meshRenderer in allMeshRenderers)
            {
                aabb.Encapsulate(meshRenderer.bounds);
            }
            var center = aabb.center;
            _camera.transform.position = center;
            
            //     m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);
            // 朝向 vMin（对应 C++ 里的 normalize(lookAt - position) + atan2/asin）
            Vector3 lookAt = center - new Vector3(aabb.extents.x, aabb.extents.y, -aabb.extents.z);
            // Vector3 dir    = (lookAt - center).normalized;
 
            if (dir != Vector3.zero)
                transform.rotation = Quaternion.LookRotation(dir);
        }


        private void OnDrawGizmos()
        {
            Gizmos.color = Color.green;
            Gizmos.DrawWireCube(aabb.center, aabb.size);
        }
    }

} 