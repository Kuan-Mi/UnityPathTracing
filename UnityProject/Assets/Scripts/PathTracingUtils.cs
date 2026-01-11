using UnityEngine;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    public static class PathTracingUtils
    {
        
        public static Matrix4x4 GetWorldToClipMatrix(Camera camera)
        {
            // Unity 的 GPU 投影矩阵（处理平台差异 & Y 翻转）
            Matrix4x4 proj = GL.GetGPUProjectionMatrix(camera.projectionMatrix, false);

            return proj * camera.worldToCameraMatrix;
        }
        
        public static Vector4 GetNrdFrustum(UniversalCameraData cameraData)
        {
            Matrix4x4 p = cameraData.xr.enabled? cameraData.xr.GetProjMatrix() : cameraData.camera.projectionMatrix;

            var isOrthographic = cameraData.camera.orthographic;

            float x0, x1, y0, y1;

            if (!isOrthographic)
            {

                float m00 = p.m00;
                float m11 = p.m11;
                float m02 = p.m02;
                float m12 = p.m12;
                
                // Debug.Log($"Proj Matrix: \n{p.m00}, {p.m01}, {p.m02}, {p.m03}\n{p.m10}, {p.m11}, {p.m12}, {p.m13}\n{p.m20}, {p.m21}, {p.m22}, {p.m23}\n{p.m30}, {p.m31}, {p.m32}, {p.m33}");

                // 计算 scale 和 offset
                // 这里的逻辑是将 uv [0,1] 映射到 view space 射线方向
                float x = -(1.0f - m02) / m00;
                float y = -(1.0f + m12) / m11;
                float z = 2.0f / m00;
                float w = 2.0f / m11;

                return new Vector4(-x, y, -z, w);
                
            }
            else
            {
                var cam = cameraData.camera;
                // 正交投影
                float halfHeight = cam.orthographicSize;
                float halfWidth = halfHeight * cam.aspect;
                x0 = -halfWidth;
                x1 = halfWidth;
                y0 = -halfHeight;
                y1 = halfHeight;
            }

            // 匹配 C++ 代码逻辑:
            // pfFrustum4[0] = -x0;
            // pfFrustum4[2] = x0 - x1;
            // 针对 D3D 风格 (Unity 在 GPU 端通常使用 D3D 类似约定):
            // pfFrustum4[1] = -y1;
            // pfFrustum4[3] = y1 - y0;


            var result = new Vector4(-x0, -y1, x0 - x1, y1 - y0);
            Debug .Log($"NRD Frustum: {result.x}, {result.y}, {result.z}, {result.w}");
            
            return new Vector4(-x0, -y1, x0 - x1, y1 - y0);
        }

    }
}