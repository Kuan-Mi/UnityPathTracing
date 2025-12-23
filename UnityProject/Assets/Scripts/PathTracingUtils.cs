using UnityEngine;

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
        
        public static Vector4 GetNrdFrustum(Camera cam)
        {
            Matrix4x4 p = cam.projectionMatrix;

            // Unity 的投影矩阵 p 的元素索引:
            // [0,0] = 2n/(r-left), [0,2] = (r+left)/(r-left)
            // [1,1] = 2n/(top-bottom), [1,2] = (top+bottom)/(top-bottom)

            float x0, x1, y0, y1;

            if (!cam.orthographic)
            {
                // 透视投影重建 (基于投影矩阵的逆推)
                // 对应 C++ 中的 x0 = vPlane[PLANE_LEFT].z / vPlane[PLANE_LEFT].x
                x0 = (-1.0f - p.m02) / p.m00;
                x1 = (1.0f - p.m02) / p.m00;
                y0 = (-1.0f - p.m12) / p.m11;
                y1 = (1.0f - p.m12) / p.m11;
            }
            else
            {
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

            return new Vector4(-x0, -y1, x0 - x1, y1 - y0);
        }

    }
}