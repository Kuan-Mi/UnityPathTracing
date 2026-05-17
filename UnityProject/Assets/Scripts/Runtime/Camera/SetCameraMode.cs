using System;
using UnityEngine;

namespace DefaultNamespace
{
    public class SetCameraMode : MonoBehaviour
    {
        public Camera targetCamera;

        private void OnEnable()
        {
            if (targetCamera != null)
            {
                Debug.Log(targetCamera.depthTextureMode);

                targetCamera.depthTextureMode = DepthTextureMode.Depth | DepthTextureMode.MotionVectors;
            }
        }
    }
}