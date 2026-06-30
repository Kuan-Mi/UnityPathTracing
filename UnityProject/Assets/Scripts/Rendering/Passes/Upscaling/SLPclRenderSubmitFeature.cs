using System;
using PathTracing.NativeInterop.Streamline;
using UnityEngine;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// Renderer-wide Reflex/PCL submit markers for Streamline. Add this once to the URP renderer
    /// instead of wiring SLPclRenderSubmitStartPass/EndPass in every rendering feature.
    /// </summary>
    public sealed class SLPclRenderSubmitFeature : ScriptableRendererFeature
    {
        [SerializeField] private bool playerOnly = true;

        private SLPclRenderSubmitStartPass _startPass;
        private SLPclRenderSubmitEndPass   _endPass;
        private int                        _lastQueuedFrameSequence;

        public override void Create()
        {
            _startPass ??= new SLPclRenderSubmitStartPass
            {
                renderPassEvent = RenderPassEvent.BeforeRendering
            };
            _endPass ??= new SLPclRenderSubmitEndPass
            {
                renderPassEvent = RenderPassEvent.AfterRendering
            };
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (playerOnly && !Application.isPlaying) return;

            var cameraData = renderingData.cameraData;
            var cam = cameraData.camera;
            if (cam == null) return;
            if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView) return;
            if (cameraData.xr.enabled && cameraData.xr.multipassId != 0) return;
            if (cameraData.renderType != CameraRenderType.Base) return;

            var token = SLStreamlineFrameLoop.CurrentFrameTokenPtr;
            var frameSequence = SLStreamlineFrameLoop.CurrentFrameSequence;
            if (token == IntPtr.Zero || frameSequence == _lastQueuedFrameSequence) return;

            var startFunc = SLStreamlineFrameLoop.GetRenderSubmitStartEventFunc();
            var endFunc   = SLStreamlineFrameLoop.GetRenderSubmitEndEventFunc();
            if (startFunc == IntPtr.Zero || endFunc == IntPtr.Zero) return;

            _startPass.Setup(startFunc, token);
            _endPass.Setup(endFunc, token);
            renderer.EnqueuePass(_startPass);
            renderer.EnqueuePass(_endPass);
            _lastQueuedFrameSequence = frameSequence;
        }

        protected override void Dispose(bool disposing)
        {
            _startPass = null;
            _endPass = null;
            _lastQueuedFrameSequence = 0;
        }
    }
}
