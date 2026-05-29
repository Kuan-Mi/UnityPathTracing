using System.Collections.Generic;
using NativeRender;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Issues the native plugin's per-frame tick exactly once per rendered frame,
    /// independent of which ScriptableRendererFeature is active.
    ///
    /// Replaces the old <c>NativeFrameTick</c> ScriptableRenderPass, which had to be
    /// manually enqueued by every feature and could be skipped by render-graph pass
    /// culling or in the editor's render-graph debug mode. The tick drains the native
    /// deferred-delete queue and recycles the transient descriptor ring / upload pool,
    /// so it must run reliably every frame.
    ///
    /// <see cref="RenderPipelineManager.endContextRendering"/> fires once per
    /// <c>RenderPipeline.Render</c> call — after all cameras in the context have been
    /// submitted — which is the natural once-per-frame boundary (a single call even in
    /// single-pass-instanced XR), so the old per-eye / per-feature de-duplication is no
    /// longer needed.
    /// </summary>
    public static class NativeFrameTickDriver
    {
        private static CommandBuffer _cmd;

#if UNITY_EDITOR
        // Also drive the tick in edit-mode rendering (e.g. Scene view), matching the
        // old pass which ran whenever a feature's AddRenderPasses executed. Without
        // this the native deferred-delete queue would not drain outside play mode.
        [UnityEditor.InitializeOnLoadMethod]
        private static void InitializeEditor() => Initialize();
#endif

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Initialize()
        {
            _cmd ??= new CommandBuffer { name = "NativeFrameTick" };

            // -= before += so we never accumulate duplicate subscriptions across
            // editor play sessions that have domain reload disabled.
            RenderPipelineManager.endContextRendering -= OnEndContextRendering;
            RenderPipelineManager.endContextRendering += OnEndContextRendering;

            Application.quitting -= Shutdown;
            Application.quitting += Shutdown;
            // Debug.Log("NativeFrameTickDriver initialized and subscribed to RenderPipelineManager.endContextRendering");
        }

        private static void OnEndContextRendering(ScriptableRenderContext context, List<Camera> cameras)
        {
            if (_cmd == null) return;

            _cmd.Clear();
            _cmd.IssuePluginEvent(NativeRenderPlugin.NR_GetFrameTickEventFunc(), 1);
            // Graphics.ExecuteCommandBuffer queues the event onto the render thread
            // after the frame's submitted work — matching the GPU-flush pattern used
            // elsewhere and avoiding a second context.Submit().
            Graphics.ExecuteCommandBuffer(_cmd); 
            // Debug.Log($"NativeFrameTickDriver issued frame tick event for {Time.frameCount}");
        }

        private static void Shutdown()
        {
            RenderPipelineManager.endContextRendering -= OnEndContextRendering;
            Application.quitting -= Shutdown;
            _cmd?.Release();
            _cmd = null;
            // Debug.Log("NativeFrameTickDriver shutdown and unsubscribed from RenderPipelineManager.endContextRendering");
        }
    }
}
