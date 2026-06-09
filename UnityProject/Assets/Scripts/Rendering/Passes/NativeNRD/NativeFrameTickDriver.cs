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
        private static void InitializeEditor()
        {
            Initialize();

            // Exiting Play mode raises Application.quitting (-> Shutdown). With
            // "Reload Domain" disabled no domain reload follows, so neither
            // InitializeOnLoadMethod nor RuntimeInitializeOnLoadMethod re-runs and the
            // edit-mode tick stays dead. Re-initialize explicitly when we return to
            // Edit mode, and do the real teardown only on actual editor quit.
            UnityEditor.EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
            UnityEditor.EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
            UnityEditor.EditorApplication.quitting             -= Shutdown;
            UnityEditor.EditorApplication.quitting             += Shutdown;
        }

        private static void OnPlayModeStateChanged(UnityEditor.PlayModeStateChange state)
        {
            if (state == UnityEditor.PlayModeStateChange.EnteredEditMode)
                Initialize();
        }
#endif

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Initialize()
        {
            _cmd ??= new CommandBuffer { name = "NativeFrameTick" };

            // -= before += so we never accumulate duplicate subscriptions across
            // editor play sessions that have domain reload disabled.
            RenderPipelineManager.endContextRendering -= OnEndContextRendering;
            RenderPipelineManager.endContextRendering += OnEndContextRendering;

#if !UNITY_EDITOR
            // In a player, Application.quitting is the genuine end-of-process signal.
            // In the editor it also fires on Play-mode exit, which must NOT tear down
            // the edit-mode tick driver — that is handled via playModeStateChanged /
            // EditorApplication.quitting in InitializeEditor instead.
            Application.quitting -= Shutdown;
            Application.quitting += Shutdown;
#endif
            Debug.Log("NativeFrameTickDriver initialized and subscribed to RenderPipelineManager.endContextRendering");
        }

        private static void OnEndContextRendering(ScriptableRenderContext context, List<Camera> cameras)
        {
            if (_cmd == null) return;

            // Asset-preview / thumbnail rendering (e.g. FBX previews via
            // PreviewRenderUtility) drives its own isolated RenderPipeline.Render with a
            // temporary camera and graphics state. Issuing our render-thread plugin event
            // in that context crashes inside Unity's GetFrameFenceImpl. The frame tick
            // only needs to run for genuine editor/game frames, so skip contexts made up
            // entirely of preview/thumbnail cameras.
            if (IsPreviewOnlyContext(cameras)) return;

            _cmd.Clear();
            _cmd.IssuePluginEvent(NativeRenderPlugin.NR_GetFrameTickEventFunc(), 1);
            // Graphics.ExecuteCommandBuffer queues the event onto the render thread
            // after the frame's submitted work — matching the GPU-flush pattern used
            // elsewhere and avoiding a second context.Submit().
            Graphics.ExecuteCommandBuffer(_cmd);
            // Debug.Log($"NativeFrameTickDriver issued frame tick event for {Time.frameCount}");
        }

        // True when every camera in the context is a preview/thumbnail camera, i.e. an
        // editor asset-preview render rather than a real Scene/Game frame.
        private static bool IsPreviewOnlyContext(List<Camera> cameras)
        {
            if (cameras == null || cameras.Count == 0) return false;
            foreach (var cam in cameras)
            {
                if (cam == null) continue;
                if (cam.cameraType != CameraType.Preview)
                    return false;
            }
            return true;
        }

        private static void Shutdown()
        {
            RenderPipelineManager.endContextRendering -= OnEndContextRendering;
            Application.quitting                      -= Shutdown;
            _cmd?.Release();
            _cmd = null;
            Debug.Log("NativeFrameTickDriver shutdown and unsubscribed from RenderPipelineManager.endContextRendering");
        }
    }
}