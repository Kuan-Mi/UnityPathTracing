using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace PathTracing
{
    /// <summary>
    /// Drives the Streamline / DLSS-G Reflex frame loop once per frame at the START of
    /// the frame, by issuing a render-thread plugin event on
    /// <see cref="RenderPipelineManager.beginContextRendering"/>.
    ///
    /// This is the C# half of the "step 2b" pacing spike (see StreamlineProbe.cpp).
    /// The native <c>BeginFrame()</c> it triggers mints the frame token and calls
    /// <c>slReflexSleep</c> + <c>eSimulationStart</c> + input tagging. Doing this at
    /// frame begin (rather than at present, as step 2a did) is what gives the SL pacer
    /// a real per-frame timeline so DLSS-G can actually generate an interpolated frame.
    ///
    /// Deliberately self-contained and separate from <see cref="NativeFrameTickDriver"/>
    /// (which fires at <c>endContextRendering</c>, the wrong end of the frame for a
    /// Reflex sleep) so this spike stays cleanly removable.
    /// </summary>
    public static class StreamlineFrameDriver
    {
        // Exported by StreamlineProbePlugin.dll (the load-on-startup SL probe plugin).
        private const string DllName = "StreamlineProbePlugin";

        [DllImport(DllName)]
        private static extern IntPtr NR_SL_GetReflexBeginEventFunc();

        [DllImport(DllName)]
        private static extern void NR_SL_SetFrameGeneration(int enable);

        [DllImport(DllName)]
        private static extern int NR_SL_IsFrameGenerationOn();

        private static CommandBuffer _cmd;
        private static IntPtr        _eventFunc = IntPtr.Zero;
        // Set false if the probe DLL is absent/disabled so we stop trying every frame.
        private static bool          _available = true;

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
        private static void InitializeEditor()
        {
            Initialize();

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
            _cmd ??= new CommandBuffer { name = "StreamlineReflexBegin" };

            RenderPipelineManager.beginContextRendering -= OnBeginContextRendering;
            RenderPipelineManager.beginContextRendering += OnBeginContextRendering;

#if !UNITY_EDITOR
            Application.quitting -= Shutdown;
            Application.quitting += Shutdown;
#endif
            Debug.Log("StreamlineFrameDriver initialized and subscribed to RenderPipelineManager.beginContextRendering");
        }

        private static void OnBeginContextRendering(ScriptableRenderContext context, List<Camera> cameras)
        {
            if (_cmd == null || !_available) return;

            // Skip editor asset-preview / thumbnail contexts (own isolated RenderPipeline.Render),
            // matching NativeFrameTickDriver — issuing a render-thread event there crashes Unity.
            if (IsPreviewOnlyContext(cameras)) return;

            if (_eventFunc == IntPtr.Zero)
            {
                try
                {
                    _eventFunc = NR_SL_GetReflexBeginEventFunc();
                }
                catch (DllNotFoundException)
                {
                    _available = false;   // probe plugin not deployed/enabled — go quiet
                    Debug.LogWarning("StreamlineFrameDriver: StreamlineProbePlugin not found; DLSS-G frame driving disabled.");
                    return;
                }
                if (_eventFunc == IntPtr.Zero) { _available = false; return; }
            }

            _cmd.Clear();
            _cmd.IssuePluginEvent(_eventFunc, 0);
            Graphics.ExecuteCommandBuffer(_cmd);
        }

        /// <summary>
        /// Enable or disable DLSS-G frame generation at runtime. Safe to call any time
        /// (e.g. from a UI toggle); the native side records the request and applies it on
        /// the present thread at the next present. No-op if the probe plugin is absent.
        /// </summary>
        public static void SetFrameGeneration(bool enable)
        {
            if (!_available) return;
            try
            {
                NR_SL_SetFrameGeneration(enable ? 1 : 0);
            }
            catch (DllNotFoundException)
            {
                _available = false;
                Debug.LogWarning("StreamlineFrameDriver: StreamlineProbePlugin not found; SetFrameGeneration ignored.");
            }
        }

        /// <summary>True if DLSS-G frame generation is currently applied on the present thread.</summary>
        public static bool IsFrameGenerationOn()
        {
            if (!_available) return false;
            try
            {
                return NR_SL_IsFrameGenerationOn() != 0;
            }
            catch (DllNotFoundException)
            {
                _available = false;
                return false;
            }
        }

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
            RenderPipelineManager.beginContextRendering -= OnBeginContextRendering;
            Application.quitting                        -= Shutdown;
            _cmd?.Release();
            _cmd = null;
            _eventFunc = IntPtr.Zero;
            Debug.Log("StreamlineFrameDriver shutdown and unsubscribed from RenderPipelineManager.beginContextRendering");
        }
    }
}
