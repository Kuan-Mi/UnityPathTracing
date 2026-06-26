using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.LowLevel;
using UnityEngine.Rendering;

namespace SLDLRR
{
    public static class SLStreamlineFrameLoop
    {
        private static IntPtr _beginFunc;
        private static IntPtr _frameToken;
        private static bool _playerLoopInstalled;

        private struct SLReflexFrameBegin { }

        public static IntPtr CurrentFrameTokenPtr => _frameToken;

        public static IntPtr GetBeginEventFunc()
        {
            if (!SLNative.Available) return IntPtr.Zero;
            if (_beginFunc != IntPtr.Zero) return _beginFunc;
            try { _beginFunc = SLNative.GetSLFGBeginFrameFunc(); }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
            return _beginFunc;
        }

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
        private static void InitializeEditor() => Initialize();
#endif

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Initialize()
        {
            RenderPipelineManager.beginContextRendering -= OnBeginContextRendering;
            RenderPipelineManager.beginContextRendering += OnBeginContextRendering;
            SLPclLatencyPing.Register();

            InstallPlayerLoop();
            SLReflexRuntime.EnsureKeyPoller();

            Application.quitting -= Teardown;
            Application.quitting += Teardown;
#if UNITY_EDITOR
            UnityEditor.AssemblyReloadEvents.beforeAssemblyReload -= Teardown;
            UnityEditor.AssemblyReloadEvents.beforeAssemblyReload += Teardown;
#endif
        }

        private static void InstallPlayerLoop()
        {
            var loop = PlayerLoop.GetCurrentPlayerLoop();
            var list = new List<PlayerLoopSystem>(loop.subSystemList);
            list.RemoveAll(s => s.type == typeof(SLReflexFrameBegin));

            int insertIndex = list.FindIndex(s => s.type == typeof(UnityEngine.PlayerLoop.EarlyUpdate));
            if (insertIndex < 0)
            {
                int initializationIndex = list.FindIndex(s => s.type == typeof(UnityEngine.PlayerLoop.Initialization));
                insertIndex = initializationIndex >= 0 ? initializationIndex + 1 : 0;
            }

            list.Insert(insertIndex, new PlayerLoopSystem
            {
                type           = typeof(SLReflexFrameBegin),
                updateDelegate = OnPlayerLoopFrameBegin,
            });
            loop.subSystemList = list.ToArray();
            PlayerLoop.SetPlayerLoop(loop);
            _playerLoopInstalled = true;
        }

        private static void RemovePlayerLoop()
        {
            if (!_playerLoopInstalled) return;
            var loop = PlayerLoop.GetCurrentPlayerLoop();
            var list = new List<PlayerLoopSystem>(loop.subSystemList);
            if (list.RemoveAll(s => s.type == typeof(SLReflexFrameBegin)) > 0)
            {
                loop.subSystemList = list.ToArray();
                PlayerLoop.SetPlayerLoop(loop);
            }
            _playerLoopInstalled = false;
        }

        private static void OnPlayerLoopFrameBegin()
        {
            if (!SLNative.Available) return;
#if UNITY_EDITOR
            if (!Application.isPlaying) return;
#endif
            try
            {
                SLPclLatencyPing.ResetFrameState();
                _frameToken = SLNative.SL_FrameBegin();
            }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
        }

        private static void OnBeginContextRendering(ScriptableRenderContext context, List<Camera> cameras)
        {
            if (!SLNative.Available) return;
            if (IsPreviewOnlyContext(cameras)) return;
            if (_frameToken == IntPtr.Zero) return;

            try { SLNative.SL_MarkSimulationEnd(_frameToken); }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
        }

        private static void Teardown()
        {
            RenderPipelineManager.beginContextRendering -= OnBeginContextRendering;
            SLPclLatencyPing.Unregister();
            RemovePlayerLoop();
            SLDlssg.Dispose();
        }

        private static bool IsPreviewOnlyContext(List<Camera> cameras)
        {
            if (cameras == null || cameras.Count == 0) return false;
            foreach (var cam in cameras)
            {
                if (cam == null) continue;
                if (cam.cameraType != CameraType.Preview) return false;
            }
            return true;
        }
    }
}
