using System;
using System.Collections.Generic;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.LowLevel;
using UnityEngine.Rendering;
using PathTracing;

namespace PathTracing.NativeInterop.Streamline
{
    public static class SLStreamlineFrameLoop
    {
        private static IntPtr _renderSubmitStartFunc;
        private static IntPtr _renderSubmitEndFunc;
        private static IntPtr _frameToken;
        private static int _frameSequence;
        private static bool _playerLoopInstalled;

        // Distinct profiler markers so the two PlayerLoop delegates don't both aggregate under the
        // generic "UpdateFunction.Invoke()" sample. The FrameBegin marker's time is dominated by
        // slReflexSleep (intentional CPU pacing), not CPU work.
        private static readonly ProfilerMarker s_frameBeginMarker = new ProfilerMarker("SL Reflex FrameBegin (Sleep+SimStart)");
        private static readonly ProfilerMarker s_simEndMarker     = new ProfilerMarker("SL Reflex SimulationEnd");

        private struct SLReflexFrameBegin { }
        private struct SLReflexSimulationEnd { }

        // This frame's Streamline token. In the PLAYER it is minted once per frame at EarlyUpdate
        // (OnPlayerLoopFrameBegin) and shared by Reflex, the RR/SR evaluate and DLSS-G present
        // tagging. In the EDITOR's edit mode it is re-minted per camera render
        // (OnEditorBeginCameraRendering), because the Scene/Game view repaint pump runs outside the
        // PlayerLoop and can render a camera several times per tick — a single shared token would be
        // tagged more than once, which Streamline forbids.
        public static IntPtr CurrentFrameTokenPtr => _frameToken;
        public static int CurrentFrameSequence => _frameSequence;

        public static IntPtr GetRenderSubmitStartEventFunc()
        {
            if (!SLNative.Available) return IntPtr.Zero;
            if (_renderSubmitStartFunc != IntPtr.Zero) return _renderSubmitStartFunc;
            try { _renderSubmitStartFunc = SLNative.GetSLRenderSubmitStartEventFunc(); }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
            return _renderSubmitStartFunc;
        }

        public static IntPtr GetRenderSubmitEndEventFunc()
        {
            if (!SLNative.Available) return IntPtr.Zero;
            if (_renderSubmitEndFunc != IntPtr.Zero) return _renderSubmitEndFunc;
            try { _renderSubmitEndFunc = SLNative.GetSLRenderSubmitEndEventFunc(); }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
            return _renderSubmitEndFunc;
        }

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
        private static void InitializeEditor() => Initialize();
#endif

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Initialize()
        {
            SLPclLatencyPing.Register();
            SLReflexFlash.Register();

            InstallPlayerLoop();
            SLReflexRuntime.EnsureKeyPoller();

            Application.quitting -= Teardown;
            Application.quitting += Teardown;
#if UNITY_EDITOR
            UnityEditor.AssemblyReloadEvents.beforeAssemblyReload -= Teardown;
            UnityEditor.AssemblyReloadEvents.beforeAssemblyReload += Teardown;
            // Application.quitting only fires on play-mode exit; EditorApplication.quitting is the
            // one that fires when the editor process itself closes. Without it, the PCL ping message
            // thread is left blocked in GetMessageW (a native call Unity can't abort), which leaves a
            // residual Unity.exe process behind after the editor window is closed.
            UnityEditor.EditorApplication.quitting -= Teardown;
            UnityEditor.EditorApplication.quitting += Teardown;

            // Edit mode only: the editor's Scene/Game view repaint pump runs OUTSIDE the PlayerLoop
            // and can render the same camera several times per PlayerLoop tick, while the shared
            // token is minted only once per tick. Sharing it across those repaints makes Streamline
            // reject the 2nd slSetConstants / PCL submit for the same (token, viewport). Mint a fresh
            // token at the start of each camera render instead. See OnEditorBeginCameraRendering.
            RenderPipelineManager.beginCameraRendering -= OnEditorBeginCameraRendering;
            RenderPipelineManager.beginCameraRendering += OnEditorBeginCameraRendering;
#endif
        }

#if UNITY_EDITOR
        // Runs immediately before each camera's AddRenderPasses (edit mode only — play mode keeps the
        // shared per-tick token that Reflex/FG depend on). Mints this camera render its own SL token
        // so the evaluate-time features (DLSS-RR/-SR) and the PCL submit markers tag a unique token
        // even when the editor repaint pump renders a camera multiple times per PlayerLoop tick.
        private static void OnEditorBeginCameraRendering(ScriptableRenderContext _, Camera cam)
        {
            if (Application.isPlaying) return;
            if (!SLNative.Available) return;
            if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView) return;
            try { _frameToken = SLNative.SL_GetNewFrameToken(); }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
        }
#endif

        private static void InstallPlayerLoop()
        {
            var loop = PlayerLoop.GetCurrentPlayerLoop();
            var list = new List<PlayerLoopSystem>(loop.subSystemList);
            list.RemoveAll(s => s.type == typeof(SLReflexFrameBegin));
            list.RemoveAll(s => s.type == typeof(SLReflexSimulationEnd));

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

            var simEndNode = new PlayerLoopSystem
            {
                type           = typeof(SLReflexSimulationEnd),
                updateDelegate = OnPlayerLoopSimulationEnd,
            };

            RemoveSystem(ref loop, typeof(SLReflexSimulationEnd));
            InsertAfter(
                ref loop,
                typeof(UnityEngine.PlayerLoop.PostLateUpdate),
                typeof(UnityEngine.PlayerLoop.PostLateUpdate.EndGraphicsJobsAfterScriptLateUpdate),
                simEndNode);

            PlayerLoop.SetPlayerLoop(loop);
            _playerLoopInstalled = true;
        }

        private static void RemovePlayerLoop()
        {
            if (!_playerLoopInstalled) return;
            var loop = PlayerLoop.GetCurrentPlayerLoop();
            var list = new List<PlayerLoopSystem>(loop.subSystemList);
            if (list.RemoveAll(s => s.type == typeof(SLReflexFrameBegin)) > 0)
                loop.subSystemList = list.ToArray();
            RemoveSystem(ref loop, typeof(SLReflexSimulationEnd));
            PlayerLoop.SetPlayerLoop(loop);
            _playerLoopInstalled = false;
        }

        private static void OnPlayerLoopFrameBegin()
        {
            if (!SLNative.Available) return;
            // Mint this frame's shared SL token every frame in BOTH the editor and the player:
            // the evaluate-time features (DLSS-RR/-SR) tag against CurrentFrameTokenPtr and run in
            // the editor's Scene/Game view (edit mode included), so the token must exist there too.
            // The Reflex sleep + simulation marker below are player-only no-ops in the editor
            // (native side gates them on s_IsPlayer), so they are harmless to call here.
            using (s_frameBeginMarker.Auto())
            {
                try
                {
                    SLPclLatencyPing.ResetFrameState();
                    _frameToken = SLNative.SL_GetNewFrameToken();
                    unchecked { _frameSequence++; }
                    if (_frameToken != IntPtr.Zero)
                    {
                        SLNative.SL_ReflexSleep(_frameToken);
                        SLNative.SL_MarkSimulationStart(_frameToken);
                    }
                }
                catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
            }
        }

        private static void OnPlayerLoopSimulationEnd()
        {
            if (!SLNative.Available) return;
            if (_frameToken == IntPtr.Zero) return;

            using (s_simEndMarker.Auto())
            {
                try { SLNative.SL_MarkSimulationEnd(_frameToken); }
                catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
            }
        }

        private static void Teardown()
        {
            SLPclLatencyPing.Unregister();
            SLReflexFlash.Unregister();
            RemovePlayerLoop();
#if UNITY_EDITOR
            RenderPipelineManager.beginCameraRendering -= OnEditorBeginCameraRendering;
#endif
            SLDlssg.Dispose();
        }

        private static bool InsertAfter(ref PlayerLoopSystem root, Type parentType, Type afterType, PlayerLoopSystem node)
        {
            if (root.subSystemList == null) return false;

            for (int i = 0; i < root.subSystemList.Length; ++i)
            {
                var child = root.subSystemList[i];
                if (child.type == parentType)
                {
                    var children = new List<PlayerLoopSystem>(child.subSystemList ?? Array.Empty<PlayerLoopSystem>());
                    children.RemoveAll(s => s.type == node.type);

                    int insertIndex = children.FindIndex(s => s.type == afterType);
                    if (insertIndex < 0) return false;

                    children.Insert(insertIndex + 1, node);
                    child.subSystemList = children.ToArray();
                    root.subSystemList[i] = child;
                    return true;
                }
            }

            return false;
        }

        private static bool RemoveSystem(ref PlayerLoopSystem root, Type type)
        {
            if (root.subSystemList == null) return false;

            bool removed = false;
            var children = new List<PlayerLoopSystem>(root.subSystemList);
            removed |= children.RemoveAll(s => s.type == type) > 0;

            for (int i = 0; i < children.Count; ++i)
            {
                var child = children[i];
                if (RemoveSystem(ref child, type))
                {
                    children[i] = child;
                    removed = true;
                }
            }

            if (removed)
                root.subSystemList = children.ToArray();
            return removed;
        }
    }
}
