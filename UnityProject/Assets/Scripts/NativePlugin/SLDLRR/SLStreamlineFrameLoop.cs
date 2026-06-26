using System;
using System.Collections.Generic;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.LowLevel;

namespace SLDLRR
{
    public static class SLStreamlineFrameLoop
    {
        private static IntPtr _renderSubmitStartFunc;
        private static IntPtr _renderSubmitEndFunc;
        private static IntPtr _frameToken;
        private static bool _playerLoopInstalled;

        // Distinct profiler markers so the two PlayerLoop delegates don't both aggregate under the
        // generic "UpdateFunction.Invoke()" sample. The FrameBegin marker's time is dominated by
        // slReflexSleep (intentional CPU pacing), not CPU work.
        private static readonly ProfilerMarker s_frameBeginMarker = new ProfilerMarker("SL Reflex FrameBegin (Sleep+SimStart)");
        private static readonly ProfilerMarker s_simEndMarker     = new ProfilerMarker("SL Reflex SimulationEnd");

        private struct SLReflexFrameBegin { }
        private struct SLReflexSimulationEnd { }

        public static IntPtr CurrentFrameTokenPtr => _frameToken;

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
#endif
        }

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
#if UNITY_EDITOR
            if (!Application.isPlaying) return;
#endif
            using (s_frameBeginMarker.Auto())
            {
                try
                {
                    SLPclLatencyPing.ResetFrameState();
                    _frameToken = SLNative.SL_GetNewFrameToken();
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
