using System;
using UnityEngine;
using UnityEngine.InputSystem;

namespace PathTracing.NativeInterop.Streamline
{
    public static class SLReflexRuntime
    {
        public enum Mode { Off = 0, On = 1, OnPlusBoost = 2 }

        public readonly struct Stats
        {
            public readonly bool lowLatencyAvailable;
            public readonly bool latencyReportAvailable;
            public readonly bool flashIndicatorDriverControlled;
            public readonly uint statsWindowMessage;

            public readonly ulong frameID;
            public readonly ulong inputSampleTime;
            public readonly ulong simStartTime;
            public readonly ulong simEndTime;
            public readonly ulong renderSubmitStartTime;
            public readonly ulong renderSubmitEndTime;
            public readonly ulong presentStartTime;
            public readonly ulong presentEndTime;
            public readonly ulong driverStartTime;
            public readonly ulong driverEndTime;
            public readonly ulong osRenderQueueStartTime;
            public readonly ulong osRenderQueueEndTime;
            public readonly ulong gpuRenderStartTime;
            public readonly ulong gpuRenderEndTime;
            public readonly uint gpuActiveRenderTimeUs;
            public readonly uint gpuFrameTimeUs;

            internal Stats(SLNative.ReflexStats native)
            {
                lowLatencyAvailable = native.lowLatencyAvailable != 0;
                latencyReportAvailable = native.latencyReportAvailable != 0;
                flashIndicatorDriverControlled = native.flashIndicatorDriverControlled != 0;
                statsWindowMessage = native.statsWindowMessage;
                frameID = native.frameID;
                inputSampleTime = native.inputSampleTime;
                simStartTime = native.simStartTime;
                simEndTime = native.simEndTime;
                renderSubmitStartTime = native.renderSubmitStartTime;
                renderSubmitEndTime = native.renderSubmitEndTime;
                presentStartTime = native.presentStartTime;
                presentEndTime = native.presentEndTime;
                driverStartTime = native.driverStartTime;
                driverEndTime = native.driverEndTime;
                osRenderQueueStartTime = native.osRenderQueueStartTime;
                osRenderQueueEndTime = native.osRenderQueueEndTime;
                gpuRenderStartTime = native.gpuRenderStartTime;
                gpuRenderEndTime = native.gpuRenderEndTime;
                gpuActiveRenderTimeUs = native.gpuActiveRenderTimeUs;
                gpuFrameTimeUs = native.gpuFrameTimeUs;
            }
        }

        public static KeyCode ReflexToggleKey = KeyCode.F8;
        public static KeyCode ReflexBoostKey  = KeyCode.F9;

        private static bool _reflexOn = true;
        private static ReflexKeyPoller _keyPoller;

        public static void SetMode(Mode mode, uint fpsCapUs = 0)
        {
            if (!SLNative.Available) return;
            try { SLNative.SL_SetReflexMode((int)mode, fpsCapUs); }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
        }

        public static int GetMode()
        {
            if (!SLNative.Available) return -1;
            try { return SLNative.SL_GetReflexMode(); }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); return -1; }
        }

        public static bool IsLowLatencyAvailable()
        {
            if (!SLNative.Available) return false;
            try { return SLNative.SL_IsReflexLowLatencyAvailable() != 0; }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); return false; }
        }

        public static bool TryGetStats(out Stats stats)
        {
            stats = default;
            if (!SLNative.Available) return false;
            try
            {
                bool ok = SLNative.SL_GetReflexStats(out var native) != 0;
                stats = new Stats(native);
                return ok;
            }
            catch (DllNotFoundException)
            {
                SLNative.MarkUnavailable();
                return false;
            }
        }

        internal static void EnsureKeyPoller()
        {
#if UNITY_EDITOR
            if (!Application.isPlaying) return;
#endif
            if (_keyPoller != null) return;
            var go = new GameObject("SLReflexKeyToggle") { hideFlags = HideFlags.HideAndDontSave };
            UnityEngine.Object.DontDestroyOnLoad(go);
            _keyPoller = go.AddComponent<ReflexKeyPoller>();
        }

        private static void ToggleReflex()
        {
            _reflexOn = !_reflexOn;
            SetMode(_reflexOn ? Mode.On : Mode.Off);
            Debug.Log($"[SLReflexRuntime] Reflex {(_reflexOn ? "ON (Low Latency)" : "OFF")} ({ReflexToggleKey}).");
        }

        private static void SetBoost()
        {
            _reflexOn = true;
            SetMode(Mode.OnPlusBoost);
            Debug.Log($"[SLReflexRuntime] Reflex ON + Boost ({ReflexBoostKey}).");
        }

        private sealed class ReflexKeyPoller : MonoBehaviour
        {
            private void Update()
            {
                if (WasPressedThisFrame(ReflexToggleKey)) ToggleReflex();
                if (WasPressedThisFrame(ReflexBoostKey))  SetBoost();
            }
        }

        private static bool WasPressedThisFrame(KeyCode key)
        {
            var keyboard = Keyboard.current;
            if (keyboard == null) return false;

            switch (key)
            {
                case KeyCode.F8: return keyboard.f8Key.wasPressedThisFrame;
                case KeyCode.F9: return keyboard.f9Key.wasPressedThisFrame;
                default: return false;
            }
        }
    }
}
