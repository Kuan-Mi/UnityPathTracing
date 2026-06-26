using System;
using UnityEngine;
using UnityEngine.InputSystem;

namespace SLDLRR
{
    public static class SLReflexRuntime
    {
        public enum Mode { Off = 0, On = 1, OnPlusBoost = 2 }

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
