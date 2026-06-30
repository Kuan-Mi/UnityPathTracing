using System;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;

namespace PathTracing.NativeInterop.Streamline
{
    // Emits the Reflex eTriggerFlash PCL marker on the frame whose input sampled a click, so the
    // Reflex Latency Analyzer (LDAT) / flash indicator can measure click-to-photon latency. Mirror
    // of SLPclLatencyPing's main-thread input hook (same Dynamic-update gating + current frame
    // token). Optional: only meaningful with the hardware analyzer or the flash-indicator overlay.
    public static class SLReflexFlash
    {
        internal static void Register()
        {
            InputSystem.onAfterUpdate -= OnAfterInputUpdate;
            InputSystem.onAfterUpdate += OnAfterInputUpdate;
        }

        internal static void Unregister()
        {
            InputSystem.onAfterUpdate -= OnAfterInputUpdate;
        }

        private static void OnAfterInputUpdate()
        {
            if (!SLNative.Available) return;
#if UNITY_EDITOR
            if (!Application.isPlaying) return;
#endif
            if (InputState.currentUpdateType != InputUpdateType.Dynamic) return;

            var mouse = Mouse.current;
            if (mouse == null || !mouse.leftButton.wasPressedThisFrame) return;

            IntPtr token = SLStreamlineFrameLoop.CurrentFrameTokenPtr;
            if (token == IntPtr.Zero) return;

            try
            {
                SLNative.SL_MarkTriggerFlash(token);
            }
            catch (DllNotFoundException)
            {
                SLNative.MarkUnavailable();
            }
        }
    }
}