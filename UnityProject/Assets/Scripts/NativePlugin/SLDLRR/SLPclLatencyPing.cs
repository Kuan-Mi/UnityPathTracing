using System;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;

namespace SLDLRR
{
    public static class SLPclLatencyPing
    {
        public static bool LastFrameHadPing { get; private set; }
        public static uint LastFramePingCount { get; private set; }

        internal static void Register()
        {
            InputSystem.onAfterUpdate -= OnAfterInputUpdate;
            InputSystem.onAfterUpdate += OnAfterInputUpdate;
        }

        internal static void Unregister()
        {
            InputSystem.onAfterUpdate -= OnAfterInputUpdate;
        }

        internal static void ResetFrameState()
        {
            LastFramePingCount = 0;
            LastFrameHadPing = false;
        }

        private static void OnAfterInputUpdate()
        {
            if (!SLNative.Available) return;
#if UNITY_EDITOR
            if (!Application.isPlaying) return;
#endif
            if (InputState.currentUpdateType != InputUpdateType.Dynamic) return;

            IntPtr token = SLStreamlineFrameLoop.CurrentFrameTokenPtr;
            if (token == IntPtr.Zero) return;

            try
            {
                LastFramePingCount = SLNative.SL_ConsumePclPingCount();
                LastFrameHadPing = LastFramePingCount != 0;
                if (LastFramePingCount != 0)
                {
                    Debug.Log($"Mark ping {LastFramePingCount}");
                    SLNative.SL_MarkPclLatencyPing(token, LastFramePingCount);
                }
                else
                {
                    Debug.Log("No ping");
                }
            }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
        }
    }
}
