using System;
using System.Runtime.InteropServices;

namespace SLDLRR
{
    internal static class SLNative
    {
        private const string DllName = "SLDenoiser";

        internal static bool Available = true;

        internal static void MarkUnavailable() => Available = false;

        [DllImport(DllName)] internal static extern IntPtr GetSLRenderSubmitStartEventFunc();
        [DllImport(DllName)] internal static extern IntPtr GetSLRenderSubmitEndEventFunc();
        [DllImport(DllName)] internal static extern IntPtr GetSLFGFrameInputsFunc();
        [DllImport(DllName)] internal static extern void   SL_SetFrameGeneration(int enable);
        [DllImport(DllName)] internal static extern int    SL_IsFrameGenerationOn();

        [DllImport(DllName)] internal static extern void   SL_SetReflexMode(int mode, uint fpsCapUs);
        [DllImport(DllName)] internal static extern int    SL_GetReflexMode();
        [DllImport(DllName)] internal static extern int    SL_IsReflexLowLatencyAvailable();

        [DllImport(DllName)] internal static extern IntPtr SL_GetNewFrameToken();
        [DllImport(DllName)] internal static extern void   SL_ReflexSleep(IntPtr frameToken);
        [DllImport(DllName)] internal static extern void   SL_MarkSimulationStart(IntPtr frameToken);
        [DllImport(DllName)] internal static extern void   SL_MarkSimulationEnd(IntPtr frameToken);
        [DllImport(DllName)] internal static extern uint   SL_ConsumePclPingCount();
        [DllImport(DllName)] internal static extern void   SL_MarkPclLatencyPing(IntPtr frameToken, uint count);
    }
}
