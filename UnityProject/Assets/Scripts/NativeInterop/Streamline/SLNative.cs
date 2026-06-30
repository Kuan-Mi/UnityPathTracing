using System;
using System.Runtime.InteropServices;

namespace PathTracing.NativeInterop.Streamline
{
    internal static class SLNative
    {
        private const string DllName = "StreamlinePlugin";

        [StructLayout(LayoutKind.Sequential)]
        internal struct ReflexStats
        {
            public int lowLatencyAvailable;
            public int latencyReportAvailable;
            public int flashIndicatorDriverControlled;
            public uint statsWindowMessage;

            public ulong frameID;
            public ulong inputSampleTime;
            public ulong simStartTime;
            public ulong simEndTime;
            public ulong renderSubmitStartTime;
            public ulong renderSubmitEndTime;
            public ulong presentStartTime;
            public ulong presentEndTime;
            public ulong driverStartTime;
            public ulong driverEndTime;
            public ulong osRenderQueueStartTime;
            public ulong osRenderQueueEndTime;
            public ulong gpuRenderStartTime;
            public ulong gpuRenderEndTime;
            public uint gpuActiveRenderTimeUs;
            public uint gpuFrameTimeUs;
        }

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
        [DllImport(DllName)] internal static extern int    SL_GetReflexStats(out ReflexStats stats);

        [DllImport(DllName)] internal static extern IntPtr SL_GetNewFrameToken();
        [DllImport(DllName)] internal static extern void   SL_ReflexSleep(IntPtr frameToken);
        [DllImport(DllName)] internal static extern void   SL_MarkSimulationStart(IntPtr frameToken);
        [DllImport(DllName)] internal static extern void   SL_MarkSimulationEnd(IntPtr frameToken);
        [DllImport(DllName)] internal static extern void   SL_SetPclPingThreadId(uint threadId);
        [DllImport(DllName)] internal static extern void   SL_MarkPclLatencyPing(IntPtr frameToken, uint count);
        [DllImport(DllName)] internal static extern void   SL_MarkTriggerFlash(IntPtr frameToken);
    }
}
