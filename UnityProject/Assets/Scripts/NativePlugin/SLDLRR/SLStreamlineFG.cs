using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.LowLevel;
using UnityEngine.Rendering;

namespace SLDLRR
{
    /// <summary>
    /// DLSS-G (Frame Generation) driver for the SLDenoiser plugin. Mirror of the legacy
    /// <c>StreamlineFrameDriver</c> (which targets StreamlineProbePlugin), but points at
    /// SLDenoiser and feeds the real path-tracer inputs. Issues the Reflex/begin tick once
    /// per frame at <see cref="RenderPipelineManager.beginContextRendering"/>; the per-frame
    /// depth/mvec/constants are pushed separately via <see cref="SLDlssgInputsPass"/>.
    ///
    /// IMPORTANT: keep mutually exclusive with StreamlineProbePlugin (one slInit per process).
    /// SLDenoiser only installs the FG present hooks in a standalone player; in the editor
    /// BeginFrame is a no-op (the swapchain is never adopted).
    /// </summary>
    public static class SLStreamlineFG
    {
        private const string DllName = "SLDenoiser";

        [DllImport(DllName)] private static extern IntPtr GetSLFGBeginFrameFunc();
        [DllImport(DllName)] private static extern IntPtr GetSLFGFrameInputsFunc();
        [DllImport(DllName)] private static extern void   SL_SetFrameGeneration(int enable);
        [DllImport(DllName)] private static extern int    SL_IsFrameGenerationOn();

        // Reflex low latency (independent of frame generation; works with FG off + in editor).
        [DllImport(DllName)] private static extern void   SL_SetReflexMode(int mode, uint fpsCapUs);
        [DllImport(DllName)] private static extern int    SL_GetReflexMode();
        [DllImport(DllName)] private static extern int    SL_IsReflexLowLatencyAvailable();

        // Reflex frame loop — called DIRECTLY on the main thread (not via render events).
        // SL_FrameBegin must run at the very top of the frame, before input; it returns the
        // frame token pointer to forward to the render-thread begin event as its data.
        [DllImport(DllName)] private static extern IntPtr SL_FrameBegin();
        [DllImport(DllName)] private static extern void   SL_MarkSimulationEnd();

        /// <summary>Reflex low-latency mode. Mirrors sl::ReflexMode.</summary>
        public enum ReflexMode { Off = 0, On = 1, OnPlusBoost = 2 }

        // D3D12_RESOURCE_STATES for the tagged depth/mvec (Unity UAV pool RTs).
        public const uint D3D12_STATE_UNORDERED_ACCESS    = 0x08;
        public const uint D3D12_STATE_ALL_SHADER_RESOURCE = 0x40 | 0x80;

        /// <summary>
        /// Mirror of native <c>SLDlssg::FrameInputs</c> (sequential, natural alignment — must
        /// match field-for-field). Matrices are Unity <see cref="Matrix4x4"/> (column-major);
        /// the native side memcpy's them into row-major sl::float4x4 (the transpose SL expects).
        /// </summary>
        [StructLayout(LayoutKind.Sequential)]
        public struct DlssgInputs
        {
            public IntPtr depth;          // linear view-Z (render res)  — ID3D12Resource*
            public IntPtr motionVectors;  // screen-space motion vectors (render res)
            public uint   depthState, mvecState;
            public uint   mvecDepthW, mvecDepthH; // render res
            public uint   colorW, colorH;         // output/display res

            public Matrix4x4 cameraViewToClip;
            public Matrix4x4 clipToCameraView;
            public Matrix4x4 clipToPrevClip;
            public Matrix4x4 prevClipToClip;

            public Vector3 cameraPos;
            public Vector3 cameraUp;
            public Vector3 cameraRight;
            public Vector3 cameraFwd;

            public float jitterX, jitterY;
            public float mvecScaleX, mvecScaleY;
            public float cameraNear, cameraFar, cameraFOV, cameraAspect;
            public int   depthInverted;
            public int   cameraMotionIncluded;
            public int   motionVectors3D;
            public int   reset;
        }

        private static bool   _available = true;
        private static IntPtr _frameInputsEventFunc = IntPtr.Zero;

        /// <summary>Render-event-and-data function for pushing per-frame FG inputs.</summary>
        public static IntPtr GetFrameInputsEventFunc()
        {
            if (!_available) return IntPtr.Zero;
            if (_frameInputsEventFunc != IntPtr.Zero) return _frameInputsEventFunc;
            try { _frameInputsEventFunc = GetSLFGFrameInputsFunc(); }
            catch (DllNotFoundException) { _available = false; }
            return _frameInputsEventFunc;
        }

        public static void SetFrameGeneration(bool enable)
        {
            if (!_available) return;
            try { SL_SetFrameGeneration(enable ? 1 : 0); }
            catch (DllNotFoundException) { _available = false; }
        }

        public static bool IsFrameGenerationOn()
        {
            if (!_available) return false;
            try { return SL_IsFrameGenerationOn() != 0; }
            catch (DllNotFoundException) { _available = false; return false; }
        }

        /// <summary>
        /// Set the Reflex low-latency mode. Applies on the next frame begin. This is the
        /// primary lever for the high-latency feel with frame generation — DLSS-G adds a
        /// frame of latency, and Reflex claws it back by pacing the CPU. <see cref="ReflexMode.OnPlusBoost"/>
        /// gives the lowest latency (at a small FPS cost). Independent of FG, so it also
        /// lowers latency with FG off. <paramref name="fpsCapUs"/> is an optional FPS cap in
        /// microseconds (0 = uncapped); works even when mode is Off.
        /// </summary>
        public static void SetReflexMode(ReflexMode mode, uint fpsCapUs = 0)
        {
            if (!_available) return;
            try { SL_SetReflexMode((int)mode, fpsCapUs); }
            catch (DllNotFoundException) { _available = false; }
        }

        /// <summary>The Reflex mode currently applied (-1 until first applied).</summary>
        public static int GetReflexMode()
        {
            if (!_available) return -1;
            try { return SL_GetReflexMode(); }
            catch (DllNotFoundException) { _available = false; return -1; }
        }

        /// <summary>
        /// sl::ReflexState::lowLatencyAvailable — use only to gate Reflex UI (hide/disable on
        /// non-NVIDIA / older hardware). Do everything else the same regardless.
        /// </summary>
        public static bool IsReflexLowLatencyAvailable()
        {
            if (!_available) return false;
            try { return SL_IsReflexLowLatencyAvailable() != 0; }
            catch (DllNotFoundException) { _available = false; return false; }
        }

        // ---- Per-frame inputs ring (kept here so the unsafe ptr math stays in this assembly) ----
        private const int                  RingCount = 3;
        private static NativeArray<DlssgInputs> _ring;

        /// <summary>
        /// Stores this frame's FG inputs in a persistent ring and returns a pointer valid until
        /// the render thread consumes the IssuePluginEventAndData event. Mirrors DlrrDenoiser.
        /// </summary>
        public static IntPtr GetInteropDataPtr(in DlssgInputs inputs, uint frameIndex)
        {
            if (!_ring.IsCreated)
                _ring = new NativeArray<DlssgInputs>(RingCount, Allocator.Persistent);
            int idx = (int)(frameIndex % RingCount);
            _ring[idx] = inputs;
            unsafe { return (IntPtr)_ring.GetUnsafePtr() + idx * sizeof(DlssgInputs); }
        }

        private static void DisposeRing()
        {
            if (_ring.IsCreated) _ring.Dispose();
        }

        // ---- Reflex frame loop ----
        // The latency-critical Reflex sleep + eSimulationStart run on the MAIN thread at the
        // very top of the frame (before input) via a PlayerLoop system — slReflexSleep paces
        // the simulation thread, so it must NOT be a render-thread plugin event. The frame
        // token minted here is forwarded to the render thread by SLReflexBeginPass (a RenderGraph
        // pass at BeforeRendering) so DLSS tagging + present markers pin to this frame; the
        // present hook then closes the PCL timeline on it.
        private static IntPtr _beginFunc  = IntPtr.Zero;
        private static IntPtr _frameToken = IntPtr.Zero;

        /// <summary>Native render-thread begin/latch event func (forward this frame's token as data).</summary>
        public static IntPtr GetBeginEventFunc()
        {
            if (!_available) return IntPtr.Zero;
            if (_beginFunc != IntPtr.Zero) return _beginFunc;
            try { _beginFunc = GetSLFGBeginFrameFunc(); }
            catch (DllNotFoundException) { _available = false; }
            return _beginFunc;
        }

        /// <summary>This frame's Streamline FrameToken (minted on the main thread); Zero if none yet.</summary>
        public static IntPtr CurrentFrameTokenPtr => _frameToken;

        // Marker type identifying our injected PlayerLoop system.
        private struct SLReflexFrameBegin { }
        private static bool _playerLoopInstalled;

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
        private static void InitializeEditor() => Initialize();
#endif

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Initialize()
        {
            RenderPipelineManager.beginContextRendering -= OnBeginContextRendering;
            RenderPipelineManager.beginContextRendering += OnBeginContextRendering;

            InstallPlayerLoop();

            Application.quitting -= Teardown;
            Application.quitting += Teardown;
#if UNITY_EDITOR
            UnityEditor.AssemblyReloadEvents.beforeAssemblyReload -= Teardown;
            UnityEditor.AssemblyReloadEvents.beforeAssemblyReload += Teardown;
#endif
        }

        // Inserts SL_FrameBegin at the very front of the PlayerLoop (before input/sim).
        private static void InstallPlayerLoop()
        {
            var loop = PlayerLoop.GetCurrentPlayerLoop();
            var list = new List<PlayerLoopSystem>(loop.subSystemList);
            list.RemoveAll(s => s.type == typeof(SLReflexFrameBegin));
            list.Insert(0, new PlayerLoopSystem
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

        // MAIN thread, top of frame. slReflexSleep + eSimulationStart for the new frame.
        // Edit mode is intentionally skipped — we don't want Reflex pacing the editor's loop
        // when not playing (DLSS-RR mints its own token natively in that case).
        private static void OnPlayerLoopFrameBegin()
        {
            if (!_available) return;
#if UNITY_EDITOR
            if (!Application.isPlaying) return;
#endif
            try { _frameToken = SL_FrameBegin(); }
            catch (DllNotFoundException) { _available = false; }
        }

        // Simulation (Update/LateUpdate) just finished and rendering is starting — close the
        // sim window on the MAIN thread. Direct P/Invoke, no command buffer. The render-thread
        // token latch is issued separately by SLReflexBeginPass inside the RenderGraph.
        private static void OnBeginContextRendering(ScriptableRenderContext context, List<Camera> cameras)
        {
            if (!_available) return;
            if (IsPreviewOnlyContext(cameras)) return;

            try { SL_MarkSimulationEnd(); }
            catch (DllNotFoundException) { _available = false; }
        }

        private static void Teardown()
        {
            RemovePlayerLoop();
            DisposeRing();
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
