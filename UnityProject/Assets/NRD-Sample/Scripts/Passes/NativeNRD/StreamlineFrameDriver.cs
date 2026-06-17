// using System;
// using System.Collections.Generic;
// using System.Runtime.InteropServices;
// using UnityEngine;
// using UnityEngine.Rendering;
//
// namespace PathTracing
// {
//     /// <summary>
//     /// Drives the Streamline / DLSS-G Reflex frame loop once per frame at the START of
//     /// the frame, by issuing a render-thread plugin event on
//     /// <see cref="RenderPipelineManager.beginContextRendering"/>.
//     ///
//     /// This is the C# half of the "step 2b" pacing spike (see StreamlineProbe.cpp).
//     /// The native <c>BeginFrame()</c> it triggers mints the frame token and calls
//     /// <c>slReflexSleep</c> + <c>eSimulationStart</c> + input tagging. Doing this at
//     /// frame begin (rather than at present, as step 2a did) is what gives the SL pacer
//     /// a real per-frame timeline so DLSS-G can actually generate an interpolated frame.
//     ///
//     /// Deliberately self-contained and separate from <see cref="NativeFrameTickDriver"/>
//     /// (which fires at <c>endContextRendering</c>, the wrong end of the frame for a
//     /// Reflex sleep) so this spike stays cleanly removable.
//     /// </summary>
//     public static class StreamlineFrameDriver
//     {
//         // Exported by StreamlineProbePlugin.dll (the load-on-startup SL probe plugin).
//         private const string DllName = "StreamlineProbePlugin";
//
//         [DllImport(DllName)]
//         private static extern IntPtr NR_SL_GetReflexBeginEventFunc();
//
//         [DllImport(DllName)]
//         private static extern void NR_SL_SetFrameGeneration(int enable);
//
//         [DllImport(DllName)]
//         private static extern int NR_SL_IsFrameGenerationOn();
//
//         [DllImport(DllName)]
//         private static extern IntPtr NR_SL_GetFrameInputsEventFunc();
//
//         private static IntPtr _frameInputsEventFunc = IntPtr.Zero;
//
//         // D3D12_RESOURCE_STATES the tagged textures are in when Streamline reads them at
//         // present. NRI SRV textures (ViewZ/Mv) sit in ALL_SHADER_RESOURCE; UAV textures
//         // (Final) in UNORDERED_ACCESS. These are best-effort steady states — if the debug
//         // layer complains about a barrier, this is the first knob to adjust.
//         public const uint D3D12_STATE_ALL_SHADER_RESOURCE = 0x40 | 0x80; // NON_PIXEL | PIXEL
//         public const uint D3D12_STATE_UNORDERED_ACCESS    = 0x08;
//
//         /// <summary>
//         /// Mirror of the native <c>StreamlineProbe::FrameInputs</c> (sequential layout, must
//         /// match field-for-field). Matrices are Unity <see cref="Matrix4x4"/> (column-major);
//         /// the native side memcpy's them into row-major sl::float4x4, which applies the
//         /// column→row-vector transpose Streamline expects — so pass Unity matrices as-is.
//         /// </summary>
//         [StructLayout(LayoutKind.Sequential)]
//         public struct DlssgInputs
//         {
//             public IntPtr color;          // HUDLessColor (final, no UI)  — ID3D12Resource*
//             public IntPtr depth;          // depth / viewZ
//             public IntPtr motionVectors;  // screen-space motion vectors
//             public uint   colorW, colorH;
//             public uint   mvecDepthW, mvecDepthH;
//             public uint   colorState, depthState, mvecState;
//
//             public Matrix4x4 cameraViewToClip; // = viewToClip
//             public Matrix4x4 clipToCameraView; // = viewToClip.inverse
//             public Matrix4x4 clipToPrevClip;   // = prevWorldToClip * worldToClip.inverse
//             public Matrix4x4 prevClipToClip;   // = worldToClip    * prevWorldToClip.inverse
//
//             public float jitterX, jitterY;
//             public float mvecScaleX, mvecScaleY;
//             public float cameraPosX, cameraPosY, cameraPosZ;
//             public float cameraUpX, cameraUpY, cameraUpZ;
//             public float cameraRightX, cameraRightY, cameraRightZ;
//             public float cameraFwdX, cameraFwdY, cameraFwdZ;
//             public float cameraNear, cameraFar, cameraFOV, cameraAspect;
//             public int   depthInverted;
//             public int   cameraMotionIncluded;
//             public int   motionVectors3D;
//             public int   reset;
//         }
//
//         /// <summary>
//         /// Render-event-and-data function for pushing per-frame DLSS-G inputs. Issue via
//         /// <c>cmd.IssuePluginEventAndData(func, 0, ptrToDlssgInputs)</c> on the render thread
//         /// (see <see cref="NrdDlssgInputsPass"/>). The data pointer must stay valid until the
//         /// render thread runs the event — back it with a persistent ring buffer. Returns
//         /// <see cref="IntPtr.Zero"/> if the probe plugin is absent (cached after first miss).
//         /// </summary>
//         public static IntPtr GetFrameInputsEventFunc()
//         {
//             if (!_available) return IntPtr.Zero;
//             if (_frameInputsEventFunc != IntPtr.Zero) return _frameInputsEventFunc;
//             try
//             {
//                 _frameInputsEventFunc = NR_SL_GetFrameInputsEventFunc();
//             }
//             catch (DllNotFoundException)
//             {
//                 _available = false;
//                 return IntPtr.Zero;
//             }
//             return _frameInputsEventFunc;
//         }
//
//         private static CommandBuffer _cmd;
//         private static IntPtr        _eventFunc = IntPtr.Zero;
//         // Set false if the probe DLL is absent/disabled so we stop trying every frame.
//         private static bool          _available = true;
//
// #if UNITY_EDITOR
//         [UnityEditor.InitializeOnLoadMethod]
//         private static void InitializeEditor()
//         {
//             Initialize();
//
//             UnityEditor.EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
//             UnityEditor.EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
//             UnityEditor.EditorApplication.quitting             -= Shutdown;
//             UnityEditor.EditorApplication.quitting             += Shutdown;
//         }
//
//         private static void OnPlayModeStateChanged(UnityEditor.PlayModeStateChange state)
//         {
//             if (state == UnityEditor.PlayModeStateChange.EnteredEditMode)
//                 Initialize();
//         }
// #endif
//
//         [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
//         private static void Initialize()
//         {
//             _cmd ??= new CommandBuffer { name = "StreamlineReflexBegin" };
//
//             RenderPipelineManager.beginContextRendering -= OnBeginContextRendering;
//             RenderPipelineManager.beginContextRendering += OnBeginContextRendering;
//
// #if !UNITY_EDITOR
//             Application.quitting -= Shutdown;
//             Application.quitting += Shutdown;
// #endif
//             Debug.Log("StreamlineFrameDriver initialized and subscribed to RenderPipelineManager.beginContextRendering");
//         }
//
//         private static void OnBeginContextRendering(ScriptableRenderContext context, List<Camera> cameras)
//         {
//             if (_cmd == null || !_available) return;
//
//             // Skip editor asset-preview / thumbnail contexts (own isolated RenderPipeline.Render),
//             // matching NativeFrameTickDriver — issuing a render-thread event there crashes Unity.
//             if (IsPreviewOnlyContext(cameras)) return;
//
//             if (_eventFunc == IntPtr.Zero)
//             {
//                 try
//                 {
//                     _eventFunc = NR_SL_GetReflexBeginEventFunc();
//                 }
//                 catch (DllNotFoundException)
//                 {
//                     _available = false;   // probe plugin not deployed/enabled — go quiet
//                     Debug.LogWarning("StreamlineFrameDriver: StreamlineProbePlugin not found; DLSS-G frame driving disabled.");
//                     return;
//                 }
//                 if (_eventFunc == IntPtr.Zero) { _available = false; return; }
//             }
//
//             _cmd.Clear();
//             _cmd.IssuePluginEvent(_eventFunc, 0);
//             Graphics.ExecuteCommandBuffer(_cmd);
//         }
//
//         /// <summary>
//         /// Enable or disable DLSS-G frame generation at runtime. Safe to call any time
//         /// (e.g. from a UI toggle); the native side records the request and applies it on
//         /// the present thread at the next present. No-op if the probe plugin is absent.
//         /// </summary>
//         public static void SetFrameGeneration(bool enable)
//         {
//             if (!_available) return;
//             try
//             {
//                 NR_SL_SetFrameGeneration(enable ? 1 : 0);
//             }
//             catch (DllNotFoundException)
//             {
//                 _available = false;
//                 Debug.LogWarning("StreamlineFrameDriver: StreamlineProbePlugin not found; SetFrameGeneration ignored.");
//             }
//         }
//
//         /// <summary>True if DLSS-G frame generation is currently applied on the present thread.</summary>
//         public static bool IsFrameGenerationOn()
//         {
//             if (!_available) return false;
//             try
//             {
//                 return NR_SL_IsFrameGenerationOn() != 0;
//             }
//             catch (DllNotFoundException)
//             {
//                 _available = false;
//                 return false;
//             }
//         }
//
//         private static bool IsPreviewOnlyContext(List<Camera> cameras)
//         {
//             if (cameras == null || cameras.Count == 0) return false;
//             foreach (var cam in cameras)
//             {
//                 if (cam == null) continue;
//                 if (cam.cameraType != CameraType.Preview)
//                     return false;
//             }
//             return true;
//         }
//
//         private static void Shutdown()
//         {
//             RenderPipelineManager.beginContextRendering -= OnBeginContextRendering;
//             Application.quitting                        -= Shutdown;
//             _cmd?.Release();
//             _cmd = null;
//             _eventFunc = IntPtr.Zero;
//             Debug.Log("StreamlineFrameDriver shutdown and unsubscribed from RenderPipelineManager.beginContextRendering");
//         }
//     }
// }
