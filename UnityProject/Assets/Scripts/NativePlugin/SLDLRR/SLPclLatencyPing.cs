using System;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.Layouts;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;
using UnityEngine.InputSystem.Utilities;

namespace SLDLRR
{
    public static class SLPclLatencyPing
    {
        private const string PclStatsPingMessageName = "PC_Latency_Stats_Ping";
        private const uint QuitMessage = 0x8000 + 0x0D11;

        [StructLayout(LayoutKind.Sequential)]
        private struct PclPingState : IInputStateTypeInfo
        {
            public static FourCC Format => new FourCC('P', 'C', 'L', 'P');
            public FourCC format => Format;

            [InputControl(name = "ping", layout = "Button", bit = 0)]
            public uint buttons;
        }

        [InputControlLayout(displayName = "PCL Ping", stateType = typeof(PclPingState))]
        private sealed class PclPingDevice : InputDevice
        {
            public ButtonControl ping { get; private set; }

            protected override void FinishSetup()
            {
                base.FinishSetup();
                ping = GetChildControl<ButtonControl>("ping");
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct MSG
        {
            public IntPtr hwnd;
            public uint message;
            public UIntPtr wParam;
            public IntPtr lParam;
            public uint time;
            public int ptX;
            public int ptY;
        }

        public static bool LastFrameHadPing { get; private set; }
        public static uint LastFramePingCount { get; private set; }

        private static PclPingDevice _device;
        private static Thread _messageThread;
        private static volatile bool _messageThreadRunning;
        private static uint _messageThreadId;
        private static uint _pclStatsPingMessage;
        private static uint _processedPingCountThisUpdate;
        private static bool _layoutRegistered;

        internal static void Register()
        {
            EnsureDevice();
            StartMessageThread();
            InputSystem.onBeforeUpdate -= OnBeforeInputUpdate;
            InputSystem.onAfterUpdate -= OnAfterInputUpdate;
            InputSystem.onEvent -= OnInputEvent;
            InputSystem.onBeforeUpdate += OnBeforeInputUpdate;
            InputSystem.onAfterUpdate += OnAfterInputUpdate;
            InputSystem.onEvent += OnInputEvent;
        }

        internal static void Unregister()
        {
            InputSystem.onBeforeUpdate -= OnBeforeInputUpdate;
            InputSystem.onAfterUpdate -= OnAfterInputUpdate;
            InputSystem.onEvent -= OnInputEvent;
            StopMessageThread();
            if (_device != null && HasDevice(_device))
                InputSystem.RemoveDevice(_device);
            _device = null;
            _processedPingCountThisUpdate = 0;
        }

        internal static void ResetFrameState()
        {
            LastFramePingCount = 0;
            LastFrameHadPing = false;
        }

        private static void OnBeforeInputUpdate()
        {
#if UNITY_EDITOR
            if (!Application.isPlaying) return;
#endif
            if (InputState.currentUpdateType != InputUpdateType.Dynamic) return;
            _processedPingCountThisUpdate = 0;
        }

        private static void OnInputEvent(InputEventPtr eventPtr, InputDevice device)
        {
            PclPingDevice pingDevice = _device;
            if (pingDevice == null || device != pingDevice) return;
            if (InputState.currentUpdateType != InputUpdateType.Dynamic) return;
            if (pingDevice.ping.ReadValueFromEvent(eventPtr, out float value) && value > 0.5f)
                ++_processedPingCountThisUpdate;
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
                LastFramePingCount = _processedPingCountThisUpdate;
                LastFrameHadPing = LastFramePingCount != 0;
                if (LastFramePingCount != 0)
                {
                    SLNative.SL_MarkPclLatencyPing(token, LastFramePingCount);
                }
            }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
        }

        private static void EnsureDevice()
        {
            if (!_layoutRegistered)
            {
                InputSystem.RegisterLayout<PclPingDevice>();
                _layoutRegistered = true;
            }

            if (_device == null || !HasDevice(_device))
                _device = InputSystem.AddDevice<PclPingDevice>();
        }

        private static bool HasDevice(InputDevice device)
        {
            foreach (InputDevice existing in InputSystem.devices)
            {
                if (ReferenceEquals(existing, device))
                    return true;
            }
            return false;
        }

        private static void StartMessageThread()
        {
            if (_messageThreadRunning) return;
            _messageThreadRunning = true;
            _messageThread = new Thread(MessageThreadMain)
            {
                IsBackground = true,
                Name = "SL PCL Ping Message Thread",
            };
            _messageThread.Start();
        }

        private static void StopMessageThread()
        {
            _messageThreadRunning = false;
            uint threadId = _messageThreadId;
            if (threadId != 0)
                PostThreadMessageW(threadId, QuitMessage, UIntPtr.Zero, IntPtr.Zero);

            if (_messageThread != null && _messageThread.IsAlive)
                _messageThread.Join(500);
            _messageThread = null;
            _messageThreadId = 0;

            try { SLNative.SL_SetPclPingThreadId(0); }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
        }

        private static void MessageThreadMain()
        {
            uint threadId = GetCurrentThreadId();
            _pclStatsPingMessage = RegisterWindowMessageW(PclStatsPingMessageName);
            PeekMessageW(out _, IntPtr.Zero, 0, 0, 0);
            _messageThreadId = threadId;

            try { SLNative.SL_SetPclPingThreadId(threadId); }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }

            while (_messageThreadRunning)
            {
                int result = GetMessageW(out MSG msg, IntPtr.Zero, 0, 0);
                if (result <= 0) break;
                if (msg.message == QuitMessage) break;
                if (msg.message == _pclStatsPingMessage)
                    QueuePingInputEvent();
            }

            try { SLNative.SL_SetPclPingThreadId(0); }
            catch (DllNotFoundException) { SLNative.MarkUnavailable(); }
        }

        private static void QueuePingInputEvent()
        {
            PclPingDevice device = _device;
            if (device == null) return;

            try
            {
                InputSystem.QueueStateEvent(device, new PclPingState { buttons = 1u });
                InputSystem.QueueStateEvent(device, default(PclPingState));
            }
            catch (InvalidOperationException) { }
        }

        [DllImport("kernel32.dll")]
        private static extern uint GetCurrentThreadId();

        [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern uint RegisterWindowMessageW(string lpString);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern int GetMessageW(out MSG lpMsg, IntPtr hWnd, uint wMsgFilterMin, uint wMsgFilterMax);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern bool PeekMessageW(out MSG lpMsg, IntPtr hWnd, uint wMsgFilterMin, uint wMsgFilterMax, uint wRemoveMsg);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern bool PostThreadMessageW(uint idThread, uint msg, UIntPtr wParam, IntPtr lParam);
    }
}
