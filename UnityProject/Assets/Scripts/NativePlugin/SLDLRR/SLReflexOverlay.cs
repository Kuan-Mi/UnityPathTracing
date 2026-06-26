using UnityEngine;
using UnityEngine.InputSystem;

namespace SLDLRR
{
    public sealed class SLReflexOverlay : MonoBehaviour
    {
        [SerializeField] private Key toggleKey = Key.F7;
        [SerializeField] private bool visible = true;
        [SerializeField] private bool showStatsReport = true;
        [SerializeField] private int reflexMode = (int)SLReflexRuntime.Mode.On;
        [SerializeField] private bool useReflexFpsCap;
        [SerializeField] private int reflexFpsCap = 60;

        private Rect _windowRect = new Rect(24, 80, 360, 300);
        private bool _windowInited;
        private GUIStyle _mono;
        private GUIStyle _header;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Bootstrap()
        {
            if (FindObjectOfType<SLReflexOverlay>() != null) return;
            var go = new GameObject("SL Reflex Overlay") { hideFlags = HideFlags.HideAndDontSave };
            DontDestroyOnLoad(go);
            go.AddComponent<SLReflexOverlay>();
        }

        private void OnEnable()
        {
            int mode = SLReflexRuntime.GetMode();
            if (mode >= 0 && mode <= 2)
                reflexMode = mode;
        }

        private void Update()
        {
            if (Keyboard.current?[toggleKey].wasPressedThisFrame == true)
                visible = !visible;
        }

        private void OnGUI()
        {
            InitStyles();

            float buttonW = 170f;
            float buttonH = 28f;
            if (GUI.Button(new Rect(16, 44, buttonW, buttonH), $"Reflex/PCL [{toggleKey}]"))
                visible = !visible;

            if (!visible) return;

            if (!_windowInited)
            {
                _windowRect = new Rect(24, 80, 370, 310);
                _windowInited = true;
            }

            _windowRect = GUI.Window(0x5A11EF, _windowRect, DrawWindow, "Streamline Reflex / PCL");
        }

        private void DrawWindow(int id)
        {
            bool hasStats = SLReflexRuntime.TryGetStats(out var stats);
            bool supported = hasStats && stats.lowLatencyAvailable;

            GUILayout.Label($"Reflex LowLatency Supported: {(supported ? "yes" : "no")}", _header);

            using (new GUILayout.HorizontalScope())
            {
                GUILayout.Label("Reflex Low Latency", GUILayout.Width(145));
                string[] modes = { "Off", "On", "On + Boost" };
                int nextMode = GUILayout.Toolbar(Mathf.Clamp(reflexMode, 0, 2), modes);
                if (nextMode != reflexMode)
                {
                    reflexMode = nextMode;
                    ApplyReflexOptions();
                }
            }

            using (new GUILayout.HorizontalScope())
            {
                bool nextCap = GUILayout.Toggle(useReflexFpsCap, "Reflex FPS Capping", GUILayout.Width(170));
                if (nextCap != useReflexFpsCap)
                {
                    useReflexFpsCap = nextCap;
                    ApplyReflexOptions();
                }

                using (new GUIEnabledScope(useReflexFpsCap))
                {
                    int nextFps = Mathf.Clamp((int)GUILayout.HorizontalSlider(reflexFpsCap, 20, 240, GUILayout.Width(95)), 20, 240);
                    string fpsText = GUILayout.TextField(nextFps.ToString(), GUILayout.Width(42));
                    if (int.TryParse(fpsText, out int typedFps))
                        nextFps = Mathf.Clamp(typedFps, 20, 240);
                    if (nextFps != reflexFpsCap)
                    {
                        reflexFpsCap = nextFps;
                        ApplyReflexOptions();
                    }
                }
            }

            showStatsReport = GUILayout.Toggle(showStatsReport, "Show Stats Report");
            if (showStatsReport)
            {
                GUILayout.Space(4);
                GUILayout.Label(BuildStatsText(hasStats, stats), _mono);
            }

            GUI.DragWindow(new Rect(0, 0, _windowRect.width, 20));
        }

        private void ApplyReflexOptions()
        {
            uint capUs = 0;
            if (useReflexFpsCap)
                capUs = (uint)Mathf.RoundToInt(1000000.0f / Mathf.Max(1, reflexFpsCap));
            SLReflexRuntime.SetMode((SLReflexRuntime.Mode)Mathf.Clamp(reflexMode, 0, 2), capUs);
        }

        private static string BuildStatsText(bool hasStats, SLReflexRuntime.Stats stats)
        {
            if (!hasStats)
                return "Latency Report Unavailable";

            if (stats.frameID == 0)
                return "Latency Report Unavailable";

            return
                $"frameID: {stats.frameID}\n" +
                $"totalGameToRenderLatencyUs: {stats.totalGameToRenderLatencyUs}\n" +
                $"simDeltaUs: {stats.simDeltaUs}\n" +
                $"renderDeltaUs: {stats.renderDeltaUs}\n" +
                $"presentDeltaUs: {stats.presentDeltaUs}\n" +
                $"driverDeltaUs: {stats.driverDeltaUs}\n" +
                $"osRenderQueueDeltaUs: {stats.osRenderQueueDeltaUs}\n" +
                $"gpuRenderDeltaUs: {stats.gpuRenderDeltaUs}\n" +
                $"gpuActiveRenderTimeUs: {stats.gpuActiveRenderTimeUs}\n" +
                $"gpuFrameTimeUs: {stats.gpuFrameTimeUs}";
        }

        private void InitStyles()
        {
            _mono ??= new GUIStyle(GUI.skin.label)
            {
                font = Font.CreateDynamicFontFromOSFont("Consolas", 12),
                fontSize = 12,
                normal = { textColor = Color.white },
                wordWrap = false
            };

            _header ??= new GUIStyle(GUI.skin.label)
            {
                fontStyle = FontStyle.Bold,
                normal = { textColor = Color.white }
            };
        }

        private readonly struct GUIEnabledScope : System.IDisposable
        {
            private readonly bool _previous;

            public GUIEnabledScope(bool enabled)
            {
                _previous = GUI.enabled;
                GUI.enabled = enabled;
            }

            public void Dispose()
            {
                GUI.enabled = _previous;
            }
        }
    }
}
