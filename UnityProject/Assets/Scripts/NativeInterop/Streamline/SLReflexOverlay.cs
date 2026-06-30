using UnityEngine;
using UnityEngine.InputSystem;
using System.Collections.Generic;

namespace PathTracing.NativeInterop.Streamline
{
    public sealed class SLReflexOverlay : MonoBehaviour
    {
        [SerializeField]
        private Key toggleKey = Key.F7;

        [SerializeField]
        private bool visible = true;

        [SerializeField]
        private bool showStatsReport = true;

        [SerializeField]
        private int reflexMode = (int)SLReflexRuntime.Mode.On;

        [SerializeField]
        private bool useReflexFpsCap;

        [SerializeField]
        private int reflexFpsCap = 60;

        [SerializeField]
        private int frameWindow = 3;

        private          Rect                        _windowRect = new Rect(24, 80, 360, 300);
        private          bool                        _windowInited;
        private          GUIStyle                    _mono;
        private          GUIStyle                    _header;
        private          GUIStyle                    _small;
        private          Vector2                     _scrollPos;
        private readonly List<SLReflexRuntime.Stats> _history = new List<SLReflexRuntime.Stats>();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Bootstrap()
        {
            if (FindFirstObjectByType<SLReflexOverlay>() != null) return;
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
                _windowRect   = new Rect(24, 80, 620, 680);
                _windowInited = true;
            }

            _windowRect = GUI.Window(0x5A11EF, _windowRect, DrawWindow, "Streamline Reflex / PCL");
        }

        private void DrawWindow(int id)
        {
            bool hasStats  = SLReflexRuntime.TryGetStats(out var stats);
            bool supported = hasStats && stats.lowLatencyAvailable;
            if (hasStats && stats.frameID != 0)
                RecordStats(stats);

            GUILayout.Label($"Reflex LowLatency Supported: {(supported ? "yes" : "no")}", _header);

            using (new GUILayout.HorizontalScope())
            {
                GUILayout.Label("Reflex Low Latency", GUILayout.Width(145));
                string[] modes    = { "Off", "On", "On + Boost" };
                int      nextMode = GUILayout.Toolbar(Mathf.Clamp(reflexMode, 0, 2), modes);
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
                    int    nextFps = Mathf.Clamp((int)GUILayout.HorizontalSlider(reflexFpsCap, 20, 240, GUILayout.Width(95)), 20, 240);
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
                using (new GUILayout.HorizontalScope())
                {
                    GUILayout.Label("Frame Window", GUILayout.Width(145));
                    int    nextWindow = Mathf.Clamp((int)GUILayout.HorizontalSlider(frameWindow, 1, 8, GUILayout.Width(120)), 1, 8);
                    string windowText = GUILayout.TextField(nextWindow.ToString(), GUILayout.Width(42));
                    if (int.TryParse(windowText, out int typedWindow))
                        nextWindow = Mathf.Clamp(typedWindow, 1, 8);
                    if (nextWindow != frameWindow)
                    {
                        frameWindow = nextWindow;
                        TrimHistory();
                    }
                }

                GUILayout.Space(4);
                _scrollPos = GUILayout.BeginScrollView(_scrollPos, GUILayout.ExpandHeight(true));
                if (_history.Count != 0)
                {
                    DrawTimelineWindow();
                    GUILayout.Space(6);
                    SLReflexRuntime.Stats latest = _history[_history.Count - 1];
                    GUILayout.Label(BuildStatsText(true, latest, GetTimelineOrigin(latest)), _mono);
                }
                else
                {
                    GUILayout.Label("Latency Report Unavailable", _mono);
                }

                GUILayout.EndScrollView();
            }

            GUI.DragWindow(new Rect(0, 0, _windowRect.width, 20));
        }

        private void RecordStats(SLReflexRuntime.Stats stats)
        {
            if (_history.Count != 0 && _history[_history.Count - 1].frameID == stats.frameID)
            {
                _history[_history.Count - 1] = stats;
                return;
            }

            _history.Add(stats);
            TrimHistory();
        }

        private void TrimHistory()
        {
            frameWindow = Mathf.Clamp(frameWindow, 1, 8);
            while (_history.Count > frameWindow)
                _history.RemoveAt(0);
        }

        private void ApplyReflexOptions()
        {
            uint capUs = 0;
            if (useReflexFpsCap)
                capUs = (uint)Mathf.RoundToInt(1000000.0f / Mathf.Max(1, reflexFpsCap));
            SLReflexRuntime.SetMode((SLReflexRuntime.Mode)Mathf.Clamp(reflexMode, 0, 2), capUs);
        }

        private static string BuildStatsText(bool hasStats, SLReflexRuntime.Stats stats, ulong origin)
        {
            if (!hasStats)
                return "Latency Report Unavailable";

            if (stats.frameID == 0)
                return "Latency Report Unavailable";

            return
                FormatStatsHeader() +
                FormatStatsRow("frameID", stats.frameID, origin, false) +
                FormatStatsRow("inputSampleTime", stats.inputSampleTime, origin, true) +
                FormatStatsRow("simStartTime", stats.simStartTime, origin, true) +
                FormatStatsRow("simEndTime", stats.simEndTime, origin, true) +
                FormatStatsRow("renderSubmitStartTime", stats.renderSubmitStartTime, origin, true) +
                FormatStatsRow("renderSubmitEndTime", stats.renderSubmitEndTime, origin, true) +
                FormatStatsRow("presentStartTime", stats.presentStartTime, origin, true) +
                FormatStatsRow("presentEndTime", stats.presentEndTime, origin, true) +
                FormatStatsRow("driverStartTime", stats.driverStartTime, origin, true) +
                FormatStatsRow("driverEndTime", stats.driverEndTime, origin, true) +
                FormatStatsRow("osRenderQueueStartTime", stats.osRenderQueueStartTime, origin, true) +
                FormatStatsRow("osRenderQueueEndTime", stats.osRenderQueueEndTime, origin, true) +
                FormatStatsRow("gpuRenderStartTime", stats.gpuRenderStartTime, origin, true) +
                FormatStatsRow("gpuRenderEndTime", stats.gpuRenderEndTime, origin, true) +
                FormatStatsRow("gpuActiveRenderTimeUs", stats.gpuActiveRenderTimeUs, origin, false) +
                FormatStatsRow("gpuFrameTimeUs", stats.gpuFrameTimeUs, origin, false);
        }

        private static string FormatStatsHeader()
        {
            return $"{"name",-28}{"timestamp",20}{"offset",12}\n";
        }

        private static string FormatStatsRow(string name, ulong value, ulong origin, bool showOffset)
        {
            string offset = showOffset && value != 0 ? Delta(value, origin).ToString() : "";
            return $"{name,-28}{value,20}{offset,12}\n";
        }

        private static string FormatStatsRow(string name, uint value, ulong origin, bool showOffset)
        {
            string offset = showOffset && value != 0 ? Delta(value, origin).ToString() : "";
            return $"{name,-28}{value,20}{offset,12}\n";
        }

        private void DrawTimeline(SLReflexRuntime.Stats stats)
        {
            Segment[] segments =
            {
                new Segment("sim", stats.simStartTime, stats.simEndTime, new Color(0.35f, 0.72f, 1.00f)),
                new Segment("render", stats.renderSubmitStartTime, stats.renderSubmitEndTime, new Color(0.48f, 0.90f, 0.48f)),
                new Segment("present", stats.presentStartTime, stats.presentEndTime, new Color(1.00f, 0.78f, 0.30f)),
                new Segment("driver", stats.driverStartTime, stats.driverEndTime, new Color(1.00f, 0.52f, 0.42f)),
                new Segment("os queue", stats.osRenderQueueStartTime, stats.osRenderQueueEndTime, new Color(0.74f, 0.55f, 1.00f)),
                new Segment("gpurender", stats.gpuRenderStartTime, stats.gpuRenderEndTime, new Color(0.42f, 1.00f, 0.86f)),
            };

            ulong origin = GetTimelineOrigin(stats, segments);
            ulong end    = MaxEnd(segments);
            ulong total  = Delta(end, origin);

            if (total == 0)
            {
                GUILayout.Label("Latency Report Unavailable", _mono);
                return;
            }

            GUILayout.Label($"Timeline: input sample -> GPU render end ({FormatUs(total)})", _header);

            foreach (var segment in segments)
                DrawStageRow(segment, origin, total);
        }

        private void DrawTimelineWindow()
        {
            List<FrameSegments> frames = new List<FrameSegments>(_history.Count);
            ulong               origin = ulong.MaxValue;
            ulong               end    = 0;

            foreach (var stats in _history)
            {
                var frame = new FrameSegments(stats.frameID, BuildSegments(stats));
                frames.Add(frame);

                ulong frameOrigin = GetTimelineOrigin(stats, frame.segments);
                ulong frameEnd    = MaxEnd(frame.segments);
                if (frameOrigin != 0 && frameOrigin < origin)
                    origin = frameOrigin;
                if (frameEnd > end)
                    end = frameEnd;
            }

            if (origin == ulong.MaxValue)
                origin = 0;

            ulong total = Delta(end, origin);
            if (total == 0)
            {
                GUILayout.Label("Latency Report Unavailable", _mono);
                return;
            }

            GUILayout.Label($"Timeline Window: {_history.Count} frame(s), {FormatUs(total)}", _header);

            if (frames.Count != 0)
            {
                for (int stageIndex = 0; stageIndex < frames[0].segments.Length; ++stageIndex)
                {
                    string stageName = frames[0].segments[stageIndex].name;
                    DrawMergedRow(stageName, frames, stageIndex, origin, total, stageName == "os queue");
                }
            }
        }

        private void DrawMergedRow(string label, List<FrameSegments> frames, int stageIndex, ulong originUs, ulong totalUs, bool markOverlaps)
        {
            Rect        row       = GUILayoutUtility.GetRect(580, 22);
            const float labelW    = 82f;
            Rect        labelRect = new Rect(row.x, row.y, labelW, row.height);
            Rect        barRect   = new Rect(row.x + labelW, row.y + 4, row.width - labelW - 4, row.height - 8);

            GUI.Label(labelRect, label, _small);
            DrawBarBackground(barRect);

            List<RowSegment> drawn = new List<RowSegment>();
            for (int frameIndex = 0; frameIndex < frames.Count; ++frameIndex)
            {
                var segments = frames[frameIndex].segments;
                int first    = stageIndex < 0 ? 0 : stageIndex;
                int last     = stageIndex < 0 ? segments.Length - 1 : stageIndex;

                for (int i = first; i <= last; ++i)
                {
                    Segment segment = segments[i];
                    if (!segment.IsValid) continue;

                    Color color = TintForFrame(segment.color, frameIndex, frames.Count);
                    DrawSegmentInRect(barRect, Delta(segment.startUs, originUs), segment.DurationUs, totalUs, color);
                    drawn.Add(new RowSegment(segment, frameIndex));
                }
            }

            if (markOverlaps)
                DrawOverlapMarks(barRect, drawn, originUs, totalUs);
        }

        private static Color TintForFrame(Color baseColor, int frameIndex, int frameCount)
        {
            if (frameCount <= 1)
                return baseColor;

            float t      = frameIndex / (float)(frameCount - 1);
            Color older  = Color.Lerp(baseColor, Color.black, 0.35f);
            Color newer  = Color.Lerp(baseColor, Color.white, 0.12f);
            Color result = Color.Lerp(older, newer, t);
            result.a = 0.72f;
            return result;
        }

        private static void DrawOverlapMarks(Rect barRect, List<RowSegment> segments, ulong originUs, ulong totalUs)
        {
            if (segments.Count < 2) return;

            for (int i = 0; i < segments.Count; ++i)
            {
                for (int j = i + 1; j < segments.Count; ++j)
                {
                    if (segments[i].frameIndex == segments[j].frameIndex)
                        continue;

                    Segment a     = segments[i].segment;
                    Segment b     = segments[j].segment;
                    ulong   start = a.startUs > b.startUs ? a.startUs : b.startUs;
                    ulong   end   = a.endUs < b.endUs ? a.endUs : b.endUs;
                    if (end <= start) continue;

                    DrawSegmentInRect(
                        new Rect(barRect.x, barRect.y, barRect.width, 3),
                        Delta(start, originUs),
                        Delta(end, start),
                        totalUs,
                        new Color(1f, 1f, 1f, 0.95f));
                }
            }
        }

        private static Segment[] BuildSegments(SLReflexRuntime.Stats stats)
        {
            return new[]
            {
                new Segment("sim", stats.simStartTime, stats.simEndTime, new Color(0.35f, 0.72f, 1.00f)),
                new Segment("render", stats.renderSubmitStartTime, stats.renderSubmitEndTime, new Color(0.48f, 0.90f, 0.48f)),
                new Segment("present", stats.presentStartTime, stats.presentEndTime, new Color(1.00f, 0.78f, 0.30f)),
                new Segment("driver", stats.driverStartTime, stats.driverEndTime, new Color(1.00f, 0.52f, 0.42f)),
                new Segment("os queue", stats.osRenderQueueStartTime, stats.osRenderQueueEndTime, new Color(0.74f, 0.55f, 1.00f)),
                new Segment("gpurender", stats.gpuRenderStartTime, stats.gpuRenderEndTime, new Color(0.42f, 1.00f, 0.86f)),
            };
        }

        private static ulong GetTimelineOrigin(SLReflexRuntime.Stats stats)
        {
            Segment[] segments =
            {
                new Segment("sim", stats.simStartTime, stats.simEndTime, Color.white),
                new Segment("render", stats.renderSubmitStartTime, stats.renderSubmitEndTime, Color.white),
                new Segment("present", stats.presentStartTime, stats.presentEndTime, Color.white),
                new Segment("driver", stats.driverStartTime, stats.driverEndTime, Color.white),
                new Segment("os queue", stats.osRenderQueueStartTime, stats.osRenderQueueEndTime, Color.white),
                new Segment("gpurender", stats.gpuRenderStartTime, stats.gpuRenderEndTime, Color.white),
            };

            return GetTimelineOrigin(stats, segments);
        }

        private static ulong GetTimelineOrigin(SLReflexRuntime.Stats stats, Segment[] segments)
        {
            return stats.inputSampleTime != 0 ? stats.inputSampleTime : MinStart(segments);
        }

        private void DrawStageRow(Segment segment, ulong originUs, ulong totalUs)
        {
            Rect        row       = GUILayoutUtility.GetRect(520, 18);
            const float labelW    = 118f;
            const float valueW    = 72f;
            Rect        labelRect = new Rect(row.x, row.y, labelW, row.height);
            Rect        barRect   = new Rect(row.x + labelW, row.y + 3, row.width - labelW - valueW - 8, row.height - 6);
            Rect        valueRect = new Rect(row.x + row.width - valueW, row.y, valueW, row.height);

            GUI.Label(labelRect, segment.name, _small);
            DrawBarBackground(barRect);
            if (segment.IsValid)
            {
                DrawSegmentInRect(barRect, Delta(segment.startUs, originUs), segment.DurationUs, totalUs, segment.color);
                GUI.Label(valueRect, FormatUs(segment.DurationUs), _mono);
            }
            else
            {
                GUI.Label(valueRect, "N/A", _mono);
            }
        }

        private static void DrawBarBackground(Rect rect)
        {
            var prev = GUI.color;
            GUI.color = new Color(0.08f, 0.08f, 0.08f, 0.85f);
            GUI.DrawTexture(rect, Texture2D.whiteTexture);
            GUI.color = prev;
        }

        private static void DrawSegmentInRect(Rect rect, ulong startUs, ulong durationUs, ulong totalUs, Color color)
        {
            if (durationUs == 0 || totalUs == 0) return;

            float x    = rect.x + rect.width * ((float)startUs / totalUs);
            float w    = Mathf.Max(1f, rect.width * ((float)durationUs / totalUs));
            var   prev = GUI.color;
            GUI.color = color;
            GUI.DrawTexture(new Rect(x, rect.y, Mathf.Min(w, rect.xMax - x), rect.height), Texture2D.whiteTexture);
            GUI.color = prev;
        }

        private static string FormatUs(ulong us)
        {
            return us >= 1000 ? $"{us / 1000.0:F2} ms" : $"{us} us";
        }

        private static ulong Delta(ulong end, ulong start)
        {
            return end >= start ? end - start : 0;
        }

        private static ulong MinStart(Segment[] segments)
        {
            ulong min = ulong.MaxValue;
            foreach (var segment in segments)
            {
                if (segment.IsValid && segment.startUs < min)
                    min = segment.startUs;
            }

            return min == ulong.MaxValue ? 0 : min;
        }

        private static ulong MaxEnd(Segment[] segments)
        {
            ulong max = 0;
            foreach (var segment in segments)
            {
                if (segment.IsValid && segment.endUs > max)
                    max = segment.endUs;
            }

            return max;
        }

        private void InitStyles()
        {
            _mono ??= new GUIStyle(GUI.skin.label)
            {
                font     = Font.CreateDynamicFontFromOSFont("Consolas", 12),
                fontSize = 12,
                normal   = { textColor = Color.white },
                wordWrap = false
            };

            _header ??= new GUIStyle(GUI.skin.label)
            {
                fontStyle = FontStyle.Bold,
                normal    = { textColor = Color.white }
            };

            _small ??= new GUIStyle(GUI.skin.label)
            {
                fontSize = 11,
                normal   = { textColor = Color.white }
            };
        }

        private readonly struct Segment
        {
            public readonly string name;
            public readonly ulong  startUs;
            public readonly ulong  endUs;
            public readonly Color  color;
            public bool IsValid => startUs != 0 && endUs > startUs;
            public ulong DurationUs => Delta(endUs, startUs);

            public Segment(string name, ulong startUs, ulong endUs, Color color)
            {
                this.name    = name;
                this.startUs = startUs;
                this.endUs   = endUs;
                this.color   = color;
            }
        }

        private readonly struct FrameSegments
        {
            public readonly ulong     frameID;
            public readonly Segment[] segments;

            public FrameSegments(ulong frameID, Segment[] segments)
            {
                this.frameID  = frameID;
                this.segments = segments;
            }
        }

        private readonly struct RowSegment
        {
            public readonly Segment segment;
            public readonly int     frameIndex;

            public RowSegment(Segment segment, int frameIndex)
            {
                this.segment    = segment;
                this.frameIndex = frameIndex;
            }
        }

        private readonly struct GUIEnabledScope : System.IDisposable
        {
            private readonly bool _previous;

            public GUIEnabledScope(bool enabled)
            {
                _previous   = GUI.enabled;
                GUI.enabled = enabled;
            }

            public void Dispose()
            {
                GUI.enabled = _previous;
            }
        }
    }
}