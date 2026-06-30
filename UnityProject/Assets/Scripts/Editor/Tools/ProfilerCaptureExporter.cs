#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.Profiling;
using UnityEditorInternal;
using UnityEngine;

namespace PathTracing.Editor.Tools
{
    /// <summary>
    /// Loads a Unity Profiler binary capture (.data) and writes a human-readable text
    /// report next to it ("&lt;name&gt;.report.txt"): a per-frame CPU/GPU summary plus a
    /// per-thread marker self-time breakdown aggregated over all frames. GPU work shows up
    /// as its own "GPU" thread group (this Unity version exposes no per-marker GPU column).
    /// The .data format itself is opaque binary; this drives the same
    /// <see cref="ProfilerDriver"/> APIs the Profiler window uses, so it stays correct
    /// without us parsing the stream by hand.
    /// </summary>
    public static class ProfilerCaptureExporter
    {
        private const string MenuRoot     = "Tools/Profiler Capture";
        private const string ReportSuffix = ".report.txt";

        // How much detail to emit.
        private const int    TopMarkersPerThread = 50;
        private const double ThreadSkipSelfMs    = 0.01; // skip near-idle threads
        private const double MarkerSkipSelfMs    = 0.001;

        // ---------------------------------------------------------------
        // Menu
        // ---------------------------------------------------------------

        [MenuItem(MenuRoot + "/Load .data And Export To Text…")]
        public static void LoadAndExport()
        {
            string startDir = DefaultCaptureDir();
            string dataPath = EditorUtility.OpenFilePanel("Open Profiler capture", startDir, "data");
            if (string.IsNullOrEmpty(dataPath)) return;

            if (!ProfilerDriver.LoadProfile(dataPath, false))
            {
                Debug.LogError($"[ProfilerCaptureExporter] Failed to load '{dataPath}'.");
                return;
            }

            string outPath = dataPath + ReportSuffix;
            ExportLoadedFrames(outPath, dataPath);
        }

        [MenuItem(MenuRoot + "/Export Currently Loaded Capture To Text")]
        public static void ExportLoaded()
        {
            if (ProfilerDriver.lastFrameIndex < 0)
            {
                Debug.LogWarning("[ProfilerCaptureExporter] No profiler data is loaded. " +
                                 "Capture or load a .data file first, or use 'Load .data And Export To Text…'.");
                return;
            }

            string outPath = Path.Combine(DefaultCaptureDir(),
                $"ProfilerReport_{DateTime.Now:yyyy-MM-dd_HH-mm-ss}{ReportSuffix}");
            ExportLoadedFrames(outPath, sourceLabel: "<currently loaded capture>");
        }

        // ---------------------------------------------------------------
        // Export
        // ---------------------------------------------------------------

        private static void ExportLoadedFrames(string outPath, string sourceLabel)
        {
            int first = ProfilerDriver.firstFrameIndex;
            int last  = ProfilerDriver.lastFrameIndex;
            if (first < 0 || last < first)
            {
                Debug.LogError("[ProfilerCaptureExporter] Loaded capture has no frames.");
                return;
            }

            var frames = new List<FrameSummary>();
            // threadKey -> markerName -> aggregate (self-time over all frames)
            var byThread     = new Dictionary<string, Dictionary<string, MarkerAgg>>();
            var threadSelf   = new Dictionary<string, double>();
            var threadFrames = new Dictionary<string, int>();

            int frameCount = last - first + 1;
            try
            {
                for (int f = first; f <= last; f++)
                {
                    EditorUtility.DisplayProgressBar("Exporting profiler capture",
                        $"Frame {f - first + 1}/{frameCount}", (float)(f - first) / Mathf.Max(1, frameCount));

                    AccumulateFrame(f, frames, byThread, threadSelf, threadFrames);
                }
            }
            finally
            {
                EditorUtility.ClearProgressBar();
            }

            string report = BuildReport(sourceLabel, first, last, frames,
                byThread, threadSelf, threadFrames);

            Directory.CreateDirectory(Path.GetDirectoryName(outPath) ?? ".");
            File.WriteAllText(outPath, report);
            AssetDatabase.Refresh();

            Debug.Log($"[ProfilerCaptureExporter] Wrote report for {frames.Count} frame(s):\n{outPath}");
            EditorUtility.OpenWithDefaultApp(outPath);
        }

        private static void AccumulateFrame(
            int frameIndex,
            List<FrameSummary> frames,
            Dictionary<string, Dictionary<string, MarkerAgg>> byThread,
            Dictionary<string, double> threadSelf,
            Dictionary<string, int> threadFrames)
        {
            var children = new List<int>();
            var stack    = new Stack<int>();
            bool gotSummary = false;

            // Thread indices are contiguous from 0 within a frame; stop at the first
            // invalid view (with a hard cap as a safety net).
            for (int t = 0; t < 1024; t++)
            {
                HierarchyFrameDataView view;
                try
                {
                    view = ProfilerDriver.GetHierarchyFrameDataView(
                        frameIndex, t, HierarchyFrameDataView.ViewModes.Default,
                        HierarchyFrameDataView.columnSelfTime, false);
                }
                catch
                {
                    break;
                }

                if (view == null || !view.valid)
                {
                    view?.Dispose();
                    if (t == 0) return; // no main thread -> nothing to read this frame
                    break;
                }

                using (view)
                {
                    string threadKey = string.IsNullOrEmpty(view.threadGroupName)
                        ? view.threadName
                        : $"{view.threadGroupName}/{view.threadName}";

                    if (t == 0)
                    {
                        frames.Add(new FrameSummary
                        {
                            Index = frameIndex,
                            CpuMs = view.frameTimeMs,
                            GpuMs = view.frameGpuTimeMs,
                        });
                        gotSummary = true;
                    }

                    if (!byThread.TryGetValue(threadKey, out var markers))
                    {
                        markers           = new Dictionary<string, MarkerAgg>();
                        byThread[threadKey] = markers;
                    }

                    double threadSelfThisFrame = 0.0;

                    // Iterative pre-order walk of the call tree.
                    stack.Clear();
                    stack.Push(view.GetRootItemID());
                    while (stack.Count > 0)
                    {
                        int id = stack.Pop();

                        float self    = view.GetItemColumnDataAsFloat(id, HierarchyFrameDataView.columnSelfTime);
                        int   calls    = (int)view.GetItemColumnDataAsFloat(id, HierarchyFrameDataView.columnCalls);
                        string name    = view.GetItemName(id);

                        if (self > 0f && !string.IsNullOrEmpty(name))
                        {
                            Accum(markers, name, self, calls);
                            threadSelfThisFrame += self;
                        }

                        view.GetItemChildren(id, children);
                        for (int i = 0; i < children.Count; i++)
                            stack.Push(children[i]);
                    }

                    threadSelf[threadKey]   = threadSelf.GetValueOrDefault(threadKey) + threadSelfThisFrame;
                    threadFrames[threadKey] = threadFrames.GetValueOrDefault(threadKey) + 1;
                }
            }

            if (!gotSummary)
                frames.Add(new FrameSummary { Index = frameIndex, CpuMs = 0f, GpuMs = 0f });
        }

        private static void Accum(Dictionary<string, MarkerAgg> map, string name, double self, int calls)
        {
            if (!map.TryGetValue(name, out var a)) a = new MarkerAgg();
            a.SelfMs += self;
            a.Calls  += calls;
            map[name] = a;
        }

        // ---------------------------------------------------------------
        // Report formatting
        // ---------------------------------------------------------------

        private static string BuildReport(
            string sourceLabel, int first, int last,
            List<FrameSummary> frames,
            Dictionary<string, Dictionary<string, MarkerAgg>> byThread,
            Dictionary<string, double> threadSelf,
            Dictionary<string, int> threadFrames)
        {
            var ci = CultureInfo.InvariantCulture;
            var sb = new StringBuilder(64 * 1024);
            int n  = frames.Count;

            sb.AppendLine("Unity Profiler Capture Report");
            sb.AppendLine("=============================");
            sb.AppendLine($"Source      : {sourceLabel}");
            sb.AppendLine($"Generated   : {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"Frames      : {n}  (profiler index {first}..{last})");
            sb.AppendLine();

            // ---- CPU / GPU frame-time summary ----
            if (n > 0)
            {
                var cpu = frames.Select(x => (double)x.CpuMs).Where(x => x > 0).ToList();
                var gpu = frames.Select(x => (double)x.GpuMs).Where(x => x > 0).ToList();
                sb.AppendLine("Frame time (ms)        min      avg      max");
                sb.AppendLine($"  CPU (main thread)  {Stat(cpu, ci)}");
                if (gpu.Count > 0)
                    sb.AppendLine($"  GPU                {Stat(gpu, ci)}");
                sb.AppendLine();
            }

            // ---- Per-frame table ----
            sb.AppendLine("Per-frame");
            sb.AppendLine("  frame        cpuMs     gpuMs");
            foreach (var fr in frames)
                sb.AppendLine($"  {fr.Index,6}  {fr.CpuMs,10:F3}  {(fr.GpuMs > 0 ? fr.GpuMs.ToString("F3", ci) : "-"),9}");
            sb.AppendLine();

            // ---- Per-thread marker breakdown (GPU work appears as a "GPU" thread) ----
            var threadsOrdered = byThread.Keys
                .OrderByDescending(k => threadSelf.GetValueOrDefault(k))
                .ToList();

            foreach (string threadKey in threadsOrdered)
            {
                int    tf       = Math.Max(1, threadFrames.GetValueOrDefault(threadKey));
                double avgThread = threadSelf.GetValueOrDefault(threadKey) / tf;
                if (avgThread < ThreadSkipSelfMs) continue;

                sb.AppendLine($"Thread: {threadKey}   (self ~{avgThread.ToString("F3", ci)} ms/frame over {tf} frame(s))");
                sb.AppendLine("    avgMs/frame    totalMs    calls   marker");

                var markers = byThread[threadKey]
                    .Where(kv => kv.Value.SelfMs / n > MarkerSkipSelfMs)
                    .OrderByDescending(kv => kv.Value.SelfMs)
                    .Take(TopMarkersPerThread);

                foreach (var kv in markers)
                    sb.AppendLine($"  {kv.Value.SelfMs / n,11:F4}  {kv.Value.SelfMs,9:F2}  {kv.Value.Calls,7}   {kv.Key}");
                sb.AppendLine();
            }

            return sb.ToString();
        }

        private static string Stat(List<double> v, CultureInfo ci)
        {
            if (v == null || v.Count == 0) return "    -        -        -";
            return $"{v.Min(),9:F3}{v.Average(),9:F3}{v.Max(),9:F3}";
        }

        // ---------------------------------------------------------------
        // Frame timeline (execution order)
        // ---------------------------------------------------------------

        /// <summary>Loaded frame range, or false when no capture is loaded.</summary>
        internal static bool TryGetRange(out int first, out int last)
        {
            first = ProfilerDriver.firstFrameIndex;
            last  = ProfilerDriver.lastFrameIndex;
            return first >= 0 && last >= first;
        }

        /// <summary>Thread names present in the given frame (in thread-index order).</summary>
        internal static List<string> GetThreadNames(int frame)
        {
            var names = new List<string>();
            for (int t = 0; t < 1024; t++)
            {
                RawFrameDataView view;
                try { view = ProfilerDriver.GetRawFrameDataView(frame, t); }
                catch { break; }

                if (view == null || !view.valid) { view?.Dispose(); break; }
                using (view)
                    names.Add(string.IsNullOrEmpty(view.threadGroupName)
                        ? view.threadName
                        : $"{view.threadGroupName}/{view.threadName}");
            }
            return names;
        }

        /// <summary>
        /// Dumps one frame's samples in execution (recording) order for threads whose name
        /// contains one of <paramref name="threadFilters"/> (empty = all threads), indented
        /// by call depth. Uses <see cref="RawFrameDataView"/>, whose sample array is stored
        /// pre-order exactly as recorded, so the output reflects the real chronological
        /// call sequence. <paramref name="dataPath"/> is only used to name the output file.
        /// </summary>
        internal static void ExportTimeline(int frame, string[] threadFilters, double minMs, string dataPath)
        {
            if (!TryGetRange(out int first, out int last))
            {
                Debug.LogError("[ProfilerCaptureExporter] Loaded capture has no frames.");
                return;
            }

            frame = Mathf.Clamp(frame, first, last);

            var sb = new StringBuilder(64 * 1024);
            sb.AppendLine("Unity Profiler Frame Timeline (execution order)");
            sb.AppendLine("===============================================");
            sb.AppendLine($"Source   : {dataPath ?? "<currently loaded capture>"}");
            sb.AppendLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"Frame    : {frame}  (loaded range {first}..{last})");
            sb.AppendLine($"Threads  : {(threadFilters.Length == 0 ? "<all>" : string.Join(", ", threadFilters))}");
            if (minMs > 0) sb.AppendLine($"Min dur  : {minMs} ms");
            sb.AppendLine();

            var available = new List<string>();
            int written   = 0;

            for (int t = 0; t < 1024; t++)
            {
                RawFrameDataView view;
                try { view = ProfilerDriver.GetRawFrameDataView(frame, t); }
                catch { break; }

                if (view == null || !view.valid)
                {
                    view?.Dispose();
                    break;
                }

                using (view)
                {
                    string full = string.IsNullOrEmpty(view.threadGroupName)
                        ? view.threadName
                        : $"{view.threadGroupName}/{view.threadName}";
                    available.Add(full);

                    if (!MatchesThread(full, threadFilters)) continue;

                    AppendThreadTimeline(sb, view, full, minMs);
                    written++;
                }
            }

            if (written == 0)
            {
                Debug.LogWarning("[ProfilerCaptureExporter] No threads matched the filter " +
                                 $"for frame {frame}. Available threads:\n  {string.Join("\n  ", available)}");
                return;
            }

            string outPath = dataPath != null
                ? $"{dataPath}.frame{frame}.timeline.txt"
                : Path.Combine(DefaultCaptureDir(),
                    $"ProfilerTimeline_frame{frame}_{DateTime.Now:yyyy-MM-dd_HH-mm-ss}.txt");

            Directory.CreateDirectory(Path.GetDirectoryName(outPath) ?? ".");
            File.WriteAllText(outPath, sb.ToString());
            AssetDatabase.Refresh();

            Debug.Log($"[ProfilerCaptureExporter] Wrote timeline for frame {frame}, {written} thread(s):\n{outPath}");
            EditorUtility.OpenWithDefaultApp(outPath);
        }

        private static bool MatchesThread(string threadName, string[] filters)
        {
            if (filters.Length == 0) return true;
            foreach (string s in filters)
                if (!string.IsNullOrEmpty(threadName) &&
                    threadName.IndexOf(s, StringComparison.OrdinalIgnoreCase) >= 0)
                    return true;
            return false;
        }

        /// <summary>Appends one thread's samples in recording order, indented by call depth.</summary>
        private static void AppendThreadTimeline(StringBuilder sb, RawFrameDataView view, string full, double minMs)
        {
            int    count      = view.sampleCount;
            double frameStart = view.frameStartTimeMs;

            sb.AppendLine($"=== Thread: {full} ===");
            sb.AppendLine($"  frameTime {view.frameTimeMs:F3} ms, {count} samples, maxDepth {view.maxDepth}");
            sb.AppendLine("    start(ms)    dur(ms)   marker");

            // Depth is reconstructed from the pre-order child counts: each open ancestor
            // tracks how many of its children are still to be visited.
            var remaining = new Stack<int>();
            for (int i = 0; i < count; i++)
            {
                while (remaining.Count > 0 && remaining.Peek() == 0) remaining.Pop();
                int depth = remaining.Count;
                if (remaining.Count > 0) remaining.Push(remaining.Pop() - 1);

                int childCount = view.GetSampleChildrenCount(i);
                if (childCount > 0) remaining.Push(childCount);

                float dur = view.GetSampleTimeMs(i);
                if (dur < minMs) continue;

                double start = view.GetSampleStartTimeMs(i) - frameStart;
                string name  = view.GetSampleName(i);
                sb.Append($"  {start,10:F3} {dur,10:F3}   ");
                sb.Append(' ', depth * 2);
                sb.AppendLine(name);
            }

            sb.AppendLine();
        }

        private static string DefaultCaptureDir()
        {
            string dir = Path.Combine(
                Path.GetDirectoryName(Application.dataPath) ?? ".", "ProfilerCaptures");
            return Directory.Exists(dir) ? dir : Application.dataPath;
        }

        // ---------------------------------------------------------------
        // Data
        // ---------------------------------------------------------------

        private struct FrameSummary
        {
            public int   Index;
            public float CpuMs;
            public float GpuMs;
        }

        private sealed class MarkerAgg
        {
            public double SelfMs;
            public long   Calls;
        }
    }
}
#endif
