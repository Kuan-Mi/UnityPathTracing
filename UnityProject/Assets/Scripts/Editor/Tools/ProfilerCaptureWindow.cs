#if UNITY_EDITOR
using System.Linq;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

namespace PathTracing.Editor.Tools
{
    /// <summary>
    /// Small front-end for <see cref="ProfilerCaptureExporter"/>: load a .data capture,
    /// pick a frame and threads, and export an execution-order timeline (or the aggregated
    /// per-thread report) to a text file. All the heavy lifting lives in the exporter.
    /// </summary>
    public sealed class ProfilerCaptureWindow : EditorWindow
    {
        [SerializeField] private string loadedPath   = "";
        [SerializeField] private int    frame        = -1;       // -1 = use last loaded frame
        [SerializeField] private string threadFilter = "Main Thread, Render Thread";
        [SerializeField] private float  minMs        = 0f;

        [MenuItem("Tools/Profiler Capture/Window")]
        public static void Open() => GetWindow<ProfilerCaptureWindow>("Profiler Capture");

        private void OnGUI()
        {
            bool hasData = ProfilerCaptureExporter.TryGetRange(out int first, out int last);

            EditorGUILayout.LabelField("Capture", EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                EditorGUILayout.TextField("Loaded .data", string.IsNullOrEmpty(loadedPath)
                    ? (hasData ? "<currently loaded capture>" : "<none>")
                    : loadedPath);
                if (GUILayout.Button("Load…", GUILayout.Width(60)))
                    LoadData();
            }

            EditorGUILayout.LabelField("Frame range",
                hasData ? $"{first} .. {last}" : "no capture loaded");

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Timeline export", EditorStyles.boldLabel);

            using (new EditorGUILayout.HorizontalScope())
            {
                frame = EditorGUILayout.IntField(
                    new GUIContent("Frame", "Frame index to dump. Leave at -1 for the last frame."), frame);
                using (new EditorGUI.DisabledScope(!hasData))
                    if (GUILayout.Button("Last", GUILayout.Width(50)))
                        frame = last;
            }

            threadFilter = EditorGUILayout.TextField(
                new GUIContent("Threads", "Comma-separated name substrings. Empty = all threads."), threadFilter);
            minMs = EditorGUILayout.FloatField(
                new GUIContent("Min sample (ms)", "Omit samples shorter than this. 0 = keep all."), minMs);

            using (new EditorGUI.DisabledScope(!hasData))
            {
                if (GUILayout.Button("List threads in frame"))
                {
                    int f = ResolveFrame(first, last);
                    var names = ProfilerCaptureExporter.GetThreadNames(f);
                    Debug.Log($"[ProfilerCapture] Threads in frame {f}:\n  {string.Join("\n  ", names)}");
                }

                if (GUILayout.Button("Export Timeline (execution order)"))
                {
                    ProfilerCaptureExporter.ExportTimeline(
                        ResolveFrame(first, last), ParseFilters(threadFilter), minMs,
                        string.IsNullOrEmpty(loadedPath) ? null : loadedPath);
                }

                EditorGUILayout.Space();
                EditorGUILayout.LabelField("Aggregated report", EditorStyles.boldLabel);
                if (GUILayout.Button("Export Aggregated Report (all frames)"))
                    ProfilerCaptureExporter.ExportLoaded();
            }

            if (!hasData)
                EditorGUILayout.HelpBox(
                    "Load a .data file above, or open a capture in the Profiler window first.",
                    MessageType.Info);
        }

        private void LoadData()
        {
            string path = EditorUtility.OpenFilePanel("Open Profiler capture", DefaultDir(), "data");
            if (string.IsNullOrEmpty(path)) return;

            if (!ProfilerDriver.LoadProfile(path, false))
            {
                Debug.LogError($"[ProfilerCapture] Failed to load '{path}'.");
                return;
            }

            loadedPath = path;
            if (ProfilerCaptureExporter.TryGetRange(out _, out int last))
                frame = last;
        }

        private int ResolveFrame(int first, int last)
            => frame < 0 ? last : Mathf.Clamp(frame, first, last);

        private static string[] ParseFilters(string raw)
            => string.IsNullOrWhiteSpace(raw)
                ? new string[0]
                : raw.Split(',').Select(s => s.Trim()).Where(s => s.Length > 0).ToArray();

        private static string DefaultDir()
        {
            string dir = System.IO.Path.Combine(
                System.IO.Path.GetDirectoryName(Application.dataPath) ?? ".", "ProfilerCaptures");
            return System.IO.Directory.Exists(dir) ? dir : Application.dataPath;
        }
    }
}
#endif
