using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Profiling;

public class GPUProfiler : MonoBehaviour
{
    public List<string> recorderNames;

    [ContextMenu("Populate All Render Pass Markers")]
    void PopulateAllMarkers()
    {
        recorderNames = new List<string>(RenderPassMarkers.AllMarkerNames);
    }

    [Tooltip("平均值统计的时间窗口（秒）")]
    public int averageWindowSeconds = 1;

    // 内部类，用于封装Recorder及其采样数据
    private class RecorderData
    {
        public Recorder recorder;
        public Queue<Sample> history = new Queue<Sample>();
        public float currentAverage = 0f;

        public struct Sample
        {
            public float time;
            public float valueMs;
        }
    }

    Dictionary<string, RecorderData> _recorderMap = new Dictionary<string, RecorderData>();

    public void EnableRecorder()
    {
        foreach (var recorderName in recorderNames)
        {
            if (_recorderMap.ContainsKey(recorderName))
                continue;

            var recorder = Recorder.Get(recorderName);
            if (recorder != null)
            {
                recorder.enabled = true;
                _recorderMap[recorderName] = new RecorderData { recorder = recorder };
            }
            else
            {
                Debug.LogError($"Recorder not found: {recorderName}");
            }
        }
    }

    private async void OnEnable()
    {
        await Awaitable.WaitForSecondsAsync(3.0f);

        EnableRecorder();
    }

    private void OnDisable()
    {
        foreach (var data in _recorderMap.Values)
        {
            data.recorder.enabled = false;
        }

        _recorderMap.Clear();
    }

    private void Update()
    {
        if (Input.GetKeyDown(toggleKey))
            _visible = !_visible;
        
        float currentTime = Time.unscaledTime;

        foreach (var item in _recorderMap)
        {
            var data = item.Value;
            if (!data.recorder.isValid)
            {
                data.history.Clear();
                data.currentAverage = 0f;
                continue;
            }

            // 获取当前帧毫秒数 (gpuElapsedNanoseconds 为 0 时不计入采样，避免干扰平均值)
            long ns = data.recorder.gpuElapsedNanoseconds;
            if (ns <= 0)
            {
                data.history.Clear();
                data.currentAverage = 0f;
                continue;
            }

            float currentMs = ns * 1e-6f;

            // 添加新样本
            data.history.Enqueue(new RecorderData.Sample { time = currentTime, valueMs = currentMs });

            // 移除窗口外的旧样本
            while (data.history.Count > 0 && data.history.Peek().time < currentTime - averageWindowSeconds)
            {
                data.history.Dequeue();
            }

            // 计算平均值
            if (data.history.Count > 0)
            {
                float sum = 0;
                foreach (var s in data.history) sum += s.valueMs;
                data.currentAverage = sum / data.history.Count;
            }
        }
    }

    [Tooltip("切换面板的按键（默认 F2）")]
    public KeyCode toggleKey = KeyCode.F2;

    private bool _visible;
    private Rect _windowRect;
    private bool _windowRectInited;
    private Vector2 _scrollPos;
 

    void OnGUI()
    {
        int h = Screen.height;
        var fontSize = h * 2 / 120;

        GUIStyle btnStyle = new GUIStyle(GUI.skin.button);
        btnStyle.fontSize = fontSize;

        if (GUI.Button(new Rect(10, 10, fontSize * 10, fontSize * 1.8f), $"GPU Profiler [{toggleKey}]", btnStyle))
            _visible = !_visible;

        if (!_visible) return;

        if (!_windowRectInited)
        {
            _windowRect       = new Rect(10, (int)(fontSize * 2.2f) + 10, fontSize * 28, h - (int)(fontSize * 2.2f) - 30);
            _windowRectInited = true;
        }

        _windowRect = GUI.Window(0xBEEF + 1, _windowRect, DrawWindow, $"GPU Profiler  ({averageWindowSeconds}s Avg)  [{toggleKey}]");
    }

    private void DrawWindow(int id)
    {
        var activeData = _recorderMap.Where(r => r.Value.recorder.isValid && r.Value.history.Count > 0).ToList();

        int h = Screen.height;
        var fontSize = h * 2 / 120;
        float lineHeight = fontSize * 1.6f;

        GUIStyle nameStyle = new GUIStyle(GUI.skin.label) { fontSize = fontSize, alignment = TextAnchor.MiddleLeft };
        GUIStyle valueStyle = new GUIStyle(nameStyle) { alignment = TextAnchor.MiddleRight };
        GUIStyle headerStyle = new GUIStyle(nameStyle) { fontStyle = FontStyle.Bold, normal = { textColor = Color.yellow } };
        GUIStyle headerValueStyle = new GUIStyle(headerStyle) { alignment = TextAnchor.MiddleRight };

        float v = fontSize / 12f;
        float nameWidth  = v * 180;
        float curWidth   = v * 80;
        float avgWidth   = v * 80;
        float rowWidth   = nameWidth + curWidth + avgWidth;

        _scrollPos = GUILayout.BeginScrollView(_scrollPos);

        // 表头
        GUILayout.BeginHorizontal();
        GUILayout.Label("Pass Name", headerStyle,      GUILayout.Width(nameWidth));
        GUILayout.Label("Current",   headerValueStyle, GUILayout.Width(curWidth));
        GUILayout.Label("Average",   headerValueStyle, GUILayout.Width(avgWidth));
        GUILayout.EndHorizontal();

        if (activeData.Count == 0)
        {
            GUILayout.Label("No active recorders.", nameStyle);
        }
        else
        {
            foreach (var item in activeData)
            {
                float curMs = item.Value.history.Last().valueMs;
                float avg   = item.Value.currentAverage;

                GUILayout.BeginHorizontal();
                GUILayout.Label(item.Key,               nameStyle,  GUILayout.Width(nameWidth));
                GUILayout.Label($"{curMs:F3} ms",       valueStyle, GUILayout.Width(curWidth));
                GUILayout.Label($"{avg:F3} ms",         valueStyle, GUILayout.Width(avgWidth));
                GUILayout.EndHorizontal();
            }

            // Total 行
            float totalCur = activeData.Sum(d => d.Value.history.Last().valueMs);
            float totalAvg = activeData.Sum(d => d.Value.currentAverage);
            GUILayout.BeginHorizontal();
            GUILayout.Label("Total",               headerStyle,      GUILayout.Width(nameWidth));
            GUILayout.Label($"{totalCur:F3} ms",   headerValueStyle, GUILayout.Width(curWidth));
            GUILayout.Label($"{totalAvg:F3} ms",   headerValueStyle, GUILayout.Width(avgWidth));
            GUILayout.EndHorizontal();
        }

        GUILayout.EndScrollView();

        GUI.DragWindow(new Rect(0, 0, _windowRect.width, lineHeight));
    }
}