using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Profiling;

public class GPUProfiler : MonoBehaviour
{
    public List<string> recorderNames;
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

    private void OnEnable()
    {
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
        float currentTime = Time.unscaledTime;

        foreach (var item in _recorderMap)
        {
            var data = item.Value;
            if (!data.recorder.isValid) continue;

            // 获取当前帧毫秒数 (gpuElapsedNanoseconds 为 0 时不计入采样，避免干扰平均值)
            long ns = data.recorder.gpuElapsedNanoseconds;
            if (ns <= 0) continue;

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

    void OnGUI()
    {
        // 过滤出有数据的Recorder
        var activeData = _recorderMap.Where(r => r.Value.recorder.isValid && r.Value.history.Count > 0).ToList();
        if (activeData.Count == 0) return;

        GUI.backgroundColor = new Color(0, 0, 0, 0.6f);

        int h = Screen.height;
        var fontSize = h * 2 / 100;

        GUIStyle nameStyle = new GUIStyle(GUI.skin.label);
        nameStyle.fontSize = fontSize;
        nameStyle.alignment = TextAnchor.MiddleLeft;

        GUIStyle valueStyle = new GUIStyle(nameStyle);
        valueStyle.alignment = TextAnchor.MiddleRight;

        // 布局参数
        float startX = 20;
        float startY = 40;
        float nameWidth = 220;    // 第一列：名称
        float currentWidth = 120; // 第二列：当前值
        float averageWidth = 120; // 第三列：平均值
        float lineHeight = fontSize * 1.6f;

        float totalWidth = nameWidth + currentWidth + averageWidth + startX * 2;
        float totalHeight = (activeData.Count + 1) * lineHeight + 50; // +1 是为了表头

        GUI.Box(new Rect(10, 10, totalWidth, totalHeight), $"GPU Profiler ({averageWindowSeconds}s Avg)");

        // 绘制表头
        float headerY = startY;
        GUI.contentColor = Color.yellow;
        GUI.Label(new Rect(startX, headerY, nameWidth, lineHeight), "Pass Name", nameStyle);
        GUI.Label(new Rect(startX + nameWidth, headerY, currentWidth, lineHeight), "Current", valueStyle);
        GUI.Label(new Rect(startX + nameWidth + currentWidth, headerY, averageWidth, lineHeight), "Average", valueStyle);
        GUI.contentColor = Color.white;

        // 绘制数据行
        int i = 0;
        foreach (var item in activeData)
        {
            float y = headerY + lineHeight + (i * lineHeight);
            var data = item.Value;

            // 获取最新一帧的值
            float curMs = data.history.Last().valueMs;

            GUI.Label(new Rect(startX, y, nameWidth, lineHeight), item.Key, nameStyle);
            GUI.Label(new Rect(startX + nameWidth, y, currentWidth, lineHeight), $"{curMs:F3} ms", valueStyle);
            GUI.Label(new Rect(startX + nameWidth + currentWidth, y, averageWidth, lineHeight), $"{data.currentAverage:F3} ms", valueStyle);

            i++;
        }
    }
}