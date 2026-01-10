using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Profiling;

public class Profiler : MonoBehaviour
{
    public List<string> recorderNames;

    Dictionary<string, Recorder> _recorders = new Dictionary<string, Recorder>();


    public void EnableRecorder()
    {
        foreach (var recorderName in recorderNames)
        {
            if (_recorders.ContainsKey(recorderName))
                continue;

            var recorder = Recorder.Get(recorderName);

            if (recorder != null)
            {
                recorder.enabled = true;
                _recorders[recorderName] = recorder;
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
        foreach (var recorder in _recorders.Values)
        {
            recorder.enabled = false;
        }

        _recorders.Clear();
    }


    void OnGUI()
    {
        GUI.backgroundColor = new Color(0, 0, 0, 0.5f);

        var validRecorders = _recorders.Where(r => r.Value.isValid && r.Value.gpuElapsedNanoseconds > 0).ToList();

        int h = Screen.height;
        var fontSize = h * 2 / 100;

        GUIStyle nameStyle = new GUIStyle(GUI.skin.label);
        nameStyle.fontSize = fontSize;
        nameStyle.alignment = TextAnchor.UpperLeft;

        GUIStyle valueStyle = new GUIStyle(nameStyle);
        valueStyle.alignment = TextAnchor.UpperRight; // 数值右对齐更美观

        float startX = 20;
        float startY = 30;
        float nameWidth = 250; // 固定第一列宽度
        float valueWidth = 150; // 固定第二列宽度
        float lineHeight = fontSize * 1.5f;


        float totalHeight = validRecorders.Count * lineHeight + startY;
        float totalWidth = nameWidth + valueWidth + startX * 2;

        GUI.Box(new Rect(10, 10, totalWidth, totalHeight), "GPU Profiler Timings");

        GUI.backgroundColor = Color.white;
        int i = 0;
        foreach (var recorder in validRecorders)
        {
            float y = startY + (i * lineHeight);

            // 绘制名称
            GUI.Label(new Rect(startX, y, nameWidth, lineHeight), recorder.Key, nameStyle);

            // 绘制数值（注意这里不再需要 ,-20 补空格了）
            string valStr = $"{recorder.Value.gpuElapsedNanoseconds * (1e-6f):F3} ms";
            GUI.Label(new Rect(startX + nameWidth, y, valueWidth, lineHeight), valStr, valueStyle);

            i++;
        }
    }
}