using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// 运行时 IMGUI 面板，通过反射显示并修改 PathTracingSetting 的所有字段。
    /// 将此组件挂在场景中任意 GameObject 上，运行时按 F1（可改）切换显示。
    /// </summary>
    public class PathTracingSettingGUI : MonoBehaviour
    {
        [Tooltip("切换面板的按键（默认 F1）")]
        public KeyCode toggleKey = KeyCode.F1;

        private bool              _visible;
        private Vector2           _scrollPos;
        private readonly Dictionary<string, bool>   _foldouts  = new();
        private readonly Dictionary<string, string> _textCache = new();

        private RtxdiFeature _feature;
        private Rect         _windowRect;
        private bool         _windowRectInited;

        // ─── GUI Styles (lazy init inside OnGUI) ────────────────────────
        private GUIStyle _boldLabel;
        private GUIStyle _label;
        private GUIStyle _labelSelected;
        private GUIStyle _tf;
        private GUIStyle _btn;
        private GUIStyle _box;
        private bool     _stylesReady;

        private void InitStyles()
        {
            if (_stylesReady) return;
            _boldLabel = new GUIStyle(GUI.skin.label)
            {
                fontStyle = FontStyle.Bold,
                fontSize  = 12,
                normal    = { textColor = new Color(1f, 0.85f, 0.3f) }
            };
            _label = new GUIStyle(GUI.skin.label)  { fontSize = 11, normal = { textColor = Color.white } };
            _labelSelected = new GUIStyle(GUI.skin.label) { fontSize = 11, fontStyle = FontStyle.Bold, normal = { textColor = new Color(0.4f, 1f, 0.4f) } };
            _tf    = new GUIStyle(GUI.skin.textField) { fontSize = 11 };
            _btn   = new GUIStyle(GUI.skin.button)    { fontSize = 11 };
            _box   = new GUIStyle(GUI.skin.box);
            _stylesReady = true;
        }

        // ─── Lifecycle ──────────────────────────────────────────────────

        private void Update()
        {
            if (Input.GetKeyDown(toggleKey))
                _visible = !_visible;
        }

        private void OnGUI()
        {
            InitStyles();

            // 右上角按钮，避免与左上角的 GPU Profiler 按钮重叠
            int h = Screen.height;
            int fontSize = h * 2 / 100;
            float btnW = fontSize * 14;
            float btnH = fontSize * 1.8f;
            var btnStyle = new GUIStyle(GUI.skin.button) { fontSize = fontSize };
            if (GUI.Button(new Rect(Screen.width - btnW - 10, 10, btnW, btnH),
                           $"PT Settings [{toggleKey}]", btnStyle))
                _visible = !_visible;

            if (!_visible) return;

            InitStyles();

            if (!_windowRectInited)
            {
                _windowRect       = new Rect(Screen.width - 500, (int)btnH + 20, 490, Screen.height - (int)btnH - 40);
                _windowRectInited = true;
            }

            if (_feature == null)
                _feature = FindFeature();

            if (_feature == null)
            {
                GUI.Box(new Rect(Screen.width - 360, (int)btnH + 20, 350, 28), "PathTracingSettingGUI: RtxdiFeature not found");
                return;
            }

            _windowRect = GUI.Window(0xBEEF, _windowRect, DrawWindow, "Path Tracing Settings  [" + toggleKey + "]");
        }

        private void DrawWindow(int id)
        {
            GUILayout.Space(2);
            _scrollPos = GUILayout.BeginScrollView(_scrollPos, GUILayout.ExpandHeight(true));

            DrawObjectFieldsInPlace(_feature.setting, typeof(PathTracingSetting), "root");

            GUILayout.EndScrollView();
            GUI.DragWindow(new Rect(0, 0, _windowRect.width, 18));
        }

        // ─── Core reflection draw ────────────────────────────────────────

        /// <summary>
        /// 遍历 type 的所有公有实例字段并在 owner 上原地修改。
        /// owner 可以是类实例，也可以是装箱的 struct（反射可原地写入）。
        /// </summary>
        private void DrawObjectFieldsInPlace(object owner, Type type, string path)
        {
            if (owner == null) return;

            FieldInfo[] fields = type.GetFields(BindingFlags.Public | BindingFlags.Instance);

            bool groupOpen = true; // 初始未遇到任何 FoldoutHeader 时默认全显

            foreach (var field in fields)
            {
                // 跳过自动属性后备字段（如 <exposure>k__BackingField）
                if (field.Name.Contains('<')) continue;

                // FoldoutHeader 触发新折叠分组
                var foldoutAttr = field.GetCustomAttribute<FoldoutHeaderAttribute>();
                if (foldoutAttr != null)
                {
                    string gKey = path + "_g_" + foldoutAttr.Name;
                    if (!_foldouts.ContainsKey(gKey)) _foldouts[gKey] = false;

                    bool open    = _foldouts[gKey];
                    bool newOpen = GUILayout.Toggle(open, (open ? "▼ " : "▶ ") + foldoutAttr.Name, _boldLabel);
                    if (newOpen != open) _foldouts[gKey] = newOpen;
                    groupOpen = newOpen;
                }

                if (!groupOpen) continue;

                DrawFieldInPlace(owner, field, path + "." + field.Name);
            }
        }

        /// <summary>绘制单个字段行，并将修改写回 owner。</summary>
        private void DrawFieldInPlace(object owner, FieldInfo field, string path)
        {
            object value = field.GetValue(owner);
            Type   ft    = field.FieldType;

            if (IsSimple(ft))
            {
                GUILayout.BeginHorizontal();
                GUILayout.Label(field.Name, _label, GUILayout.Width(215));
                object newValue = DrawSimpleValue(ft, value, field, path);
                GUILayout.EndHorizontal();

                if (!Equals(newValue, value))
                    field.SetValue(owner, newValue);
            }
            else if (ft.IsEnum)
            {
                string[] names = Enum.GetNames(ft);
                Array    vals  = Enum.GetValues(ft);
                int      idx   = Array.IndexOf(names, value.ToString());
                if (idx < 0) idx = 0;

                string dropKey = path + "_drop";
                if (!_foldouts.ContainsKey(dropKey)) _foldouts[dropKey] = false;
                bool isOpen = _foldouts[dropKey];

                GUILayout.BeginHorizontal();
                GUILayout.Label(field.Name, _label, GUILayout.Width(215));
                if (GUILayout.Button(names[idx] + "  ▼", _tf, GUILayout.Width(185)))
                    _foldouts[dropKey] = !isOpen;
                GUILayout.EndHorizontal();

                if (isOpen)
                {
                    GUILayout.BeginVertical(_box);
                    for (int i = 0; i < names.Length; i++)
                    {
                        bool sel = i == idx;
                        if (GUILayout.Button((sel ? "● " : "  ") + names[i], sel ? _labelSelected : _label))
                        {
                            field.SetValue(owner, vals.GetValue(i));
                            _foldouts[dropKey] = false;
                        }
                    }
                    GUILayout.EndVertical();
                }
            }
            else if (ft.IsValueType && !ft.IsPrimitive && !ft.IsEnum)
            {
                // 嵌套 struct：折叠展开
                string fkey = path + "_fold";
                if (!_foldouts.ContainsKey(fkey)) _foldouts[fkey] = false;

                bool open    = _foldouts[fkey];
                bool newOpen = GUILayout.Toggle(open, (open ? "▼ " : "▷ ") + field.Name, _boldLabel);
                if (newOpen != open) _foldouts[fkey] = newOpen;

                if (newOpen)
                {
                    GUILayout.BeginVertical(_box);
                    DrawObjectFieldsInPlace(value, ft, path);
                    GUILayout.EndVertical();
                    // 将修改后的装箱 struct 写回父对象
                    field.SetValue(owner, value);
                }
            }
            else
            {
                // 其他类型：只读显示
                GUILayout.BeginHorizontal();
                GUILayout.Label(field.Name, _label, GUILayout.Width(215));
                GUILayout.Label(value?.ToString() ?? "null", _label);
                GUILayout.EndHorizontal();
            }
        }

        // ─── Simple type detection ───────────────────────────────────────

        private static bool IsSimple(Type t)
        {
            return t == typeof(bool)   ||
                   t == typeof(float)  || t == typeof(int)    || t == typeof(uint)  ||
                   t == typeof(short)  || t == typeof(ushort) || t == typeof(byte)  ||
                   t == typeof(string) ||
                   t == typeof(float2) || t == typeof(float3) || t == typeof(float4) ||
                   t == typeof(int2)   || t == typeof(uint2)  ||
                   t == typeof(Vector2)|| t == typeof(Vector3)|| t == typeof(Vector4);
        }

        // ─── Simple value drawing ────────────────────────────────────────

        private object DrawSimpleValue(Type ft, object value, FieldInfo field, string path)
        {
            var range = field?.GetCustomAttribute<RangeAttribute>();

            // [Toggle] 特殊处理：将整数/数值字段显示为勾选框（0 = false, 非0 = true）
            if (field?.GetCustomAttribute<ToggleAttribute>() != null)
            {
                bool cur = !Equals(value, Convert.ChangeType(0, ft));
                bool nxt = GUILayout.Toggle(cur, "");
                return nxt ? Convert.ChangeType(1, ft) : Convert.ChangeType(0, ft);
            }

            if (ft == typeof(bool))
            {
                return GUILayout.Toggle((bool)value, "");
            }
            if (ft == typeof(float))
            {
                float v = (float)value;
                if (range != null)
                {
                    float nv = GUILayout.HorizontalSlider(v, range.min, range.max, GUILayout.Width(145));
                    GUILayout.Label(nv.ToString("F3"), _label, GUILayout.Width(50));
                    return nv;
                }
                return FloatField(path, v);
            }
            if (ft == typeof(int))
            {
                int v = (int)value;
                if (range != null)
                {
                    int nv = Mathf.RoundToInt(GUILayout.HorizontalSlider(v, range.min, range.max, GUILayout.Width(145)));
                    GUILayout.Label(nv.ToString(), _label, GUILayout.Width(40));
                    return nv;
                }
                return IntField(path, v, 120);
            }
            if (ft == typeof(uint))
            {
                uint v = (uint)value;
                if (range != null)
                {
                    uint nv = (uint)Mathf.RoundToInt(GUILayout.HorizontalSlider(v, range.min, range.max, GUILayout.Width(145)));
                    GUILayout.Label(nv.ToString(), _label, GUILayout.Width(40));
                    return nv;
                }
                return UIntField(path, v, 120);
            }
            if (ft == typeof(short))   return IntField(path, (short)value,  120);
            if (ft == typeof(ushort))  return (ushort)Mathf.Max(0, IntField(path, (ushort)value, 120));
            if (ft == typeof(byte))    return (byte)Mathf.Clamp(IntField(path, (byte)value, 120), 0, 255);
            if (ft == typeof(float2))
            {
                float2 v = (float2)value;
                return new float2(CompactFloat(path + ".x", v.x), CompactFloat(path + ".y", v.y));
            }
            if (ft == typeof(float3))
            {
                float3 v = (float3)value;
                return new float3(CompactFloat(path + ".x", v.x), CompactFloat(path + ".y", v.y), CompactFloat(path + ".z", v.z));
            }
            if (ft == typeof(float4))
            {
                float4 v = (float4)value;
                return new float4(CompactFloat(path + ".x", v.x), CompactFloat(path + ".y", v.y),
                                  CompactFloat(path + ".z", v.z), CompactFloat(path + ".w", v.w));
            }
            if (ft == typeof(int2))
            {
                int2 v = (int2)value;
                return new int2(IntField(path + ".x", v.x, 58), IntField(path + ".y", v.y, 58));
            }
            if (ft == typeof(uint2))
            {
                uint2 v = (uint2)value;
                return new uint2(UIntField(path + ".x", v.x, 58), UIntField(path + ".y", v.y, 58));
            }
            if (ft == typeof(Vector2))
            {
                Vector2 v = (Vector2)value;
                return new Vector2(CompactFloat(path + ".x", v.x), CompactFloat(path + ".y", v.y));
            }
            if (ft == typeof(Vector3))
            {
                Vector3 v = (Vector3)value;
                return new Vector3(CompactFloat(path + ".x", v.x), CompactFloat(path + ".y", v.y), CompactFloat(path + ".z", v.z));
            }
            if (ft == typeof(Vector4))
            {
                Vector4 v = (Vector4)value;
                return new Vector4(CompactFloat(path + ".x", v.x), CompactFloat(path + ".y", v.y),
                                   CompactFloat(path + ".z", v.z), CompactFloat(path + ".w", v.w));
            }
            if (ft == typeof(string))
            {
                string s   = (string)value ?? "";
                string cur = GetTextCache(path, s);
                GUI.SetNextControlName(path);
                string next = GUILayout.TextField(cur, _tf, GUILayout.Width(185));
                _textCache[path] = next;
                return next;
            }

            // 其他：只读
            GUILayout.Label(value?.ToString() ?? "null", _label, GUILayout.Width(185));
            return value;
        }

        // ─── Text-field helpers (focus-aware to avoid cursor jumping) ────

        private float FloatField(string key, float v)
        {
            string cur = GetTextCache(key, v.ToString("G6"));
            GUI.SetNextControlName(key);
            string s = GUILayout.TextField(cur, _tf, GUILayout.Width(120));
            _textCache[key] = s;
            return float.TryParse(s,
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out float r) ? r : v;
        }

        private int IntField(string key, int v, int width)
        {
            string cur = GetTextCache(key, v.ToString());
            GUI.SetNextControlName(key);
            string s = GUILayout.TextField(cur, _tf, GUILayout.Width(width));
            _textCache[key] = s;
            return int.TryParse(s, out int r) ? r : v;
        }

        private uint UIntField(string key, uint v, int width)
        {
            string cur = GetTextCache(key, v.ToString());
            GUI.SetNextControlName(key);
            string s = GUILayout.TextField(cur, _tf, GUILayout.Width(width));
            _textCache[key] = s;
            return uint.TryParse(s, out uint r) ? r : v;
        }

        /// <summary>紧凑的小数字段，用于 float2/3/4 的每个分量。</summary>
        private float CompactFloat(string key, float v)
        {
            string cur = GetTextCache(key, v.ToString("G5"));
            GUI.SetNextControlName(key);
            string s = GUILayout.TextField(cur, _tf, GUILayout.Width(55));
            _textCache[key] = s;
            return float.TryParse(s,
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out float r) ? r : v;
        }

        /// <summary>
        /// 仅当控件未获得焦点时用实际值刷新缓存，
        /// 正在输入时保持用户键入的临时字符串。
        /// </summary>
        private string GetTextCache(string key, string actualValue)
        {
            if (GUI.GetNameOfFocusedControl() != key)
                _textCache[key] = actualValue;
            return _textCache.TryGetValue(key, out var cached) ? cached : actualValue;
        }

        // ─── Feature finder ──────────────────────────────────────────────

        private static RtxdiFeature FindFeature()
        {
            var cam = Camera.main;
            if (cam == null) return null;

            var uca = cam.GetComponent<UniversalAdditionalCameraData>();
            if (uca == null) return null;

            var renderer = uca.scriptableRenderer;

            // 尝试公开属性（URP 2021+）
            var prop = typeof(ScriptableRenderer).GetProperty("rendererFeatures",
                BindingFlags.Public | BindingFlags.Instance);
            if (prop != null)
            {
                var list = prop.GetValue(renderer) as List<ScriptableRendererFeature>;
                if (list != null)
                {
                    foreach (var f in list)
                        if (f is RtxdiFeature r) return r;
                    return null;
                }
            }

            // 回退：通过私有字段
            var fi = typeof(ScriptableRenderer).GetField("m_RendererFeatures",
                BindingFlags.NonPublic | BindingFlags.Instance);
            if (fi == null) return null;

            var flist = fi.GetValue(renderer) as List<ScriptableRendererFeature>;
            if (flist == null) return null;

            foreach (var f in flist)
                if (f is RtxdiFeature r) return r;

            return null;
        }
    }
}
