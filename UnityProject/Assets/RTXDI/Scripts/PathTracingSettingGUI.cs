using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// RTXDI 管线的运行时设置面板：在场景中查找 <see cref="UnityRtxdiFeature"/>
    /// 并通过反射显示/编辑其 setting。通用反射绘制逻辑在基类
    /// <see cref="SettingsReflectionGUI"/>（Core 程序集）中。
    /// </summary>
    public class PathTracingSettingGUI : SettingsReflectionGUI
    {
        private UnityRtxdiFeature _feature;

        protected override string NotFoundMessage => "PathTracingSettingGUI: RtxdiFeature not found";

        protected override object GetSettings()
        {
            if (_feature == null)
                _feature = FindFeature();
            return _feature != null ? _feature.setting : null;
        }

        // ─── Feature finder ──────────────────────────────────────────────

        private static UnityRtxdiFeature FindFeature()
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
                        if (f is UnityRtxdiFeature r) return r;
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
                if (f is UnityRtxdiFeature r) return r;

            return null;
        }
    }
}
