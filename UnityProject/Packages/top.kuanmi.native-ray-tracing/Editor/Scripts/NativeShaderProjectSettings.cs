using System;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace NativeRender
{
    /// <summary>
    /// Project-wide settings applied to every NativeComputeShader / RayTraceShader import.
    /// Stored at <c>ProjectSettings/NativeShaderSettings.asset</c> and version-controlled with
    /// the project.  Individual importer settings are merged on top of these globals.
    /// </summary>
    [Serializable]
    public class NativeShaderProjectSettings
    {
        private const string SettingsPath = "ProjectSettings/NativeShaderSettings.asset";

        // ── Singleton ──────────────────────────────────────────────────────────
        private static NativeShaderProjectSettings _instance;

        public static NativeShaderProjectSettings instance
        {
            get
            {
                if (_instance == null)
                    _instance = Load();
                return _instance;
            }
        }

        // ── Data ───────────────────────────────────────────────────────────────
        [Tooltip("Preprocessor defines added to every shader import (e.g. FOO=1, BAR). " +
                 "Per-asset defines are appended after these.")]
        public string[] globalDefines = Array.Empty<string>();

        [Tooltip("Additional #include search paths added to every shader import. " +
                 "Supports environment variables and project-relative paths. " +
                 "Per-asset paths are appended after these.")]
        public string[] globalIncludePaths = Array.Empty<string>();

        [Tooltip("Extra DXC compiler arguments added to every shader import (e.g. -HV 2021). " +
                 "Per-asset args are appended after these.")]
        public string[] globalExtraArgs = Array.Empty<string>();

        // ── Persistence ────────────────────────────────────────────────────────
        private static NativeShaderProjectSettings Load()
        {
            var settings = new NativeShaderProjectSettings();
            if (File.Exists(SettingsPath))
            {
                try
                {
                    var json = File.ReadAllText(SettingsPath);
                    JsonUtility.FromJsonOverwrite(json, settings);
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[NativeShaderProjectSettings] Failed to load {SettingsPath}: {e.Message}");
                }
            }
            return settings;
        }

        public void Save()
        {
            try
            {
                var json = JsonUtility.ToJson(this, prettyPrint: true);
                File.WriteAllText(SettingsPath, json);
            }
            catch (Exception e)
            {
                Debug.LogError($"[NativeShaderProjectSettings] Failed to save {SettingsPath}: {e.Message}");
            }
        }

        /// <summary>Invalidate the cached instance so the next access re-reads from disk.</summary>
        public static void Invalidate() => _instance = null;
    }

    // ── Settings Provider (Project Settings window) ────────────────────────────
    internal static class NativeShaderProjectSettingsProvider
    {
        private const string MenuPath = "Project/Native Shader Settings";

        [SettingsProvider]
        public static SettingsProvider Create()
        {
            return new SettingsProvider(MenuPath, SettingsScope.Project)
            {
                label = "Native Shader Settings",
                activateHandler = (_, rootElement) => { /* UIElements mode, handled via OnGUI */ },
                guiHandler = _ => DrawGUI(),
                keywords = new System.Collections.Generic.HashSet<string>
                    { "native", "shader", "hlsl", "define", "include", "dxc" }
            };
        }

        // Serialized-object wrapper so we get Unity's standard array drawers for free.
        private static SerializedObject _so;
        private static NativeShaderProjectSettingsWrapper _wrapper;

        private static void DrawGUI()
        {
            if (_wrapper == null)
            {
                _wrapper = ScriptableObject.CreateInstance<NativeShaderProjectSettingsWrapper>();
                _wrapper.Sync(NativeShaderProjectSettings.instance);
                _so = new SerializedObject(_wrapper);
            }

            EditorGUI.BeginChangeCheck();

            _so.Update();
            EditorGUILayout.PropertyField(_so.FindProperty("globalDefines"),      new GUIContent("Global Defines"));
            EditorGUILayout.PropertyField(_so.FindProperty("globalIncludePaths"), new GUIContent("Global Include Paths"));
            EditorGUILayout.PropertyField(_so.FindProperty("globalExtraArgs"),    new GUIContent("Global Extra Args"));
            _so.ApplyModifiedProperties();

            if (EditorGUI.EndChangeCheck())
            {
                _wrapper.ApplyTo(NativeShaderProjectSettings.instance);
                NativeShaderProjectSettings.instance.Save();
                // Force reimport of all affected assets on next domain reload if needed.
                NativeShaderProjectSettings.Invalidate();
                _wrapper = null; // rebuild wrapper with fresh data next frame
            }

            EditorGUILayout.Space(8);
            EditorGUILayout.HelpBox(
                "These settings are prepended to every NativeComputeShader and RayTraceShader import. " +
                "Per-asset defines / include paths / extra args are appended after the globals.",
                MessageType.Info);
        }
    }

    /// <summary>
    /// Thin ScriptableObject wrapper so <see cref="SerializedObject"/> can drive the
    /// standard Unity array PropertyField drawers inside the Settings window.
    /// </summary>
    [Serializable]
    internal class NativeShaderProjectSettingsWrapper : ScriptableObject
    {
        public string[] globalDefines      = Array.Empty<string>();
        public string[] globalIncludePaths = Array.Empty<string>();
        public string[] globalExtraArgs    = Array.Empty<string>();

        public void Sync(NativeShaderProjectSettings src)
        {
            globalDefines      = src.globalDefines      ?? Array.Empty<string>();
            globalIncludePaths = src.globalIncludePaths ?? Array.Empty<string>();
            globalExtraArgs    = src.globalExtraArgs    ?? Array.Empty<string>();
        }

        public void ApplyTo(NativeShaderProjectSettings dst)
        {
            dst.globalDefines      = globalDefines      ?? Array.Empty<string>();
            dst.globalIncludePaths = globalIncludePaths ?? Array.Empty<string>();
            dst.globalExtraArgs    = globalExtraArgs    ?? Array.Empty<string>();
        }
    }
}
