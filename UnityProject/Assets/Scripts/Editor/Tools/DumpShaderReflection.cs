#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using NativeRender;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace PathTracing.Editor.Tools
{
    /// <summary>
    /// Dumps each native shader asset's reflection JSON to a sibling
    /// "&lt;name&gt;.&lt;ext&gt;.reflection.json" file for easy browsing during development.
    ///
    /// Covers every native shader asset type that exposes reflection: compute, ray-trace,
    /// hit-group and raster. New types only need an extra <see cref="Kind"/> entry below.
    /// </summary>
    public static class DumpShaderReflection
    {
        private const string MenuRoot         = "Tools/Native Shader Reflection";
        private const string ReflectionSuffix = ".reflection.json";

        /// <summary>Describes one native shader asset type for the table-driven menu actions.</summary>
        private sealed class Kind
        {
            public readonly string                   TypeName; // for AssetDatabase.FindAssets("t:...")
            public readonly string                   Extension; // source asset extension, e.g. ".computeshader"
            public readonly Func<string, bool, bool> Dump; // (assetPath, force) => wrote a file

            public Kind(string typeName, string extension, Func<string, bool, bool> dump)
            {
                TypeName  = typeName;
                Extension = extension;
                Dump      = dump;
            }
        }

        /// <summary>All native shader asset types that carry reflection JSON.</summary>
        private static readonly Kind[] Kinds =
        {
            new(nameof(NativeComputeShader), ".computeshader",
                (p, f) => Dump<NativeComputeShader>(p, f, s => s.ReflectionJson, s => s.ForceRecompile())),
            new(nameof(RayTraceShader), ".rayshader",
                (p, f) => Dump<RayTraceShader>(p, f, s => s.ReflectionJson, s => s.ForceRecompile())),
            new(nameof(HitGroupShader), ".hitgroupshader",
                (p, f) => Dump<HitGroupShader>(p, f, s => s.ReflectionJson, s => s.ForceRecompile())),
            new(nameof(NativeRasterShader), ".rastershader",
                (p, f) => Dump<NativeRasterShader>(p, f, s => s.ReflectionJson, s => s.ForceRecompile())),
        };

        // -------------------------------------------------------------------
        // Reimport
        // -------------------------------------------------------------------

        [MenuItem(MenuRoot + "/Reimport All Native Shaders")]
        public static void ReimportAll()
        {
            List<string> paths = CollectAssetPaths(searchFolders: null);
            Reimport(paths);
            Debug.Log($"[DumpShaderReflection] Reimported {paths.Count} native shader(s).");
        }

        [MenuItem(MenuRoot + "/Reimport Selected Folder Shaders")]
        public static void ReimportSelectedFolder()
        {
            string folderPath = FirstSelectedFolder();
            if (folderPath == null)
            {
                Debug.LogWarning("[DumpShaderReflection] Please select a folder in the Project window.");
                return;
            }

            List<string> paths = CollectAssetPaths(new[] { folderPath });
            Reimport(paths);
            Debug.Log($"[DumpShaderReflection] Reimported {paths.Count} native shader(s) from '{folderPath}'.");
        }

        // -------------------------------------------------------------------
        // Dump
        // -------------------------------------------------------------------

        [MenuItem(MenuRoot + "/Dump All To Sibling Files")]
        public static void DumpAll()
        {
            List<string> paths   = CollectAssetPaths(searchFolders: null);
            int          written = 0, skipped = 0;
            try
            {
                AssetDatabase.StartAssetEditing();
                for (int i = 0; i < paths.Count; i++)
                {
                    EditorUtility.DisplayProgressBar(
                        "Dumping shader reflection", paths[i], (float)i / Mathf.Max(1, paths.Count));
                    if (DumpPath(paths[i], force: false)) written++;
                    else skipped++;
                }
            }
            finally
            {
                AssetDatabase.StopAssetEditing();
                EditorUtility.ClearProgressBar();
                AssetDatabase.Refresh();
            }

            Debug.Log($"[DumpShaderReflection] Wrote {written}, skipped {skipped} (no reflection yet).");
        }

        [MenuItem(MenuRoot + "/Dump Selected To Sibling File")]
        public static void DumpSelected()
        {
            int written = 0;
            foreach (Object obj in Selection.objects)
            {
                string p = AssetDatabase.GetAssetPath(obj);
                if (string.IsNullOrEmpty(p)) continue;
                if (DumpPath(p, force: true)) written++;
            }

            AssetDatabase.Refresh();
            Debug.Log($"[DumpShaderReflection] Wrote {written} file(s) for selection.");
        }

        [MenuItem(MenuRoot + "/Delete All Sibling Reflection Files")]
        public static void DeleteAll()
        {
            int    deleted    = 0;
            string assetsRoot = Application.dataPath;
            foreach (Kind kind in Kinds)
            {
                string pattern = "*" + kind.Extension + ReflectionSuffix;
                foreach (string path in Directory.EnumerateFiles(assetsRoot, pattern, SearchOption.AllDirectories))
                {
                    File.Delete(path);
                    string meta = path + ".meta";
                    if (File.Exists(meta)) File.Delete(meta);
                    deleted++;
                }
            }

            AssetDatabase.Refresh();
            Debug.Log($"[DumpShaderReflection] Deleted {deleted} reflection file(s).");
        }

        // -------------------------------------------------------------------
        // Helpers
        // -------------------------------------------------------------------

        /// <summary>Gathers asset paths for every <see cref="Kind"/>, optionally limited to folders.</summary>
        private static List<string> CollectAssetPaths(string[] searchFolders)
        {
            var paths = new List<string>();
            foreach (Kind kind in Kinds)
            {
                string[] guids = searchFolders == null
                    ? AssetDatabase.FindAssets("t:" + kind.TypeName)
                    : AssetDatabase.FindAssets("t:" + kind.TypeName, searchFolders);
                foreach (string guid in guids)
                    paths.Add(AssetDatabase.GUIDToAssetPath(guid));
            }

            return paths;
        }

        private static void Reimport(IReadOnlyList<string> paths)
        {
            try
            {
                AssetDatabase.StartAssetEditing();
                for (int i = 0; i < paths.Count; i++)
                {
                    EditorUtility.DisplayProgressBar(
                        "Reimporting native shaders", paths[i], (float)i / Mathf.Max(1, paths.Count));
                    AssetDatabase.ImportAsset(paths[i], ImportAssetOptions.ForceUpdate);
                }
            }
            finally
            {
                AssetDatabase.StopAssetEditing();
                EditorUtility.ClearProgressBar();
                AssetDatabase.Refresh();
            }
        }

        private static string FirstSelectedFolder()
        {
            foreach (Object obj in Selection.objects)
            {
                string p = AssetDatabase.GetAssetPath(obj);
                if (!string.IsNullOrEmpty(p) && AssetDatabase.IsValidFolder(p))
                    return p;
            }

            return null;
        }

        /// <summary>Dispatches a single asset path to the matching <see cref="Kind"/> by extension.</summary>
        private static bool DumpPath(string assetPath, bool force)
        {
            foreach (Kind kind in Kinds)
                if (assetPath.EndsWith(kind.Extension, StringComparison.OrdinalIgnoreCase))
                    return kind.Dump(assetPath, force);
            return false;
        }

        /// <summary>
        /// Loads the asset, obtains its reflection JSON (force-recompiling when empty and
        /// <paramref name="force"/> is set), and writes the pretty-printed sibling file.
        /// </summary>
        private static bool Dump<T>(string assetPath, bool force,
            Func<T, string> getJson, Action<T> recompile) where T : Object
        {
            var shader = AssetDatabase.LoadAssetAtPath<T>(assetPath);
            if (shader == null) return false;

            string json = getJson(shader);
            if (string.IsNullOrEmpty(json))
            {
                if (!force) return false;
                recompile(shader);
                json = getJson(shader);
                if (string.IsNullOrEmpty(json)) return false;
            }

            string outPath = assetPath + ReflectionSuffix;
            string absPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", outPath));
            File.WriteAllText(absPath, PrettyPrint(json));
            return true;
        }

        private static string PrettyPrint(string json)
        {
            var  sb    = new System.Text.StringBuilder(json.Length + 256);
            int  ind   = 0;
            bool inStr = false;
            for (int i = 0; i < json.Length; i++)
            {
                char c                                                 = json[i];
                if (c == '"' && (i == 0 || json[i - 1] != '\\')) inStr = !inStr;

                if (!inStr)
                {
                    switch (c)
                    {
                        case '{':
                        case '[':
                        {
                            // Keep empty objects/arrays on one line: "{}" / "[]".
                            char close = c == '{' ? '}' : ']';
                            int  j     = i + 1;
                            while (j < json.Length && char.IsWhiteSpace(json[j])) j++;
                            if (j < json.Length && json[j] == close)
                            {
                                sb.Append(c);
                                sb.Append(close);
                                i = j;
                                continue;
                            }

                            sb.Append(c);
                            sb.Append('\n');
                            sb.Append(' ', ++ind * 2);
                            continue;
                        }
                        case '}':
                        case ']':
                            sb.Append('\n');
                            sb.Append(' ', --ind * 2);
                            sb.Append(c);
                            continue;
                        case ',':
                            sb.Append(c);
                            sb.Append('\n');
                            sb.Append(' ', ind * 2);
                            continue;
                        case ':':
                            sb.Append(": ");
                            continue;
                    }
                }

                sb.Append(c);
            }

            return sb.ToString();
        }
    }
}
#endif