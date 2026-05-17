#if UNITY_EDITOR
using System.IO;
using NativeRender;
using UnityEditor;
using UnityEngine;

namespace ProjectTools.Editor
{
    /// <summary>
    /// Dumps each NativeComputeShader's reflection JSON to a sibling
    /// "<name>.computeshader.reflection.json" file for easy browsing during development.
    /// </summary>
    public static class DumpShaderReflection
    {
        private const string MenuRoot = "Tools/Native Shader Reflection";

        [MenuItem(MenuRoot + "/Reimport All Native Shaders")]
        public static void ReimportAll()
        {
            string[] computeGuids  = AssetDatabase.FindAssets("t:" + nameof(NativeComputeShader));
            string[] raytraceGuids = AssetDatabase.FindAssets("t:" + nameof(RayTraceShader));
            int      total         = computeGuids.Length + raytraceGuids.Length;
            int      done          = 0;
            try
            {
                AssetDatabase.StartAssetEditing();
                foreach (string guid in computeGuids)
                {
                    string assetPath = AssetDatabase.GUIDToAssetPath(guid);
                    EditorUtility.DisplayProgressBar(
                        "Reimporting native shaders", assetPath, (float)done / Mathf.Max(1, total));
                    AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
                    done++;
                }

                foreach (string guid in raytraceGuids)
                {
                    string assetPath = AssetDatabase.GUIDToAssetPath(guid);
                    EditorUtility.DisplayProgressBar(
                        "Reimporting native shaders", assetPath, (float)done / Mathf.Max(1, total));
                    AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
                    done++;
                }
            }
            finally
            {
                AssetDatabase.StopAssetEditing();
                EditorUtility.ClearProgressBar();
                AssetDatabase.Refresh();
            }

            Debug.Log($"[DumpShaderReflection] Reimported {done} native shader(s) ({computeGuids.Length} compute, {raytraceGuids.Length} raytrace).");
        }

        [MenuItem(MenuRoot + "/Reimport Selected Folder Shaders")]
        public static void ReimportSelectedFolder()
        {
            string folderPath = null;
            foreach (var obj in Selection.objects)
            {
                string p = AssetDatabase.GetAssetPath(obj);
                if (!string.IsNullOrEmpty(p) && AssetDatabase.IsValidFolder(p))
                {
                    folderPath = p;
                    break;
                }
            }

            if (folderPath == null)
            {
                Debug.LogWarning("[DumpShaderReflection] Please select a folder in the Project window.");
                return;
            }

            string[] computeGuids  = AssetDatabase.FindAssets("t:" + nameof(NativeComputeShader), new[] { folderPath });
            string[] raytraceGuids = AssetDatabase.FindAssets("t:" + nameof(RayTraceShader), new[] { folderPath });
            int      total         = computeGuids.Length + raytraceGuids.Length;
            int      done          = 0;
            try
            {
                AssetDatabase.StartAssetEditing();
                foreach (string guid in computeGuids)
                {
                    string assetPath = AssetDatabase.GUIDToAssetPath(guid);
                    EditorUtility.DisplayProgressBar(
                        "Reimporting native shaders", assetPath, (float)done / Mathf.Max(1, total));
                    AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
                    done++;
                }

                foreach (string guid in raytraceGuids)
                {
                    string assetPath = AssetDatabase.GUIDToAssetPath(guid);
                    EditorUtility.DisplayProgressBar(
                        "Reimporting native shaders", assetPath, (float)done / Mathf.Max(1, total));
                    AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);
                    done++;
                }
            }
            finally
            {
                AssetDatabase.StopAssetEditing();
                EditorUtility.ClearProgressBar();
                AssetDatabase.Refresh();
            }

            Debug.Log($"[DumpShaderReflection] Reimported {done} native shader(s) from '{folderPath}' ({computeGuids.Length} compute, {raytraceGuids.Length} raytrace).");
        }

        [MenuItem(MenuRoot + "/Dump All To Sibling Files")]
        public static void DumpAll()
        {
            var guids   = AssetDatabase.FindAssets("t:" + nameof(NativeComputeShader));
            int written = 0, skipped = 0;
            AssetDatabase.StartAssetEditing();
            for (int i = 0; i < guids.Length; i++)
            {
                string assetPath = AssetDatabase.GUIDToAssetPath(guids[i]);
                EditorUtility.DisplayProgressBar(
                    "Dumping shader reflection",
                    assetPath, (float)i / Mathf.Max(1, guids.Length));

                if (DumpComputeShader(assetPath, force: false)) written++;
                else skipped++;
            }

            Debug.Log($"[DumpShaderReflection] Wrote {written}, skipped {skipped} (no reflection yet).");

            var raytraceGuids = AssetDatabase.FindAssets("t:" + nameof(RayTraceShader));
            written = skipped = 0;

            for (int i = 0; i < raytraceGuids.Length; i++)
            {
                string assetPath = AssetDatabase.GUIDToAssetPath(raytraceGuids[i]);
                EditorUtility.DisplayProgressBar(
                    "Dumping shader reflection",
                    assetPath, (float)i / Mathf.Max(1, raytraceGuids.Length));
                if (DumpRayShader(assetPath, force: false)) written++;
                else skipped++;
            }

            AssetDatabase.StopAssetEditing();
            EditorUtility.ClearProgressBar();
            AssetDatabase.Refresh();

            Debug.Log($"[DumpShaderReflection] Wrote {written}, skipped {skipped} (no reflection yet).");
        }

        [MenuItem(MenuRoot + "/Dump Selected To Sibling File")]
        public static void DumpSelected()
        {
            int written = 0;
            foreach (var obj in Selection.objects)
            {
                string p = AssetDatabase.GetAssetPath(obj);
                if (string.IsNullOrEmpty(p)) continue;
                if (!p.EndsWith(".computeshader")) continue;
                if (DumpComputeShader(p, force: true)) written++;
            }

            AssetDatabase.Refresh();
            Debug.Log($"[DumpShaderReflection] Wrote {written} file(s) for selection.");
        }

        [MenuItem(MenuRoot + "/Delete All Sibling Reflection Files")]
        public static void DeleteAll()
        {
            int    deleted    = 0;
            string assetsRoot = Application.dataPath;
            foreach (var path in Directory.EnumerateFiles(
                         assetsRoot, "*.computeshader.reflection.json", SearchOption.AllDirectories))
            {
                File.Delete(path);
                string meta = path + ".meta";
                if (File.Exists(meta)) File.Delete(meta);
                deleted++;
            }

            AssetDatabase.Refresh();
            Debug.Log($"[DumpShaderReflection] Deleted {deleted} reflection file(s).");
        }
        
        
        private static bool DumpRayShader(string assetPath, bool force)
        {
            var shader = AssetDatabase.LoadAssetAtPath<RayTraceShader>(assetPath);
            if (shader == null) return false;

            string json = shader.ReflectionJson;
            if (string.IsNullOrEmpty(json))
            {
                if (!force) return false;
                shader.ForceRecompile();
                json = shader.ReflectionJson;
                if (string.IsNullOrEmpty(json)) return false;
            }

            string outPath = assetPath + ".reflection.json";
            string absPath = Path.GetFullPath(Path.Combine(
                Application.dataPath, "..", outPath));

            // Pretty-print: insert newlines after object/array separators for readability.
            string pretty = PrettyPrint(json);
            File.WriteAllText(absPath, pretty);
            return true;
        }

        private static bool DumpComputeShader(string assetPath, bool force)
        {
            var shader = AssetDatabase.LoadAssetAtPath<NativeComputeShader>(assetPath);
            if (shader == null) return false;

            string json = shader.ReflectionJson;
            if (string.IsNullOrEmpty(json))
            {
                if (!force) return false;
                shader.ForceRecompile();
                json = shader.ReflectionJson;
                if (string.IsNullOrEmpty(json)) return false;
            }

            string outPath = assetPath + ".reflection.json";
            string absPath = Path.GetFullPath(Path.Combine(
                Application.dataPath, "..", outPath));

            // Pretty-print: insert newlines after object/array separators for readability.
            string pretty = PrettyPrint(json);
            File.WriteAllText(absPath, pretty);
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
                            sb.Append(c);
                            sb.Append('\n');
                            sb.Append(' ', ++ind * 2);
                            continue;
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