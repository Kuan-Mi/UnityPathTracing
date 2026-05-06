using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

/// <summary>
/// 批量替换指定文件夹内材质所用 Shader 的编辑器工具窗口。
/// 菜单路径：Tools / Material Shader Replacer
/// </summary>
public class MaterialShaderReplacer : EditorWindow
{
    private DefaultAsset _targetFolder;
    private Shader _originalShader;
    private Shader _replacementShader;

    private Vector2 _scrollPos;
    private readonly List<string> _log = new List<string>();

    [MenuItem("Tools/Material Shader Replacer")]
    public static void OpenWindow()
    {
        GetWindow<MaterialShaderReplacer>("Material Shader Replacer");
    }

    private void OnGUI()
    {
        EditorGUILayout.Space(6);
        EditorGUILayout.LabelField("批量替换材质 Shader", EditorStyles.boldLabel);
        EditorGUILayout.Space(4);

        _targetFolder = (DefaultAsset)EditorGUILayout.ObjectField(
            "目标文件夹", _targetFolder, typeof(DefaultAsset), false);

        _originalShader = (Shader)EditorGUILayout.ObjectField(
            "原始 Shader", _originalShader, typeof(Shader), false);

        _replacementShader = (Shader)EditorGUILayout.ObjectField(
            "替换 Shader", _replacementShader, typeof(Shader), false);
 
        EditorGUILayout.Space(8);

        bool canExecute = _targetFolder != null && _originalShader != null && _replacementShader != null;
        using (new EditorGUI.DisabledScope(!canExecute))
        {
            if (GUILayout.Button("开始替换", GUILayout.Height(30)))
            {
                Execute();
            }
        }

        if (!canExecute)
        {
            EditorGUILayout.HelpBox("请填写所有三个字段后再执行。", MessageType.Warning);
        }

        if (_log.Count > 0)
        {
            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField("执行日志", EditorStyles.boldLabel);
            _scrollPos = EditorGUILayout.BeginScrollView(_scrollPos, GUILayout.MinHeight(120));
            foreach (string line in _log)
            {
                EditorGUILayout.LabelField(line, EditorStyles.wordWrappedLabel);
            }
            EditorGUILayout.EndScrollView();

            if (GUILayout.Button("清空日志"))
            {
                _log.Clear();
            }
        }
    }

    private void Execute()
    {
        _log.Clear();

        // 获取文件夹的相对路径（Assets/...）
        string folderPath = AssetDatabase.GetAssetPath(_targetFolder);
        if (string.IsNullOrEmpty(folderPath) || !AssetDatabase.IsValidFolder(folderPath))
        {
            _log.Add("[错误] 所选对象不是有效文件夹。");
            Repaint();
            return;
        }

        // 递归查找该目录下全部 .mat 文件的 GUID
        string absoluteFolder = Path.GetFullPath(Path.Combine(Application.dataPath, "..", folderPath));
        string[] matGuids = AssetDatabase.FindAssets("t:Material", new[] { folderPath });

        if (matGuids.Length == 0)
        {
            _log.Add("[信息] 文件夹内未找到任何材质文件。");
            Repaint();
            return;
        }

        int replacedCount = 0;

        try
        {
            AssetDatabase.StartAssetEditing();

            foreach (string guid in matGuids)
            {
                string assetPath = AssetDatabase.GUIDToAssetPath(guid);
                Material mat = AssetDatabase.LoadAssetAtPath<Material>(assetPath);

                if (mat == null)
                {
                    _log.Add($"[跳过] 无法加载材质：{assetPath}");
                    continue;
                }

                if (mat.shader == null || mat.shader != _originalShader)
                {
                    continue;
                }

                mat.shader = _replacementShader;
                EditorUtility.SetDirty(mat);
                _log.Add($"[已替换] {assetPath}");
                replacedCount++;
            }
        }
        finally
        {
            AssetDatabase.StopAssetEditing();
            AssetDatabase.SaveAssets();
        }

        _log.Add($"\n完成：共替换 {replacedCount} / {matGuids.Length} 个材质。");
        Repaint();
        Debug.Log($"[MaterialShaderReplacer] 完成，共替换 {replacedCount} 个材质。");
    }
}
