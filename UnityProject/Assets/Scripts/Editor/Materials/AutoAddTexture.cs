using UnityEditor;
using UnityEngine;

namespace PathTracing
{
    public class AutoAddTexture : EditorWindow
    {
        public string folderPath;
        public string materialFolderPath;

        [MenuItem("Tools/Auto Add Texture")]
        public static void ShowWindow()
        {
            GetWindow<AutoAddTexture>("Auto Add Texture");
        }

        public void OnGUI()
        {
            // 选择纹理文件夹
            GUILayout.Label("Select Texture Folder", EditorStyles.boldLabel);
            if (GUILayout.Button("Select Folder"))
            {
                folderPath = EditorUtility.OpenFolderPanel("Select Texture Folder", "", "");
            }

            GUILayout.Label($"Selected Folder: {folderPath}");
            
            // 选择材质文件夹
            GUILayout.Space(20);
            GUILayout.Label("Select Material Folder", EditorStyles.boldLabel);
            if (GUILayout.Button("Select Material Folder"))
            {
                materialFolderPath = EditorUtility.OpenFolderPanel("Select Material Folder", "", "");
            }
            GUILayout.Label($"Selected Material Folder: {materialFolderPath}");

            if (GUILayout.Button("Merge Textures"))
            {
                MergeTextures();
            }
            
            if (GUILayout.Button("Auto Add Textures to Materials"))
            {
                AutoAddTexturesToMaterials();
            }
        }

        // 合并指定文件夹下的纹理贴图
        // xxx_albedo.png + xxx_opacity.png -> xxx_baseMap.png
        // xxx_metallic.png + xxx_roughness.png -> xxx_maskMap.png
 
        private void MergeTextures()
        {
            if (string.IsNullOrEmpty(folderPath))
            {
                Debug.LogError("No folder selected!");
                return;
            }

            Debug.Log("Merge Textures");

            string[] textureFiles = System.IO.Directory.GetFiles(folderPath, "*.*", System.IO.SearchOption.AllDirectories);
            Debug.Log($"Found {textureFiles.Length} texture files in {folderPath}");

            foreach (string file in textureFiles)
            {
                string dir = System.IO.Path.GetDirectoryName(file);
                string fileName = System.IO.Path.GetFileNameWithoutExtension(file);

                // 1. 合并 Albedo + Opacity -> BaseMap
                if (file.EndsWith("_albedo.jpeg", System.StringComparison.OrdinalIgnoreCase) || 
                    file.EndsWith("_albedo.png", System.StringComparison.OrdinalIgnoreCase) || 
                    file.EndsWith("_albedo.jpg", System.StringComparison.OrdinalIgnoreCase))
                {
                    string baseName = fileName.Replace("_albedo", "");
                    string opacityFile = FindFileWithExtensions(dir, baseName + "_opacity", new[] { ".jpeg", ".jpg", ".png" });

                    if (!string.IsNullOrEmpty(opacityFile))
                    {
                        Debug.Log($"Processing BaseMap: {baseName}");
                        string outputFile = System.IO.Path.Combine(dir, baseName + "_baseMap.png");
                        MergeChannels(file, opacityFile, outputFile, MergeType.BaseMap);
                    }
                }

                // 2. 合并 Metallic + Roughness -> MaskMap
                // 逻辑更新：支持仅有 Metallic 或仅有 Roughness 的情况
                bool isMetallic = file.EndsWith("_metallic.jpeg", System.StringComparison.OrdinalIgnoreCase) || 
                                  file.EndsWith("_metallic.png", System.StringComparison.OrdinalIgnoreCase) || 
                                  file.EndsWith("_metallic.jpg", System.StringComparison.OrdinalIgnoreCase);
                bool isRoughness = file.EndsWith("_roughness.jpeg", System.StringComparison.OrdinalIgnoreCase) || 
                                   file.EndsWith("_roughness.png", System.StringComparison.OrdinalIgnoreCase) || 
                                   file.EndsWith("_roughness.jpg", System.StringComparison.OrdinalIgnoreCase);

                if (isMetallic || isRoughness)
                {
                    string baseName;
                    string metallicFile = null;
                    string roughnessFile = null;

                    if (isMetallic)
                    {
                        baseName = fileName.Replace("_metallic", "");
                        metallicFile = file;
                        // 尝试找 roughness，如果没有则为 null
                        roughnessFile = FindFileWithExtensions(dir, baseName + "_roughness", new[] { ".jpeg", ".jpg", ".png" });
                    }
                    else // isRoughness
                    {
                        baseName = fileName.Replace("_roughness", "");
                        
                        // 检查是否已经存在 update metallic 文件，如果存在，则该组合应该在 isMetallic 分支处理过，跳过
                        string existingMetallic = FindFileWithExtensions(dir, baseName + "_metallic", new[] { ".jpeg", ".jpg", ".png" });
                        if (!string.IsNullOrEmpty(existingMetallic)) 
                        {
                            continue;
                        }
                        
                        roughnessFile = file;
                        // metallicFile 保持为 null
                    }

                    Debug.Log($"Processing MaskMap: {baseName}");
                    string outputFile = System.IO.Path.Combine(dir, baseName + "_maskMap.png");
                    
                    // 传入可能为 null 的路径，MergeChannels 内部处理默认值
                    MergeChannels(metallicFile, roughnessFile, outputFile, MergeType.MaskMap);
                }
            }

            AssetDatabase.Refresh();
        }

        private void MergeChannels(string file1, string file2, string outputPath, MergeType type)
        {
            Texture2D tex1 = null; // 对应 Albedo 或 Metallic
            Texture2D tex2 = null; // 对应 Opacity 或 Roughness

            // 加载 Texture 1 (若存在)
            if (!string.IsNullOrEmpty(file1) && System.IO.File.Exists(file1))
            {
                byte[] data = System.IO.File.ReadAllBytes(file1);
                tex1 = new Texture2D(2, 2);
                tex1.LoadImage(data);
            }

            // 加载 Texture 2 (若存在)
            if (!string.IsNullOrEmpty(file2) && System.IO.File.Exists(file2))
            {
                byte[] data = System.IO.File.ReadAllBytes(file2);
                tex2 = new Texture2D(2, 2);
                tex2.LoadImage(data);
            }

            // 两个都不存在则无法合并
            if (tex1 == null && tex2 == null) return;

            // 确定尺寸：以存在的那个为准
            int width = tex1 != null ? tex1.width : tex2.width;
            int height = tex1 != null ? tex1.height : tex2.height;

            // 如果两个都存在，检查尺寸是否匹配
            if (tex1 != null && tex2 != null)
            {
                if (tex1.width != width || tex1.height != height || tex2.width != width || tex2.height != height)
                {
                    Debug.LogError($"Size mismatch for merging: {file1} vs {file2}");
                    // 清理内存
                    if(tex1) DestroyImmediate(tex1);
                    if(tex2) DestroyImmediate(tex2);
                    return;
                }
            }
 
            Color[] cols1 = tex1 != null ? tex1.GetPixels() : GetDefaultColors(width * height, Color.black);
            Color[] cols2 = tex2 != null ? tex2.GetPixels() : GetDefaultColors(width * height, Color.white);
            
            Color[] newCols = new Color[cols1.Length];

            for (int i = 0; i < cols1.Length; i++)
            {
                if (type == MergeType.BaseMap)
                {
                    // BaseMap: RGB 来自 Albedo (cols1), A 来自 Opacity (cols2.r)
                    newCols[i] = new Color(cols1[i].r, cols1[i].g, cols1[i].b, cols2[i].r);
                }
                else if (type == MergeType.MaskMap)
                {
                    // MaskMap:
                    // R: Metallic (cols1.r). 缺失默认为 0.
                    // G: Occlusion (Set to 1)
                    // B: Detail (Set to 0)
                    // A: Smoothness (1 - Roughness (cols2.r)). 
                    //    若 Roughness 缺失默认为 1，则 Smoothness 为 0。
                    float metallic = cols1[i].r;
                    float roughness = cols2[i].r;
                    float smoothness = 1.0f - roughness;

                    newCols[i] = new Color(metallic, 1.0f, 0.0f, smoothness);
                }
            }

            Texture2D outputTex = new Texture2D(width, height);
            outputTex.SetPixels(newCols);
            outputTex.Apply();

            byte[] bytes = outputTex.EncodeToPNG();
            System.IO.File.WriteAllBytes(outputPath, bytes);

            if(tex1) DestroyImmediate(tex1);
            if(tex2) DestroyImmediate(tex2);
            DestroyImmediate(outputTex);
        }

        private Color[] GetDefaultColors(int count, Color color)
        {
            Color[] cols = new Color[count];
            for (int i = 0; i < count; i++)
            {
                cols[i] = color;
            }
            return cols;
        }
        private string FindFileWithExtensions(string dir, string fileNameWithoutExt, string[] extensions)
        {
            foreach (var ext in extensions)
            {
                string path = System.IO.Path.Combine(dir, fileNameWithoutExt + ext);
                if (System.IO.File.Exists(path)) return path;
            }
            return null;
        }

        private enum MergeType { BaseMap, MaskMap }
 

        // 为选定的材质自动添加纹理（通过在指定文件夹下搜索同名的纹理）
        
          
        private void AutoAddTexturesToMaterials()
        {
            if (string.IsNullOrEmpty(materialFolderPath) || string.IsNullOrEmpty(folderPath))
            {
                Debug.LogError("请先选择材质文件夹和纹理文件夹！");
                return;
            }

            Debug.Log("开始自动为材质赋予纹理...");
            
            // 获取材质文件夹下的所有材质文件
            string[] materialFiles = System.IO.Directory.GetFiles(materialFolderPath, "*.mat", System.IO.SearchOption.AllDirectories);

            foreach (string matFile in materialFiles)
            {
                // 将系统绝对路径转换为Unity Asset路径
                string assetPath = GetRelativeAssetPath(matFile);
                if (string.IsNullOrEmpty(assetPath)) continue;

                Material mat = AssetDatabase.LoadAssetAtPath<Material>(assetPath);
                if (mat == null) continue;

                // 简单的名称匹配逻辑：材质名 "MyProp" -> 寻找 "MyProp_baseMap.png" 等
                string matName = mat.name;

                // 1. Base Map (Albedo)
                // 优先找 MergeTextures 生成的 _baseMap，其次找原始的 _albedo
                string baseMapFile = FindTextureFile(matName, "_baseMap", folderPath);
                if (string.IsNullOrEmpty(baseMapFile)) baseMapFile = FindTextureFile(matName, "_albedo", folderPath);

                // 2. Mask Map (Metallic + Smoothness)
                string maskMapFile = FindTextureFile(matName, "_maskMap", folderPath);

                // 3. Normal Map
                string normalMapFile = FindTextureFile(matName, "_normal", folderPath);

                bool changed = false;

                if (!string.IsNullOrEmpty(baseMapFile))
                {
                    Texture2D tex = AssetDatabase.LoadAssetAtPath<Texture2D>(GetRelativeAssetPath(baseMapFile));
                    if (tex != null)
                    {
                        Undo.RecordObject(mat, "Assign BaseMap");
                        mat.SetTexture("_BaseMap", tex);
                        changed = true;
                    }
                }

                if (!string.IsNullOrEmpty(maskMapFile))
                {
                    Texture2D tex = AssetDatabase.LoadAssetAtPath<Texture2D>(GetRelativeAssetPath(maskMapFile));
                    if (tex != null)
                    {
                        Undo.RecordObject(mat, "Assign MaskMap");
                        mat.SetTexture("_MetallicGlossMap", tex);
                        // mat.EnableKeyword("_METALLICSPECGLOSSMAP"); // 必须启用此关键字才能使用金属度贴图
                        
                        // 如果 MaskMap 包含 AO (通常在 G 通道)，也可以赋值给 OcclusionMap
                        // URP Lit 默认从 G 通道读取 AO
                        mat.SetTexture("_OcclusionMap", tex);
                        // mat.EnableKeyword("_OCCLUSIONMAP"); // 取决于具体版本需求，通常赋值贴图即可
                        
                        changed = true;
                    }
                }

                if (!string.IsNullOrEmpty(normalMapFile))
                {
                    Texture2D tex = AssetDatabase.LoadAssetAtPath<Texture2D>(GetRelativeAssetPath(normalMapFile));
                    if (tex != null)
                    {
                        Undo.RecordObject(mat, "Assign NormalMap");
                        mat.SetTexture("_BumpMap", tex);
                        // mat.EnableKeyword("_NORMALMAP");
                        changed = true;
                    }
                }

                // 4. Emissive Map (_emissive)
                // 如果存在以 _emissive 结尾的贴图，则赋值为材质的发光贴图
                string emissiveMapFile = FindTextureFile(matName, "_emissive", folderPath);
                if (!string.IsNullOrEmpty(emissiveMapFile))
                {
                    Texture2D emitTex = AssetDatabase.LoadAssetAtPath<Texture2D>(GetRelativeAssetPath(emissiveMapFile));
                    if (emitTex != null)
                    {
                        Undo.RecordObject(mat, "Assign EmissionMap");
                        // 常用属性名为 _EmissionMap，设置白色作为默认的 EmissionColor 以确保可见
                        mat.SetTexture("_EmissionMap", emitTex);
                        mat.SetColor("_EmissionColor", Color.white);
                        // 启用发光关键字（根据 Unity 版本/Shader 可能不同，但通常可用）
                        mat.EnableKeyword("_EMISSION");
                        changed = true;
                    }
                }

                if (changed)
                {
                    Debug.Log($"Updated Material: {matName}");
                    EditorUtility.SetDirty(mat);
                }
            }

            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log("材质纹理自动匹配完成。");
        }

        // 辅助方法：在指定文件夹搜索匹配特定后缀的文件
        private string FindTextureFile(string baseName, string suffix, string searchDir)
        {
            // 在文件夹中搜索所有文件
            // 搜索模式示例: Name_suffix.*
            string searchPattern = $"{baseName}{suffix}.*"; 
            
            // 注意：Directory.GetFiles 的 searchPattern 匹配比较宽泛，需要精确检查
            string[] files = System.IO.Directory.GetFiles(searchDir, searchPattern, System.IO.SearchOption.AllDirectories);

            foreach (string file in files)
            {
                string fileName = System.IO.Path.GetFileNameWithoutExtension(file);
                // 确保文件名精确匹配 (忽略大小写)
                if (fileName.Equals($"{baseName}{suffix}", System.StringComparison.OrdinalIgnoreCase))
                {
                    string ext = System.IO.Path.GetExtension(file).ToLower();
                    // 仅支持常见纹理格式
                    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tga" || ext == ".tif" || ext == ".tiff")
                    {
                        return file;
                    }
                }
            }
            return null;
        }

        // 辅助方法：将绝对路径转换为 Assets/ 开头的相对路径
        private string GetRelativeAssetPath(string absolutePath)
        {
            absolutePath = absolutePath.Replace("\\", "/");
            int index = absolutePath.IndexOf("Assets/");
            if (index >= 0)
            {
                return absolutePath.Substring(index);
            }
            // 如果已经在 Assets 文件夹外，无法加载
            Debug.LogWarning($"Path is not inside Assets folder: {absolutePath}");
            return null;
        }
        
    }
}