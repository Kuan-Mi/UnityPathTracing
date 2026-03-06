using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Unity.VisualScripting;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DefaultNamespace.Editor
{
    public class FindSameMesh : EditorWindow
    {
        [MenuItem("Tools/Find Same Mesh")]
        public static void ShowWindow()
        {
            GetWindow<FindSameMesh>("Find Same Mesh");
        }

        public Texture2D whiteTexture;
        public Texture2D normalTexture;

        private void OnGUI()
        {
            whiteTexture =
                (Texture2D)EditorGUILayout.ObjectField("White Texture", whiteTexture, typeof(Texture2D), false);
            normalTexture =
                (Texture2D)EditorGUILayout.ObjectField("Normal Texture", normalTexture, typeof(Texture2D), false);

            if (GUILayout.Button("Find Same Mesh"))
            {
                FindSameMeshInScene();
            }

            if (GUILayout.Button("Find Same Skin Mesh"))
            {
                FindSameSkinMeshInScene();
            }

            if (GUILayout.Button("Find Same Mat"))
            {
                FindSameMatScene();
            }

            if (GUILayout.Button("Find Mat without AO"))
            {
                FindMatWithoutAO();
            }
            
            if (GUILayout.Button("Find Same Texture"))
            {
                FindSameTextureInScene();
            }
            
            if (GUILayout.Button("Find Same Mesh Collider"))
            {
                FindSameMeshColliderInScene();
            }
        }

        private void FindSameMeshColliderInScene()
        {
            var allMeshColliders = GameObject.FindObjectsOfType<MeshCollider>(true);


            foreach (var meshCollider in allMeshColliders)
            {
                var meshFilter = meshCollider.GetComponent<MeshFilter>();
                if (meshFilter == null || meshFilter.sharedMesh == null)
                {
                    continue;
                }
                
                if(meshCollider.sharedMesh!= meshFilter.sharedMesh)
                {
                    var hash = GetHash(meshFilter.sharedMesh);
                    var colliderHash = GetHash(meshCollider.sharedMesh);

                    if (hash == colliderHash)
                    {
                         meshCollider.sharedMesh = meshFilter.sharedMesh;
                         Debug.Log($"MeshCollider {meshCollider.name} has same mesh hash: {colliderHash}");
                    }
                    else
                    {
                        Debug.LogWarning($"MeshCollider {meshCollider.name} has different mesh hash: {colliderHash} vs {hash}");
                    }
                    
                }
                
                
                
                
            }
        }

        private void FindMatWithoutAO()
        {
            var allMeshRenderers = GameObject.FindObjectsOfType<MeshRenderer>(true).ToList();

            var allSkinnedMeshRenderers = GameObject.FindObjectsOfType<SkinnedMeshRenderer>(true);

            var allRenderers = new List<Renderer>();
            allRenderers.AddRange(allSkinnedMeshRenderers);
            allRenderers.AddRange(allMeshRenderers);

            foreach (var meshRenderer in allRenderers)
            {
                var materials = meshRenderer.sharedMaterials;
                for (var i = 0; i < materials.Length; i++)
                {
                    if (materials[i] == null) continue;

                    if (
                        materials[i].shader.name != "Universal Render Pipeline/Lit")
                    {
                        continue;
                    }

                    var metallicSmoothness = materials[i].GetTexture("_MetallicGlossMap");

                    if (metallicSmoothness == null)
                    {
                        var metallic = materials[i].GetFloat("_Metallic");

                        var MSTexture = new Texture2D(2, 2);
                        var c = new Color(metallic, 1, 1, 1);
                        MSTexture.SetPixel(0, 0, c);
                        MSTexture.SetPixel(1, 0, c);
                        MSTexture.SetPixel(0, 1, c);
                        MSTexture.SetPixel(1, 1, c);

                        MSTexture.Apply();
                        var textureName =
                            $"{materials[i].name}_MetallicSmoothness_{Guid.NewGuid().ToString()[..8]}.png";

                        var path = $"Assets/Textures/{textureName}";

                        if (!AssetDatabase.IsValidFolder("Assets/Textures"))
                        {
                            AssetDatabase.CreateFolder("Assets", "Textures");
                        }

                        var bytes = MSTexture.EncodeToPNG();
                        System.IO.File.WriteAllBytes(path, bytes);

                        AssetDatabase.Refresh();

                        var savedTexture = AssetDatabase.LoadAssetAtPath<Texture2D>($"Assets/Textures/{textureName}");
                        materials[i].SetTexture("_MetallicGlossMap", savedTexture);

                        Debug.Log(
                            $"GO {meshRenderer.name} has material without MetallicSmoothness at index {i}: {materials[i].name}");
                    }

                    if (!materials[i].GetTexture("_OcclusionMap"))
                    {
                        materials[i].SetTexture("_OcclusionMap", whiteTexture);
                        Debug.Log($"GO {meshRenderer.name} has material without AO at index {i}: {materials[i].name}");
                    }

                    if (!materials[i].GetTexture("_BumpMap"))
                    {
                        materials[i].SetTexture("_BumpMap", normalTexture);
                        Debug.Log(
                            $"GO {meshRenderer.name} has material without Normal at index {i}: {materials[i].name}");
                    }

                    if (!materials[i].GetTexture("_BaseMap"))
                    {
                        materials[i].SetTexture("_BaseMap", whiteTexture);
                        Debug.Log(
                            $"GO {meshRenderer.name} has material without Albedo at index {i}: {materials[i].name}");
                    }
                }
            }
        }

        private void FindSameMatScene()
        {
            var allRenderers = FindObjectsOfType<MeshRenderer>(true);

            foreach (var meshRenderer in allRenderers)
            {
                var otherMeshRenderers = allRenderers.Where(mr => mr != meshRenderer);

                var materials = meshRenderer.sharedMaterials;
                foreach (var material in materials)
                {
                    foreach (var otherMeshRenderer in otherMeshRenderers)
                    {
                        var otherMaterials = otherMeshRenderer.sharedMaterials;
                        for (var i = 0; i < otherMaterials.Length; i++)
                        {
                            if (otherMaterials[i] != material && otherMaterials[i].name == material.name)
                            {
                                Debug.Log(
                                    $"GO {meshRenderer.name} has same material with {otherMeshRenderer.name} at index {i}: {material.name}");
                                otherMaterials[i] = material;
                                otherMeshRenderer.sharedMaterials = otherMaterials;
                            }
                        }
                    }
                }
            }

            EditorSceneManager.MarkSceneDirty(UnityEngine.SceneManagement.SceneManager.GetActiveScene());
        }

        public int GetHash(Mesh mesh)
        {
            var allVertices = mesh.vertices;
            var allNormals = mesh.normals;
            var allTangents = mesh.tangents;
            var allUVs = mesh.uv;
            var allUV2S = mesh.uv2;


            var hash = 17;

            foreach (var vertex in allVertices)
            {
                hash = hash * 31 + vertex.GetHashCode();
            }

            foreach (var normal in allNormals)
            {
                hash = hash * 31 + normal.GetHashCode();
            }

            foreach (var tangent in allTangents)
            {
                hash = hash * 31 + tangent.GetHashCode();
            }

            foreach (var uv in allUVs)
            {
                hash = hash * 31 + uv.GetHashCode();
            }

            foreach (var uv2 in allUV2S)
            {
                hash = hash * 31 + uv2.GetHashCode();
            }

            return hash;
        }

        private void FindSameMeshInScene()
        {
            var meshes = new Dictionary<int, List<MeshFilter>>();
            var keyToMesh = new Dictionary<int, Mesh>();


            var allObjects = FindObjectsOfType<MeshFilter>(true);

            foreach (var meshFilter in allObjects)
            {
                if (meshFilter.sharedMesh == null) continue;

                var hash = GetHash(meshFilter.sharedMesh);

                if (!meshes.ContainsKey(hash))
                {
                    meshes[hash] = new List<MeshFilter>();
                }

                meshes[hash].Add(meshFilter);

                if (!keyToMesh.ContainsKey(hash))
                {
                    keyToMesh[hash] = meshFilter.sharedMesh;
                }
            }


            foreach (var pair in meshes)
            {
                if (pair.Value.Count > 1)
                {
                    Debug.Log($"Mesh: {pair.Key} has {pair.Value.Count} instances.");

                    foreach (var meshFilter in pair.Value)
                    {
                        Debug.Log($" - {meshFilter.name}");
                        meshFilter.sharedMesh = keyToMesh[pair.Key];
                    }
                }
            }

            EditorSceneManager.MarkSceneDirty(UnityEngine.SceneManagement.SceneManager.GetActiveScene());
        }
        
        
        
        private void FindSameTextureInScene()
        {
            var mats = new Dictionary<int, List<(Material,int)>>();
            var keyToTexture = new Dictionary<int, Texture2D>();


            var allObjects = FindObjectsOfType<MeshRenderer>(true);
            
            var allMaterials = new HashSet<Material>();
            
            foreach (var meshRenderer in allObjects)
            {
                allMaterials.AddRange(meshRenderer.sharedMaterials);
            }
            
            
            Debug.Log($"Found {allMaterials.Count} materials in scene.");

            foreach (var material in allMaterials)
            {
                if (material == null) continue;
                var shader = material.shader;
                if (shader.name != "Standard")
                {
                    continue;
                }
                
                var textureIDs = material.GetTexturePropertyNameIDs();
                
                foreach (var id in textureIDs)
                {
                    var texture = material.GetTexture(id) as Texture2D;

                    if (texture == null) continue;

                    var hash = GetTextureHash(texture);

                    if (!mats.ContainsKey(hash))
                    {
                        mats[hash] = new List<(Material,int)>();
                    }
                    
                    mats[hash].Add( (material, id));

                    keyToTexture.TryAdd(hash, texture);
                }
            }

            foreach (var pair in mats)
            {
                if (pair.Value.Count > 1)
                {
                    Debug.Log($"Texture key: {pair.Key} has {pair.Value.Count} instances.");

                    foreach (var (material, id) in pair.Value)
                    {
                        Debug.Log($" - Material: {material.name}, ID: {id} - Texture: {keyToTexture[pair.Key].name}");
                        material.SetTexture(id, keyToTexture[pair.Key]);
                    }
                }
            } 


            // foreach (var pair in meshes)
            // {
            //     if (pair.Value.Count > 1)
            //     {
            //         Debug.Log($"Mesh: {pair.Key} has {pair.Value.Count} instances.");
            //
            //         foreach (var meshFilter in pair.Value)
            //         {
            //             Debug.Log($" - {meshFilter.name}");
            //             meshFilter.sharedMesh = keyToMesh[pair.Key];
            //         }
            //     }
            // }

            EditorSceneManager.MarkSceneDirty(UnityEngine.SceneManagement.SceneManager.GetActiveScene());
        }

        private int GetTextureHash(Texture2D texture)
        {
            if (texture == null) return 0;

            var path = AssetDatabase.GetAssetPath(texture);
            if (string.IsNullOrEmpty(path))
            {
                Debug.LogWarning($"Texture '{texture.name}' does not have a valid asset path.");
                return 0;
            }
            
            using (var stream = File.OpenRead(path))
            using (var md5 = MD5.Create())
            {
                var hash = md5.ComputeHash(stream);
                return  hash.Aggregate(0, (current, b) => current * 31 + b.GetHashCode());
            }
        }
        
        // private int GetMatHash(Material mat)
        // {
        //     var hash = 17;
        //     
        //     var textureIDs = mat.GetTexturePropertyNameIDs();
        //     foreach (var id in textureIDs)
        //     {
        //         var texture = mat.GetTexture(id);
        //
        //         if (texture is not Texture2D)
        //         {
        //             Debug.Log($"Material {mat.name} has non-Texture2D texture with ID {id}");
        //             continue;
        //         }
        //         
        //         hash = hash * 31 + id.GetHashCode();
        //         if (texture != null)
        //         {
        //             hash = hash * 31 + GetTextureHash(texture as Texture2D);
        //         }
        //     }
        //     
        //     
        //     
        // }

        private void FindSameSkinMeshInScene()
        {
            var meshes = new Dictionary<int, List<SkinnedMeshRenderer>>();
            var keyToMesh = new Dictionary<int, Mesh>();


            var allObjects = GameObject.FindObjectsOfType<SkinnedMeshRenderer>(true);

            foreach (var meshFilter in allObjects)
            {
                if (meshFilter.sharedMesh == null) continue;

                var hash = GetHash(meshFilter.sharedMesh);

                if (!meshes.ContainsKey(hash))
                {
                    meshes[hash] = new List<SkinnedMeshRenderer>();
                }

                meshes[hash].Add(meshFilter);

                if (!keyToMesh.ContainsKey(hash))
                {
                    keyToMesh[hash] = meshFilter.sharedMesh;
                }
            }


            foreach (var pair in meshes)
            {
                if (pair.Value.Count > 1)
                {
                    Debug.Log($"Mesh: {pair.Key} has {pair.Value.Count} instances.");

                    foreach (var meshFilter in pair.Value)
                    {
                        Debug.Log($" - {meshFilter.name}");


                        meshFilter.sharedMesh = keyToMesh[pair.Key];
                    }
                }
            }

            EditorSceneManager.MarkSceneDirty(UnityEngine.SceneManagement.SceneManager.GetActiveScene());
        }
    }
}
