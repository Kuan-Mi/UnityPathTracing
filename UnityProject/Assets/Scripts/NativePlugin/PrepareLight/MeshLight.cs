using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using RTXDI;

[DisallowMultipleComponent]
[ExecuteAlways]
public class MeshLight : MonoBehaviour
{
    private void OnEnable()
    {
        GPUScene.RegisterMeshLight(this);
    }

    private void OnDisable()
    {
        GPUScene.UnregisterMeshLight(this);
    }

    private List<Color> lastEmitColors = new List<Color>();
    private List<Texture2D> lastEmitTextures = new List<Texture2D>();
    
    
    private void Update()
    {
        var renderer = Renderer;
        if (renderer)
        {
            var materials = renderer.sharedMaterials;
            var emitColors = new List<Color>();
            var emitTextures = new List<Texture2D>();
            
            foreach (var mat in materials)
            {
                if (mat.HasProperty("_EmissionColor"))
                {
                    emitColors.Add(mat.GetColor("_EmissionColor"));
                }
                else
                {
                    emitColors.Add(Color.black);
                }

                if (mat.HasProperty("_EmissionMap"))
                {
                    emitTextures.Add(mat.GetTexture("_EmissionMap") as Texture2D);
                }
                else
                {
                    emitTextures.Add(null);
                }
            }
            
            if (!emitColors.SequenceEqual(lastEmitColors) || !emitTextures.SequenceEqual(lastEmitTextures))
            {
                GPUScene.Instance?.MarkSceneDirty();
                lastEmitColors = emitColors;
                lastEmitTextures = emitTextures;
            }
        }
    }

    public MeshRenderer Renderer => GetComponent<MeshRenderer>();
    public MeshFilter Filter => GetComponent<MeshFilter>();
    public Material[] Materials => Renderer ? Renderer.sharedMaterials : null;
    public Mesh Mesh => Filter ? Filter.sharedMesh : null;
}
