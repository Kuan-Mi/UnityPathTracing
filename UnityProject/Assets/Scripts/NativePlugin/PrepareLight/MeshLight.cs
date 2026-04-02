using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using RTXDI;

[DisallowMultipleComponent]
[ExecuteAlways]
public class MeshLight : MonoBehaviour
{
    private static readonly int EmissionColor = Shader.PropertyToID("_EmissionColor");
    private static readonly int EmissionMap = Shader.PropertyToID("_EmissionMap");

    private void OnEnable()
    {
        GPUScene.RegisterMeshLight(this);
    }

    private void OnDisable()
    {
        GPUScene.UnregisterMeshLight(this);
    }

    private List<Color> lastEmitColors = new List<Color>();
    private List<Texture> lastEmitTextures = new List<Texture>();
    

    private List<Color> emitColors = new List<Color>();
    private List<Texture> emitTextures = new List<Texture>();
    
    
    private void Update()
    {
        if (!Renderer) return;
        
        var materials = Renderer.sharedMaterials;
        emitColors.Clear();
        emitTextures.Clear();
            
        foreach (var mat in materials)
        {
            emitColors.Add(mat.HasProperty(EmissionColor) ? mat.GetColor(EmissionColor) : Color.black);
            emitTextures.Add(mat.HasProperty(EmissionMap) ? mat.GetTexture(EmissionMap) : null);
        }
            
        if (!emitColors.SequenceEqual(lastEmitColors) || !emitTextures.SequenceEqual(lastEmitTextures))
        {
            GPUScene.Instance?.MarkSceneDirty();
            lastEmitColors = emitColors;
            lastEmitTextures = emitTextures;
        }
    }
    
    private MeshRenderer m_renderer;

    public MeshRenderer Renderer => m_renderer ? m_renderer : (m_renderer = GetComponent<MeshRenderer>());
    public MeshFilter Filter => GetComponent<MeshFilter>();
    public Material[] Materials => Renderer ? Renderer.sharedMaterials : null;
    public Mesh Mesh => Filter ? Filter.sharedMesh : null;
}
