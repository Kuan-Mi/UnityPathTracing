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

    public MeshRenderer Renderer => GetComponent<MeshRenderer>();
    public MeshFilter Filter => GetComponent<MeshFilter>();
    public Material[] Materials => Renderer ? Renderer.sharedMaterials : null;
    public Mesh Mesh => Filter ? Filter.sharedMesh : null;
}
