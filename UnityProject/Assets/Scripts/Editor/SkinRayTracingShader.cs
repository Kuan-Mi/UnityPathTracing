using System;
using UnityEditor;
using UnityEditor.Rendering;
using UnityEditor.Rendering.Universal.ShaderGUI;
using UnityEngine;
using UnityEngine.Rendering;

public class SkinRayTracingShader : BaseShaderGUI
{
    // Skin-specific properties
    private MaterialProperty smoothnessProp;
    private MaterialProperty metallicProp;
    private MaterialProperty metallicGlossMapProp;
    private MaterialProperty bumpScaleProp;
    private MaterialProperty bumpMapProp;
    private MaterialProperty microNormalMapProp;
    private MaterialProperty microNormalStrengthProp;
    private MaterialProperty microNormalTilingProp;
    private MaterialProperty SSSProp;
    private MaterialProperty SkinnedMeshProp;

    // collect properties from the material properties
    public override void FindProperties(MaterialProperty[] properties)
    {
        base.FindProperties(properties);
        smoothnessProp          = FindProperty("_Smoothness",          properties, false);
        metallicProp            = FindProperty("_Metallic",            properties, false);
        metallicGlossMapProp    = FindProperty("_MetallicGlossMap",    properties, false);
        bumpScaleProp           = FindProperty("_BumpScale",           properties, false);
        bumpMapProp             = FindProperty("_BumpMap",             properties, false);
        microNormalMapProp      = FindProperty("_MicroNormalMap",      properties, false);
        microNormalStrengthProp = FindProperty("_MicroNormalStrength", properties, false);
        microNormalTilingProp   = FindProperty("_MicroNormalTiling",   properties, false);
        SSSProp                 = FindProperty("_SSS",                 properties, false);
        SkinnedMeshProp         = FindProperty("_SKINNEDMESH",         properties, false);
    }

    // material changed check
    public override void ValidateMaterial(Material material)
    {
        SetMaterialKeywords(material);
        CoreUtils.SetKeyword(material, "_SSS",         material.HasProperty("_SSS")         && material.GetFloat("_SSS")         > 0.5f);
        CoreUtils.SetKeyword(material, "_SKINNEDMESH", material.HasProperty("_SKINNEDMESH") && material.GetFloat("_SKINNEDMESH") > 0.5f);
    }

    // material main surface inputs
    public override void DrawSurfaceInputs(Material material)
    {
        base.DrawSurfaceInputs(material);

        // Metallic / Smoothness
        if (metallicGlossMapProp != null)
        {
            materialEditor.TexturePropertySingleLine(
                new GUIContent("Metallic Map", "Metallic (R) and Smoothness (A) map."),
                metallicGlossMapProp,
                metallicGlossMapProp.textureValue != null ? null : metallicProp);
        }

        if (smoothnessProp != null)
            materialEditor.ShaderProperty(smoothnessProp,
                new GUIContent("Smoothness", "Controls the spread of highlights and reflections on the surface."));

        // Normal map
        if (bumpMapProp != null)
        {
            materialEditor.TexturePropertySingleLine(
                new GUIContent("Normal Map"),
                bumpMapProp,
                bumpMapProp.textureValue != null ? bumpScaleProp : null);
        }

        // Micro Normal (skin pore detail) section
        if (microNormalMapProp != null)
        {
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Micro Normal (Skin Pore Detail)", EditorStyles.boldLabel);
            materialEditor.TexturePropertySingleLine(
                new GUIContent("Micro Normal Map", "Micro-detail normal map for skin pore simulation."),
                microNormalMapProp,
                microNormalStrengthProp);

            if (microNormalTilingProp != null)
                materialEditor.ShaderProperty(microNormalTilingProp,
                    new GUIContent("Micro Normal Tiling", "UV tiling multiplier for the micro normal map."));
        }
    }

    // material main advanced options
    public override void DrawAdvancedOptions(Material material)
    {
        if (SSSProp != null)
            materialEditor.ShaderProperty(SSSProp, "SSS (Ray Tracing)");

        if (SkinnedMeshProp != null)
            materialEditor.ShaderProperty(SkinnedMeshProp, "Skinned Mesh (Ray Tracing)");

        base.DrawAdvancedOptions(material);
    }

    public override void AssignNewShaderToMaterial(Material material, Shader oldShader, Shader newShader)
    {
        if (material == null)
            throw new ArgumentNullException("material");

        base.AssignNewShaderToMaterial(material, oldShader, newShader);
    }
}
