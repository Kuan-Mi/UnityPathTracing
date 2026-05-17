using System;
using UnityEditor;
using UnityEditor.Rendering;
using UnityEditor.Rendering.Universal.ShaderGUI;
using UnityEngine;
using UnityEngine.Rendering;

public class LitRayTracingShader : BaseShaderGUI
{
    static readonly string[] workflowModeNames = Enum.GetNames(typeof(LitGUI.WorkflowMode));

    private LitGUI.LitProperties litProperties;
    private LitRayDetailGUI.LitProperties litDetailProperties;
    private MaterialProperty SSSProp;
    private MaterialProperty SSSScatteringColorProp;
    private MaterialProperty SSSScatteringScaleProp;
    private MaterialProperty SkinnedMeshProp;

    public override void FillAdditionalFoldouts(MaterialHeaderScopeList materialScopesList)
    {
        materialScopesList.RegisterHeaderScope(LitRayDetailGUI.Styles.detailInputs, Expandable.Details, _ => LitRayDetailGUI.DoDetailArea(litDetailProperties, materialEditor));
    }

    // collect properties from the material properties
    public override void FindProperties(MaterialProperty[] properties)
    {
        base.FindProperties(properties);
        litProperties = new LitGUI.LitProperties(properties);
        litDetailProperties = new LitRayDetailGUI.LitProperties(properties);
        SSSProp = FindProperty("_SSS", properties, false);
        SSSScatteringColorProp = FindProperty("_SSSScatteringColor", properties, false);
        SSSScatteringScaleProp = FindProperty("_SSSScatteringScale", properties, false);
        SkinnedMeshProp = FindProperty("_SKINNEDMESH", properties, false);
    }

    // material changed check
    public override void ValidateMaterial(Material material)
    {
        SetMaterialKeywords(material, LitGUI.SetMaterialKeywords, LitRayDetailGUI.SetMaterialKeywords);
        CoreUtils.SetKeyword(material, "_SSS", material.HasProperty("_SSS") && material.GetFloat("_SSS") > 0.5f);
        CoreUtils.SetKeyword(material, "_SKINNEDMESH", material.HasProperty("_SKINNEDMESH") && material.GetFloat("_SKINNEDMESH") > 0.5f);

        // GBufferRaster Y-flip: swap Front<->Back, keep Off unchanged
        if (material.HasProperty("_Cull") && material.HasProperty("_CullGBuffer"))
        {
            int cull = (int)material.GetFloat("_Cull");
            int cullGBuffer = cull == 0 ? 0 : (3 - cull); // 1->2, 2->1, 0->0
            material.SetFloat("_CullGBuffer", cullGBuffer);
        }
    }

    // material main surface options
    public override void DrawSurfaceOptions(Material material)
    {
        // Use default labelWidth
        EditorGUIUtility.labelWidth = 0f;

        if (litProperties.workflowMode != null)
            DoPopup(LitGUI.Styles.workflowModeText, litProperties.workflowMode, workflowModeNames);

        base.DrawSurfaceOptions(material);
        
        if (SSSProp != null)
            materialEditor.ShaderProperty(SSSProp, "SSS (Ray Tracing)");

        if (SkinnedMeshProp != null)
            materialEditor.ShaderProperty(SkinnedMeshProp, "Skinned Mesh (Ray Tracing)");
    }

    // material main surface inputs
    public override void DrawSurfaceInputs(Material material)
    {
        base.DrawSurfaceInputs(material);
        LitGUI.Inputs(litProperties, materialEditor, material);
        DrawEmissionProperties(material, true);
        DrawTileOffset(materialEditor, baseMapProp);
        

        if (SSSProp != null)
        {
            if (SSSScatteringColorProp != null && material.GetFloat("_SSS") > 0.5f)
                materialEditor.ShaderProperty(SSSScatteringColorProp, "SSS Scattering Color");
            if (SSSScatteringScaleProp != null && material.GetFloat("_SSS") > 0.5f)
                materialEditor.ShaderProperty(SSSScatteringScaleProp, "SSS Scattering Scale");
        }
    }

    // material main advanced options
    public override void DrawAdvancedOptions(Material material)
    {
        if (litProperties.reflections != null && litProperties.highlights != null)
        {
            materialEditor.ShaderProperty(litProperties.highlights, LitGUI.Styles.highlightsText);
            materialEditor.ShaderProperty(litProperties.reflections, LitGUI.Styles.reflectionsText);
        }

        base.DrawAdvancedOptions(material);
    }

    public override void AssignNewShaderToMaterial(Material material, Shader oldShader, Shader newShader)
    {
        if (material == null)
            throw new ArgumentNullException("material");

        // _Emission property is lost after assigning Standard shader to the material
        // thus transfer it before assigning the new shader
        if (material.HasProperty("_Emission"))
        {
            material.SetColor("_EmissionColor", material.GetColor("_Emission"));
        }

        base.AssignNewShaderToMaterial(material, oldShader, newShader);

        if (oldShader == null || !oldShader.name.Contains("Legacy Shaders/"))
        {
            SetupMaterialBlendMode(material);
            return;
        }

        SurfaceType surfaceType = SurfaceType.Opaque;
        BlendMode blendMode = BlendMode.Alpha;
        if (oldShader.name.Contains("/Transparent/Cutout/"))
        {
            surfaceType = SurfaceType.Opaque;
            material.SetFloat("_AlphaClip", 1);
        }
        else if (oldShader.name.Contains("/Transparent/"))
        {
            // NOTE: legacy shaders did not provide physically based transparency
            // therefore Fade mode
            surfaceType = SurfaceType.Transparent;
            blendMode = BlendMode.Alpha;
        }

        material.SetFloat("_Blend", (float)blendMode);

        material.SetFloat("_Surface", (float)surfaceType);
        if (surfaceType == SurfaceType.Opaque)
        {
            material.DisableKeyword("_SURFACE_TYPE_TRANSPARENT");
        }
        else
        {
            material.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
        }

        if (oldShader.name.Equals("Standard (Specular setup)"))
        {
            material.SetFloat("_WorkflowMode", (float)LitGUI.WorkflowMode.Specular);
            Texture texture = material.GetTexture("_SpecGlossMap");
            if (texture != null)
                material.SetTexture("_MetallicSpecGlossMap", texture);
        }
        else
        {
            material.SetFloat("_WorkflowMode", (float)LitGUI.WorkflowMode.Metallic);
            Texture texture = material.GetTexture("_MetallicGlossMap");
            if (texture != null)
                material.SetTexture("_MetallicSpecGlossMap", texture);
        }
    }
}