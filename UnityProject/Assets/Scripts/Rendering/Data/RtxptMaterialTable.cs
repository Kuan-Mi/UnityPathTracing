using System;
using System.Collections.Generic;
using UnityEngine;

namespace PathTracing
{
    /// <summary>
    /// Per-rebuild builder for the PT material array and the bindless texture pointer list.
    /// Dedups on the shared <see cref="RtxptMaterial"/> asset (identical references collapse to
    /// one PTMaterialData entry, keeping MaterialCount = unique materials, matching the C++
    /// baker, Sample.cpp:2095) and on Texture assets for bindless slots.
    ///
    /// <see cref="RefreshScalars"/> is the single source of truth for every scalar/color/flag
    /// field: the initial bake (<see cref="GetOrAdd"/>) sets only the texture indices and then
    /// runs it, and the lightweight per-frame material-edit path runs it again on the existing
    /// entry — so the two can never drift apart.
    /// </summary>
    internal sealed class RtxptMaterialTable
    {
        private readonly Dictionary<int, int> _materialIndexByAssetId = new(); // RtxptMaterial id → Materials index
        private readonly Dictionary<int, int> _textureSlotByAssetId   = new(); // Texture id → bindless slot

        public readonly List<PTMaterialData> Materials   = new();
        public readonly List<IntPtr>         TexturePtrs = new();

        /// <summary>Returns the material index for the asset, baking it on first sight.</summary>
        public int GetOrAdd(RtxptMaterial asset)
        {
            int assetId = asset.GetInstanceID();
            if (_materialIndexByAssetId.TryGetValue(assetId, out int existing))
                return existing;

            // Texture registration order matches the original baker (base, ORM, normal,
            // emissive, transmission) so bindless slot assignment stays PIX-comparable.
            var data = new PTMaterialData
            {
                _padding0                        = 42,
                _padding1                        = 42f,
                BaseOrDiffuseTextureIndex        = AddTexture(asset.BaseOrDiffuseTexture),
                MetalRoughOrSpecularTextureIndex = AddTexture(asset.OcclusionRoughnessMetallicTexture),
                NormalTextureIndex               = AddTexture(asset.NormalTexture),
                EmissiveTextureIndex             = AddTexture(asset.EmissiveTexture),
                TransmissionTextureIndex         = AddTexture(asset.TransmissionTexture),
                // C++ FillData never writes this field; PTMaterialData is zero-initialized so it
                // stays 0 (occlusion is packed into ORM, UseOcclusionTexture is disabled).
                // See MaterialsBaker.cpp:516/907.
                OcclusionTextureIndex            = 0u,
            };
            RefreshScalars(asset, ref data);

            int idx = Materials.Count;
            Materials.Add(data);
            _materialIndexByAssetId[assetId] = idx;
            return idx;
        }

        private uint AddTexture(Texture tex)
        {
            if (tex == null) return 0xFFFFFFFFu;
            int texId = tex.GetInstanceID();
            if (_textureSlotByAssetId.TryGetValue(texId, out int slot)) return (uint)slot;
            slot = TexturePtrs.Count;
            TexturePtrs.Add(tex.GetNativeTexturePtr());
            _textureSlotByAssetId[texId] = slot;
            return (uint)slot;
        }

        /// <summary>
        /// Writes all scalar/color/flag fields of <paramref name="data"/> from <paramref name="slot"/>,
        /// preserving the texture index fields (set only during a full scene rebuild). Texture-use
        /// flags combine the loaded state (index != invalid) with the slot's enable toggles.
        /// </summary>
        public static void RefreshScalars(RtxptMaterial slot, ref PTMaterialData data)
        {
            uint flags = 0;
            if (slot.UseSpecularGlossModel) flags |= PTMaterialFlags.UseSpecularGlossModel;
            if (data.BaseOrDiffuseTextureIndex        != 0xFFFFFFFFu && slot.EnableBaseTexture)                                    flags |= PTMaterialFlags.UseBaseOrDiffuseTexture;
            if (data.MetalRoughOrSpecularTextureIndex != 0xFFFFFFFFu && slot.EnableOcclusionRoughnessMetallicTexture)              flags |= PTMaterialFlags.UseMetalRoughOrSpecularTexture;
            if (data.EmissiveTextureIndex             != 0xFFFFFFFFu && slot.EnableEmissiveTexture)                                flags |= PTMaterialFlags.UseEmissiveTexture;
            if (data.NormalTextureIndex               != 0xFFFFFFFFu && slot.EnableNormalTexture)                                  flags |= PTMaterialFlags.UseNormalTexture;
            if (data.TransmissionTextureIndex         != 0xFFFFFFFFu && slot.EnableTransmissionTexture && slot.EnableTransmission) flags |= PTMaterialFlags.UseTransmissionTexture;
            if (slot.MetalnessInRedChannel)                       flags |= PTMaterialFlags.MetalnessInRedChannel;
            if (slot.ThinSurface || !slot.EnableTransmission)     flags |= PTMaterialFlags.ThinSurface;
            if (slot.PSDExclude)                                  flags |= PTMaterialFlags.PSDExclude;
            if (slot.EnableAsAnalyticLightProxy)                  flags |= PTMaterialFlags.EnableAsAnalyticLightProxy;
            if (slot.IgnoreMeshTangentSpace)                      flags |= PTMaterialFlags.IgnoreMeshTangentSpace;
            if (slot.PSDBlockMotionVectorsAtSurfaceType % 2 != 0) flags |= PTMaterialFlags.PSDBlockMVsAtSurfaceTypeB0;
            if (slot.PSDBlockMotionVectorsAtSurfaceType / 2 != 0) flags |= PTMaterialFlags.PSDBlockMVsAtSurfaceTypeB1;
            flags |= (uint)Mathf.Clamp(slot.NestedPriority, 0, 14) << PTMaterialFlags.NestedPriorityShift;
            flags |= (uint)Mathf.Clamp(slot.PSDDominantDeltaLobe + 1, 0, 7) << PTMaterialFlags.PSDDominantDeltaLobeP1Shift;

            data.Flags                     = flags;
            data.BaseOrDiffuseColor        = new Vector3(slot.BaseColorFactor.r, slot.BaseColorFactor.g, slot.BaseColorFactor.b);
            data.SpecularColor             = new Vector3(slot.SpecularColor.r, slot.SpecularColor.g, slot.SpecularColor.b);
            data.EmissiveColor             = new Vector3(slot.EmissiveColor.r, slot.EmissiveColor.g, slot.EmissiveColor.b) * slot.EmissiveIntensity;
            data.ShadowNoLFadeout          = Mathf.Clamp(slot.ShadowNoLFadeout, 0f, 0.25f);
            data.Opacity                   = slot.Opacity;
            data.Roughness                 = slot.Roughness;
            data.Metalness                 = slot.Metalness;
            data.NormalTextureScale        = slot.NormalTextureScale;
            data.AlphaCutoff               = slot.AlphaCutoff;
            data.TransmissionFactor        = slot.EnableTransmission ? slot.TransmissionFactor        : 0f;
            data.DiffuseTransmissionFactor = slot.EnableTransmission ? slot.DiffuseTransmissionFactor : 0f;
            data.IoR                       = slot.IoR;
            data.ThicknessFactor           = slot.ThicknessFactor;
            data.VolumeAttenuationColor    = new Vector3(slot.VolumeAttenuationColor.r, slot.VolumeAttenuationColor.g, slot.VolumeAttenuationColor.b);
            data.VolumeAttenuationDistance = slot.VolumeAttenuationDistance;
        }
    }
}
