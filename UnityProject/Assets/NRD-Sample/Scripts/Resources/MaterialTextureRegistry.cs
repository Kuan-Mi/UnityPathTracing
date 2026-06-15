using System.Collections.Generic;
using Rendering.Resources;
using UnityEngine;

namespace NativeRender
{
    internal sealed class MaterialTextureRegistry
    {
        private const int TexturesPerMaterial = NRDSampleResource.TexturesPerMaterial;

        private readonly Dictionary<Material, int> _materialSlots = new();
        private readonly Dictionary<Material, int> _materialRefCounts = new();
        private readonly Queue<int> _freeMatSlots = new();

        public BindlessTexture Textures { get; private set; }

        public void UploadTextureArray(List<Texture> texPtrs, bool preserveTextures)
        {
            if (preserveTextures) return;

            int texCount = Mathf.Max(texPtrs.Count, 1);
            Textures = new BindlessTexture(texCount);
            for (int i = 0; i < texPtrs.Count; i++)
                Textures[i] = texPtrs[i];
        }

        public void DisposeTextures(bool preserveTextures)
        {
            if (preserveTextures) return;

            Textures?.Dispose();
            Textures = null;
        }

        public void ClearSlots(bool preserveTextures)
        {
            if (preserveTextures) return;

            _materialSlots.Clear();
            _materialRefCounts.Clear();
            _freeMatSlots.Clear();
        }

        /// <summary>
        /// Returns the material slot index for <paramref name="matData"/>, registering it if new.
        /// Bulk mode appends textures to <paramref name="texPtrs"/>; incremental mode writes into
        /// the live bindless texture array and reuses freed material slots.
        /// </summary>
        public int GetOrAdd(SubmeshMaterialData matData, List<Texture> texPtrs)
        {
            Material mat = matData?.material;

            if (mat != null && _materialSlots.TryGetValue(mat, out int existing))
            {
                _materialRefCounts[mat] = (_materialRefCounts.TryGetValue(mat, out int rc) ? rc : 0) + 1;
                return existing;
            }

            int slot;
            if (_freeMatSlots.Count > 0)
                slot = _freeMatSlots.Dequeue();
            else if (texPtrs != null)
                slot = _materialSlots.Count;
            else
                slot = Textures != null ? Textures.Capacity / TexturesPerMaterial : _materialSlots.Count;

            if (mat != null)
            {
                _materialSlots[mat]     = slot;
                _materialRefCounts[mat] = 1;
            }

            if (texPtrs != null && matData != null)
            {
                for (int i = 0; i < TexturesPerMaterial; i++)
                    texPtrs.Add(GetTextureOrDefault(matData, i));
            }
            else if (Textures != null && matData != null)
            {
                int baseIndex = slot * TexturesPerMaterial;
                int need      = baseIndex + TexturesPerMaterial;
                if (need > Textures.Capacity)
                    Textures.Resize(need);

                for (int i = 0; i < TexturesPerMaterial; i++)
                    Textures[baseIndex + i] = GetTextureOrDefault(matData, i);
            }

            return slot;
        }

        public void Release(Material mat)
        {
            if (mat == null || !_materialSlots.TryGetValue(mat, out int slotIdx)) return;

            int newRc = _materialRefCounts.GetValueOrDefault(mat, 1) - 1;
            if (newRc > 0)
            {
                _materialRefCounts[mat] = newRc;
                return;
            }

            _materialSlots.Remove(mat);
            _materialRefCounts.Remove(mat);
            _freeMatSlots.Enqueue(slotIdx);

            if (Textures != null)
            {
                int baseIndex = slotIdx * TexturesPerMaterial;
                for (int i = 0; i < TexturesPerMaterial; i++)
                    Textures[baseIndex + i] = null;
            }
        }

        private static Texture GetTextureOrDefault(SubmeshMaterialData matData, int index)
        {
            Texture tex = matData.textures[index];
            if (tex != null) return tex;

            return index switch
            {
                0 => Texture2D.whiteTexture,
                1 => Texture2D.blackTexture,
                2 => Texture2D.normalTexture,
                3 => Texture2D.blackTexture,
                _ => null,
            };
        }
    }
}
