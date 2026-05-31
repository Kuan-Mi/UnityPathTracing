using System;
using System.Collections.Generic;
using UnityEngine;

namespace NativeRender
{
    /// <summary>
    /// A reusable static-sampler definition, authored once as a standalone asset and
    /// referenced by any number of native shaders. Editing this asset updates the static
    /// sampler of every shader that references it — the configuration lives in one place
    /// instead of being duplicated inline on each shader (the old <c>SamplerHint</c> override).
    ///
    /// Create via <c>Assets &gt; Create &gt; Native Render &gt; Sampler</c>. A shader's reflected
    /// <c>SamplerState</c> picks one of these in its importer inspector; at pipeline-build time
    /// the reference is resolved live (see <see cref="ToHint"/>) and sent to the native plugin,
    /// so changes are picked up on the next pipeline rebuild without reimporting the shader.
    /// </summary>
    [CreateAssetMenu(fileName = "Sampler", menuName = "Native Render/Sampler")]
    public sealed class NativeSampler : ScriptableObject
    {
        public SamplerFilter  Filter        = SamplerFilter.Linear;

        [Tooltip("Address (wrap) mode for the U/V/W texture axes. Usually all three are equal; " +
                 "they differ only for cases like an equirectangular sampler (U=Wrap, V/W=Clamp).")]
        public SamplerAddress AddressU      = SamplerAddress.Clamp;
        public SamplerAddress AddressV      = SamplerAddress.Clamp;
        public SamplerAddress AddressW      = SamplerAddress.Clamp;

        [Tooltip("Sample mip levels (MaxLOD 16) when true; clamp to mip 0 otherwise.")]
        public bool           Mips          = false;

        [Tooltip("Max anisotropy; used only when Filter is Anisotropic.")]
        public uint           MaxAnisotropy = 16;

        /// <summary>Resolves this definition into the wire-format <see cref="SamplerHint"/> for the
        /// given HLSL sampler variable name (which the native plugin matches by name).</summary>
        public SamplerHint ToHint(string hlslName) => new SamplerHint
        {
            Name          = hlslName,
            Filter        = Filter,
            AddressU      = AddressU,
            AddressV      = AddressV,
            AddressW      = AddressW,
            Mips          = Mips,
            MaxAnisotropy = MaxAnisotropy,
        };
    }

    /// <summary>
    /// Binds one of a shader's reflected HLSL samplers (by variable name) to a shared
    /// <see cref="NativeSampler"/> asset. Authored on the shader importer and serialized onto the
    /// shader asset. A null <see cref="Sampler"/> means "no override" — the native plugin then
    /// infers the sampler attributes from the variable name (sampler_LinearClamp, …).
    /// </summary>
    [Serializable]
    public struct SamplerBinding
    {
        /// <summary>HLSL sampler variable name to match exactly.</summary>
        public string        Name;
        /// <summary>The shared sampler definition; null falls back to native name-inference.</summary>
        public NativeSampler Sampler;
    }

    /// <summary>
    /// Resolves an array of <see cref="SamplerBinding"/> (HLSL name + shared sampler asset)
    /// into the wire-format <see cref="SamplerHint"/>[] consumed by the pipelines' BuildHintsJson.
    /// Bindings with no assigned sampler are skipped, leaving them to native name-inference.
    /// Shared by all three shader assets so the resolution rule lives in one place.
    /// </summary>
    internal static class SamplerBindingResolver
    {
        public static SamplerHint[] Resolve(SamplerBinding[] bindings)
        {
            if (bindings == null || bindings.Length == 0) return Array.Empty<SamplerHint>();
            var hints = new List<SamplerHint>(bindings.Length);
            foreach (var b in bindings)
                if (b.Sampler != null && !string.IsNullOrEmpty(b.Name))
                    hints.Add(b.Sampler.ToHint(b.Name));
            return hints.Count == 0 ? Array.Empty<SamplerHint>() : hints.ToArray();
        }
    }
}
