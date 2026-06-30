using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// URP <see cref="VolumeComponent"/> that exposes <see cref="RtxptFeature"/>'s environment-map
    /// controls to the Volume framework, so different scenes or world regions can override the sky /
    /// HDRI via global or box/sphere-collider Volumes instead of editing the feature's inspector
    /// defaults.
    ///
    /// Each parameter has an independent <c>overrideState</c>: when off, the authored inspector value
    /// from <see cref="RtxptSetting"/> is kept; when on, the (blended) Volume value wins. The
    /// authored settings object is never mutated — <see cref="ApplyOverrides"/> mutates only the
    /// per-frame clone supplied by the caller.
    /// </summary>
    [Serializable]
    [VolumeComponentMenu("Path Tracing/RTXPT Environment Map")]
    [SupportedOnRenderPipeline(typeof(UniversalRenderPipelineAsset))]
    public sealed class RtxptEnvironmentMapVolume : VolumeComponent
    {
        /// <summary>Enable the baked environment map as a distant light / sky.</summary>
        public BoolParameter environmentMapEnabled = new BoolParameter(true);

        /// <summary>HDR environment map: an equirectangular Texture2D or a Cubemap.</summary>
        public TextureParameter environmentMap = new TextureParameter(null);

        /// <summary>Linear intensity multiplier applied to the environment radiance.</summary>
        public MinFloatParameter environmentMapIntensity = new MinFloatParameter(1.0f, 0f);

        public ClampedFloatParameter environmentMapRotationY = new ClampedFloatParameter(0f, 0f, 360f);

        /// <summary>Color tint multiplied into the environment radiance.</summary>
        public ColorParameter environmentMapTint = new ColorParameter(Color.white, hdr: false, showAlpha: false, showEyeDropper: true);

        /// <summary>MIP level sampled for diffuse environment lighting.</summary>
        public ClampedIntParameter environmentMapDiffuseSampleMIPLevel = new ClampedIntParameter(2, 0, 5);

        /// <summary>True when at least one parameter is overridden by this (blended) Volume component.</summary>
        public bool IsActive() =>
            environmentMapEnabled.overrideState || environmentMap.overrideState ||
            environmentMapIntensity.overrideState || environmentMapTint.overrideState ||
            environmentMapRotationY.overrideState ||
            environmentMapDiffuseSampleMIPLevel.overrideState;

        /// <summary>
        /// Overlays the active (blended) environment-map overrides onto the per-frame settings. Uses a
        /// clone-on-write pattern: <paramref name="setting"/> is replaced with a clone of
        /// <paramref name="authored"/> the first time any override is applied, so the serialized
        /// inspector object is never mutated and chained volume overrides share one clone. No-op when
        /// the Volume stack isn't ready or no override is active.
        /// </summary>
        public static void ApplyOverrides(ref RtxptSetting setting, RtxptSetting authored)
        {
            if (authored == null)
                return;

            // VolumeManager isn't ready during early editor/domain-reload frames; querying the stack
            // before it's initialized throws (baseComponentTypeArray). Skip overrides until then.
            var mgr = VolumeManager.instance;
            if (mgr == null || !mgr.isInitialized || mgr.stack == null)
                return;

            var v = mgr.stack.GetComponent<RtxptEnvironmentMapVolume>();
            if (v == null || !v.IsActive())
                return;

            if (ReferenceEquals(setting, authored))
                setting = authored.Clone();

            var s                                                                                          = setting;
            if (v.environmentMapEnabled.overrideState) s.environmentMapEnabled                             = v.environmentMapEnabled.value;
            if (v.environmentMap.overrideState) s.environmentMap                                           = v.environmentMap.value;
            if (v.environmentMapIntensity.overrideState) s.environmentMapIntensity                         = v.environmentMapIntensity.value;
            if (v.environmentMapRotationY.overrideState) s.environmentMapRotationY                         = v.environmentMapRotationY.value;
            if (v.environmentMapTint.overrideState) s.environmentMapTint                                   = v.environmentMapTint.value;
            if (v.environmentMapDiffuseSampleMIPLevel.overrideState) s.environmentMapDiffuseSampleMIPLevel = v.environmentMapDiffuseSampleMIPLevel.value;
        }
    }
}