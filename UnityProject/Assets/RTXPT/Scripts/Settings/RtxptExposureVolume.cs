using System;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace PathTracing
{
    /// <summary>
    /// URP <see cref="VolumeComponent"/> that exposes <see cref="RtxptFeature"/>'s exposure /
    /// auto-exposure controls to the Volume framework, so different scenes or world regions can
    /// override them via global or box/sphere-collider Volumes (with blending) instead of editing
    /// the feature's inspector defaults.
    ///
    /// Each parameter has an independent <c>overrideState</c>: when off, the authored inspector value
    /// from <see cref="RtxptSetting"/> is kept; when on, the (blended) Volume value wins. The
    /// authored settings object is never mutated — <see cref="ApplyOverrides"/> returns a per-frame
    /// clone only when at least one override is active.
    /// </summary>
    [Serializable]
    [VolumeComponentMenu("Path Tracing/RTXPT Exposure")]
    [SupportedOnRenderPipeline(typeof(UniversalRenderPipelineAsset))]
    public sealed class RtxptExposureVolume : VolumeComponent
    {
        /// <summary>Enable histogram-free auto-exposure (geometric-mean luminance).</summary>
        public BoolParameter autoExposure = new BoolParameter(false);

        /// <summary>Exposure compensation in stops (EV). Applies in both auto and manual modes.</summary>
        public ClampedFloatParameter exposureCompensation = new ClampedFloatParameter(1.5f, -8f, 8f);

        /// <summary>Auto-exposure lower clamp, in EV.</summary>
        public FloatParameter exposureValueMin = new FloatParameter(-16f);

        /// <summary>Auto-exposure upper clamp, in EV.</summary>
        public FloatParameter exposureValueMax = new FloatParameter(16f);

        /// <summary>Manual exposure value (EV) used to derive shutter/aperture when auto-exposure is off.</summary>
        public FloatParameter exposureValue = new FloatParameter(0f);

        /// <summary>ISO film speed (manual exposure path).</summary>
        public MinFloatParameter filmSpeed = new MinFloatParameter(100f, 1f);

        /// <summary>Aperture f-number (manual exposure path).</summary>
        public MinFloatParameter fNumber = new MinFloatParameter(1f, 0.01f);

        /// <summary>Shutter time (manual exposure path).</summary>
        public MinFloatParameter shutter = new MinFloatParameter(1f, 0.0001f);

        /// <summary>True when at least one parameter is overridden by this (blended) Volume component.</summary>
        public bool IsActive() =>
            autoExposure.overrideState || exposureCompensation.overrideState ||
            exposureValueMin.overrideState || exposureValueMax.overrideState ||
            exposureValue.overrideState || filmSpeed.overrideState ||
            fNumber.overrideState || shutter.overrideState;

        /// <summary>
        /// Overlays the active (blended) exposure overrides onto the per-frame settings. Uses a
        /// clone-on-write pattern: <paramref name="setting"/> is replaced with a clone of
        /// <paramref name="authored"/> the first time any override is applied, so the serialized
        /// inspector object is never mutated and chained volume overrides share one clone. URP updates
        /// <c>VolumeManager.instance.stack</c> for the rendering camera before render passes set up, so
        /// this reflects the camera's Volumes.
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

            var v = mgr.stack.GetComponent<RtxptExposureVolume>();
            if (v == null || !v.IsActive())
                return;

            if (ReferenceEquals(setting, authored))
                setting = authored.Clone();

            var s                                                            = setting;
            if (v.autoExposure.overrideState) s.autoExposure                 = v.autoExposure.value;
            if (v.exposureCompensation.overrideState) s.exposureCompensation = v.exposureCompensation.value;
            if (v.exposureValueMin.overrideState) s.exposureValueMin         = v.exposureValueMin.value;
            if (v.exposureValueMax.overrideState) s.exposureValueMax         = v.exposureValueMax.value;
            if (v.exposureValue.overrideState) s.exposureValue               = v.exposureValue.value;
            if (v.filmSpeed.overrideState) s.filmSpeed                       = v.filmSpeed.value;
            if (v.fNumber.overrideState) s.fNumber                           = v.fNumber.value;
            if (v.shutter.overrideState) s.shutter                           = v.shutter.value;
        }
    }
}