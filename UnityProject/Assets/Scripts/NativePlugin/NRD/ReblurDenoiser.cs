using System;
using System.Collections.Generic;

namespace Nrd
{
    // -----------------------------------------------------------------------
    // REBLUR – diffuse / specular radiance + hit-distance denoiser
    // -----------------------------------------------------------------------
    public sealed class ReblurDenoiser : NrdDenoiser
    {
        private readonly Denoiser _denoiser;

        /// <summary>Per-frame input for REBLUR denoising.</summary>
        public struct FrameInput
        {
            public CommonFrameInput common;

            // REBLUR-specific settings
            public CheckerboardMode              checkerboardMode;
            public HitDistanceReconstructionMode hitDistanceReconstructionMode;
            public uint                          maxAccumulatedFrameNum;
            public uint                          maxFastAccumulatedFrameNum;
            public uint                          maxStabilizedFrameNum;

            /// <summary>Default: 1.5f</summary>
            public float fastHistoryClampingSigmaScale;
        }

        private static readonly HashSet<Denoiser> ValidDenoisers = new()
        {
            Denoiser.REBLUR_DIFFUSE,
            Denoiser.REBLUR_DIFFUSE_OCCLUSION,
            Denoiser.REBLUR_DIFFUSE_SH,
            Denoiser.REBLUR_SPECULAR,
            Denoiser.REBLUR_SPECULAR_OCCLUSION,
            Denoiser.REBLUR_SPECULAR_SH,
            Denoiser.REBLUR_DIFFUSE_SPECULAR,
            Denoiser.REBLUR_DIFFUSE_SPECULAR_OCCLUSION,
            Denoiser.REBLUR_DIFFUSE_SPECULAR_SH,
            Denoiser.REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION,
        };

        public ReblurDenoiser(string camName, Denoiser denoiser)
            : base(camName, new NrdDenoiserDesc(denoiser))
        {
            if (!ValidDenoisers.Contains(denoiser))
                throw new ArgumentException(
                    $"ReblurNrdDenoiser requires a REBLUR_* denoiser, got {denoiser}.", nameof(denoiser));
            _denoiser = denoiser;
        }

        public IntPtr GetInteropDataPtr(FrameInput fi)
        {
            var data = NrdFrameData._default;
            FillCommonSettings(ref data, fi.common);

            data.denoiserCount = 1;
            ref var entry = ref NrdFrameData.GetEntry(ref data, 0);
            entry.identifier = 0;
            entry.denoiser   = _denoiser;

            var s = ReblurSettings._default;
            s.checkerboardMode              = fi.checkerboardMode;
            s.minMaterialForDiffuse         = 0;
            s.minMaterialForSpecular        = 1;
            s.hitDistanceReconstructionMode = fi.hitDistanceReconstructionMode;
            s.maxAccumulatedFrameNum        = fi.maxAccumulatedFrameNum;
            s.maxFastAccumulatedFrameNum    = fi.maxFastAccumulatedFrameNum;
            s.maxStabilizedFrameNum         = fi.maxStabilizedFrameNum;
            s.fastHistoryClampingSigmaScale = fi.fastHistoryClampingSigmaScale > 0f ? fi.fastHistoryClampingSigmaScale : 1.5f;
            entry.Write(s);

            return StoreAndGetPtr(data, fi.common.frameIndex);
        }
    }
}