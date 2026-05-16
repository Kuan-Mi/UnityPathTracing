using System;
using Unity.Mathematics;

namespace Nrd
{

    // -----------------------------------------------------------------------
    // SIGMA – shadow / translucency denoiser
    // -----------------------------------------------------------------------
    public sealed class SigmaDenoiser : NrdDenoiser
    {
        private readonly Denoiser _denoiser;

        /// <summary>Per-frame input for SIGMA denoising.</summary>
        public struct FrameInput
        {
            public CommonFrameInput common;
            // SIGMA-specific settings
            public float3 lightDirection;
            /// <summary>Default: 0.02f</summary>
            public float  planeDistanceSensitivity;
            /// <summary>Default: 5</summary>
            public uint   maxStabilizedFrameNum;
        }

        public SigmaDenoiser(string camName, Denoiser denoiser)
            : base(camName, new NrdDenoiserDesc(denoiser))
        {
            if (denoiser != Denoiser.SIGMA_SHADOW && denoiser != Denoiser.SIGMA_SHADOW_TRANSLUCENCY)
                throw new ArgumentException(
                    $"SigmaNrdDenoiser requires a SIGMA_* denoiser, got {denoiser}.", nameof(denoiser));
            _denoiser   = denoiser;
        }

        public IntPtr GetInteropDataPtr(FrameInput fi)
        {
            var data = NrdFrameData._default;
            FillCommonSettings(ref data, fi.common);

            data.denoiserCount = 1;
            ref var entry = ref NrdFrameData.GetEntry(ref data, 0);
            entry.identifier = 0;
            entry.denoiser   = _denoiser;

            var s = SigmaSettings._default;
            s.lightDirection           = fi.lightDirection;
            s.planeDistanceSensitivity = fi.planeDistanceSensitivity > 0f ? fi.planeDistanceSensitivity : 0.02f;
            s.maxStabilizedFrameNum    = fi.maxStabilizedFrameNum > 0 ? fi.maxStabilizedFrameNum : 5;
            entry.Write(s);

            return StoreAndGetPtr(data, fi.common.frameIndex);
        }
    }

}