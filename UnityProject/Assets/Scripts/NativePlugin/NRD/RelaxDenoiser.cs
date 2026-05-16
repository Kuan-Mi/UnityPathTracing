using System;
using System.Collections.Generic;

namespace Nrd
{
    // -----------------------------------------------------------------------
    // RELAX – spatiotemporal accumulation denoiser
    // -----------------------------------------------------------------------
    public sealed class RelaxDenoiser : NrdDenoiser
    {
        private readonly Denoiser _denoiser;

        /// <summary>Per-frame input for RELAX denoising.</summary>
        public struct FrameInput
        {
            public CommonFrameInput common;
            // RELAX currently uses default settings.
            // Extend this struct if per-frame tuning is needed.
        }

        private static readonly HashSet<Denoiser> ValidDenoisers = new()
        {
            Denoiser.RELAX_DIFFUSE,
            Denoiser.RELAX_DIFFUSE_SH,
            Denoiser.RELAX_SPECULAR,
            Denoiser.RELAX_SPECULAR_SH,
            Denoiser.RELAX_DIFFUSE_SPECULAR,
            Denoiser.RELAX_DIFFUSE_SPECULAR_SH,
        };

        public RelaxDenoiser(string camName, Denoiser denoiser)
            : base(camName, new NrdDenoiserDesc(denoiser))
        {
            if (!ValidDenoisers.Contains(denoiser))
                throw new ArgumentException(
                    $"RelaxNrdDenoiser requires a RELAX_* denoiser, got {denoiser}.", nameof(denoiser));
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
            entry.Write(RelaxSettings._default);

            return StoreAndGetPtr(data, fi.common.frameIndex);
        }
    }
}