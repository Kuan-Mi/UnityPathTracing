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

        public unsafe IntPtr GetInteropDataPtr(CommonSettings common, RelaxSettings settings)
        {
            var data = NrdFrameData._default;
            data.instanceId     = _nrdInstanceId;
            data.width          = common.resourceSize[0];
            data.height         = common.resourceSize[1];
            data.commonSettings = common;

            data.denoiserCount = 1;
            ref var entry = ref NrdFrameData.GetEntry(ref data, 0);
            entry.identifier = 0;
            entry.denoiser   = _denoiser;
            entry.Write(settings);

            return StoreAndGetPtr(data, common.frameIndex);
        }
    }
}